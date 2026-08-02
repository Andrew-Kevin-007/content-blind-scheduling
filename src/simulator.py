"""Discrete-event simulator for a continuous-batching LLM inference replica.

Models a vLLM-style engine with iteration-level scheduling: requests are
admitted into a running batch subject to a KV-cache capacity bound, a prefill
step is charged when new requests are admitted, and every decode iteration
emits one token for each request in the batch.

The scheduler's only lever is ADMISSION ORDER. That is deliberate: it isolates
the effect of knowing (or predicting) output length, which is the paper's
question, from the confound of preemption machinery.

The cost model is analytical, not measured on our own hardware. Every headline
number is therefore repeated under the sensitivity sweep in config.SENSITIVITY.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass, field

import numpy as np

from config import MODEL, PERF, SLO_TBT_MS, SLO_TTFT_MS, WARMUP_FRACTION


def kv_bytes_per_token(model=MODEL) -> int:
    """K and V, per layer, per token."""
    return (
        2
        * model["n_layers"]
        * model["n_kv_heads"]
        * model["head_dim"]
        * model["dtype_bytes"]
    )


def kv_capacity_tokens(model=MODEL) -> int:
    free_bytes = (model["gpu_memory_gb"] - model["weight_memory_gb"]) * 1e9
    return int(free_bytes // kv_bytes_per_token(model))


def service_work_ms(ctx: np.ndarray, gen: np.ndarray, perf=PERF) -> np.ndarray:
    """Per-request work, used only to calibrate the arrival rate to a target load.

    Decode is charged at its MARGINAL per-token cost, because the per-iteration
    base cost is amortised across the whole batch and is a property of the
    engine rather than of any one request.
    """
    return (
        perf["prefill_fixed_ms"]
        + perf["prefill_ms_per_token"] * ctx
        + perf["decode_ms_per_request"] * gen
    )


def measure_capacity(ctx, gen, perf=PERF, model=MODEL, sample=4000) -> float:
    """Maximum sustainable throughput (req/s) for this request mix.

    Measured, not derived: the engine is driven with an unbounded backlog (all
    arrivals at t=0) under FCFS, which saturates the batch and exposes the true
    service rate including the per-iteration base cost. Deriving capacity
    analytically understates it, because the fixed decode cost is amortised over
    a batch whose size is itself load-dependent.
    """
    k = min(sample, len(ctx))
    c, g = ctx[:k], gen[:k]
    t = np.zeros(k, dtype="float64")
    _, meta = simulate(t, c, g, np.arange(k, dtype="float64"), perf, model)
    return k / (meta["wall_ms"] / 1000.0)


def rescale_arrivals(t_ms, capacity_rps, target_load):
    """Stretch/compress the arrival timeline to a target fraction of capacity.

    Shape of the arrival process and the request mix are preserved exactly; only
    the clock changes. Offered load is therefore interpretable as rho = lambda/mu.
    """
    n = len(t_ms)
    span = max(t_ms[-1] - t_ms[0], 1e-9)
    span_target_ms = 1000.0 * n / (target_load * capacity_rps)
    return (t_ms - t_ms[0]) * (span_target_ms / span)


@dataclass
class Request:
    idx: int
    arrival: float
    ctx: int
    gen: int
    priority: float          # scheduler key; lower is admitted sooner
    admitted: float = -1.0
    first_token: float = -1.0
    finished: float = -1.0
    produced: int = 0


def simulate(t_ms, ctx, gen, priority, perf=PERF, model=MODEL):
    """Run one replica to completion. Returns a per-request record array.

    `priority` is the scheduler's admission key. FCFS passes arrival order,
    Oracle-SJF passes true output length, and the proposed policy passes a
    predicted bucket. Ties break on arrival time, so every policy degrades
    gracefully to FCFS when its key carries no information.
    """
    n = len(t_ms)
    kv_cap = kv_capacity_tokens(model)
    max_batch = perf["max_batch_size"]

    reqs = [
        Request(i, float(t_ms[i]), int(ctx[i]), int(gen[i]), float(priority[i]))
        for i in range(n)
    ]

    now = 0.0
    next_arrival = 0          # index of next request to arrive
    pending: list[tuple] = []  # heap of (priority, arrival, idx, Request)
    running: list[Request] = []
    kv_used = 0
    done = 0
    busy_ms = 0.0

    while done < n:
        # Admit everything that has arrived by `now`.
        while next_arrival < n and reqs[next_arrival].arrival <= now:
            r = reqs[next_arrival]
            heapq.heappush(pending, (r.priority, r.arrival, r.idx, r))
            next_arrival += 1

        # Idle: jump to the next arrival rather than spinning.
        if not running and not pending:
            if next_arrival < n:
                now = reqs[next_arrival].arrival
                continue
            break

        # Admission control: fill the batch in scheduler-priority order.
        # Head-of-line: if the highest-priority waiter does not fit, we stop
        # rather than scanning past it. Real engines behave this way, and it is
        # precisely this behaviour that makes admission order matter.
        newly = []
        while pending:
            r = pending[0][3]
            if (
                len(running) + len(newly) < max_batch
                and kv_used + r.ctx + r.gen <= kv_cap
            ):
                heapq.heappop(pending)
                newly.append(r)
                kv_used += r.ctx + r.gen
            else:
                break

        if newly:
            # Prefill step for the newly admitted requests.
            cost = perf["prefill_fixed_ms"] + perf["prefill_ms_per_token"] * sum(
                r.ctx for r in newly
            )
            now += cost
            busy_ms += cost
            for r in newly:
                r.admitted = now
                r.first_token = now
                r.produced = 1
                running.append(r)
            # A request whose entire output is one token is already complete.
            still = []
            for r in running:
                if r.produced >= r.gen:
                    r.finished = now
                    kv_used -= r.ctx + r.gen
                    done += 1
                else:
                    still.append(r)
            running = still
            continue

        if not running:
            # Nothing admissible and nothing running: advance to next arrival.
            if next_arrival < n:
                now = max(now, reqs[next_arrival].arrival)
                continue
            break

        # Decode iteration: one token for every running request.
        cost = perf["decode_ms_base"] + perf["decode_ms_per_request"] * len(running)
        now += cost
        busy_ms += cost
        still = []
        for r in running:
            r.produced += 1
            if r.produced >= r.gen:
                r.finished = now
                kv_used -= r.ctx + r.gen
                done += 1
            else:
                still.append(r)
        running = still

    return _collect(reqs, now, busy_ms)


def _collect(reqs, wall_ms, busy_ms):
    n = len(reqs)
    out = np.zeros(
        n,
        dtype=[
            ("arrival", "f8"), ("ttft", "f8"), ("latency", "f8"),
            ("norm_latency", "f8"), ("tbt", "f8"),
            ("ctx", "i4"), ("gen", "i4"),
        ],
    )
    for i, r in enumerate(reqs):
        out[i] = (
            r.arrival,
            r.first_token - r.arrival,
            r.finished - r.arrival,
            (r.finished - r.arrival) / r.gen,
            (r.finished - r.first_token) / max(r.gen - 1, 1),
            r.ctx,
            r.gen,
        )
    return out, dict(wall_ms=wall_ms, busy_ms=busy_ms, utilisation=busy_ms / wall_ms)


def metrics(rec, meta, warmup=WARMUP_FRACTION):
    """Summarise a run, discarding a warm-up prefix so results are steady-state."""
    k = int(len(rec) * warmup)
    r = rec[k:]
    return {
        "n": int(len(r)),
        "utilisation": float(meta["utilisation"]),
        "throughput_rps": float(len(r) / (meta["wall_ms"] / 1000.0)),
        "ttft_p50": float(np.percentile(r["ttft"], 50)),
        "ttft_p95": float(np.percentile(r["ttft"], 95)),
        "ttft_p99": float(np.percentile(r["ttft"], 99)),
        "latency_p50": float(np.percentile(r["latency"], 50)),
        "latency_p95": float(np.percentile(r["latency"], 95)),
        "latency_p99": float(np.percentile(r["latency"], 99)),
        "norm_latency_mean": float(r["norm_latency"].mean()),
        "norm_latency_p95": float(np.percentile(r["norm_latency"], 95)),
        "norm_latency_p99": float(np.percentile(r["norm_latency"], 99)),
        "slo_ttft_attain": float((r["ttft"] <= SLO_TTFT_MS).mean()),
        "slo_tbt_attain": float((r["tbt"] <= SLO_TBT_MS).mean()),
        "slo_both_attain": float(
            ((r["ttft"] <= SLO_TTFT_MS) & (r["tbt"] <= SLO_TBT_MS)).mean()
        ),
    }
