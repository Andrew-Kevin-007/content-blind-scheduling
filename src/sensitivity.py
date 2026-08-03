"""Robustness of the central claim.

An earlier version of this sweep varied the marginal decode COST. That was the
wrong knob: the binding constraint in the simulator is KV-cache capacity, whose
per-request footprint is x_i + y_i, and scaling a cost coefficient leaves that
untouched. The result was invariant for an uninteresting reason.

This version scales OUTPUT LENGTH instead. That moves the decode share of work
and the KV footprint together, and it is the direction that matters practically:
reasoning and agentic workloads generate far more tokens per request than the
2024 traces do. If the conclusion survives here, it survives for a reason.

We additionally sweep the fraction of prefill that is actually executed, which
models automatic prefix caching. Multi-turn conversation re-sends its history,
so production stacks serve much of the prompt from cache; that shrinks exactly
the component this paper relies on being large.
"""

from __future__ import annotations

import itertools
import json
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

import predictor as P
import simulator as S
from config import PERF, REQUESTS_PER_RUN, RESULTS, TRACES
from data import load_trace, pick_windows, window_frame
from run import TRAIN_HOURS, TRAIN_SAMPLE, estimated_work

GEN_SCALES = [1.0, 2.0, 4.0, 8.0, 16.0]
CACHE_HIT = [0.0, 0.5, 0.75, 0.9]     # fraction of prompt served from cache
LOAD = 0.92
POLICIES = ["fcfs", "sjf_context", "cb_sjf_work", "oracle_work",
            "oracle_work_cacheaware"]
MAX_WINDOWS = 8


def key_for(policy, t_ms, ctx_seen, ctx_exec, gen, pred):
    """Ordering key. ctx_seen is what a content-blind scheduler observes.

    Under prefix caching the content-blind scheduler still sees the FULL context
    length, because cache-hit length depends on token identity and computing it
    would require reading content. The engine meanwhile executes the reduced
    prefill (ctx_exec).

    oracle_work is handicapped the same way, which is why it converges with
    CB-SJF-Work under caching. oracle_work_cacheaware is the honest upper bound:
    it sees the executed prefill and therefore keeps the advantage that
    content-blindness actually forfeits.
    """
    if policy == "fcfs":
        return t_ms
    if policy == "sjf_context":
        return ctx_seen.astype("float64")
    if policy == "cb_sjf_work":
        return estimated_work(ctx_seen.astype("float64"), pred)
    if policy == "oracle_work":
        return estimated_work(ctx_seen.astype("float64"), gen.astype("float64"))
    if policy == "oracle_work_cacheaware":
        return estimated_work(ctx_exec.astype("float64"), gen.astype("float64"))
    raise ValueError(policy)


def one(args):
    wl, widx, gen_scale, hit, policy, t_ms, ctx, gen, pred = args
    gen_s = np.maximum(1, np.round(gen * gen_scale)).astype("int64")
    ctx_exec = np.maximum(1, np.round(ctx * (1.0 - hit))).astype("int64")

    # Clamp the sequence length so no request exceeds the KV budget on its own.
    # Real engines impose exactly such a bound; without it, scaling output length
    # produces requests that can never be admitted, and the run would report
    # metrics over requests that never finished.
    kv_cap = S.kv_capacity_tokens()
    budget = int(kv_cap * 0.95)
    over = (ctx_exec + gen_s) > budget
    if over.any():
        gen_s = np.where(over, np.maximum(1, budget - ctx_exec), gen_s)
    gen_s = gen_s.astype("int32")
    ctx_exec = ctx_exec.astype("int32")

    cap = S.measure_capacity(ctx_exec, gen_s)
    t = S.rescale_arrivals(t_ms, cap, LOAD)
    prio = key_for(policy, t, ctx, ctx_exec, gen_s, pred * gen_scale)
    rec, meta = S.simulate(t, ctx_exec, gen_s, prio)
    m = S.metrics(rec, meta)

    pre = PERF["prefill_ms_per_token"] * ctx_exec.astype("float64").sum()
    dec = PERF["decode_ms_per_request"] * gen_s.astype("float64").sum()
    return {
        "workload": wl, "window": widx, "gen_scale": gen_scale, "cache_hit": hit,
        "policy": policy, "clamped_frac": float(over.mean()),
        "prefill_share_pct": 100.0 * pre / (pre + dec),
        "prefill_time_share": m["prefill_time_share"],
        "norm_latency_mean": m["norm_latency_mean"],
        "latency_p99": m["latency_p99"], "capacity_rps": cap,
    }


def build_jobs():
    jobs = []
    for wl in TRACES:
        df = load_trace(wl)
        tr = df[df.t_ms < TRAIN_HOURS * 3.6e6].head(TRAIN_SAMPLE)
        model = P.train(tr)
        starts = [s for s in pick_windows(df) if s >= TRAIN_HOURS * 3.6e6]
        for widx, s in enumerate(starts[:MAX_WINDOWS]):
            w = window_frame(df, s).head(REQUESTS_PER_RUN).reset_index(drop=True)
            ctx = w["ContextTokens"].values.copy()
            gen = w["GeneratedTokens"].values.copy()
            pred = P.predict(model, w).astype("float64")
            t_ms = w["t_ms"].values.astype("float64").copy()
            for g, pol in itertools.product(GEN_SCALES, POLICIES):
                jobs.append((wl, widx, g, 0.0, pol, t_ms, ctx, gen, pred))
            for h, pol in itertools.product(CACHE_HIT[1:], POLICIES):
                jobs.append((wl, widx, 1.0, h, pol, t_ms, ctx, gen, pred))
        del df, tr, model
    return jobs


def summarise(df, group_col):
    out = []
    for (wl, val), cell in df.groupby(["workload", group_col]):
        piv = cell.pivot_table(index="window", columns="policy",
                               values="norm_latency_mean")
        if not {"fcfs", "cb_sjf_work", "oracle_work"} <= set(piv.columns):
            continue
        base, prop, orac = piv["fcfs"], piv["cb_sjf_work"], piv["oracle_work"]
        denom = (base - orac).mean()
        row = {
            "workload": wl, group_col: val,
            "clamped_frac": float(cell.clamped_frac.mean()),
            "prefill_share_pct": cell.prefill_share_pct.mean(),
            "prefill_time_share_pct": 100 * cell.prefill_time_share.mean(),
            "improvement_pct": 100.0 * (base - prop).mean() / base.mean(),
            "oracle_gap_recovered_pct": (
                100.0 * (base - prop).mean() / denom if abs(denom) > 1e-12 else np.nan
            ),
            # How much worse than a perfectly-informed scheduler we are, in
            # absolute terms. Unlike the recovered-gap percentage, this does not
            # flatter us when FCFS also degrades.
            "cb_over_oracle_ratio": float((prop / orac).mean()),
        }
        if "sjf_context" in piv:
            row["predictor_gain_pct"] = float(
                100.0 * (piv["sjf_context"] - prop).mean() / piv["sjf_context"].mean()
            )
        if "oracle_work_cacheaware" in piv:
            ca = piv["oracle_work_cacheaware"]
            denom_ca = (base - ca).mean()
            row["cb_over_cacheaware_oracle_ratio"] = float((prop / ca).mean())
            row["oracle_gap_recovered_cacheaware_pct"] = (
                100.0 * (base - prop).mean() / denom_ca
                if abs(denom_ca) > 1e-12 else np.nan
            )
        out.append(row)
    return pd.DataFrame(out).sort_values(["workload", group_col])


def main() -> int:
    jobs = build_jobs()
    print(f"running {len(jobs)} sensitivity simulations")
    rows = []
    with ProcessPoolExecutor(max_workers=6) as ex:
        for k, r in enumerate(ex.map(one, jobs, chunksize=1), 1):
            rows.append(r)
            if k % 100 == 0:
                print(f"  {k}/{len(jobs)}", flush=True)

    df = pd.DataFrame(rows)
    (RESULTS / "raw" / "sensitivity.json").write_text(df.to_json(orient="records"))

    gen = summarise(df[df.cache_hit == 0.0], "gen_scale")
    gen.to_csv(RESULTS / "sensitivity_genlen.csv", index=False)
    cache = summarise(df[df.gen_scale == 1.0], "cache_hit")
    cache.to_csv(RESULTS / "sensitivity_cache.csv", index=False)

    print("\n=== output-length scaling (reasoning/agentic direction) ===")
    print(gen.to_string(index=False, float_format=lambda v: f"{v:.4g}"))
    print("\n=== prefix-cache hit rate ===")
    print(cache.to_string(index=False, float_format=lambda v: f"{v:.4g}"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
