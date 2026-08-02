"""Experiment matrix: workloads x policies x offered loads x windows.

Every run writes a self-describing JSON record into results/raw/. Nothing is
ever transcribed by hand into the paper; aggregate.py is the only path from
these records to a table or a figure.
"""

from __future__ import annotations

import gc
import itertools
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

import predictor as P
import simulator as S
from config import (
    N_WINDOWS,
    PERF,
    REQUESTS_PER_RUN,
    RESULTS,
    TARGET_LOADS,
    TRACES,
)
from data import load_trace, pick_windows, window_frame

# Training period: an early slice of the trace, disjoint from every evaluation
# window (windows are selected from busy periods at 38h+ for conv, 114h+ for
# code). A deployed predictor is trained on history, so ours is too.
TRAIN_HOURS = 24.0
TRAIN_SAMPLE = 1_500_000

POLICIES = [
    "fcfs",
    "sjf_context",
    "predicted_sjf",
    "cb_sjf_work",
    "oracle_sjf",
    "oracle_work",
]

POLICY_LABEL = {
    "fcfs": "FCFS (engine default)",
    "sjf_context": "SJF-Ctx (prefill proxy)",
    "predicted_sjf": "SJF-Pred (predicted output length)",
    "cb_sjf_work": "CB-SJF-Work (proposed)",
    "oracle_sjf": "Oracle-SJF (true output length)",
    "oracle_work": "Oracle-Work (true total work)",
}

# Policies that are upper bounds, not deployable systems.
ORACLE_POLICIES = {"oracle_sjf", "oracle_work"}


def estimated_work(ctx, gen_est):
    """Total schedulable service work in ms.

    Prefill is charged at its true per-token rate and is exactly observable at
    admission. Decode is charged at its MARGINAL per-token cost, since the
    per-iteration base cost is a property of the batch, not of any one request.
    """
    return (
        PERF["prefill_fixed_ms"]
        + PERF["prefill_ms_per_token"] * ctx
        + PERF["decode_ms_per_request"] * gen_est
    )


def priority_from_arrays(policy, t_ms, ctx, gen, pred):
    """Admission key for each policy. Lower is served sooner. Ties break on
    arrival time inside the simulator, so an uninformative key degrades to FCFS."""
    if policy == "fcfs":
        return t_ms
    if policy == "sjf_context":
        return ctx.astype("float64")
    if policy == "predicted_sjf":
        return pred
    if policy == "cb_sjf_work":
        return estimated_work(ctx.astype("float64"), pred)
    if policy == "oracle_sjf":
        return gen.astype("float64")
    if policy == "oracle_work":
        return estimated_work(ctx.astype("float64"), gen.astype("float64"))
    raise ValueError(policy)


def one_run(args):
    """Worker entry point. Receives only plain numpy arrays: a DataFrame per job
    would be pickled once per job and, with the parent still holding the full
    27M-row trace, was enough to exhaust memory and break the pool."""
    workload, widx, load, policy, t_ms, ctx, gen, pred, capacity = args
    prio = priority_from_arrays(policy, t_ms, ctx, gen, pred)
    t = S.rescale_arrivals(t_ms, capacity, load)

    t0 = time.time()
    rec, meta = S.simulate(t, ctx, gen, prio)
    m = S.metrics(rec, meta)

    return {
        "workload": workload,
        "window": widx,
        "target_load": load,
        "policy": policy,
        "policy_label": POLICY_LABEL[policy],
        "n_requests": int(len(ctx)),
        "sim_seconds": round(time.time() - t0, 2),
        **m,
    }


def prepare(workload: str):
    """Load trace, train the predictor on early history, cut evaluation windows."""
    df = load_trace(workload)
    train_cut = TRAIN_HOURS * 3.6e6
    train = df[df.t_ms < train_cut]
    if len(train) > TRAIN_SAMPLE:
        train = train.iloc[:: len(train) // TRAIN_SAMPLE].head(TRAIN_SAMPLE)
    print(f"  {workload}: training on {len(train):,} requests from first {TRAIN_HOURS}h")

    model = P.train(train)

    starts = pick_windows(df)
    assert min(starts) >= train_cut, "evaluation window overlaps training period"

    windows, evals = [], []
    for i, s in enumerate(starts):
        w = window_frame(df, s).head(REQUESTS_PER_RUN).reset_index(drop=True)
        pred = P.predict(model, w)
        ctx = w["ContextTokens"].values.copy()
        gen = w["GeneratedTokens"].values.copy()
        cap = S.measure_capacity(ctx, gen)
        print(f"    window {i}: measured capacity {cap:.2f} req/s", flush=True)
        windows.append(
            dict(t_ms=w["t_ms"].values.astype("float64").copy(), ctx=ctx, gen=gen,
                 pred=pred.astype("float64"), cap=cap)
        )
        evals.append(
            {"workload": workload, "window": i, "capacity_rps": cap,
             **P.evaluate(model, w)}
        )

    imp = P.feature_importance(model, train)

    # Release the full trace before the pool is forked; otherwise the parent's
    # resident set is copied into the accounting of every spawned worker.
    del df, train, model
    gc.collect()
    return windows, evals, imp


def main() -> int:
    only = sys.argv[1] if len(sys.argv) > 1 else None
    all_rows, all_evals, importances = [], [], {}

    jobs = []
    for workload in TRACES:
        if only and workload != only:
            continue
        print(f"\n=== preparing {workload} ===")
        windows, evals, imp = prepare(workload)
        all_evals.extend(evals)
        importances[workload] = imp
        for widx, load, policy in itertools.product(
            range(N_WINDOWS), TARGET_LOADS, POLICIES
        ):
            d = windows[widx]
            jobs.append(
                (workload, widx, load, policy,
                 d["t_ms"], d["ctx"], d["gen"], d["pred"], d["cap"])
            )

    n_workers = max(1, min(8, (os.cpu_count() or 4) - 2))
    print(f"\n=== running {len(jobs)} simulations on {n_workers} workers ===")
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        for k, row in enumerate(ex.map(one_run, jobs, chunksize=1), 1):
            all_rows.append(row)
            print(
                f"  [{k:>3}/{len(jobs)}] {row['workload']:<5} w{row['window']} "
                f"load={row['target_load']:<5} {row['policy']:<14} "
                f"util={row['utilisation']:.2f} "
                f"p99lat={row['latency_p99']/1000:8.2f}s "
                f"({row['sim_seconds']}s)",
                flush=True,
            )
    print(f"total {time.time()-t0:.1f}s")

    (RESULTS / "raw" / "runs.json").write_text(json.dumps(all_rows, indent=2))
    (RESULTS / "raw" / "predictor_eval.json").write_text(json.dumps(all_evals, indent=2))
    (RESULTS / "raw" / "feature_importance.json").write_text(
        json.dumps(importances, indent=2)
    )
    print(f"\nwrote {len(all_rows)} run records to results/raw/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
