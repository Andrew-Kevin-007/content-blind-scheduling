"""Robustness of the central claim to the analytical cost model.

The paper's mechanism is that prefill dominates schedulable work. That claim
rests on the ratio between the per-token prefill cost and the marginal
per-token decode cost, which we model rather than measure. This sweep varies
that ratio over a 16x range and asks whether the conclusion survives.

If the conclusion only holds at our chosen coefficients, we need to know it, and
say so, rather than presenting a single operating point as general.
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
from config import N_WINDOWS, PERF, REQUESTS_PER_RUN, RESULTS, TRACES
from data import load_trace, pick_windows, window_frame
from run import TRAIN_HOURS, TRAIN_SAMPLE

# Multipliers on the marginal decode cost. Raising it makes the *unobservable*
# component of service time more important, which is the adversarial direction
# for our claim.
DECODE_SCALES = [0.5, 1.0, 2.0, 4.0, 8.0]
LOAD = 0.92
POLICIES = ["fcfs", "cb_sjf_work", "oracle_work"]


def perf_for(scale: float) -> dict:
    p = dict(PERF)
    p["decode_ms_per_request"] = PERF["decode_ms_per_request"] * scale
    return p


def key_for(policy, t_ms, ctx, gen, pred, perf):
    if policy == "fcfs":
        return t_ms
    y = pred if policy == "cb_sjf_work" else gen.astype("float64")
    return (
        perf["prefill_fixed_ms"]
        + perf["prefill_ms_per_token"] * ctx.astype("float64")
        + perf["decode_ms_per_request"] * y
    )


def one(args):
    wl, widx, scale, policy, t_ms, ctx, gen, pred, base_cap = args
    perf = perf_for(scale)
    # Capacity must be re-measured: changing the cost model changes throughput.
    cap = S.measure_capacity(ctx, gen, perf=perf)
    t = S.rescale_arrivals(t_ms, cap, LOAD)
    prio = key_for(policy, t, ctx, gen, pred, perf)
    rec, meta = S.simulate(t, ctx, gen, prio, perf=perf)
    m = S.metrics(rec, meta)
    pre = perf["prefill_ms_per_token"] * ctx.astype("float64").sum()
    dec = perf["decode_ms_per_request"] * gen.astype("float64").sum()
    return {
        "workload": wl, "window": widx, "decode_scale": scale, "policy": policy,
        "prefill_share_pct": 100.0 * pre / (pre + dec),
        "capacity_rps": cap, "norm_latency_mean": m["norm_latency_mean"],
        "latency_p99": m["latency_p99"], "slo_both_attain": m["slo_both_attain"],
    }


def main() -> int:
    jobs = []
    for wl in TRACES:
        df = load_trace(wl)
        tr = df[df.t_ms < TRAIN_HOURS * 3.6e6]
        tr = tr.iloc[:: max(1, len(tr) // TRAIN_SAMPLE)].head(TRAIN_SAMPLE)
        model = P.train(tr)
        for widx, s in enumerate(pick_windows(df)):
            w = window_frame(df, s).head(REQUESTS_PER_RUN).reset_index(drop=True)
            ctx = w["ContextTokens"].values.copy()
            gen = w["GeneratedTokens"].values.copy()
            pred = P.predict(model, w).astype("float64")
            t_ms = w["t_ms"].values.astype("float64").copy()
            for scale, pol in itertools.product(DECODE_SCALES, POLICIES):
                jobs.append((wl, widx, scale, pol, t_ms, ctx, gen, pred, None))
        del df, tr, model

    print(f"running {len(jobs)} sensitivity simulations")
    rows = []
    with ProcessPoolExecutor(max_workers=6) as ex:
        for k, r in enumerate(ex.map(one, jobs, chunksize=1), 1):
            rows.append(r)
            if k % 30 == 0:
                print(f"  {k}/{len(jobs)}", flush=True)

    df = pd.DataFrame(rows)
    (RESULTS / "raw" / "sensitivity.json").write_text(df.to_json(orient="records"))

    out = []
    for (wl, sc), cell in df.groupby(["workload", "decode_scale"]):
        piv = cell.pivot_table(index="window", columns="policy",
                               values="norm_latency_mean")
        base, prop, orac = piv["fcfs"], piv["cb_sjf_work"], piv["oracle_work"]
        denom = (base - orac).mean()
        out.append({
            "workload": wl,
            "decode_scale": sc,
            "prefill_share_pct": cell.prefill_share_pct.mean(),
            "improvement_pct": 100.0 * (base - prop).mean() / base.mean(),
            "oracle_gap_recovered_pct": (
                100.0 * (base - prop).mean() / denom if abs(denom) > 1e-12 else np.nan
            ),
        })
    summ = pd.DataFrame(out).sort_values(["workload", "decode_scale"])
    summ.to_csv(RESULTS / "sensitivity.csv", index=False)
    print("\n=== sensitivity to decode cost ===")
    print(summ.to_string(index=False, float_format=lambda v: f"{v:.4g}"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
