"""Aggregate raw run records into the numbers the paper reports.

This is the ONLY path from results/raw/ to a table or figure. Nothing is
transcribed by hand, so the abstract cannot drift from Table I.

Every comparison is paired across evaluation windows: the same five windows are
replayed under every policy, so a paired test is both valid and much more
powerful than an unpaired one at n=5.
"""

from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd
from scipy import stats

from config import RESULTS

PRIMARY = "norm_latency_mean"   # mean normalised latency (ms per output token)

SECONDARY = [
    "norm_latency_p95",
    "latency_p99",
    "ttft_p95",
    "slo_both_attain",
    "utilisation",
    "throughput_rps",
]

BASELINE = "fcfs"
PROPOSED = "cb_sjf_work"
ORACLE = "oracle_work"


def load_runs() -> pd.DataFrame:
    rows = json.loads((RESULTS / "raw" / "runs.json").read_text())
    return pd.DataFrame(rows)


def summarise(df: pd.DataFrame) -> pd.DataFrame:
    """Mean and std across windows for every (workload, load, policy) cell."""
    metrics = [PRIMARY] + SECONDARY
    g = df.groupby(["workload", "target_load", "policy", "policy_label"])[metrics]
    out = g.agg(["mean", "std", "count"])
    out.columns = [f"{m}_{s}" for m, s in out.columns]
    return out.reset_index()


def paired_tests(df: pd.DataFrame) -> pd.DataFrame:
    """Paired comparisons against FCFS and against the oracle, per cell.

    Reports the Wilcoxon signed-rank p-value where the sample size allows it.
    With n=5 windows the smallest attainable two-sided p is 0.0625, so we report
    the statistic honestly rather than claiming significance we cannot reach:
    the effect sizes, not the stars, carry the argument.
    """
    recs = []
    for (wl, load), cell in df.groupby(["workload", "target_load"]):
        piv = cell.pivot_table(
            index="window", columns="policy", values=PRIMARY
        ).sort_index()
        if BASELINE not in piv or PROPOSED not in piv:
            continue

        base, prop = piv[BASELINE].values, piv[PROPOSED].values
        rec = {
            "workload": wl,
            "target_load": load,
            "n_windows": len(piv),
            "fcfs_mean": base.mean(),
            "proposed_mean": prop.mean(),
            "improvement_pct": 100.0 * (base - prop).mean() / base.mean(),
        }

        try:
            rec["wilcoxon_p"] = float(
                stats.wilcoxon(base, prop, alternative="greater").pvalue
            )
        except ValueError:
            rec["wilcoxon_p"] = float("nan")

        # Cohen's d for paired samples.
        d = base - prop
        rec["cohens_d"] = float(d.mean() / d.std(ddof=1)) if d.std(ddof=1) > 0 else np.nan

        if ORACLE in piv:
            orac = piv[ORACLE].values
            denom = (base - orac).mean()
            rec["oracle_mean"] = orac.mean()
            # Fraction of the achievable oracle improvement that the content-blind
            # policy actually captures. This is the paper's headline quantity.
            rec["oracle_gap_recovered_pct"] = (
                100.0 * (base - prop).mean() / denom if abs(denom) > 1e-12 else np.nan
            )
        recs.append(rec)
    return pd.DataFrame(recs)


def slo_table(df: pd.DataFrame) -> pd.DataFrame:
    piv = df.pivot_table(
        index=["workload", "target_load"],
        columns="policy",
        values="slo_both_attain",
        aggfunc="mean",
    )
    return piv.reset_index()


def main() -> int:
    df = load_runs()
    print(f"loaded {len(df)} runs")

    summ = summarise(df)
    summ.to_csv(RESULTS / "summary.csv", index=False)

    tests = paired_tests(df)
    tests.to_csv(RESULTS / "paired_tests.csv", index=False)

    slo = slo_table(df)
    slo.to_csv(RESULTS / "slo.csv", index=False)

    pred = pd.DataFrame(json.loads((RESULTS / "raw" / "predictor_eval.json").read_text()))
    pred_summ = pred.groupby("workload")[
        ["spearman", "pairwise_accuracy", "mae", "mae_log", "capacity_rps"]
    ].agg(["mean", "std"])
    pred_summ.to_csv(RESULTS / "predictor_summary.csv")

    # Composition of schedulable work. This is the paper's mechanism, so it is
    # computed here (once, from the full traces) and consumed by both the text
    # and Fig. 1 - never recomputed independently in either place.
    from config import PERF, PROC

    work_share = {}
    for wl in df.workload.unique():
        d = pd.read_parquet(
            PROC / f"{wl}.parquet", columns=["ContextTokens", "GeneratedTokens"]
        )
        pre = PERF["prefill_ms_per_token"] * d["ContextTokens"].astype("float64").sum()
        dec = PERF["decode_ms_per_request"] * d["GeneratedTokens"].astype("float64").sum()
        work_share[wl] = {
            "prefill_pct": 100.0 * pre / (pre + dec),
            "decode_pct": 100.0 * dec / (pre + dec),
            "n_requests": int(len(d)),
            "mean_context_tokens": float(d["ContextTokens"].mean()),
            "mean_output_tokens": float(d["GeneratedTokens"].mean()),
        }
    (RESULTS / "work_share.json").write_text(json.dumps(work_share, indent=2))

    # Headline numbers, written to a single file the paper reads from.
    head = {}
    for wl in df.workload.unique():
        sub = tests[tests.workload == wl]
        hi = sub[sub.target_load >= 0.85]
        head[wl] = {
            "improvement_pct_high_load_mean": float(hi.improvement_pct.mean()),
            "improvement_pct_max": float(sub.improvement_pct.max()),
            "oracle_gap_recovered_pct_high_load": float(
                hi.oracle_gap_recovered_pct.mean()
            ),
            "predictor_spearman": float(pred[pred.workload == wl].spearman.mean()),
            "predictor_pairwise_acc": float(
                pred[pred.workload == wl].pairwise_accuracy.mean()
            ),
            "capacity_rps": float(pred[pred.workload == wl].capacity_rps.mean()),
            **work_share[wl],
        }
    (RESULTS / "headline.json").write_text(json.dumps(head, indent=2))

    print("\n=== paired tests (primary metric: mean normalised latency) ===")
    print(tests.to_string(index=False, float_format=lambda v: f"{v:.4g}"))
    print("\n=== headline ===")
    print(json.dumps(head, indent=2))
    print("\nwrote summary.csv, paired_tests.csv, slo.csv, predictor_summary.csv, headline.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
