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
        # Report BOTH definitions. The ratio of means is dominated by whichever
        # window has the worst FCFS run; the mean of per-window ratios is what
        # the paired test actually operates on. Quoting only the first would
        # overstate the effect.
        per_window = 100.0 * (base - prop) / base
        rec = {
            "workload": wl,
            "target_load": load,
            "n_windows": len(piv),
            "fcfs_mean": base.mean(),
            "proposed_mean": prop.mean(),
            "improvement_pct": 100.0 * (base - prop).mean() / base.mean(),
            "improvement_pct_perwindow_mean": float(per_window.mean()),
            "improvement_pct_perwindow_min": float(per_window.min()),
            "improvement_pct_perwindow_max": float(per_window.max()),
            "windows_improved": int((per_window > 0).sum()),
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


def tail_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """Where the proposed policy LOSES.

    Shortest-first ordering minimises the mean by deferring long requests, which
    necessarily lengthens the tail. This function measures that cost per cell so
    it appears in the paper instead of being discovered by a reviewer reading
    Table II.
    """
    recs = []
    for (wl, load), cell in df.groupby(["workload", "target_load"]):
        piv = cell.pivot_table(index="window", columns="policy", values="latency_p99")
        if BASELINE not in piv or PROPOSED not in piv:
            continue
        base, prop = piv[BASELINE].values, piv[PROPOSED].values
        ratio = prop / base
        rec = {
            "workload": wl,
            "target_load": load,
            "n_windows": len(piv),
            "fcfs_p99_s": base.mean() / 1000.0,
            "proposed_p99_s": prop.mean() / 1000.0,
            "p99_ratio_mean": float(ratio.mean()),
            "p99_ratio_max": float(ratio.max()),
            "windows_tail_worse": int((prop > base).sum()),
        }
        for pol in ("oracle_sjf", "oracle_work", "sjf_context"):
            if pol in piv:
                rec[f"{pol}_p99_s"] = piv[pol].values.mean() / 1000.0
        recs.append(rec)
    return pd.DataFrame(recs)


def sjf_pred_vs_fcfs(df: pd.DataFrame) -> pd.DataFrame:
    """Test the claim that ordering by predicted output length is worse than FCFS.

    This appears in the abstract and the conclusion, so it needs the same test we
    apply to results that favour us. Asserting it untested while demanding
    p<0.01 of the predictor's gain would be a double standard.
    """
    recs = []
    for (wl, load), cell in df.groupby(["workload", "target_load"]):
        piv = cell.pivot_table(
            index="window", columns="policy", values=PRIMARY
        ).sort_index()
        if BASELINE not in piv or "predicted_sjf" not in piv:
            continue
        base, pred = piv[BASELINE].values, piv["predicted_sjf"].values
        rec = {
            "workload": wl, "target_load": load, "n_windows": len(piv),
            "fcfs_mean": float(base.mean()), "sjf_pred_mean": float(pred.mean()),
            "windows_pred_worse": int((pred > base).sum()),
        }
        try:
            rec["wilcoxon_p_onesided"] = float(
                stats.wilcoxon(pred, base, alternative="greater").pvalue
            )
        except ValueError:
            rec["wilcoxon_p_onesided"] = float("nan")
        recs.append(rec)
    return pd.DataFrame(recs)


def predictor_contribution(df: pd.DataFrame) -> pd.DataFrame:
    """Does the learned predictor earn its place, or is context length enough?

    Compares the proposed policy against SJF-Ctx, which is the same policy with
    the predicted-decode term removed. If this is zero the paper should say so.
    """
    recs = []
    for (wl, load), cell in df.groupby(["workload", "target_load"]):
        piv = cell.pivot_table(
            index="window", columns="policy", values=PRIMARY
        ).sort_index()
        if "sjf_context" not in piv or PROPOSED not in piv:
            continue
        ctx, prop = piv["sjf_context"].values, piv[PROPOSED].values
        gain = 100.0 * (ctx - prop) / ctx
        rec = {
            "workload": wl, "target_load": load, "n_windows": len(piv),
            "sjf_ctx_mean": ctx.mean(), "proposed_mean": prop.mean(),
            "predictor_gain_pct": float(gain.mean()),
            "windows_predictor_helps": int((gain > 0).sum()),
        }
        try:
            rec["wilcoxon_p_onesided"] = float(
                stats.wilcoxon(ctx, prop, alternative="greater").pvalue
            )
        except ValueError:
            rec["wilcoxon_p_onesided"] = float("nan")
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

    tails = tail_analysis(df)
    tails.to_csv(RESULTS / "tail_analysis.csv", index=False)

    # Normalised latency divides by output tokens, so one-token requests
    # contribute their full latency and can dominate the mean. Quantify that,
    # and re-test the headline on requests generating at least 8 tokens, where
    # the metric is not degenerate.
    rob = []
    for (wl, load), cell in df.groupby(["workload", "target_load"]):
        pa = cell.pivot_table(index="window", columns="policy", values=PRIMARY)
        pl = cell.pivot_table(
            index="window", columns="policy", values="norm_latency_mean_gen_ge8"
        )
        if BASELINE not in pa or PROPOSED not in pa:
            continue
        base_f = cell[cell.policy == BASELINE]
        rob.append({
            "workload": wl, "target_load": load,
            "frac_requests_gen_le4": float(base_f.frac_requests_gen_le4.mean()),
            "frac_normlat_from_gen_le4": float(base_f.frac_normlat_from_gen_le4.mean()),
            "improvement_all_pct":
                100.0 * (pa[BASELINE] - pa[PROPOSED]).mean() / pa[BASELINE].mean(),
            "improvement_gen_ge8_pct":
                100.0 * (pl[BASELINE] - pl[PROPOSED]).mean() / pl[BASELINE].mean(),
            "windows_improved_gen_ge8": int((pl[PROPOSED] < pl[BASELINE]).sum()),
            "n_windows": int(len(pl)),
        })
    robust = pd.DataFrame(rob)
    robust.to_csv(RESULTS / "metric_robustness.csv", index=False)

    contrib = predictor_contribution(df)
    contrib.to_csv(RESULTS / "predictor_contribution.csv", index=False)

    predtest = sjf_pred_vs_fcfs(df)
    predtest.to_csv(RESULTS / "sjf_pred_vs_fcfs.csv", index=False)

    # SLO attainment across loads, so the paper quotes measured values rather
    # than any that survive from an earlier revision.
    slo_by_load = df[df.policy == PROPOSED].pivot_table(
        index="target_load", columns="workload", values="slo_both_attain"
    )
    slo_by_load.to_csv(RESULTS / "slo_by_load.csv")

    # Measured engine-time split: where the replica actually spends wall clock,
    # including the per-iteration base cost that the marginal work model omits.
    time_split = df[df.policy == BASELINE].pivot_table(
        index="target_load", columns="workload", values="prefill_time_share"
    )
    time_split.to_csv(RESULTS / "prefill_time_share.csv")

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
        t_hi = tails[(tails.workload == wl) & (tails.target_load >= 0.85)]
        c_hi = contrib[(contrib.workload == wl) & (contrib.target_load >= 0.85)]
        head[wl] = {
            "improvement_pct_high_load_mean": float(hi.improvement_pct.mean()),
            "improvement_pct_max": float(sub.improvement_pct.max()),
            "improvement_perwindow_mean_high_load": float(
                hi.improvement_pct_perwindow_mean.mean()
            ),
            "improvement_perwindow_min_high_load": float(
                hi.improvement_pct_perwindow_min.min()
            ),
            "improvement_perwindow_max_high_load": float(
                hi.improvement_pct_perwindow_max.max()
            ),
            "p99_ratio_vs_fcfs_high_load": float(t_hi.p99_ratio_mean.mean()),
            "windows_tail_worse_high_load": int(t_hi.windows_tail_worse.sum()),
            "windows_total_high_load": int(t_hi.n_windows.sum()),
            "predictor_gain_pct_high_load": float(c_hi.predictor_gain_pct.mean()),
            "measured_prefill_time_share_092": float(
                df[(df.workload == wl) & (df.policy == BASELINE)
                   & (df.target_load == 0.92)].prefill_time_share.mean()
            ),
            "measured_utilisation_060": float(
                df[(df.workload == wl) & (df.policy == BASELINE)
                   & (df.target_load == 0.60)].utilisation.mean()
            ),
            "frac_normlat_from_gen_le4": float(
                robust[robust.workload == wl].frac_normlat_from_gen_le4.mean()
            ),
            "improvement_gen_ge8_high_load": float(
                robust[(robust.workload == wl)
                       & (robust.target_load >= 0.85)].improvement_gen_ge8_pct.mean()
            ),
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
    print("\n=== TAIL COST (where the proposed policy loses) ===")
    print(tails.to_string(index=False, float_format=lambda v: f"{v:.4g}"))
    print("\n=== predictor contribution over SJF-Ctx ===")
    print(contrib.to_string(index=False, float_format=lambda v: f"{v:.4g}"))
    print("\n=== measured prefill share of ENGINE TIME (fcfs) ===")
    print(time_split.round(3).to_string())
    print("\n=== headline ===")
    print(json.dumps(head, indent=2))
    print("\nwrote summary.csv, paired_tests.csv, tail_analysis.csv, "
          "predictor_contribution.csv, prefill_time_share.csv, slo.csv, "
          "predictor_summary.csv, work_share.json, headline.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
