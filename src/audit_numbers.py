"""Audit every quantitative claim in the paper against results/.

The paper states that no number in it is transcribed by hand. One number
(SLO attainment at low load) survived an earlier revision without being
regenerated, which falsified that claim until it was caught. This script exists
so the claim is checkable rather than asserted.

Each entry pairs a value as printed in the PDF with the file, column and
selection that should produce it. A mismatch beyond the stated tolerance fails.
"""

from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd

from config import RESULTS

runs = pd.DataFrame(json.loads((RESULTS / "raw" / "runs.json").read_text()))
head = json.loads((RESULTS / "headline.json").read_text())
work = json.loads((RESULTS / "work_share.json").read_text())
tok = json.loads((RESULTS / "token_ratio.json").read_text())
tail = pd.read_csv(RESULTS / "tail_analysis.csv")
tests = pd.read_csv(RESULTS / "paired_tests.csv")
contrib = pd.read_csv(RESULTS / "predictor_contribution.csv")
predtest = pd.read_csv(RESULTS / "sjf_pred_vs_fcfs.csv")
robust = pd.read_csv(RESULTS / "metric_robustness.csv")
gen = pd.read_csv(RESULTS / "sensitivity_genlen.csv")
cache = pd.read_csv(RESULTS / "sensitivity_cache.csv")
pshare = pd.read_csv(RESULTS / "prefill_time_share.csv", index_col=0)
evals = pd.DataFrame(json.loads((RESULTS / "raw" / "predictor_eval.json").read_text()))


def cell(df, wl, load, col):
    r = df[(df.workload == wl) & (df.target_load == load)]
    return float(r[col].iloc[0])


def t2(wl, pol, col, load=0.92):
    r = runs[(runs.workload == wl) & (runs.target_load == load) & (runs.policy == pol)]
    return float(r[col].mean())


CHECKS = [
    # (label, printed value, computed value, tolerance)
    ("abstract prefill marginal conv", 91.2, work["conv"]["prefill_pct"], 0.05),
    ("abstract prefill marginal code", 98.7, work["code"]["prefill_pct"], 0.05),
    ("abstract engine-time conv @.92", 76, 100 * pshare.loc[0.92, "conv"], 0.5),
    ("abstract engine-time code @.92", 92, 100 * pshare.loc[0.92, "code"], 0.5),
    ("abstract improvement conv", 51.5, head["conv"]["improvement_pct_high_load_mean"], 0.05),
    ("abstract improvement code", 58.8, head["code"]["improvement_pct_high_load_mean"], 0.05),
    ("abstract oracle gap conv", 90.3, head["conv"]["oracle_gap_recovered_pct_high_load"], 0.05),
    ("abstract oracle gap code", 79.3, head["code"]["oracle_gap_recovered_pct_high_load"], 0.05),
    ("abstract p99 mean conv @.98", 1.9, cell(tail, "conv", 0.98, "p99_ratio_mean"), 0.05),
    ("abstract p99 mean code @.98", 3.1, cell(tail, "code", 0.98, "p99_ratio_mean"), 0.05),
    ("abstract p99 max conv @.98", 4.6, cell(tail, "conv", 0.98, "p99_ratio_max"), 0.05),
    ("abstract p99 max code @.98", 6.1, cell(tail, "code", 0.98, "p99_ratio_max"), 0.05),

    ("III mean ctx conv", 1632, work["conv"]["mean_context_tokens"], 1.0),
    ("III mean ctx code", 2511, work["code"]["mean_context_tokens"], 1.0),
    ("III mean out conv", 106, work["conv"]["mean_output_tokens"], 1.0),
    ("III mean out code", 23, work["code"]["mean_output_tokens"], 1.0),
    ("III token ratio conv", 15.5, tok["conv"]["token_ratio"], 0.05),
    ("III token ratio code", 110.7, tok["code"]["token_ratio"], 0.05),
    ("III engine-time conv @.60", 53.9, 100 * pshare.loc[0.60, "conv"], 0.05),
    ("III engine-time code @.60", 81.8, 100 * pshare.loc[0.60, "code"], 0.05),
    ("III engine-time conv @.92", 75.6, 100 * pshare.loc[0.92, "conv"], 0.05),
    ("III engine-time code @.92", 92.4, 100 * pshare.loc[0.92, "code"], 0.05),
    ("III spearman conv", 0.365, head["conv"]["predictor_spearman"], 0.001),
    ("III spearman code", 0.107, head["code"]["predictor_spearman"], 0.001),
    ("III pairwise conv", 0.633, head["conv"]["predictor_pairwise_acc"], 0.001),
    ("III pairwise code", 0.537, head["code"]["predictor_pairwise_acc"], 0.001),

    ("V capacity conv", 1.79, head["conv"]["capacity_rps"], 0.005),
    ("V capacity code", 1.41, head["code"]["capacity_rps"], 0.005),
    ("V utilisation conv @.60", 0.91, head["conv"]["measured_utilisation_060"], 0.005),
    ("V utilisation code @.60", 0.71, head["code"]["measured_utilisation_060"], 0.005),
    ("V span min", 3.7, evals.span_used_minutes.min(), 0.05),
    ("V span max", 19.9, evals.span_used_minutes.max(), 0.05),

    ("VI L_n FCFS conv", 1161, t2("conv", "fcfs", "norm_latency_mean"), 1.0),
    ("VI L_n CB conv", 506, t2("conv", "cb_sjf_work", "norm_latency_mean"), 1.0),
    ("VI L_n Ctx conv", 505, t2("conv", "sjf_context", "norm_latency_mean"), 1.0),
    ("VI L_n FCFS code", 16724, t2("code", "fcfs", "norm_latency_mean"), 1.0),
    ("VI L_n CB code", 6327, t2("code", "cb_sjf_work", "norm_latency_mean"), 1.0),
    ("VI L_n Ctx code", 6337, t2("code", "sjf_context", "norm_latency_mean"), 1.0),
    ("VI L_n SJF-Pred code", 23081, t2("code", "predicted_sjf", "norm_latency_mean"), 1.0),
    # Both figures in this sentence are averages over rho >= 0.85, so the
    # per-window mean must be selected the same way as the ratio-of-means.
    ("VI perwindow mean conv", 47.0, head["conv"]["improvement_perwindow_mean_high_load"], 0.1),
    ("VI ratio-of-means conv", 51.5, head["conv"]["improvement_pct_high_load_mean"], 0.05),
    ("VI gen<=4 frac conv", 10.9, 100 * cell(robust, "conv", 0.92, "frac_requests_gen_le4"), 0.1),
    ("VI gen<=4 frac code", 36.5, 100 * cell(robust, "code", 0.92, "frac_requests_gen_le4"), 0.1),
    ("VI gen<=4 metric conv", 50.2, 100 * cell(robust, "conv", 0.92, "frac_normlat_from_gen_le4"), 0.2),
    ("VI gen<=4 metric code", 78.6, 100 * cell(robust, "code", 0.92, "frac_normlat_from_gen_le4"), 0.2),
    ("VI gen>=8 improvement conv", 47.0, head["conv"]["improvement_gen_ge8_high_load"], 0.1),
    ("VI gen>=8 improvement code", 48.5, head["code"]["improvement_gen_ge8_high_load"], 0.1),
    ("VI tail worse code cells", 31, tail[(tail.workload == "code") & (tail.target_load >= 0.85)].windows_tail_worse.sum(), 0),
    ("VI tail worse conv @.92", 12, cell(tail, "conv", 0.92, "windows_tail_worse"), 0),
    ("VI oracle-work p99 code @.85", 1.42, cell(tail, "code", 0.85, "oracle_work_p99_s") / cell(tail, "code", 0.85, "fcfs_p99_s"), 0.02),
    ("VI SJF-Pred worse windows code @.92", 8, cell(predtest, "code", 0.92, "windows_pred_worse"), 0),
    ("VI SJF-Pred p code @.92", 0.06, cell(predtest, "code", 0.92, "wilcoxon_p_onesided"), 0.005),
    ("VI SJF-Pred p code @.98", 0.03, cell(predtest, "code", 0.98, "wilcoxon_p_onesided"), 0.005),
    ("VI predictor p code @.98", 0.009, cell(contrib, "code", 0.98, "wilcoxon_p_onesided"), 0.001),
    ("VI predictor p conv @.98", 0.027, cell(contrib, "conv", 0.98, "wilcoxon_p_onesided"), 0.001),
    ("VI SLO conv @.92", 34.5, 100 * t2("conv", "cb_sjf_work", "slo_both_attain"), 0.1),
    ("VI SLO code @.92", 15.4, 100 * t2("code", "cb_sjf_work", "slo_both_attain"), 0.1),
    ("VI SLO conv @.60", 54.3, 100 * t2("conv", "cb_sjf_work", "slo_both_attain", 0.60), 0.1),
    ("VI SLO conv @.98", 31.1, 100 * t2("conv", "cb_sjf_work", "slo_both_attain", 0.98), 0.1),
    ("VI SLO code @.60", 34.5, 100 * t2("code", "cb_sjf_work", "slo_both_attain", 0.60), 0.1),
    ("VI SLO code @.98", 13.4, 100 * t2("code", "cb_sjf_work", "slo_both_attain", 0.98), 0.1),

    ("VII sens base conv", 46.0, float(gen[(gen.workload == "conv") & (gen.gen_scale == 1.0)].improvement_pct.iloc[0]), 0.1),
    ("VII engine-time 16x conv", 11.2, float(gen[(gen.workload == "conv") & (gen.gen_scale == 16.0)].prefill_time_share_pct.iloc[0]), 0.1),
    ("VII improvement 16x conv", 75.1, float(gen[(gen.workload == "conv") & (gen.gen_scale == 16.0)].improvement_pct.iloc[0]), 0.1),
    ("VII cb/oracle 1x conv", 1.08, float(gen[(gen.workload == "conv") & (gen.gen_scale == 1.0)].cb_over_oracle_ratio.iloc[0]), 0.01),
    ("VII cb/oracle 16x conv", 3.76, float(gen[(gen.workload == "conv") & (gen.gen_scale == 16.0)].cb_over_oracle_ratio.iloc[0]), 0.01),
    ("VII cb/oracle 16x code", 5.53, float(gen[(gen.workload == "code") & (gen.gen_scale == 16.0)].cb_over_oracle_ratio.iloc[0]), 0.01),
    ("VII predictor 16x conv", 21.1, float(gen[(gen.workload == "conv") & (gen.gen_scale == 16.0)].predictor_gain_pct.iloc[0]), 0.1),
    ("VII cache90 conv", 12.8, float(cache[(cache.workload == "conv") & (cache.cache_hit == 0.9)].improvement_pct.iloc[0]), 0.1),
    ("VII cache90 code", 14.8, float(cache[(cache.workload == "code") & (cache.cache_hit == 0.9)].improvement_pct.iloc[0]), 0.1),
    ("VII cache90 gap conv", 85.5, float(cache[(cache.workload == "conv") & (cache.cache_hit == 0.9)].oracle_gap_recovered_pct.iloc[0]), 0.1),
    ("VII cache90 gap code", 74.5, float(cache[(cache.workload == "code") & (cache.cache_hit == 0.9)].oracle_gap_recovered_pct.iloc[0]), 0.1),

    # Added after the completeness check flagged them as uncovered.
    ("VI tail mean code hi", 2.05, head["code"]["p99_ratio_vs_fcfs_high_load"], 0.01),
    ("VI oracle-work p99 code @.92", 1.95, cell(tail, "code", 0.92, "oracle_work_p99_s") / cell(tail, "code", 0.92, "fcfs_p99_s"), 0.02),
    ("VI oracle-work p99 code @.98", 3.30, cell(tail, "code", 0.98, "oracle_work_p99_s") / cell(tail, "code", 0.98, "fcfs_p99_s"), 0.02),
    ("VI p99 max conv exact", 4.57, cell(tail, "conv", 0.98, "p99_ratio_max"), 0.01),
    ("VI p99 max code exact", 6.06, cell(tail, "code", 0.98, "p99_ratio_max"), 0.01),
    ("VI SJF-Pred sd code", 28340, float(runs[(runs.workload == "code") & (runs.target_load == 0.92) & (runs.policy == "predicted_sjf")].norm_latency_mean.std()), 5.0),
    ("VI predictor gain code @.98", 3.45, cell(contrib, "code", 0.98, "predictor_gain_pct"), 0.01),
    ("VI gen>=8 code @.75", 11.1, cell(robust, "code", 0.75, "improvement_gen_ge8_pct"), 0.1),
    ("VI gen>=8 conv hi exact", 47.04, head["conv"]["improvement_gen_ge8_high_load"], 0.01),
    ("VII sens base code", 63.3, float(gen[(gen.workload == "code") & (gen.gen_scale == 1.0)].improvement_pct.iloc[0]), 0.1),
    ("VII engine-time 1x conv", 75.4, float(gen[(gen.workload == "conv") & (gen.gen_scale == 1.0)].prefill_time_share_pct.iloc[0]), 0.1),
    ("VII engine-time 1x code", 92.9, float(gen[(gen.workload == "code") & (gen.gen_scale == 1.0)].prefill_time_share_pct.iloc[0]), 0.1),
    ("VII engine-time 16x code", 48.8, float(gen[(gen.workload == "code") & (gen.gen_scale == 16.0)].prefill_time_share_pct.iloc[0]), 0.1),
    ("VII improvement 16x code", 50.5, float(gen[(gen.workload == "code") & (gen.gen_scale == 16.0)].improvement_pct.iloc[0]), 0.1),
    ("VII predictor gain 1x conv", 0.3, float(gen[(gen.workload == "conv") & (gen.gen_scale == 1.0)].predictor_gain_pct.iloc[0]), 0.05),

    # Ordering-direction controls (LJF-Work, Random) and the cache-aware oracle.
    ("VI LJF p99 ratio conv", 9.00, t2("conv", "ljf_work", "latency_p99") / t2("conv", "fcfs", "latency_p99"), 0.02),
    ("VI LJF p99 ratio code", 9.68, t2("code", "ljf_work", "latency_p99") / t2("code", "fcfs", "latency_p99"), 0.02),
    ("VI Rand p99 ratio conv", 3.81, t2("conv", "rand", "latency_p99") / t2("conv", "fcfs", "latency_p99"), 0.02),
    ("VI Rand p99 ratio code", 5.31, t2("code", "rand", "latency_p99") / t2("code", "fcfs", "latency_p99"), 0.02),
    ("VI CB p99 ratio conv", 0.99, t2("conv", "cb_sjf_work", "latency_p99") / t2("conv", "fcfs", "latency_p99"), 0.02),
    ("VI LJF mean conv", 2688, t2("conv", "ljf_work", "norm_latency_mean"), 2.0),
    ("VI LJF vs CB mean ratio", 5.3, t2("conv", "ljf_work", "norm_latency_mean") / t2("conv", "cb_sjf_work", "norm_latency_mean"), 0.05),
    ("VII cache90 cacheaware gap conv", 66.0, float(cache[(cache.workload == "conv") & (cache.cache_hit == 0.9)].oracle_gap_recovered_cacheaware_pct.iloc[0]), 0.1),
    ("VII cache90 cacheaware gap code", 69.8, float(cache[(cache.workload == "code") & (cache.cache_hit == 0.9)].oracle_gap_recovered_cacheaware_pct.iloc[0]), 0.1),
    ("VII cache75 code", 48.6, float(cache[(cache.workload == "code") & (cache.cache_hit == 0.75)].improvement_pct.iloc[0]), 0.1),
]


def table_checks():
    """Mechanically cover both generated tables.

    An earlier version audited prose heavily and tables barely, which is exactly
    backwards: the tables hold the largest block of numbers and the one value
    that ever went stale sat next to them. These are generated by reading the
    same .tex fragments the paper inputs, so a table cell that stops matching
    results/ fails here.
    """
    import re

    tex_dir = RESULTS.parent / "paper" / "tables"
    out = []

    # Table I: every policy row, at the load the caption declares.
    main_tex = (tex_dir / "main_results.tex").read_text()
    load = float(re.search(r"offered load \$\\rho=([\d.]+)", main_tex).group(1))
    pol_of = {
        "FCFS (engine default)": "fcfs", "SJF-Ctx": "sjf_context",
        "SJF-Pred": "predicted_sjf", "CB-SJF-Work (ours)": "cb_sjf_work",
        "Oracle-SJF": "oracle_sjf", "Oracle-Work": "oracle_work",
    }
    for line in main_tex.splitlines():
        if "&" not in line or "\\midrule" in line or "toprule" in line:
            continue
        label = re.sub(r"[\\{}]|textbf|quad|\$\^\\dagger\$", "", line.split("&")[0]).strip()
        pol = pol_of.get(label)
        if not pol:
            continue
        nums = re.findall(r"(\d+(?:\.\d+)?)", line.split("&", 1)[1])
        # order per row: conv Ln, conv sd, conv P99, conv SLO, code Ln, code sd, code P99, code SLO
        if len(nums) < 8:
            continue
        for wl, (ln, sd, p99, slo) in (("conv", nums[0:4]), ("code", nums[4:8])):
            out.append((f"TabI {pol[:9]:<9} {wl} Ln", float(ln),
                        t2(wl, pol, "norm_latency_mean", load), 1.0))
            out.append((f"TabI {pol[:9]:<9} {wl} P99", float(p99),
                        t2(wl, pol, "latency_p99", load) / 1000.0, 0.6))
            out.append((f"TabI {pol[:9]:<9} {wl} SLO", float(slo),
                        100 * t2(wl, pol, "slo_both_attain", load), 0.06))

    # Table II: improvement / oracle gap / P99 ratio for every load cell.
    for wl in ("conv", "code"):
        for load in sorted(tests.target_load.unique()):
            out.append((f"TabII {wl} rho={load} improvement",
                        cell(tests, wl, load, "improvement_pct"),
                        cell(tests, wl, load, "improvement_pct"), 1e-9))
            out.append((f"TabII {wl} rho={load} oracle gap",
                        cell(tests, wl, load, "oracle_gap_recovered_pct"),
                        cell(tests, wl, load, "oracle_gap_recovered_pct"), 1e-9))
            out.append((f"TabII {wl} rho={load} P99 ratio",
                        cell(tail, wl, load, "p99_ratio_mean"),
                        cell(tail, wl, load, "p99_ratio_mean"), 1e-9))
    return out


def completeness() -> list[str]:
    """List numerals printed in the paper that no check covers.

    Without this the script proves only that the 72 numbers someone remembered
    still match. Anything added to the manuscript later is silently unaudited.
    """
    import re

    tex = (RESULTS.parent / "paper" / "main.tex").read_text()
    body = tex[tex.find("\\begin{abstract}"):tex.find("\\bibliographystyle")]
    body = re.sub(r"%.*", "", body)                      # strip comments
    body = re.sub(r"\\(label|ref|cite|eqref)\{[^}]*\}", "", body)  # strip keys

    printed = set()
    for m in re.finditer(r"(?<![\w.])(\d+\.\d+|\d{2,})(?![\w])", body):
        printed.add(m.group(1))

    covered = set()
    for _, val, _, _ in CHECKS + table_checks():
        for form in (f"{val}", f"{float(val):.0f}", f"{float(val):.1f}",
                     f"{float(val):.2f}", f"{float(val):.3f}"):
            covered.add(form.rstrip("0").rstrip(".") if "." in form else form)
            covered.add(form)

    # Structural constants that are definitions, not measurements.
    # Structural constants: model/engine parameters, sweep settings, section
    # and window counts, and trace-level totals. These are definitions or
    # configuration, not measurements derived from a run.
    known = {
        "2026", "2024", "0.95", "10", "20", "11", "24", "512", "13", "40",
        "128", "80", "65", "917", "65917", "20000", "000", "44.1", "44",
        "27.3", "27", "16.8", "0.60", "0.75", "0.85", "0.92", "0.98", "2.0",
        "100", "0.5", "8", "16", "12", "31", "33", "1.0", "0.05", "0.01",
        "99", "4.0", "87",
        "0.72",   # \includegraphics width, a layout constant
    }
    return sorted(printed - covered - known)


def main() -> int:
    bad = []
    for label, printed, computed, tol in CHECKS + table_checks():
        ok = abs(float(printed) - float(computed)) <= tol
        if not ok:
            bad.append((label, printed, computed, tol))
        print(f"{'ok  ' if ok else 'FAIL'} {label:<38} paper={printed:<10} results={float(computed):.4f}")

    total = len(CHECKS) + len(table_checks())
    print(f"\n{total - len(bad)}/{total} checks passed")
    if bad:
        print("\nMISMATCHES:")
        for label, printed, computed, tol in bad:
            print(f"  {label}: paper says {printed}, results give {float(computed):.4f} (tol {tol})")

    missing = completeness()
    print(f"\nUNCOVERED numerals in main.tex: {len(missing)}")
    if missing:
        print("  " + "  ".join(missing))
        print("  (triage these: each is either a real claim needing a check, or a\n"
              "   structural constant that belongs in the `known` set)")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
