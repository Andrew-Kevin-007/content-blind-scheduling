"""Figures for the paper.

Design constraints, in priority order:
  1. IEEE prints in GREYSCALE. Identity is therefore carried by marker shape and
     line style; colour is redundant reinforcement, never the sole encoding.
  2. Column widths are fixed (3.5in single, 7.16in double). Fonts must match the
     body text size after LaTeX places the figure at natural size, so no
     post-hoc scaling is applied.
  3. One y-axis per panel. Never a second scale.

Palette is the Okabe-Ito colour-blind-safe set, ordered so that adjacent series
pass the CVD separation check; the ordering was validated rather than eyeballed.
"""

from __future__ import annotations

import json
import sys

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from config import FIGURES, RESULTS, TARGET_LOADS  # noqa: E402

COL1, COL2 = 3.5, 7.16

plt.rcParams.update({
    # Type 42 (TrueType) rather than matplotlib's default Type 3 bitmap fonts.
    # IEEE PDF eXpress rejects Type 3, and the failure only surfaces at upload
    # time, so it is fixed here at source.
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "mathtext.fontset": "stix",
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 8,
    "legend.fontsize": 7,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "axes.linewidth": 0.6,
    "grid.linewidth": 0.4,
    "lines.markersize": 4,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})

# policy -> (label, colour, marker, linestyle, linewidth)
STYLE = {
    "fcfs":          ("FCFS (default)",      "#0072B2", "o", "-",  1.3),
    "sjf_context":   ("SJF-Ctx",             "#E69F00", "s", "-",  1.3),
    "predicted_sjf": ("SJF-Pred",            "#009E73", "^", "-",  1.3),
    "cb_sjf_work":   ("CB-SJF-Work (ours)",  "#CC79A7", "D", "-",  2.2),
    "oracle_sjf":    ("Oracle-SJF",          "#56B4E9", "v", "--", 1.1),
    "oracle_work":   ("Oracle-Work",         "#D55E00", "x", ":",  1.1),
}
ORDER = list(STYLE)

WORKLOAD_TITLE = {"conv": "Conversation", "code": "Code"}


def _grid(ax):
    ax.grid(True, alpha=0.25, linewidth=0.4)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def fig_work_breakdown(runs: pd.DataFrame):
    """Why the paper works: prefill dominates schedulable work.

    Percentages are read from results/work_share.json, the same file the text
    cites, so the figure and the prose cannot disagree.
    """
    shares = json.loads((RESULTS / "work_share.json").read_text())
    order = [w for w in ("conv", "code") if w in shares]

    fig, ax = plt.subplots(figsize=(COL1, 1.9))
    labels = [WORKLOAD_TITLE[w] for w in order]
    pre = [shares[w]["prefill_pct"] for w in order]
    dec = [shares[w]["decode_pct"] for w in order]
    y = np.arange(len(labels))

    ax.barh(y, pre, height=0.5, color="#0072B2", label="Prefill (observable)",
            edgecolor="white", linewidth=0.8)
    ax.barh(y, dec, left=pre, height=0.5, color="#E69F00",
            label="Decode (unobservable)", edgecolor="white", linewidth=0.8)
    # Texture on the second segment so the split survives greyscale.
    ax.barh(y, dec, left=pre, height=0.5, color="none", hatch="///",
            edgecolor="white", linewidth=0)

    for i, (p, d_) in enumerate(zip(pre, dec)):
        ax.text(p / 2, i, f"{p:.1f}%", ha="center", va="center",
                color="white", fontsize=7)
        # A segment only a couple of percent wide cannot hold a centred label,
        # so anything under 5% is annotated outside the bar with a leader.
        if d_ >= 5:
            ax.text(p + d_ / 2, i, f"{d_:.1f}%", ha="center", va="center",
                    color="#222222", fontsize=7)
        else:
            ax.annotate(
                f"{d_:.1f}%", xy=(100, i), xytext=(88, i + 0.34),
                fontsize=7, color="#222222", ha="right", va="bottom",
                arrowprops=dict(arrowstyle="-", linewidth=0.5, color="#666666",
                                shrinkA=0, shrinkB=1),
            )

    ax.set_yticks(y, labels)
    ax.set_xlabel("Share of total schedulable work (%)")
    ax.set_xlim(0, 100)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=2, frameon=False)
    _grid(ax)
    ax.grid(axis="y", visible=False)
    out = FIGURES / "fig1_work_breakdown.pdf"
    fig.savefig(out)
    plt.close(fig)
    return out


def fig_predictability():
    """Output length vs context decile: signal in conv, none in code."""
    fig, ax = plt.subplots(figsize=(COL1, 2.0))
    for wl, colour, marker, ls in (
        ("conv", "#0072B2", "o", "-"), ("code", "#E69F00", "s", "--")
    ):
        d = pd.read_parquet(
            RESULTS.parent / "data" / "processed" / f"{wl}.parquet",
            columns=["ContextTokens", "GeneratedTokens"],
        )
        d = d.head(3_000_000)
        dec = pd.qcut(d["ContextTokens"], 10, labels=False, duplicates="drop")
        m = d.groupby(dec, observed=True)["GeneratedTokens"].mean()
        ax.plot(np.arange(1, len(m) + 1), m.values, color=colour, marker=marker,
                linestyle=ls, linewidth=1.4, label=WORKLOAD_TITLE[wl])
    ax.set_xlabel("Context-length decile")
    ax.set_ylabel("Mean output tokens")
    ax.legend(frameon=False, loc="upper left")
    _grid(ax)
    out = FIGURES / "fig2_predictability.pdf"
    fig.savefig(out)
    plt.close(fig)
    return out


def fig_main(runs: pd.DataFrame):
    """Headline: normalised latency vs offered load, per workload."""
    fig, axes = plt.subplots(1, 2, figsize=(COL2, 2.45), sharex=True)
    for ax, wl in zip(axes, ("conv", "code")):
        for pol in ORDER:
            sub = runs[(runs.workload == wl) & (runs.policy == pol)]
            if sub.empty:
                continue
            g = sub.groupby("target_load")["norm_latency_mean"]
            mean, std = g.mean(), g.std()
            label, colour, marker, ls, lw = STYLE[pol]
            ax.errorbar(mean.index, mean.values, yerr=std.values, color=colour,
                        marker=marker, linestyle=ls, linewidth=lw, capsize=2,
                        elinewidth=0.6, label=label)
        ax.set_yscale("log")
        ax.set_title(WORKLOAD_TITLE[wl])
        ax.set_xlabel(r"Offered load $\rho$")
        ax.set_xticks(TARGET_LOADS)
        _grid(ax)
    axes[0].set_ylabel("Mean normalised latency\n(ms per output token)")
    handles, labels = axes[0].get_legend_handles_labels()
    # The figure is placed at 0.72\textwidth, so fonts shrink by the same factor.
    # Reserve real space beneath the axes for the legend rather than letting it
    # ride up against the x-axis labels.
    fig.subplots_adjust(bottom=0.30)
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.02),
               ncol=3, frameon=False, columnspacing=1.4, handlelength=2.0)
    out = FIGURES / "fig3_main.pdf"
    fig.savefig(out)
    plt.close(fig)
    return out


def fig_oracle_gap(tests: pd.DataFrame):
    """How much of the achievable oracle benefit the content-blind policy keeps."""
    fig, ax = plt.subplots(figsize=(COL1, 2.0))
    width = 0.36
    x = np.arange(len(TARGET_LOADS))
    for i, (wl, colour, hatch) in enumerate(
        (("conv", "#0072B2", ""), ("code", "#E69F00", "///"))
    ):
        sub = tests[tests.workload == wl].sort_values("target_load")
        vals, imps = [], []
        for L in TARGET_LOADS:
            row = sub[sub.target_load == L]
            vals.append(float(row.oracle_gap_recovered_pct.iloc[0]) if not row.empty else np.nan)
            imps.append(float(row.improvement_pct.iloc[0]) if not row.empty else np.nan)
        # A recovered-gap percentage is a ratio of two quantities that both go to
        # zero at low load, so it reads as a large bar where nothing is actually
        # happening. Annotating each bar with the absolute improvement it
        # corresponds to prevents that misreading.
        ax.bar(x + (i - 0.5) * width, vals, width, color=colour, hatch=hatch,
               edgecolor="white", linewidth=0.7, label=WORKLOAD_TITLE[wl])
        for xi, v, im in zip(x + (i - 0.5) * width, vals, imps):
            ax.text(xi, v + 2, f"{im:.0f}", ha="center", va="bottom",
                    fontsize=5.5, color="#333333", rotation=90)
    ax.axhline(100, color="#444444", linewidth=0.8, linestyle=":")
    ax.text(-0.45, 101.5, "oracle", fontsize=6, ha="left", color="#444444")
    ax.set_xticks(x, [f"{L:.2f}" for L in TARGET_LOADS])
    ax.set_xlabel(r"Offered load $\rho$")
    ax.set_ylabel("Oracle gap recovered (%)")
    ax.set_ylim(0, 132)
    # Every bar sits between 75% and 98%, so there is no clear interior space;
    # the legend goes above the axes rather than on top of the data.
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=2, frameon=False)
    _grid(ax)
    out = FIGURES / "fig4_oracle_gap.pdf"
    fig.savefig(out)
    plt.close(fig)
    return out


def main() -> int:
    runs = pd.DataFrame(json.loads((RESULTS / "raw" / "runs.json").read_text()))
    tests = pd.read_csv(RESULTS / "paired_tests.csv")
    for fn, arg in (
        (fig_work_breakdown, runs), (fig_predictability, None),
        (fig_main, runs), (fig_oracle_gap, tests),
    ):
        out = fn(arg) if arg is not None else fn()
        print(f"wrote {out.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
