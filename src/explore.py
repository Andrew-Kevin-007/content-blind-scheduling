"""Fast first-look characterisation of the Azure LLM inference traces.

Answers the one question that decides whether the paper has a contribution:
is GeneratedTokens predictable from information available at admission time?
Reads a bounded prefix so it runs on a partially-downloaded file.
"""

import sys

import numpy as np
import pandas as pd

from config import TRACES

NROWS = 3_000_000


def load(path, nrows=NROWS):
    df = pd.read_csv(
        path,
        nrows=nrows,
        on_bad_lines="skip",
        dtype={"ContextTokens": "Int64", "GeneratedTokens": "Int64"},
    )
    df = df.dropna().astype({"ContextTokens": "int64", "GeneratedTokens": "int64"})
    df["TIMESTAMP"] = pd.to_datetime(df["TIMESTAMP"], format="ISO8601")
    return df


def describe(name, df):
    ctx, gen = df["ContextTokens"], df["GeneratedTokens"]
    span = (df["TIMESTAMP"].max() - df["TIMESTAMP"].min()).total_seconds()

    print(f"\n=== {name} ===")
    print(f"rows                 {len(df):,}")
    print(f"span                 {span/3600:.2f} h   ({df['TIMESTAMP'].min()} -> {df['TIMESTAMP'].max()})")
    print(f"arrival rate         {len(df)/span:.2f} req/s")

    for label, s in (("ContextTokens", ctx), ("GeneratedTokens", gen)):
        q = s.quantile([0.5, 0.9, 0.99]).astype(int)
        print(
            f"{label:<20} mean={s.mean():8.1f}  p50={q[0.5]:>6}  "
            f"p90={q[0.9]:>6}  p99={q[0.99]:>6}  max={s.max():>7}"
        )

    # Head-of-line blocking potential: how skewed is the output length?
    print(f"output len CV        {gen.std()/gen.mean():.3f}")
    top = gen.sort_values(ascending=False)
    print(f"top 1% of requests hold {100*top.head(len(gen)//100).sum()/gen.sum():.1f}% of all decode work")

    # The crux: can context length alone say anything about output length?
    pear = np.corrcoef(ctx, gen)[0, 1]
    spear = pd.Series(ctx).corr(pd.Series(gen), method="spearman")
    print(f"corr(ctx,gen)        pearson={pear:+.4f}  spearman={spear:+.4f}")

    # Conditional signal: does mean output vary across context deciles?
    dec = pd.qcut(ctx, 10, duplicates="drop")
    grp = gen.groupby(dec, observed=True).agg(["mean", "median", "count"])
    print("mean output by context decile:")
    print(f"  {' '.join(f'{v:7.1f}' for v in grp['mean'].values)}")
    spread = grp["mean"].max() / max(grp["mean"].min(), 1e-9)
    print(f"  max/min ratio across deciles: {spread:.2f}x")
    return dict(rows=len(df), rate=len(df) / span, spearman=spear, decile_spread=spread)


def main():
    out = {}
    for name, path in TRACES.items():
        if not path.exists():
            print(f"\n=== {name} === MISSING ({path.name}) - skipping")
            continue
        out[name] = describe(name, load(path))

    if len(out) == 2:
        print("\n=== cross-workload ===")
        print("If the two workloads differ materially in output distribution, then")
        print("workload identity alone is a usable content-blind feature.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
