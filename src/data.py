"""Deterministic trace loading and evaluation-window extraction.

Running this twice must produce byte-identical outputs. Windows are chosen by a
fixed rule from the trace itself, never by a random draw, so the "seeds" in the
experiment matrix select *different real periods of production traffic* rather
than different random perturbations of one period.
"""

from __future__ import annotations

import hashlib
import sys

import numpy as np
import pandas as pd

from config import (
    MIN_WINDOW_SEPARATION_MINUTES,
    N_WINDOWS,
    PROC,
    REQUESTS_PER_RUN,
    TRACES,
    WINDOW_MINUTES,
)

USECOLS = ["TIMESTAMP", "ContextTokens", "GeneratedTokens"]


def _cache_path(name: str):
    return PROC / f"{name}.parquet"


def load_trace(name: str, force: bool = False) -> pd.DataFrame:
    """Load a full trace, caching a compact typed copy on first use."""
    cache = _cache_path(name)
    if cache.exists() and not force:
        # Read the workload label back as a categorical rather than as 27M
        # separate Python strings, which alone cost over a gigabyte of RSS.
        df = pd.read_parquet(
            cache, columns=["TIMESTAMP", "ContextTokens", "GeneratedTokens", "t_ms"]
        )
        df["workload"] = pd.Categorical([name] * len(df), categories=list(TRACES))
        return df

    src = TRACES[name]
    if not src.exists():
        raise FileNotFoundError(f"{src} missing - run fetch_data.py")

    df = pd.read_csv(
        src,
        usecols=USECOLS,
        on_bad_lines="skip",
        dtype={"ContextTokens": "Int64", "GeneratedTokens": "Int64"},
    )
    before = len(df)
    df = df.dropna()
    df = df.astype({"ContextTokens": "int32", "GeneratedTokens": "int32"})
    # A request cannot generate zero tokens in this trace's semantics; drop the
    # handful of degenerate rows rather than special-casing them downstream.
    df = df[(df.ContextTokens > 0) & (df.GeneratedTokens > 0)]
    df["TIMESTAMP"] = pd.to_datetime(df["TIMESTAMP"], format="ISO8601", utc=True)
    df = df.sort_values("TIMESTAMP", kind="stable").reset_index(drop=True)
    df["t_ms"] = (
        (df["TIMESTAMP"] - df["TIMESTAMP"].iloc[0]).dt.total_seconds() * 1000.0
    )
    df["workload"] = name

    dropped = before - len(df)
    if dropped:
        print(f"  {name}: dropped {dropped:,} malformed/degenerate rows of {before:,}")

    cache.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache, index=False)
    return df


def pick_windows(
    df: pd.DataFrame,
    n: int = N_WINDOWS,
    minutes: int = WINDOW_MINUTES,
    min_sep_minutes: int = MIN_WINDOW_SEPARATION_MINUTES,
):
    """Sample n windows at even intervals across the whole trace.

    An earlier version took the n busiest windows. That was unnecessary and
    invited a cherry-picking objection: because arrivals are rescaled to a
    target offered load before simulation, a window's absolute arrival rate is
    normalised away entirely, and window choice affects only the request mix.
    Even sampling covers diurnal and weekday/weekend variation instead.

    Windows are required to be at least `min_sep_minutes` apart so that no two
    are adjacent in time and each is a genuinely separate traffic epoch.
    """
    width_ms = minutes * 60_000.0
    sep_ms = max(min_sep_minutes * 60_000.0, width_ms)
    horizon = float(df["t_ms"].iloc[-1])

    # Evenly spaced anchors across the usable span, snapped to a deterministic
    # grid so reruns are byte-identical.
    usable = horizon - width_ms
    anchors = np.linspace(0.0, usable, n)
    grid = np.round(anchors / width_ms) * width_ms

    t = df["t_ms"].values
    chosen: list[float] = []
    for s in grid:
        s = float(min(max(s, 0.0), usable))
        if any(abs(s - c) < sep_ms for c in chosen):
            continue
        # Every run consumes REQUESTS_PER_RUN requests, so a window holding
        # fewer would silently produce a shorter, non-comparable run.
        cnt = np.searchsorted(t, s + width_ms) - np.searchsorted(t, s)
        if cnt < REQUESTS_PER_RUN:
            continue
        chosen.append(s)
    return sorted(chosen)


def window_frame(df: pd.DataFrame, start_ms: float, minutes: int = WINDOW_MINUTES):
    end = start_ms + minutes * 60_000.0
    w = df[(df.t_ms >= start_ms) & (df.t_ms < end)].copy()
    w["t_ms"] = w["t_ms"] - start_ms
    return w.reset_index(drop=True)


def digest(df: pd.DataFrame) -> str:
    """Content hash used to prove the pipeline is deterministic across runs."""
    h = hashlib.sha256()
    for col in ("t_ms", "ContextTokens", "GeneratedTokens"):
        h.update(np.ascontiguousarray(df[col].values).tobytes())
    return h.hexdigest()[:16]


def main() -> int:
    for name in TRACES:
        df = load_trace(name)
        span_h = df["t_ms"].iloc[-1] / 3.6e6
        print(f"\n{name}: {len(df):,} requests over {span_h:.1f} h  digest={digest(df)}")
        for i, s in enumerate(pick_windows(df)):
            w = window_frame(df, s)
            rate = len(w) / (WINDOW_MINUTES * 60)
            print(
                f"  window {i}  t0={s/3.6e6:6.2f}h  n={len(w):>7,}  "
                f"{rate:6.1f} req/s  digest={digest(w)}"
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
