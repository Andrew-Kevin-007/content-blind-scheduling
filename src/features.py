"""Content-blind feature construction.

The defining constraint of this work: the scheduler may not read prompt text. It
sees only what a privacy-respecting front end can expose - token counts, arrival
timing, workload identity, and statistics over requests that have ALREADY
completed.

Two invariants are enforced here, because violating either would silently
manufacture the paper's headline result:
  1. No feature may depend on the request's own GeneratedTokens.
  2. Every rolling statistic is shifted by one, so a request never sees its own
     outcome or any future outcome.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from config import CONTENT_BLIND_FEATURES

RECENT_WINDOW = 512   # requests of history the front end keeps


def build(df: pd.DataFrame) -> pd.DataFrame:
    """Return a frame of content-blind features aligned to df's rows."""
    out = pd.DataFrame(index=df.index)

    ctx = df["ContextTokens"].astype("float64")
    out["context_tokens"] = ctx
    out["log_context_tokens"] = np.log1p(ctx)
    out["workload_is_code"] = (df["workload"] == "code").astype("float64")

    ts = df["TIMESTAMP"]
    out["hour_of_day"] = ts.dt.hour.astype("float64")
    out["day_of_week"] = ts.dt.dayofweek.astype("float64")

    # Causal rolling statistics: shift(1) guarantees the current request's own
    # outcome is excluded from its own features.
    gen = df["GeneratedTokens"].astype("float64")
    past_gen = gen.shift(1)
    past_ctx = ctx.shift(1)

    roll = past_gen.rolling(RECENT_WINDOW, min_periods=8)
    out["recent_mean_output"] = roll.mean()
    out["recent_p90_output"] = roll.quantile(0.90)
    out["recent_mean_context"] = past_ctx.rolling(RECENT_WINDOW, min_periods=8).mean()

    inter = df["t_ms"].diff()
    out["inter_arrival_ms"] = inter
    # Arrival rate over the recent window, in requests/second.
    out["recent_arrival_rate"] = 1000.0 / inter.shift(1).rolling(
        RECENT_WINDOW, min_periods=8
    ).mean().replace(0, np.nan)

    out = out[CONTENT_BLIND_FEATURES]

    # Early rows lack history; fill with column medians so the first requests in
    # a window are still schedulable rather than dropped.
    return out.fillna(out.median(numeric_only=True)).fillna(0.0)


def assert_content_blind(features: pd.DataFrame, df: pd.DataFrame) -> None:
    """Guard against the one bug that would invalidate the paper.

    If any feature is near-perfectly rank-correlated with the target, we are
    almost certainly leaking the label rather than predicting it.
    """
    target = df["GeneratedTokens"].astype("float64")
    for col in features.columns:
        s = features[col]
        if s.nunique() < 2:
            continue
        rho = abs(s.corr(target, method="spearman"))
        if rho > 0.95:
            raise AssertionError(
                f"feature '{col}' has |spearman|={rho:.4f} against the target; "
                "this indicates label leakage, not predictive signal"
            )
