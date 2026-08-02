"""Content-blind output-length predictor.

Trained on an earlier, disjoint period of the same trace and applied to later
evaluation windows - the way a deployed model actually works. Never trained on
the window it is evaluated on.

For scheduling, only the induced ORDER matters, so the model is evaluated by
rank correlation against true output length in addition to error magnitude.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error

import features as F
from config import CONTENT_BLIND_FEATURES, RESULTS


def train(df_train: pd.DataFrame, seed: int = 0):
    """Fit on log1p(output length); log space keeps the heavy tail from
    dominating the fit, and is monotone so the induced ranking is unchanged."""
    X = F.build(df_train)
    F.assert_content_blind(X, df_train)
    y = np.log1p(df_train["GeneratedTokens"].astype("float64").values)

    model = HistGradientBoostingRegressor(
        max_iter=300,
        learning_rate=0.08,
        max_depth=8,
        min_samples_leaf=200,
        l2_regularization=1.0,
        random_state=seed,
    )
    model.fit(X.values, y)
    return model


def predict(model, df: pd.DataFrame) -> np.ndarray:
    X = F.build(df)
    return np.expm1(model.predict(X.values))


def evaluate(model, df: pd.DataFrame) -> dict:
    """Quality of the predictor in its own right, independent of the scheduler."""
    yhat = predict(model, df)
    y = df["GeneratedTokens"].astype("float64").values

    spearman = pd.Series(yhat).corr(pd.Series(y), method="spearman")
    # Fraction of randomly drawn pairs the predictor orders correctly - this is
    # what actually determines how well shortest-job-first can work.
    rng = np.random.default_rng(0)
    n = len(y)
    i, j = rng.integers(0, n, 200_000), rng.integers(0, n, 200_000)
    ok = (y[i] != y[j])
    concordant = ((yhat[i] < yhat[j]) == (y[i] < y[j]))[ok].mean()

    return {
        "spearman": float(spearman),
        "pairwise_accuracy": float(concordant),
        "mae": float(mean_absolute_error(y, yhat)),
        "mae_log": float(mean_absolute_error(np.log1p(y), np.log1p(yhat))),
        "mean_true": float(y.mean()),
        "mean_pred": float(yhat.mean()),
    }


def feature_importance(model, df: pd.DataFrame, seed: int = 0) -> dict:
    """Permutation importance on a subsample - which metadata actually carries
    the signal. Reported in the paper so the result is interpretable, not a
    black box."""
    from sklearn.inspection import permutation_importance

    sub = df.sample(n=min(20_000, len(df)), random_state=seed)
    X = F.build(sub)
    y = np.log1p(sub["GeneratedTokens"].astype("float64").values)
    r = permutation_importance(
        model, X.values, y, n_repeats=5, random_state=seed, scoring="r2"
    )
    return {
        name: float(val)
        for name, val in zip(CONTENT_BLIND_FEATURES, r.importances_mean)
    }


def save(obj, name: str):
    path = RESULTS / name
    path.write_text(json.dumps(obj, indent=2))
    return path
