"""Compact causal estimator hierarchy used identically on reference and synthetic data."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

X_COLUMNS = [
    "age_group",
    "female",
    "education",
    "income",
    "bmi",
    "active",
    "diabetes",
    "general_health",
]


@dataclass
class Estimate:
    estimator: str
    estimate: float
    se: float | None
    ci_low: float | None
    ci_high: float | None
    n: int
    metadata: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def _ci(estimate: float, se: float | None, level: float = 0.95) -> tuple[float | None, float | None]:
    if se is None or not np.isfinite(se):
        return None, None
    z = norm.ppf(0.5 + level / 2)
    return estimate - z * se, estimate + z * se


def _xy(data: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = data[X_COLUMNS].to_numpy(dtype=float)
    d = data["treatment"].to_numpy(dtype=int)
    y = data["outcome"].to_numpy(dtype=float)
    return x, d, y


def crude_difference(data: pd.DataFrame) -> Estimate:
    _, d, y = _xy(data)
    y1, y0 = y[d == 1], y[d == 0]
    if len(y1) < 2 or len(y0) < 2:
        raise ValueError("Both treatment groups need at least two observations")
    tau = float(y1.mean() - y0.mean())
    se = float(np.sqrt(y1.var(ddof=1) / len(y1) + y0.var(ddof=1) / len(y0)))
    lo, hi = _ci(tau, se)
    return Estimate("crude", tau, se, lo, hi, len(data), "Welch-style mean-difference SE")


def _propensity_model(seed: int) -> object:
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, solver="lbfgs", random_state=seed))


def ipw(data: pd.DataFrame, seed: int = 1, clip: float = 0.02) -> Estimate:
    x, d, y = _xy(data)
    model = _propensity_model(seed)
    model.fit(x, d)
    e = np.clip(model.predict_proba(x)[:, 1], clip, 1 - clip)
    w1 = d / e
    w0 = (1 - d) / (1 - e)
    mu1 = float(np.sum(w1 * y) / np.sum(w1))
    mu0 = float(np.sum(w0 * y) / np.sum(w0))
    tau = mu1 - mu0
    influence = (w1 * (y - mu1) / np.mean(w1)) - (w0 * (y - mu0) / np.mean(w0))
    se = float(np.std(influence, ddof=1) / np.sqrt(len(data)))
    lo, hi = _ci(tau, se)
    return Estimate("ipw", tau, se, lo, hi, len(data), f"Hájek-normalized IPW; propensity clipped to [{clip}, {1-clip}]")


def _outcome_model(seed: int) -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(learning_rate=0.06, max_iter=120, max_leaf_nodes=15, l2_regularization=0.5, random_state=seed)


def g_computation(data: pd.DataFrame, seed: int = 1, n_splits: int = 5) -> Estimate:
    x, d, y = _xy(data)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    effect_pred = np.empty(len(data), dtype=float)
    for fold, (train, test) in enumerate(kf.split(x)):
        model = _outcome_model(seed + fold)
        model.fit(np.column_stack([x[train], d[train]]), y[train])
        x0 = np.column_stack([x[test], np.zeros(len(test))])
        x1 = np.column_stack([x[test], np.ones(len(test))])
        effect_pred[test] = model.predict(x1) - model.predict(x0)
    tau = float(np.mean(effect_pred))
    return Estimate("gcomp", tau, None, None, None, len(data), "cross-fitted HGB outcome regression; no naive plug-in CI reported")


def aipw(data: pd.DataFrame, seed: int = 1, n_splits: int = 5, clip: float = 0.02) -> Estimate:
    x, d, y = _xy(data)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    e_hat = np.empty(len(data), dtype=float)
    m0_hat = np.empty(len(data), dtype=float)
    m1_hat = np.empty(len(data), dtype=float)
    for fold, (train, test) in enumerate(kf.split(x)):
        p_model = HistGradientBoostingClassifier(learning_rate=0.06, max_iter=100, max_leaf_nodes=15, l2_regularization=0.5, random_state=seed + 100 + fold)
        p_model.fit(x[train], d[train])
        e_hat[test] = p_model.predict_proba(x[test])[:, 1]
        y_model = _outcome_model(seed + 200 + fold)
        y_model.fit(np.column_stack([x[train], d[train]]), y[train])
        m0_hat[test] = y_model.predict(np.column_stack([x[test], np.zeros(len(test))]))
        m1_hat[test] = y_model.predict(np.column_stack([x[test], np.ones(len(test))]))
    e_hat = np.clip(e_hat, clip, 1 - clip)
    pseudo = m1_hat - m0_hat + d * (y - m1_hat) / e_hat - (1 - d) * (y - m0_hat) / (1 - e_hat)
    tau = float(np.mean(pseudo))
    se = float(np.std(pseudo - tau, ddof=1) / np.sqrt(len(data)))
    lo, hi = _ci(tau, se)
    return Estimate("aipw", tau, se, lo, hi, len(data), f"5-fold cross-fitted HGB nuisances; propensity clipped to [{clip}, {1-clip}]")


def estimate_all(data: pd.DataFrame, seed: int = 1, n_splits: int = 5) -> list[Estimate]:
    return [crude_difference(data), g_computation(data, seed=seed, n_splits=n_splits), ipw(data, seed=seed), aipw(data, seed=seed, n_splits=n_splits)]


def fitted_propensity(data: pd.DataFrame, seed: int = 1) -> np.ndarray:
    x, d, _ = _xy(data)
    model = _propensity_model(seed)
    model.fit(x, d)
    return model.predict_proba(x)[:, 1]
