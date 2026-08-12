"""Conventional fidelity, predictive utility, mechanism preservation, and causal utility."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .estimators import X_COLUMNS
from .synthesize import CATEGORICAL_COLUMNS, CONTINUOUS_COLUMNS


def total_variation(real: pd.Series, synthetic: pd.Series) -> float:
    categories = sorted(set(real.unique()) | set(synthetic.unique()))
    p = real.value_counts(normalize=True).reindex(categories, fill_value=0.0)
    q = synthetic.value_counts(normalize=True).reindex(categories, fill_value=0.0)
    return float(0.5 * np.abs(p - q).sum())


def _numeric_corr_frame(df: pd.DataFrame) -> pd.DataFrame:
    # Codes are aligned by their original BRFSS coding, so numeric correlations
    # are a transparent descriptive dependence diagnostic, not a causal metric.
    return df[CATEGORICAL_COLUMNS + CONTINUOUS_COLUMNS].astype(float)


def distinguishability(
    real: pd.DataFrame,
    synthetic: pd.DataFrame,
    seed: int = 1,
    max_rows: int = 3000,
) -> dict[str, float]:
    """Real-vs-synthetic logistic classifier diagnostics (AUC and raw pMSE)."""
    rng = np.random.default_rng(seed)
    nr = min(len(real), max_rows)
    ns = min(len(synthetic), max_rows)
    r = real.iloc[rng.choice(len(real), nr, replace=False)].copy()
    s = synthetic.iloc[rng.choice(len(synthetic), ns, replace=False)].copy()
    r["__synthetic"] = 0
    s["__synthetic"] = 1
    both = pd.concat([r, s], ignore_index=True)
    y = both.pop("__synthetic").to_numpy()

    cat = [c for c in CATEGORICAL_COLUMNS if c in both.columns]
    cont = [c for c in CONTINUOUS_COLUMNS if c in both.columns]
    pre = ColumnTransformer(
        [("cat", OneHotEncoder(handle_unknown="ignore"), cat), ("cont", StandardScaler(), cont)],
        remainder="drop",
    )
    model = make_pipeline(
        pre,
        LogisticRegression(max_iter=2000, solver="liblinear", random_state=seed),
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    p = cross_val_predict(model, both, y, cv=cv, method="predict_proba")[:, 1]
    c = float(y.mean())
    auc = float(roc_auc_score(y, p))
    symmetric_auc = max(auc, 1.0 - auc)
    return {
        "real_vs_synthetic_auc": symmetric_auc,
        "pmse_raw": float(np.mean((p - c) ** 2)),
    }


def fidelity_report(
    real: pd.DataFrame,
    synthetic: pd.DataFrame,
    seed: int = 1,
) -> dict[str, float]:
    """Compact statistical-fidelity scorecard."""
    metrics: dict[str, float] = {}
    for c in CONTINUOUS_COLUMNS:
        metrics[f"ks_{c}"] = float(ks_2samp(real[c], synthetic[c]).statistic)
        metrics[f"mean_abs_std_diff_{c}"] = float(
            abs(real[c].mean() - synthetic[c].mean()) / max(real[c].std(ddof=1), 1e-9)
        )
    for c in CATEGORICAL_COLUMNS:
        metrics[f"tv_{c}"] = total_variation(real[c], synthetic[c])

    r_corr = _numeric_corr_frame(real).corr().to_numpy()
    s_corr = _numeric_corr_frame(synthetic).corr().to_numpy()
    metrics["correlation_frobenius_per_variable"] = float(
        np.linalg.norm(r_corr - s_corr, ord="fro") / r_corr.shape[0]
    )
    metrics.update(distinguishability(real, synthetic, seed=seed))

    marginal_terms = [
        value
        for key, value in metrics.items()
        if key.startswith(("ks_", "tv_", "mean_abs_std_diff_"))
    ]
    metrics["descriptive_fidelity_distance"] = float(np.mean(marginal_terms))
    return metrics


def predictive_utility(
    real: pd.DataFrame,
    synthetic: pd.DataFrame,
    seed: int = 1,
) -> dict[str, float]:
    """Descriptive train-on-synthetic, test-on-reference predictive utility."""
    x_syn = synthetic[X_COLUMNS].to_numpy(dtype=float)
    x_real = real[X_COLUMNS].to_numpy(dtype=float)
    d_syn = synthetic["treatment"].to_numpy(dtype=int)
    d_real = real["treatment"].to_numpy(dtype=int)
    y_syn = synthetic["outcome"].to_numpy(dtype=float)
    y_real = real["outcome"].to_numpy(dtype=float)

    clf = HistGradientBoostingClassifier(max_iter=80, max_leaf_nodes=15, random_state=seed)
    clf.fit(x_syn, d_syn)
    p = clf.predict_proba(x_real)[:, 1]

    reg = HistGradientBoostingRegressor(
        max_iter=80,
        max_leaf_nodes=15,
        random_state=seed + 1,
    )
    reg.fit(np.column_stack([x_syn, d_syn]), y_syn)
    yhat = reg.predict(np.column_stack([x_real, d_real]))

    return {
        "tstr_treatment_auc": float(roc_auc_score(d_real, p)),
        "tstr_outcome_rmse": float(np.sqrt(mean_squared_error(y_real, yhat))),
    }


def mechanism_preservation(
    real: pd.DataFrame,
    synthetic: pd.DataFrame,
    seed: int = 1,
) -> dict[str, float]:
    """Compare learned treatment and outcome mechanisms on a common real-X grid."""
    x_real = real[X_COLUMNS].to_numpy(dtype=float)
    d_real = real["treatment"].to_numpy(dtype=int)
    y_real = real["outcome"].to_numpy(dtype=float)
    x_syn = synthetic[X_COLUMNS].to_numpy(dtype=float)
    d_syn = synthetic["treatment"].to_numpy(dtype=int)
    y_syn = synthetic["outcome"].to_numpy(dtype=float)

    def propensity_model(seed_: int):
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, random_state=seed_),
        )

    pr = propensity_model(seed)
    ps = propensity_model(seed + 1)
    pr.fit(x_real, d_real)
    ps.fit(x_syn, d_syn)
    er = pr.predict_proba(x_real)[:, 1]
    es = ps.predict_proba(x_real)[:, 1]

    def outcome_model(seed_: int):
        return HistGradientBoostingRegressor(
            max_iter=80,
            max_leaf_nodes=15,
            random_state=seed_,
        )

    yr = outcome_model(seed + 2)
    ys = outcome_model(seed + 3)
    yr.fit(np.column_stack([x_real, d_real]), y_real)
    ys.fit(np.column_stack([x_syn, d_syn]), y_syn)
    x0 = np.column_stack([x_real, np.zeros(len(real))])
    x1 = np.column_stack([x_real, np.ones(len(real))])
    contrast_real = yr.predict(x1) - yr.predict(x0)
    contrast_syn = ys.predict(x1) - ys.predict(x0)

    return {
        "treatment_mechanism_rmse": float(np.sqrt(mean_squared_error(er, es))),
        "treatment_mechanism_correlation": float(np.corrcoef(er, es)[0, 1]),
        "outcome_contrast_rmse": float(
            np.sqrt(mean_squared_error(contrast_real, contrast_syn))
        ),
        "outcome_contrast_correlation": float(
            np.corrcoef(contrast_real, contrast_syn)[0, 1]
        ),
    }


def truth_mechanism_diagnostics(
    reference_truth: pd.DataFrame,
    analysis_data: pd.DataFrame,
    seed: int = 1,
) -> dict[str, float]:
    """Compare learned mechanisms with the known semi-synthetic DGP truth."""
    x_eval = reference_truth[X_COLUMNS].to_numpy(dtype=float)
    true_p = reference_truth["true_propensity"].to_numpy(dtype=float)
    true_tau = reference_truth["individual_effect"].to_numpy(dtype=float)

    x_train = analysis_data[X_COLUMNS].to_numpy(dtype=float)
    d_train = analysis_data["treatment"].to_numpy(dtype=int)
    y_train = analysis_data["outcome"].to_numpy(dtype=float)

    p_model = HistGradientBoostingClassifier(
        learning_rate=0.06,
        max_iter=100,
        max_leaf_nodes=15,
        l2_regularization=0.5,
        random_state=seed,
    )
    p_model.fit(x_train, d_train)
    p_hat = p_model.predict_proba(x_eval)[:, 1]

    y_model = HistGradientBoostingRegressor(
        learning_rate=0.06,
        max_iter=100,
        max_leaf_nodes=15,
        l2_regularization=0.5,
        random_state=seed + 1,
    )
    y_model.fit(np.column_stack([x_train, d_train]), y_train)
    x0 = np.column_stack([x_eval, np.zeros(len(x_eval))])
    x1 = np.column_stack([x_eval, np.ones(len(x_eval))])
    tau_hat = y_model.predict(x1) - y_model.predict(x0)

    metrics = {
        "oracle_propensity_rmse": float(np.sqrt(mean_squared_error(true_p, p_hat))),
        "oracle_propensity_correlation": float(np.corrcoef(true_p, p_hat)[0, 1]),
        "oracle_effect_contrast_bias": float(np.mean(tau_hat - true_tau)),
        "oracle_effect_contrast_mae": float(np.mean(np.abs(tau_hat - true_tau))),
        "oracle_effect_contrast_rmse": float(
            np.sqrt(mean_squared_error(true_tau, tau_hat))
        ),
    }
    if np.std(true_tau) > 1e-10 and np.std(tau_hat) > 1e-10:
        metrics["oracle_effect_contrast_correlation"] = float(
            np.corrcoef(true_tau, tau_hat)[0, 1]
        )
    return metrics


def add_causal_errors(estimates: pd.DataFrame) -> pd.DataFrame:
    """Add estimand-specific errors to replicate-level estimates."""
    out = estimates.copy()
    out["error"] = out["estimate"] - out["true_ate"]
    out["absolute_error"] = out["error"].abs()
    out["relative_absolute_error"] = (
        out["absolute_error"] / out["true_ate"].abs().replace(0, np.nan)
    )
    out["covered"] = np.where(
        out["ci_low"].notna(),
        (out["ci_low"] <= out["true_ate"]) & (out["ci_high"] >= out["true_ate"]),
        np.nan,
    )
    out["sign_preserved"] = np.sign(out["estimate"]) == np.sign(out["true_ate"])
    return out


def summarize_causal_utility(estimates: pd.DataFrame) -> pd.DataFrame:
    """Aggregate Monte Carlo bias, RMSE, uncertainty calibration, and coverage."""
    rows = []
    for keys, g in estimates.groupby(
        ["scenario", "generator", "estimator"],
        dropna=False,
    ):
        scenario, generator, estimator = keys
        empirical_sd = float(g["estimate"].std(ddof=1)) if len(g) > 1 else np.nan
        mean_se = float(g["se"].dropna().mean()) if g["se"].notna().any() else np.nan
        covered = g["covered"].dropna()
        rows.append(
            {
                "scenario": scenario,
                "generator": generator,
                "estimator": estimator,
                "replications": int(len(g)),
                "true_ate": float(g["true_ate"].mean()),
                "mean_estimate": float(g["estimate"].mean()),
                "bias": float(g["error"].mean()),
                "mean_absolute_error": float(g["absolute_error"].mean()),
                "rmse": float(np.sqrt(np.mean(g["error"] ** 2))),
                "empirical_sd": empirical_sd,
                "mean_estimated_se": mean_se,
                "se_ratio": (
                    mean_se / empirical_sd
                    if empirical_sd and np.isfinite(mean_se)
                    else np.nan
                ),
                "coverage_95": float(covered.mean()) if len(covered) else np.nan,
                "mean_ci_width": (
                    float((g["ci_high"] - g["ci_low"]).dropna().mean())
                    if g["ci_low"].notna().any()
                    else np.nan
                ),
                "sign_preservation": float(g["sign_preserved"].mean()),
            }
        )
    return pd.DataFrame(rows)


def summarize_fidelity(long_metrics: pd.DataFrame) -> pd.DataFrame:
    """Summarize replicate-level fidelity and mechanism diagnostics."""
    return (
        long_metrics.groupby(["scenario", "generator", "family", "metric"])["value"]
        .agg(mean="mean", std="std", count="count")
        .reset_index()
    )
