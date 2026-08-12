"""Semi-synthetic causal data-generating process with known potential outcomes."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.special import expit

AGE_MIDPOINTS = np.array([21, 27, 32, 37, 42, 47, 52, 57, 62, 67, 72, 77, 82], dtype=float)


@dataclass(frozen=True)
class Scenario:
    """Configuration for one causal simulation scenario."""

    name: str
    propensity_scale: float = 1.0
    target_treatment_prevalence: float = 0.45
    treatment_effect: float = 2.0
    outcome_noise_sd: float = 5.0
    heterogeneous_effect: bool = False


SCENARIOS = {
    "standard": Scenario(name="standard", propensity_scale=1.0),
    "weak_overlap": Scenario(name="weak_overlap", propensity_scale=2.0),
}


def _safe_z(series: pd.Series) -> np.ndarray:
    x = series.to_numpy(dtype=float)
    sd = x.std(ddof=0)
    if sd == 0:
        return np.zeros_like(x)
    return (x - x.mean()) / sd


def build_dgp_features(x: pd.DataFrame) -> pd.DataFrame:
    """Construct standardized features used only by the known structural DGP."""
    out = pd.DataFrame(index=x.index)
    age_idx = x["age_group"].astype(int).to_numpy() - 1
    if np.any((age_idx < 0) | (age_idx >= len(AGE_MIDPOINTS))):
        raise ValueError("age_group must be coded 1..13")
    out["age"] = AGE_MIDPOINTS[age_idx]
    for c in ["education", "income", "bmi", "general_health"]:
        out[c] = x[c].astype(float)
    for c in ["female", "active", "diabetes"]:
        out[c] = x[c].astype(float)

    for c in ["age", "education", "income", "bmi", "general_health"]:
        out[f"{c}_z"] = _safe_z(out[c])
    return out


def _calibrated_propensity(score: np.ndarray, scale: float, target: float) -> tuple[np.ndarray, float]:
    if not 0 < target < 1:
        raise ValueError("target treatment prevalence must lie in (0,1)")

    def objective(alpha: float) -> float:
        return float(expit(alpha + scale * score).mean() - target)

    alpha = brentq(objective, -20.0, 20.0)
    return expit(alpha + scale * score), float(alpha)


def simulate_reference(x: pd.DataFrame, scenario: Scenario, seed: int) -> pd.DataFrame:
    """Simulate treatment and continuous potential outcomes on empirical X.

    Variable roles are intentionally explicit:
    * Confounders: age, BMI, income, activity, diabetes, general health.
    * Treatment-only predictor: sex (female indicator).
    * Outcome-only predictor: education.
    * Effect modification: absent in the primary protocol; optional in config.

    The observed outcome is a simulated health-risk score in arbitrary units.
    It is not a clinical endpoint and should not be interpreted as one.
    """
    f = build_dgp_features(x)
    rng = np.random.default_rng(seed)

    treatment_score = (
        0.35 * f["age_z"]
        + 0.25 * f["bmi_z"]
        - 0.30 * f["income_z"]
        + 0.35 * f["general_health_z"]
        + 0.45 * f["diabetes"]
        - 0.25 * f["active"]
        + 0.10 * f["female"]
        + 0.18 * f["age_z"] * f["diabetes"]
        + 0.12 * (f["bmi_z"] ** 2 - 1.0)
    ).to_numpy()
    propensity, intercept = _calibrated_propensity(
        treatment_score,
        scenario.propensity_scale,
        scenario.target_treatment_prevalence,
    )
    treatment = rng.binomial(1, propensity)

    mu0 = (
        50.0
        + 2.8 * f["general_health_z"]
        + 1.5 * f["bmi_z"]
        + 1.1 * f["age_z"]
        - 1.0 * f["income_z"]
        - 1.2 * f["active"]
        + 2.0 * f["diabetes"]
        - 0.6 * f["education_z"]
        + 0.8 * (f["bmi_z"] ** 2 - 1.0)
        + 0.7 * f["age_z"] * f["general_health_z"]
    ).to_numpy()

    if scenario.heterogeneous_effect:
        tau_i = (
            scenario.treatment_effect
            + 0.40 * f["diabetes"].to_numpy()
            - 0.30 * f["active"].to_numpy()
        )
    else:
        tau_i = np.full(len(x), scenario.treatment_effect, dtype=float)

    epsilon = rng.normal(0.0, scenario.outcome_noise_sd, size=len(x))
    y0 = mu0 + epsilon
    y1 = mu0 + tau_i + epsilon
    outcome = np.where(treatment == 1, y1, y0)

    out = x.copy().reset_index(drop=True)
    out["treatment"] = treatment.astype(int)
    out["outcome"] = outcome.astype(float)
    out["true_propensity"] = propensity.astype(float)
    out["y0"] = y0.astype(float)
    out["y1"] = y1.astype(float)
    out["individual_effect"] = tau_i.astype(float)
    out.attrs["true_ate"] = float(np.mean(tau_i))
    out.attrs["propensity_intercept"] = intercept
    out.attrs["scenario"] = scenario.name
    return out


def observed_analysis_frame(reference: pd.DataFrame) -> pd.DataFrame:
    """Drop counterfactual/truth columns before data synthesis or estimation."""
    keep = [
        "age_group",
        "female",
        "education",
        "income",
        "bmi",
        "active",
        "diabetes",
        "general_health",
        "treatment",
        "outcome",
    ]
    return reference.loc[:, keep].copy()


def overlap_summary(propensity: np.ndarray, treatment: np.ndarray | None = None) -> dict[str, float]:
    """Summarize practical positivity and, optionally, realized treatment rate."""
    p = np.asarray(propensity, dtype=float)
    result = {
        "propensity_mean": float(np.mean(p)),
        "propensity_min": float(np.min(p)),
        "propensity_p01": float(np.quantile(p, 0.01)),
        "propensity_p05": float(np.quantile(p, 0.05)),
        "propensity_p50": float(np.quantile(p, 0.50)),
        "propensity_p95": float(np.quantile(p, 0.95)),
        "propensity_p99": float(np.quantile(p, 0.99)),
        "propensity_max": float(np.max(p)),
        "share_outside_0.1_0.9": float(np.mean((p < 0.1) | (p > 0.9))),
    }
    if treatment is not None:
        result["treated_fraction"] = float(np.mean(treatment))
    return result
