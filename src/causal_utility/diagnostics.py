"""Diagnostics that connect causal identification to synthetic-data changes."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .estimators import fitted_propensity


def estimated_overlap(data: pd.DataFrame, seed: int = 1) -> dict[str, float]:
    p = fitted_propensity(data, seed=seed)
    d = data["treatment"].to_numpy(dtype=int)
    weights = d / np.clip(p, 0.02, 0.98) + (1 - d) / np.clip(1 - p, 0.02, 0.98)
    ess = float(weights.sum() ** 2 / np.sum(weights**2))
    return {
        "estimated_propensity_mean": float(np.mean(p)),
        "estimated_propensity_p01": float(np.quantile(p, 0.01)),
        "estimated_propensity_p05": float(np.quantile(p, 0.05)),
        "estimated_propensity_p50": float(np.quantile(p, 0.50)),
        "estimated_propensity_p95": float(np.quantile(p, 0.95)),
        "estimated_propensity_p99": float(np.quantile(p, 0.99)),
        "estimated_share_outside_0.1_0.9": float(np.mean((p < 0.1) | (p > 0.9))),
        "ipw_effective_sample_size": ess,
        "treated_fraction": float(np.mean(d)),
    }
