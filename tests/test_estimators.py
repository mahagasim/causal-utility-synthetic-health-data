import numpy as np
import pandas as pd

from causal_utility.dgp import SCENARIOS, observed_analysis_frame, simulate_reference
from causal_utility.estimators import aipw, estimate_all


def randomized_like_covariates(n=1200):
    rng = np.random.default_rng(13)
    return pd.DataFrame(
        {
            "age_group": rng.integers(1, 14, n),
            "female": rng.integers(0, 2, n),
            "education": rng.integers(1, 7, n),
            "income": rng.integers(1, 12, n),
            "bmi": np.clip(rng.normal(27, 4, n), 15, 50),
            "active": rng.integers(0, 2, n),
            "diabetes": rng.binomial(1, 0.12, n),
            "general_health": rng.integers(1, 6, n),
        }
    )


def test_estimator_hierarchy_returns_finite_point_estimates():
    data = observed_analysis_frame(
        simulate_reference(
            randomized_like_covariates(),
            SCENARIOS["standard"],
            seed=14,
        )
    )
    estimates = estimate_all(data, seed=15, n_splits=3)
    assert [e.estimator for e in estimates] == ["crude", "gcomp", "ipw", "aipw"]
    assert all(np.isfinite(e.estimate) for e in estimates)
    assert all(e.n == len(data) for e in estimates)


def test_aipw_is_reproducible_and_reports_finite_uncertainty():
    data = observed_analysis_frame(
        simulate_reference(
            randomized_like_covariates(),
            SCENARIOS["standard"],
            seed=21,
        )
    )
    first = aipw(data, seed=22, n_splits=3)
    second = aipw(data, seed=22, n_splits=3)

    assert np.isclose(first.estimate, second.estimate)
    assert np.isclose(first.se, second.se)
    assert first.se is not None and np.isfinite(first.se) and first.se > 0
    assert first.ci_low is not None and first.ci_high is not None
    assert first.ci_low < first.estimate < first.ci_high
