import numpy as np
import pandas as pd

from causal_utility.dgp import SCENARIOS, observed_analysis_frame, simulate_reference


def covariates(n=800):
    rng = np.random.default_rng(4)
    return pd.DataFrame({"age_group": rng.integers(1, 14, n), "female": rng.integers(0, 2, n), "education": rng.integers(1, 7, n), "income": rng.integers(1, 12, n), "bmi": np.clip(rng.normal(28, 5, n), 15, 55), "active": rng.integers(0, 2, n), "diabetes": rng.binomial(1, 0.15, n), "general_health": rng.integers(1, 6, n)})


def test_known_constant_ate_and_potential_outcomes():
    ref = simulate_reference(covariates(), SCENARIOS["standard"], seed=11)
    assert np.allclose(ref["y1"] - ref["y0"], 2.0)
    assert np.isclose(ref.attrs["true_ate"], 2.0)
    observed = np.where(ref.treatment.eq(1), ref.y1, ref.y0)
    assert np.allclose(ref.outcome, observed)


def test_truth_columns_do_not_enter_analysis_frame():
    ref = simulate_reference(covariates(100), SCENARIOS["standard"], seed=12)
    analysis = observed_analysis_frame(ref)
    assert not {"true_propensity", "y0", "y1", "individual_effect"}.intersection(analysis.columns)
    assert {"treatment", "outcome"}.issubset(analysis.columns)


def test_weak_overlap_is_more_extreme_than_standard():
    x = covariates(2000)
    standard = simulate_reference(x, SCENARIOS["standard"], seed=1)
    weak = simulate_reference(x, SCENARIOS["weak_overlap"], seed=1)
    std_extreme = np.mean((standard.true_propensity < 0.1) | (standard.true_propensity > 0.9))
    weak_extreme = np.mean((weak.true_propensity < 0.1) | (weak.true_propensity > 0.9))
    assert weak_extreme > std_extreme
