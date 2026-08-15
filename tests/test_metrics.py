import numpy as np
import pandas as pd

from causal_utility.utility_metrics import (
    add_causal_errors,
    summarize_causal_utility,
    total_variation,
)


def test_total_variation_zero_for_identical_distribution():
    x = pd.Series([0, 0, 1, 1])
    assert total_variation(x, x.copy()) == 0.0


def test_causal_error_and_summary():
    rows = pd.DataFrame(
        {
            "scenario": ["standard"] * 2,
            "generator": ["g"] * 2,
            "estimator": ["aipw"] * 2,
            "estimate": [1.8, 2.2],
            "reference_estimate": [1.9, 2.1],
            "true_ate": [2.0, 2.0],
            "se": [0.2, 0.2],
            "ci_low": [1.4, 1.8],
            "ci_high": [2.2, 2.6],
        }
    )
    enriched = add_causal_errors(rows)
    assert np.allclose(enriched["absolute_error"], 0.2)
    assert np.allclose(enriched["reference_estimate_distortion"], [-0.1, 0.1])

    summary = summarize_causal_utility(enriched)
    assert np.isclose(summary.loc[0, "bias"], 0.0)
    assert np.isclose(summary.loc[0, "rmse"], 0.2)
    assert np.isclose(summary.loc[0, "coverage_95"], 1.0)
    assert np.isclose(summary.loc[0, "mean_reference_estimate_distortion"], 0.0)
    assert np.isclose(
        summary.loc[0, "mean_absolute_reference_estimate_distortion"],
        0.1,
    )
    assert np.isclose(
        summary.loc[0, "rmse_reference_estimate_distortion"],
        0.1,
    )
    assert np.isclose(summary.loc[0, "reference_estimate_correlation"], 1.0)
