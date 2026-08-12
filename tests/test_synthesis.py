import numpy as np
import pandas as pd

from causal_utility.synthesize import CATEGORICAL_COLUMNS, CONTINUOUS_COLUMNS, EmpiricalGaussianCopula


def analysis_frame(n=500):
    rng = np.random.default_rng(8)
    return pd.DataFrame({"age_group": rng.integers(1, 14, n), "female": rng.integers(0, 2, n), "education": rng.integers(1, 7, n), "income": rng.integers(1, 12, n), "active": rng.integers(0, 2, n), "diabetes": rng.binomial(1, 0.15, n), "general_health": rng.integers(1, 6, n), "treatment": rng.integers(0, 2, n), "bmi": rng.normal(28, 5, n), "outcome": rng.normal(52, 7, n)})[CATEGORICAL_COLUMNS + CONTINUOUS_COLUMNS]


def test_gaussian_copula_schema_categories_and_reproducibility():
    real = analysis_frame()
    model = EmpiricalGaussianCopula(CATEGORICAL_COLUMNS, CONTINUOUS_COLUMNS, seed=9).fit(real)
    a = model.sample(300, seed=10)
    b = model.sample(300, seed=10)
    pd.testing.assert_frame_equal(a, b)
    assert list(a.columns) == list(real.columns)
    assert len(a) == 300
    for col in CATEGORICAL_COLUMNS:
        assert set(a[col].unique()).issubset(set(real[col].unique()))
    assert not a.isna().any().any()
