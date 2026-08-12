import numpy as np
import pandas as pd
import pytest

pytest.importorskip("ctgan")

from causal_utility.synthesize import (  # noqa: E402
    CATEGORICAL_COLUMNS,
    CONTINUOUS_COLUMNS,
    synthesize_ctgan,
)


def test_ctgan_tiny_end_to_end_synthesis():
    """Exercise the optional CTGAN path without requiring BRFSS data."""
    rng = np.random.default_rng(42)
    n = 120
    real = pd.DataFrame(
        {
            "age_group": rng.integers(1, 14, n),
            "female": rng.integers(0, 2, n),
            "education": rng.integers(1, 7, n),
            "income": rng.integers(1, 12, n),
            "active": rng.integers(0, 2, n),
            "diabetes": rng.binomial(1, 0.15, n),
            "general_health": rng.integers(1, 6, n),
            "treatment": rng.integers(0, 2, n),
            "bmi": np.clip(rng.normal(28.0, 5.0, n), 15.0, 55.0),
            "outcome": rng.normal(52.0, 7.0, n),
        }
    )[CATEGORICAL_COLUMNS + CONTINUOUS_COLUMNS]

    synthetic = synthesize_ctgan(real, seed=43, epochs=1)

    assert len(synthetic) == len(real)
    assert list(synthetic.columns) == list(real.columns)
    assert not synthetic.isna().any().any()
    for column in CATEGORICAL_COLUMNS:
        assert set(synthetic[column].unique()).issubset(set(real[column].unique()))
