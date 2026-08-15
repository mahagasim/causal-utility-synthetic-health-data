import numpy as np
import pandas as pd
import pytest

from causal_utility.data import (
    ANALYTIC_COLUMNS,
    data_audit,
    prepare_brfss,
    sample_covariates,
)


def toy_raw():
    return pd.DataFrame(
        {
            "_ageg5yr": [1, 14, 13],
            "_sex": [1, 2, 2],
            "educa": [6, 4, 9],
            "income3": [11, 77, 5],
            "_bmi5": [2500, 3000, 7000],
            "_totinda": [1, 2, 9],
            "diabete4": [3, 1, 9],
            "genhlth": [2, 1, 7],
        }
    )


def test_prepare_brfss_restricts_schema_and_invalid_codes():
    out = prepare_brfss(toy_raw())
    assert list(out.columns) == list(ANALYTIC_COLUMNS)
    assert len(out) == 1
    row = out.iloc[0]
    assert row["age_group"] == 1
    assert row["female"] == 0
    assert row["active"] == 1
    assert row["diabetes"] == 0
    assert np.isclose(row["bmi"], 25.0)


def test_prepare_brfss_rejects_missing_schema():
    with pytest.raises(ValueError):
        prepare_brfss(pd.DataFrame({"_ageg5yr": [1]}))


def test_data_audit_reports_screening_before_complete_cases():
    raw = toy_raw()
    cleaned = prepare_brfss(raw)
    audit = data_audit(raw, cleaned)

    age = audit.loc[audit["raw_variable"] == "_ageg5yr"].iloc[0]
    assert np.isclose(age["raw_missing_fraction"], 0.0)
    assert np.isclose(age["invalid_or_out_of_range_fraction"], 1 / 3)
    assert np.isclose(age["usable_fraction_before_complete_case"], 2 / 3)

    bmi = audit.loc[audit["raw_variable"] == "_bmi5"].iloc[0]
    assert np.isclose(bmi["invalid_or_out_of_range_fraction"], 1 / 3)
    assert np.isclose(bmi["usable_fraction_before_complete_case"], 2 / 3)

    sample = audit.loc[audit["raw_variable"] == "__sample__"].iloc[0]
    assert np.isclose(sample["complete_case_pool_fraction"], 1 / 3)
    assert np.isclose(sample["retained_nonmissing_fraction"], 1 / 3)


def test_sample_covariates_reproducible():
    pool = pd.DataFrame({c: np.arange(10) for c in ANALYTIC_COLUMNS})
    a = sample_covariates(pool, 5, seed=7)
    b = sample_covariates(pool, 5, seed=7)
    pd.testing.assert_frame_equal(a, b)
