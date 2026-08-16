"""BRFSS data loading and scientifically conservative preprocessing."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

RAW_TO_ANALYTIC = {
    "_ageg5yr": "age_group",
    "_sex": "female",
    "educa": "education",
    "income3": "income",
    "_bmi5": "bmi",
    "_totinda": "active",
    "diabete4": "diabetes",
    "genhlth": "general_health",
}

REQUIRED_RAW_COLUMNS = tuple(RAW_TO_ANALYTIC)
ANALYTIC_COLUMNS = tuple(RAW_TO_ANALYTIC.values())

# 2022 BRFSS special codes used in the user's prior coursework and checked against
# the official 2022 documentation before this project was built.
INVALID_CODES: dict[str, Iterable[float]] = {
    "_ageg5yr": (14,),
    "educa": (9,),
    "income3": (77, 99),
    "_totinda": (9,),
    "diabete4": (7, 9),
    "genhlth": (7, 9),
}

# Study-specific plausibility screen for the empirical covariate pool. This is
# not presented as an official BRFSS validity range; it should be reported as an
# analysis choice and can be varied in a later sensitivity analysis.
BMI_MIN_HUNDREDTHS = 1200
BMI_MAX_HUNDREDTHS = 6000


def load_brfss(path: str | Path) -> pd.DataFrame:
    """Load a BRFSS reduced extract from xlsx, csv, parquet, or Stata format."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".xlsx":
        return pd.read_excel(path)
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".dta":
        return pd.read_stata(path)
    raise ValueError(f"Unsupported BRFSS input format: {suffix}")


def validate_required_columns(df: pd.DataFrame) -> None:
    """Fail clearly when the reduced BRFSS schema is not available."""
    missing = sorted(set(REQUIRED_RAW_COLUMNS) - set(df.columns))
    if missing:
        raise ValueError(f"Input is missing required BRFSS variables: {missing}")


def _screen_raw_covariates(df: pd.DataFrame) -> pd.DataFrame:
    """Apply special-code and plausibility screens before analytic recoding."""
    validate_required_columns(df)
    x = df.loc[:, REQUIRED_RAW_COLUMNS].copy()

    for column, invalid in INVALID_CODES.items():
        x[column] = x[column].replace(list(invalid), np.nan)

    x.loc[
        (x["_bmi5"] < BMI_MIN_HUNDREDTHS)
        | (x["_bmi5"] > BMI_MAX_HUNDREDTHS),
        "_bmi5",
    ] = np.nan
    return x


def prepare_brfss(df: pd.DataFrame) -> pd.DataFrame:
    """Create the eight-variable empirical baseline covariate pool.

    The project intentionally uses complete cases only after limiting the raw
    file to the covariates that enter the DGP. This avoids dropping records for
    variables that are irrelevant to the causal experiment.

    Notes
    -----
    * ``smokday2`` is not used in the MVP because the reduced extract contains
      extensive structural/questionnaire-skip missingness.
    * ``diabetes`` is a simple indicator for a current diabetes diagnosis
      (DIABETE4 == 1). Other valid categories remain 0; the variable is used
      only as a realistic baseline feature, not for a clinical claim.
    * ``active`` equals 1 when _TOTINDA == 1 (reported leisure-time activity).
    * BMI values outside 12--60 kg/m^2 are screened as a study-specific
      plausibility choice, not as an official BRFSS validity rule.
    """
    x = _screen_raw_covariates(df)

    out = pd.DataFrame(index=x.index)
    out["age_group"] = x["_ageg5yr"]
    out["female"] = (x["_sex"] == 2).astype(float)
    out["education"] = x["educa"]
    out["income"] = x["income3"]
    out["bmi"] = x["_bmi5"] / 100.0
    out["active"] = (x["_totinda"] == 1).astype(float)
    out["diabetes"] = (x["diabete4"] == 1).astype(float)
    out["general_health"] = x["genhlth"]

    for raw, analytic in (
        ("_sex", "female"),
        ("_totinda", "active"),
        ("diabete4", "diabetes"),
    ):
        out.loc[x[raw].isna(), analytic] = np.nan

    out = out.dropna().reset_index(drop=True)

    for c in [
        "age_group",
        "female",
        "education",
        "income",
        "active",
        "diabetes",
        "general_health",
    ]:
        out[c] = out[c].astype(int)
    out["bmi"] = out["bmi"].astype(float)

    return out.loc[:, ANALYTIC_COLUMNS]


def data_audit(raw: pd.DataFrame, cleaned: pd.DataFrame) -> pd.DataFrame:
    """Return source missingness, screening loss, and complete-case retention."""
    screened = _screen_raw_covariates(raw)
    pool_fraction = len(cleaned) / len(raw) if len(raw) else np.nan
    rows = []
    for raw_col, analytic_col in RAW_TO_ANALYTIC.items():
        raw_missing = raw[raw_col].isna()
        screened_missing = screened[raw_col].isna()
        invalid_or_out_of_range = (~raw_missing) & screened_missing
        usable_fraction = float(screened[raw_col].notna().mean())
        rows.append(
            {
                "raw_variable": raw_col,
                "analytic_variable": analytic_col,
                "raw_missing_fraction": float(raw_missing.mean()),
                "invalid_or_out_of_range_fraction": float(
                    invalid_or_out_of_range.mean()
                ),
                "usable_fraction_before_complete_case": usable_fraction,
                # Retained for backward-compatible output naming. For variable
                # rows this now means usable after variable-specific screening,
                # rather than the uninformative post-complete-case value 1.0.
                "retained_nonmissing_fraction": usable_fraction,
                "complete_case_pool_fraction": pool_fraction,
            }
        )
    rows.append(
        {
            "raw_variable": "__sample__",
            "analytic_variable": "complete_case_pool",
            "raw_missing_fraction": np.nan,
            "invalid_or_out_of_range_fraction": np.nan,
            "usable_fraction_before_complete_case": np.nan,
            "retained_nonmissing_fraction": pool_fraction,
            "complete_case_pool_fraction": pool_fraction,
        }
    )
    return pd.DataFrame(rows)


def sample_covariates(pool: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """Sample covariates with replacement from the empirical BRFSS pool."""
    if n <= 0:
        raise ValueError("n must be positive")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(pool), size=n)
    return pool.iloc[idx].reset_index(drop=True).copy()
