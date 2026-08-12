# Data provenance and access

## Source

The empirical covariate distribution is derived from the **2022 Behavioral Risk Factor Surveillance System (BRFSS)**. The working source for this repository is the author's pre-existing reduced coursework extract, `Demo+Health data.xlsx`, which contains selected demographic and health variables from the public-use BRFSS file.

The raw workbook is **not committed**. Put it at:

```text
data/raw/Demo+Health data.xlsx
```

The first run creates a local cache at `data/processed/brfss_covariates.csv`; this cache is also ignored by git because it contains respondent-level rows.

Official source/documentation: https://www.cdc.gov/brfss/annual_data/annual_2022.html

## Variables used as baseline X

| Raw variable | Analytic name | Role |
|---|---|---|
| `_ageg5yr` | `age_group` | baseline demographic/confounder |
| `_sex` | `female` | treatment-only predictor in the DGP |
| `educa` | `education` | outcome-only predictor in the DGP |
| `income3` | `income` | socioeconomic confounder |
| `_bmi5` | `bmi` | continuous health confounder |
| `_totinda` | `active` | behavioral confounder |
| `diabete4` | `diabetes` | baseline health confounder |
| `genhlth` | `general_health` | baseline health confounder |

The project deliberately excludes `smokday2` from the MVP because the reduced extract has extensive missingness consistent with questionnaire skip structure. It also avoids using both BMI and BMI category simultaneously.

## Critical design distinction

BRFSS supplies **only X**. Treatment `D`, potential outcomes `Y(0), Y(1)`, and observed `Y` are simulated. Therefore the project does not estimate or claim a real BRFSS treatment effect.

## Survey design

The primary estimand is defined over the empirical reference covariate distribution used in the simulation, not the U.S. adult population. The MVP therefore does not make population-representative BRFSS claims and does not use survey weights in the causal estimator. A survey-weighted target-population extension is explicitly deferred.
