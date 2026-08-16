# Reproducibility guide

## Environment

Core package:

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

For the optional CTGAN integration:

```bash
pip install -e ".[dev,ctgan]"
```

## Data

Place the pre-existing reduced BRFSS workbook at:

```text
data/raw/Demo+Health data.xlsx
```

Raw and processed respondent-level data are ignored by git. The first full read may create the local cache `data/processed/brfss_covariates.csv`; that cache is also excluded from version control.

## Checks

```bash
python -m compileall -q src
pytest
ruff check src tests
```

## Experiments

Execution smoke test:

```bash
causal-utility run --config configs/smoke.yaml --output results/smoke --figures figures/smoke
```

Small computational pilot (not for inference):

```bash
causal-utility run --config configs/pilot.yaml --output results/pilot --figures figures/pilot
```

Completed primary Gaussian-copula protocol:

```bash
causal-utility run --config configs/main_gaussian.yaml --output results/main_gaussian --figures figures/main_gaussian
```

The committed primary results use `n=4000`, 100 replications per scenario, two overlap scenarios, the empirical Gaussian copula, and base seed `20260811`.

`configs/full.yaml` is retained only as an optional broader Gaussian-copula + CTGAN benchmark. It is **not** the source of the reported primary results and no full CTGAN scientific result is claimed in this repository.

Reference-only large-sample sanity check:

```bash
python scripts/reference_validation.py
```

## Seeds and leakage safeguards

All experiment layers receive explicit seeds. Counterfactual values, true propensities, and individual effects exist only in the reference truth object and are removed by `observed_analysis_frame()` before synthesis or estimation. Tests explicitly enforce this boundary.

The main run is reproducible from the committed code, `configs/main_gaussian.yaml`, the source workbook, and the base seed. Aggregate result tables are versioned; respondent-level BRFSS rows, processed respondent rows, and row-level synthetic releases are not.

## Result verification

The primary result directory is:

```text
results/main_gaussian/
```

Key files:

- `run_config.json` — executed configuration;
- `data_audit.csv` — preprocessing and retention audit;
- `main_results_table.csv` — compact causal results;
- `causal_summary.csv` — full causal Monte Carlo summary;
- `fidelity_summary.csv` — conventional, predictive, and mechanism diagnostics;
- `overlap_summary.csv` — overlap stress-test summary;
- `aipw_release_level.csv` — release-level AIPW preservation diagnostics.

The final execution path was cross-checked by rerunning the original three-replication pilot and reproducing its committed estimator summaries exactly.

## CI

`.github/workflows/ci.yml` runs compilation, Ruff, and pytest over Python 3.10, 3.12, and 3.14. CTGAN has a separate manually triggered smoke workflow so routine CI does not train a GAN on every push.
