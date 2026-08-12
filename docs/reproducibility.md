# Reproducibility guide

## Environment

Core package:

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\\Scripts\\activate
pip install -e ".[dev]"
```

For CTGAN:

```bash
pip install -e ".[dev,ctgan]"
```

## Data

Place the pre-existing reduced BRFSS workbook at:

```text
data/raw/Demo+Health data.xlsx
```

Raw and processed respondent-level data are ignored by git.

## Checks

```bash
python -m compileall -q src
pytest
ruff check src tests
```

## Experiments

```bash
causal-utility run --config configs/smoke.yaml --output results/smoke --figures figures/smoke
causal-utility run --config configs/pilot.yaml --output results/pilot --figures figures/pilot
causal-utility run --config configs/full.yaml --output results/full --figures figures/full
```

Reference-only large-sample sanity check:

```bash
python scripts/reference_validation.py
```

## Seeds and leakage safeguards

All experiment layers receive explicit seeds. Counterfactual values, true propensities, and individual effects exist only in the reference truth object and are removed by `observed_analysis_frame()` before synthesis or estimation. Tests explicitly enforce this boundary.

## CI

`.github/workflows/ci.yml` runs compilation, Ruff, and pytest over Python 3.10, 3.12, and 3.14. CTGAN has a separate manually triggered smoke workflow so routine CI does not train a GAN on every push.
