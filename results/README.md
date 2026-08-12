# Results

Only aggregate experiment outputs belong in version control. Respondent-level BRFSS records and row-level synthetic datasets are deliberately excluded.

- `pilot/` contains the small Gaussian-copula computational validation committed with the repository.
- `reference_large_sample_check.csv` records the reference-only causal sanity check.
- Smoke-test outputs are generated locally and are not versioned.
- The full Gaussian-copula + CTGAN Monte Carlo results should be generated from `configs/full.yaml` only after the optional CTGAN dependency is installed.
