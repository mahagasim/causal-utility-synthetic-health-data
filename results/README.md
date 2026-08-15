# Results

Only non-identifying experiment outputs belong in version control. Respondent-level BRFSS records and row-level synthetic datasets are deliberately excluded.

- `main_gaussian/` contains the completed primary Gaussian-copula experiment: 100 replications × 4,000 observations × two overlap scenarios. Aggregate tables and a compact release-level AIPW/fidelity/overlap table are committed directly. Full replicate-level tables are reproducible from the committed code and configuration. No microdata are included.
- `pilot/` contains the earlier 3-replication Gaussian-copula computational validation. It is retained for pipeline validation only and is not the basis for substantive conclusions.
- `reference_large_sample_check.csv` records the reference-only large-sample estimator sanity check.
- Smoke-test outputs are generated locally and are not versioned.
- CTGAN remains an optional extension and no CTGAN scientific result is reported unless it is explicitly executed and audited.
