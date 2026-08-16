# Results

Only aggregate or release-level non-respondent experiment outputs belong in version control. Respondent-level BRFSS records, processed respondent rows, and row-level synthetic datasets are deliberately excluded.

- `main_gaussian/` contains the completed primary Gaussian-copula Monte Carlo study: 100 replications per scenario, n=4,000, standard and weak overlap, four causal estimators, and the committed reviewer-facing summaries.
- `pilot/` contains the earlier three-replication Gaussian-copula computational validation. It is retained for provenance but is not the evidence base for the main conclusions.
- `reference_large_sample_check.csv` records the reference-only causal sanity check.
- Smoke-test outputs are generated locally and are not versioned.
- `configs/full.yaml` defines an optional broader Gaussian-copula + CTGAN benchmark. No full CTGAN scientific result is claimed unless that experiment is actually run.

The completed primary results should be interpreted together with `../docs/protocol.md`, `../docs/limitations.md`, and the application/interview brief at `../docs/interview_prep.md`.
