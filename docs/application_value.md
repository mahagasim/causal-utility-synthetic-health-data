# Application value

This repository is designed to demonstrate a research workflow rather than a generic machine-learning exercise.

It shows the ability to:

- formulate a causal estimand before choosing models;
- distinguish predictive/distributional fidelity from causal and inferential utility;
- construct a semi-synthetic DGP with known potential outcomes and controlled overlap;
- implement g-computation, IPW, and cross-fitted AIPW in a common evaluation framework;
- evaluate tabular synthesis at the marginal, dependence, mechanism, estimand, and uncertainty levels;
- use realistic health-data covariates without making unsupported clinical claims;
- write modular Python research software with tests, configuration files, deterministic seeds, CI, provenance documentation, and explicit limitations.

The strongest methodological message is not “CTGAN works” or “Gaussian copula wins.” It is that a synthetic dataset can look statistically convincing while its usefulness for a causal target must still be evaluated **at the estimand and inferential levels**.
