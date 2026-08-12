# Limitations and robustness roadmap

## Current limitations

1. **Semi-synthetic outcome and treatment.** Real BRFSS covariates improve realism, but the treatment/outcome mechanisms are designed rather than clinical mechanisms. This is a feature for ground-truth evaluation, but external clinical validity is not claimed.
2. **Empirical reference population.** The MVP targets the sampled BRFSS covariate distribution and does not use survey weights to target the U.S. adult population.
3. **Complete-case covariate pool.** Missing-data modeling is deliberately separated from the primary causal-utility question.
4. **Two DGP scenarios.** The project prioritizes interpretable standard and weak-overlap settings rather than an exhaustive simulation factorial.
5. **Finite generator set.** Gaussian copula and CTGAN represent a transparent statistical baseline and a widely used deep tabular generator; the project is not a generator leaderboard.
6. **Synthetic-data inference.** Naively treating one synthetic dataset as observed can misrepresent uncertainty. This project therefore reports both point-estimand distortion and Monte Carlo uncertainty calibration rather than assuming ordinary model SEs remain valid.
7. **No formal privacy evaluation.** Privacy-risk attacks and differential privacy are outside the MVP.

## High-value extensions

The following are deliberately deferred until the primary results justify them:

- heterogeneous treatment effects/CATE as a robustness target;
- survey-weighted target population;
- multiple synthetic releases and combining rules for inference;
- membership/attribute inference and distance-to-closest-record diagnostics;
- differentially private generation;
- hybrid causal synthesis that models X, D|X, and Y|D,X separately;
- external validation using controlled-access clinical or routine-care data;
- TMLE/DML only if estimator sensitivity becomes a substantive result.
