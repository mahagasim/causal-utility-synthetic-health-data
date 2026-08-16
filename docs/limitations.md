# Limitations and robustness roadmap

## Current limitations

1. **Semi-synthetic outcome and treatment.** Real BRFSS covariates improve realism, but the treatment/outcome mechanisms are designed rather than clinical mechanisms. This is a feature for ground-truth evaluation, but external clinical validity is not claimed.
2. **Empirical reference population.** The MVP targets the sampled BRFSS covariate distribution and does not use survey weights to target the U.S. adult population.
3. **Complete-case covariate pool.** Missing-data modeling is deliberately separated from the primary causal-utility question. BMI outside 12–60 kg/m² is screened as a study-specific plausibility choice rather than an official BRFSS validity range.
4. **Two DGP scenarios.** The project prioritizes interpretable standard and weak-overlap settings rather than an exhaustive simulation factorial.
5. **Finite generator set.** The empirical Gaussian copula is the transparent primary baseline. CTGAN integration is available and smoke-tested, but CTGAN results are not claimed unless the full benchmark is actually run.
6. **Estimator misspecification is partly deliberate.** The IPW benchmark uses a compact main-effects logistic propensity model even though the treatment DGP contains nonlinear terms and an interaction. Persistent IPW bias should therefore be interpreted as misspecification sensitivity, not automatically as synthesis failure. The cross-fitted AIPW estimator is the primary estimator.
7. **Synthetic-data inference.** Naively treating one synthetic dataset as observed can misrepresent uncertainty. The within-dataset SEs reported here are analyst-facing model-based SEs and do not include synthesis-model or original-sample uncertainty. The study therefore emphasizes Monte Carlo empirical SD, SE/SD ratios, coverage, and synthetic-reference point-estimate distortion.
8. **Mechanism-correlation diagnostics are secondary under the constant-effect DGP.** The true individual treatment effect is constant at 2, so any heterogeneity in fitted treatment contrasts is estimation noise/model approximation rather than true effect modification. Correlations between fitted real- and synthetic-data treatment contrasts can therefore be unstable or hard to interpret; contrast RMSE and the prespecified ATE-preservation metrics are more informative for the primary question.
9. **Finite Monte Carlo precision.** The primary application configuration uses 100 replications. For a nominal 95% coverage rate, the Monte Carlo standard error is about 2.2 percentage points, so small coverage differences should not be overinterpreted.
10. **No formal privacy evaluation.** Privacy-risk attacks, formal anonymity guarantees, and differential privacy are outside the MVP.

## High-value extensions

The following are deliberately deferred until the primary results justify them:

- heterogeneous treatment effects/CATE as a robustness target;
- survey-weighted target population;
- BMI-screen sensitivity and explicit missing-data modeling;
- multiple synthetic releases and combining rules for inference;
- membership/attribute inference and distance-to-closest-record diagnostics;
- differentially private generation;
- hybrid causal synthesis that models X, D|X, and Y|D,X separately;
- external validation using controlled-access clinical or routine-care data;
- TMLE/DML only if estimator sensitivity becomes a substantive result.
