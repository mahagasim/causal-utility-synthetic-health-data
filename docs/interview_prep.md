# Interview brief: causal utility of synthetic health data

This note is a compact guide for explaining and defending the project. It is not an additional analysis and does not add claims beyond the committed results.

## One-sentence summary

I used real BRFSS baseline covariates inside a semi-synthetic causal simulation with known ground truth to test whether a synthetic dataset that looks statistically similar to the reference data also preserves a downstream causal ATE analysis; the answer was: not necessarily, especially when overlap is weak.

## 30-second explanation

The project asks whether conventional synthetic-data fidelity is enough for causal research. I sample eight baseline covariates from a cleaned 2022 BRFSS extract, then simulate treatment and outcome so the true ATE is exactly 2. I compare the same causal estimators on the reference and Gaussian-copula synthetic datasets under standard and weak overlap. The synthetic data look very similar by conventional diagnostics—real-versus-synthetic classifier AUC is about 0.52—but AIPW synthetic-reference distortion is still meaningful and increases from RMSE 0.230 to 0.312 under weak overlap. My conclusion is that synthetic health data should be evaluated against the downstream causal question, not only marginal or predictive fidelity.

## Two-minute explanation

The motivation is that synthetic data are often evaluated using distributional similarity or predictive performance, while a researcher may actually care about a causal estimand. I therefore built a controlled semi-synthetic benchmark where the covariate distribution is realistic but the causal truth is known.

The empirical X distribution comes from eight variables in a pre-existing 2022 BRFSS coursework extract. After invalid-code screening, a study-specific BMI plausibility screen, and complete-case restriction, the pool contains 324,636 observations. In each Monte Carlo replicate I sample 4,000 X rows, simulate a binary treatment with nonlinear confounding and target prevalence around 45%, and simulate a continuous outcome with a constant treatment effect of 2. The weak-overlap scenario doubles the scale of the treatment score while leaving the causal truth unchanged.

I apply crude difference in means, cross-fitted g-computation, IPW, and cross-fitted AIPW to the reference data and to an empirical Gaussian-copula synthetic release. AIPW is the primary estimator because the nuisance models are flexible and cross-fitted. IPW is intentionally misspecification-sensitive because its logistic propensity model omits nonlinear terms that are present in the treatment DGP.

The main experiment has 100 replications per scenario. Under standard overlap, reference AIPW RMSE relative to truth is 0.191 and Gaussian-synthetic AIPW RMSE is 0.261. The synthetic-reference AIPW distortion RMSE is 0.230. Under weak overlap it rises to 0.312. At the same time, the discriminator AUC is only about 0.516–0.517 and marginal fidelity distance is around 0.011. So conventional fidelity is excellent, but it does not reliably rank releases by causal preservation. The practical message is that utility assessment should be estimand-specific and should include inferential behavior, especially under positivity problems.

## Technical explanation

### Identification and DGP

The estimand is

\[
\tau = E[Y(1)-Y(0)] = 2.
\]

The design assumes consistency, conditional exchangeability given X, and positivity by construction. BRFSS supplies only X. Treatment D is simulated from a nonlinear propensity model with age, BMI, income, general health, diabetes, activity, sex, an age-by-diabetes interaction, and a quadratic BMI term. The untreated outcome model is also nonlinear and uses overlapping but non-identical predictors. The observed outcome is generated from the corresponding potential outcome plus noise.

Because both potential outcomes and the true propensity are generated internally, truth is known for evaluation. Those truth columns are then removed before synthesis or estimation, preventing leakage.

### Why semi-synthetic rather than fully synthetic simulation?

A fully parametric simulation would make the causal truth easy to control but could produce an unrealistic covariate distribution. Using empirical BRFSS X preserves realistic mixed-type dependence, skewness, and prevalence patterns while retaining a known treatment effect. The trade-off is that the study does not estimate a real BRFSS causal effect and does not claim clinical external validity.

### Why a Gaussian copula?

It is a transparent baseline for mixed tabular synthesis. The aim of this project is not to win a generator leaderboard; it is to make the distinction between statistical fidelity and causal utility easy to inspect. The repository contains optional CTGAN integration, but no CTGAN scientific result is claimed because the completed primary experiment uses only the Gaussian baseline.

### Why AIPW?

AIPW combines an outcome model and propensity model and is doubly robust under standard conditions. Here both nuisance components use flexible histogram gradient boosting and are cross-fitted, reducing overfitting bias. It is the primary estimator because it is more appropriate for the nonlinear DGP than the deliberately compact IPW propensity model.

### Why include crude, g-computation, and IPW?

They create an estimator hierarchy and help distinguish synthesis distortion from estimator behavior. Crude difference is deliberately confounded. G-computation tests outcome-model dependence. IPW is deliberately sensitive to propensity misspecification. If an estimator is already biased on the reference data, that bias should not automatically be attributed to synthesis.

### What exactly is synthetic-reference distortion?

For each scenario, replicate, and estimator:

\[
\Delta_{syn-ref}=\hat\tau_{syn}-\hat\tau_{ref}.
\]

This isolates how much the synthetic release changes the analysis relative to the matching reference sample. It is distinct from truth-relative error. A synthetic estimate can accidentally move closer to the known truth while still changing the original analysis substantially.

### What happened under weak overlap?

The share of reference observations with true propensity outside 0.1–0.9 rises from about 1.84% to 26.03%. Mean estimated IPW effective sample size falls from roughly 3,120 to 1,610 out of 4,000. Under that stress test, AIPW synthetic-reference distortion RMSE increases from 0.230 to 0.312 and IPW distortion RMSE from 0.286 to 0.420.

### What does the coverage result mean?

Reference AIPW coverage is 0.96 under standard overlap and 0.93 under weak overlap. Gaussian-synthetic AIPW coverage is 0.86 in both scenarios. The synthetic-data confidence intervals use analyst-facing within-dataset standard errors; they do not include synthesis-model or original-sample uncertainty. Coverage is therefore a diagnostic of inferential preservation, not evidence that the reported SE is a complete synthetic-data uncertainty procedure. With only 100 Monte Carlo replications, coverage estimates themselves have Monte Carlo standard error of about 2.2 percentage points near 95%.

### Why not interpret the fitted outcome-contrast correlation as a primary result?

The true individual treatment effect is constant at 2. Any heterogeneity in fitted treatment contrasts is therefore model noise or approximation error rather than true effect modification. Correlation of two noisy fitted contrast vectors can be unstable even when the ATE is reasonably preserved. The primary mechanism diagnostics are therefore secondary to prespecified ATE distortion, contrast RMSE, and truth-relative performance. A heterogeneous-effect extension would make contrast correlation more substantively interpretable.

## Numbers to remember

| Quantity | Standard overlap | Weak overlap |
|---|---:|---:|
| True ATE | 2.000 | 2.000 |
| Reference AIPW mean | 1.989 | 2.082 |
| Gaussian AIPW mean | 1.960 | 2.113 |
| Reference AIPW RMSE vs truth | 0.191 | 0.285 |
| Gaussian AIPW RMSE vs truth | 0.261 | 0.307 |
| AIPW RMSE vs matching reference | 0.230 | 0.312 |
| Gaussian AIPW 95% coverage | 0.86 | 0.86 |
| Real-vs-synthetic AUC | 0.516 | 0.517 |
| Descriptive fidelity distance | 0.0110 | 0.0114 |
| True propensity outside 0.1–0.9 | 1.84% | 26.03% |
| Mean estimated IPW ESS | ~3,120 | ~1,610 |

## Likely interview questions

### "Does this prove synthetic data are bad for causal inference?"

No. It shows that conventional statistical fidelity is not sufficient evidence of causal utility. The result is conditional on this DGP, generator, estimators, and sample size. A different synthesis method could perform better or worse.

### "Why did some synthetic estimates become closer to the truth?"

Because synthesis can change the joint distribution in a way that attenuates an estimator's pre-existing bias. That does not mean the original analysis was preserved. This is exactly why I report both truth-relative error and synthetic-reference distortion.

### "Is the synthetic data private?"

The project does not establish that. Synthetic data are not automatically anonymous or differentially private. Privacy-risk evaluation is explicitly outside the primary study.

### "Why no BRFSS survey weights?"

The target estimand is defined over the empirical covariate distribution sampled for the simulation, not the U.S. adult population. Survey-weighted population generalization is a separate extension.

### "Why complete cases?"

The primary question is causal utility after synthesis, not missing-data methodology. Complete-case restriction makes that layer explicit and keeps missing-data modeling from becoming another source of variation. It is a limitation, not a claim that complete-case analysis is universally preferable.

### "Why 100 replications?"

It is a pragmatic application-stage Monte Carlo size that gives stable point-performance summaries while keeping computation manageable. It is enough to see the main preservation pattern, but small coverage differences should not be overinterpreted because Monte Carlo uncertainty remains about 2.2 percentage points around 95% coverage.

### "What would you do next?"

The highest-value next steps are: add heterogeneous treatment effects so CATE/contrast preservation becomes meaningful; compare multiple synthesis mechanisms including CTGAN or causal-structure-aware generators; use multiple synthetic releases with principled combining rules for inference; evaluate survey-weighted targets; and add explicit disclosure/privacy-risk diagnostics so utility and privacy can be studied jointly rather than conflated.

## Claims to avoid

Do **not** say:

- the project estimates a real causal effect in BRFSS;
- the Gaussian copula is state of the art;
- synthetic data are private or anonymized by default;
- CTGAN outperforms or underperforms anything in the main study;
- 0.86 coverage is a precisely estimated universal property of synthetic data;
- better statistical fidelity causes worse causal utility because the observed release-level correlations are descriptive and small;
- the results generalize to all health datasets, generators, estimators, or causal questions.

## Best closing sentence

The project changed the question from "Does the synthetic dataset look like the original?" to "Does it preserve the scientific answer the analyst actually cares about?"—and the experiment shows why those are not equivalent questions.
