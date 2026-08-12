# Targeted literature review

The project uses the literature to determine evaluation dimensions, not to justify a large model zoo.

## 1. General vs specific utility

Snoke et al. (2018) distinguish general distributional utility from analysis-specific utility and develop propensity-score mean-squared error (pMSE) for synthetic data. This motivates our real-vs-synthetic classifier, pMSE-style diagnostic, marginal comparisons, and the explicit decision **not** to treat general fidelity as sufficient evidence of causal validity.

## 2. Tabular synthesis benchmark

Xu et al. (2019) introduce CTGAN for mixed continuous/discrete tabular data. CTGAN is therefore included as one deep-generative comparator alongside a more transparent Gaussian-copula baseline. The study question is not which generator wins globally.

## 3. Inferential utility

Decruyenaere et al. (2024) show that naive inference from synthetic data can have poor type-I-error behavior and underestimated uncertainty even when point estimates appear acceptable. This is why the causal scorecard includes empirical SD, estimated SE, SE ratio, confidence-interval coverage, and CI width rather than reporting only ATE bias.

## 4. Causal utility in medical synthesis

Amad et al. (2025, preprint) propose that treatment-containing synthetic medical data should preserve the covariate distribution, treatment assignment mechanism, and outcome generation mechanism. The present project operationalizes those three layers through distributional diagnostics plus mechanism-specific diagnostics and an estimand-level scorecard.

## 5. Fidelity may not protect the ATE

Xu (2026, preprint) directly studies cases where generative synthetic data can have high predictive fidelity while distorting causal estimands and discusses positivity problems. That motivates our central “fidelity vs causal error” figure and the weak-overlap stress scenario.

## Interpretation rule

Peer-reviewed sources anchor the conventional/inferential utility components. Recent causal-synthesis preprints motivate emerging causal-utility questions and are labeled as preprints rather than treated as settled evidence.
