"""Large-sample reference-only sanity check for the known causal DGP."""

from pathlib import Path

import pandas as pd

from causal_utility.data import sample_covariates
from causal_utility.dgp import SCENARIOS, observed_analysis_frame, simulate_reference
from causal_utility.estimators import estimate_all


def main() -> None:
    cache = Path("data/processed/brfss_covariates.csv")
    if not cache.exists():
        raise FileNotFoundError("Create the analytic cache by running a configured experiment first.")
    pool = pd.read_csv(cache)
    rows = []
    for scenario_name in ["standard", "weak_overlap"]:
        x = sample_covariates(pool, n=8000, seed=731)
        reference = simulate_reference(x, SCENARIOS[scenario_name], seed=732)
        observed = observed_analysis_frame(reference)
        for estimate in estimate_all(observed, seed=733, n_splits=5):
            rows.append(
                {
                    "scenario": scenario_name,
                    "n": len(observed),
                    "true_ate": reference.attrs["true_ate"],
                    "estimator": estimate.estimator,
                    "estimate": estimate.estimate,
                    "se": estimate.se,
                    "absolute_error": abs(estimate.estimate - reference.attrs["true_ate"]),
                }
            )
    out = Path("results/reference_large_sample_check.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(out)


if __name__ == "__main__":
    main()
