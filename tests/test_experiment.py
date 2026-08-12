from pathlib import Path

import pandas as pd

from causal_utility.experiment import run_experiment


def test_end_to_end_on_analytic_cache(tmp_path: Path):
    rows = []
    for i in range(160):
        rows.append({"age_group": i % 13 + 1, "female": i % 2, "education": i % 6 + 1, "income": i % 11 + 1, "bmi": 20.0 + (i % 20) * 0.6, "active": (i // 2) % 2, "diabetes": 1 if i % 9 == 0 else 0, "general_health": i % 5 + 1})
    cache = tmp_path / "pool.csv"
    pd.DataFrame(rows).to_csv(cache, index=False)
    cfg = {"data": {"path": str(tmp_path / "not_needed.xlsx"), "cache_path": str(cache)}, "experiment": {"n": 120, "repetitions": 1, "seed": 123, "n_splits": 2, "scenarios": ["standard"], "generators": ["gaussian_copula"], "ctgan_epochs": 1}}
    result = run_experiment(cfg)
    assert {"estimates", "causal_summary", "fidelity", "overlap"}.issubset(result)
    assert set(result["estimates"]["generator"]) == {"reference", "gaussian_copula"}
    assert set(result["estimates"]["estimator"]) == {"crude", "gcomp", "ipw", "aipw"}
