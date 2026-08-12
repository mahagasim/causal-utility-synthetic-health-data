"""End-to-end Monte Carlo experiment orchestration."""

from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from .data import data_audit, load_brfss, prepare_brfss, sample_covariates
from .dgp import SCENARIOS, Scenario, observed_analysis_frame, overlap_summary, simulate_reference
from .diagnostics import estimated_overlap
from .estimators import estimate_all
from .plotting import (
    plot_bmi_distribution,
    plot_causal_bias,
    plot_dag,
    plot_fidelity_vs_causal_error,
    plot_propensity_overlap,
)
from .synthesize import synthesize
from .utility_metrics import (
    add_causal_errors,
    fidelity_report,
    mechanism_preservation,
    predictive_utility,
    truth_mechanism_diagnostics,
    summarize_causal_utility,
    summarize_fidelity,
)


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError("Experiment config must be a YAML mapping")
    return cfg


def _scenario_from_config(name: str, overrides: dict[str, Any] | None = None) -> Scenario:
    if name not in SCENARIOS:
        raise ValueError(f"Unknown scenario {name!r}; choose from {sorted(SCENARIOS)}")
    base = asdict(SCENARIOS[name])
    if overrides:
        base.update(overrides)
    base["name"] = name
    return Scenario(**base)


def _estimate_rows(
    data: pd.DataFrame,
    *,
    scenario: str,
    generator: str,
    replicate: int,
    true_ate: float,
    seed: int,
    n_splits: int,
    reference_estimates: dict[str, float] | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for est in estimate_all(data, seed=seed, n_splits=n_splits):
        row = est.to_dict()
        row.update(
            {
                "scenario": scenario,
                "generator": generator,
                "replicate": replicate,
                "true_ate": true_ate,
                "reference_estimate": (
                    reference_estimates.get(est.estimator) if reference_estimates is not None else est.estimate
                ),
            }
        )
        row["reference_estimate_distortion"] = row["estimate"] - row["reference_estimate"]
        rows.append(row)
    return rows


def _long_metric_rows(
    metrics: dict[str, float], *, scenario: str, generator: str, replicate: int, family: str
) -> list[dict[str, Any]]:
    return [
        {
            "scenario": scenario,
            "generator": generator,
            "replicate": replicate,
            "family": family,
            "metric": key,
            "value": float(value),
        }
        for key, value in metrics.items()
        if np.isfinite(value)
    ]


def run_experiment(config: dict[str, Any]) -> dict[str, pd.DataFrame]:
    """Run the configured experiment and return all result tables.

    No respondent-level reference or synthetic records are written by this
    function. Only aggregate/audit tables are returned for persistence.
    """
    raw_path = Path(config["data"]["path"])
    cache_path = config.get("data", {}).get("cache_path")
    cache = Path(cache_path) if cache_path else None
    if cache is not None and cache.exists():
        pool = pd.read_csv(cache)
        audit = pd.DataFrame(
            [
                {"raw_variable": "__sample__", "analytic_variable": "complete_case_pool", "raw_missing_fraction": np.nan, "retained_nonmissing_fraction": np.nan, "retained_rows": len(pool), "source_path": str(raw_path), "cache_used": True}
            ]
        )
    else:
        raw = load_brfss(raw_path)
        pool = prepare_brfss(raw)
        audit = data_audit(raw, pool)
        audit["retained_rows"] = len(pool)
        audit["source_path"] = str(raw_path)
        audit["cache_used"] = False
        if cache is not None:
            cache.parent.mkdir(parents=True, exist_ok=True)
            pool.to_csv(cache, index=False)

    exp = config["experiment"]
    n = int(exp["n"])
    repetitions = int(exp["repetitions"])
    base_seed = int(exp.get("seed", 20260811))
    n_splits = int(exp.get("n_splits", 5))
    generators = list(exp.get("generators", ["gaussian_copula"]))
    scenarios = list(exp.get("scenarios", ["standard", "weak_overlap"]))
    ctgan_epochs = int(exp.get("ctgan_epochs", 150))
    scenario_overrides = config.get("scenario_overrides", {})

    estimate_rows: list[dict[str, Any]] = []
    fidelity_rows: list[dict[str, Any]] = []
    overlap_rows: list[dict[str, Any]] = []
    timing_rows: list[dict[str, Any]] = []
    first_example: dict[str, pd.DataFrame | np.ndarray] = {}

    for scenario_idx, scenario_name in enumerate(scenarios):
        scenario = _scenario_from_config(scenario_name, scenario_overrides.get(scenario_name))
        for replicate in range(repetitions):
            replicate_seed = base_seed + scenario_idx * 100_000 + replicate * 1_000
            x = sample_covariates(pool, n=n, seed=replicate_seed)
            reference_truth = simulate_reference(x, scenario=scenario, seed=replicate_seed + 1)
            reference = observed_analysis_frame(reference_truth)
            true_ate = float(reference_truth.attrs["true_ate"])

            truth_overlap = overlap_summary(reference_truth["true_propensity"].to_numpy(), reference_truth["treatment"].to_numpy())
            estimated_ref_overlap = estimated_overlap(reference, seed=replicate_seed + 2)
            overlap_rows.extend(_long_metric_rows({**{f"true_{k}": v for k, v in truth_overlap.items()}, **estimated_ref_overlap}, scenario=scenario_name, generator="reference", replicate=replicate, family="overlap"))
            fidelity_rows.extend(_long_metric_rows(truth_mechanism_diagnostics(reference_truth, reference, seed=replicate_seed + 20), scenario=scenario_name, generator="reference", replicate=replicate, family="oracle_mechanism"))

            t0 = time.perf_counter()
            ref_rows = _estimate_rows(reference, scenario=scenario_name, generator="reference", replicate=replicate, true_ate=true_ate, seed=replicate_seed + 3, n_splits=n_splits)
            timing_rows.append({"scenario": scenario_name, "generator": "reference", "replicate": replicate, "seconds": time.perf_counter() - t0})
            estimate_rows.extend(ref_rows)
            reference_estimates = {r["estimator"]: float(r["estimate"]) for r in ref_rows}

            if not first_example:
                first_example["reference"] = reference.copy()
                first_example["truth_propensity"] = reference_truth["true_propensity"].to_numpy()
                first_example["truth_treatment"] = reference_truth["treatment"].to_numpy()

            for generator_idx, generator in enumerate(generators):
                gen_seed = replicate_seed + 100 + generator_idx * 100
                t0 = time.perf_counter()
                synthetic = synthesize(reference, method=generator, seed=gen_seed, ctgan_epochs=ctgan_epochs)
                synth_seconds = time.perf_counter() - t0
                if len(synthetic) != len(reference):
                    raise RuntimeError(f"{generator} returned {len(synthetic)} rows for {len(reference)} reference rows")
                t1 = time.perf_counter()
                synth_rows = _estimate_rows(synthetic, scenario=scenario_name, generator=generator, replicate=replicate, true_ate=true_ate, seed=gen_seed + 1, n_splits=n_splits, reference_estimates=reference_estimates)
                estimate_rows.extend(synth_rows)
                fidelity_rows.extend(_long_metric_rows(fidelity_report(reference, synthetic, seed=gen_seed + 2), scenario=scenario_name, generator=generator, replicate=replicate, family="fidelity"))
                fidelity_rows.extend(_long_metric_rows(predictive_utility(reference, synthetic, seed=gen_seed + 3), scenario=scenario_name, generator=generator, replicate=replicate, family="predictive_utility"))
                fidelity_rows.extend(_long_metric_rows(mechanism_preservation(reference, synthetic, seed=gen_seed + 4), scenario=scenario_name, generator=generator, replicate=replicate, family="mechanism_preservation"))
                fidelity_rows.extend(_long_metric_rows(truth_mechanism_diagnostics(reference_truth, synthetic, seed=gen_seed + 40), scenario=scenario_name, generator=generator, replicate=replicate, family="oracle_mechanism"))
                synth_overlap = estimated_overlap(synthetic, seed=gen_seed + 5)
                overlap_rows.extend(_long_metric_rows(synth_overlap, scenario=scenario_name, generator=generator, replicate=replicate, family="overlap"))
                timing_rows.append({"scenario": scenario_name, "generator": generator, "replicate": replicate, "seconds": synth_seconds + (time.perf_counter() - t1)})
                if replicate == 0 and scenario_idx == 0:
                    first_example[f"synthetic_{generator}"] = synthetic.copy()

    estimates = add_causal_errors(pd.DataFrame(estimate_rows))
    fidelity = pd.DataFrame(fidelity_rows)
    overlap = pd.DataFrame(overlap_rows)
    timings = pd.DataFrame(timing_rows)
    causal_summary = summarize_causal_utility(estimates)
    fidelity_summary = summarize_fidelity(fidelity)
    return {"data_audit": audit, "estimates": estimates, "causal_summary": causal_summary, "fidelity": fidelity, "fidelity_summary": fidelity_summary, "overlap": overlap, "timings": timings, "_examples": first_example}


def save_results(results: dict[str, pd.DataFrame], output_dir: str | Path) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    for name, table in results.items():
        if name.startswith("_"):
            continue
        table.to_csv(output / f"{name}.csv", index=False)


def save_run_metadata(config: dict[str, Any], output_dir: str | Path) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    with (output / "run_config.json").open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, sort_keys=True)


def make_figures(results: dict[str, pd.DataFrame], figure_dir: str | Path) -> None:
    figure_dir = Path(figure_dir)
    figure_dir.mkdir(parents=True, exist_ok=True)
    plot_dag(figure_dir / "dag.png")
    plot_causal_bias(results["causal_summary"], figure_dir / "causal_utility_mae.png")
    plot_fidelity_vs_causal_error(results["fidelity"], results["estimates"], figure_dir / "fidelity_vs_causal_error.png")
    examples = results.get("_examples", {})
    reference = examples.get("reference")
    if isinstance(reference, pd.DataFrame):
        p = examples.get("truth_propensity")
        d = examples.get("truth_treatment")
        if isinstance(p, np.ndarray) and isinstance(d, np.ndarray):
            plot_propensity_overlap(p, d, figure_dir / "reference_propensity_overlap.png")
        for key, synthetic in examples.items():
            if key.startswith("synthetic_") and isinstance(synthetic, pd.DataFrame):
                generator = key.removeprefix("synthetic_")
                plot_bmi_distribution(reference, synthetic, generator, figure_dir / f"bmi_{generator}.png")
