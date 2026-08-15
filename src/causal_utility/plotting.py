"""Publication-oriented plots. Each function creates one figure with one axes."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _finish(fig, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_dag(path: str | Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    positions = {"X": (0.18, 0.55), "D": (0.55, 0.72), "Y": (0.82, 0.55)}
    labels = {
        "X": "X\nBaseline covariates",
        "D": "D\nTreatment",
        "Y": "Y\nOutcome",
    }
    for start_node, end_node in [("X", "D"), ("X", "Y"), ("D", "Y")]:
        ax.annotate(
            "",
            xy=positions[end_node],
            xytext=positions[start_node],
            arrowprops={
                "arrowstyle": "->",
                "lw": 1.7,
                "shrinkA": 44,
                "shrinkB": 44,
            },
            zorder=1,
        )
    for key, (x, y) in positions.items():
        ax.scatter([x], [y], s=3000, zorder=2)
        ax.text(x, y, labels[key], ha="center", va="center", fontsize=9.5, zorder=3)
    ax.text(
        0.50,
        0.18,
        "Identification target: ATE = E[Y(1) - Y(0)]",
        ha="center",
        fontsize=11,
    )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    _finish(fig, path)


def plot_bmi_distribution(
    real: pd.DataFrame,
    synthetic: pd.DataFrame,
    generator: str,
    path: str | Path,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    bins = np.linspace(
        min(real.bmi.min(), synthetic.bmi.min()),
        max(real.bmi.max(), synthetic.bmi.max()),
        35,
    )
    ax.hist(
        real["bmi"],
        bins=bins,
        density=True,
        histtype="step",
        linewidth=1.8,
        label="Reference",
    )
    ax.hist(
        synthetic["bmi"],
        bins=bins,
        density=True,
        histtype="step",
        linewidth=1.8,
        label=generator,
    )
    ax.set_xlabel("BMI")
    ax.set_ylabel("Density")
    ax.set_title("Marginal fidelity: BMI")
    ax.legend()
    _finish(fig, path)


def plot_propensity_overlap(
    propensity: np.ndarray,
    treatment: np.ndarray,
    path: str | Path,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    bins = np.linspace(0, 1, 31)
    ax.hist(
        propensity[treatment == 0],
        bins=bins,
        density=True,
        histtype="step",
        linewidth=1.8,
        label="Control",
    )
    ax.hist(
        propensity[treatment == 1],
        bins=bins,
        density=True,
        histtype="step",
        linewidth=1.8,
        label="Treated",
    )
    ax.set_xlabel("True treatment propensity")
    ax.set_ylabel("Density")
    ax.set_title("Reference overlap diagnostic")
    ax.legend()
    _finish(fig, path)


def plot_causal_bias(summary: pd.DataFrame, path: str | Path) -> None:
    s = summary[summary["generator"] != "reference"].copy()
    if s.empty:
        return
    s["label"] = s["scenario"] + " | " + s["generator"] + " | " + s["estimator"]
    s = s.sort_values("mean_absolute_error")
    fig, ax = plt.subplots(figsize=(8, max(4.5, 0.28 * len(s))))
    ax.barh(s["label"], s["mean_absolute_error"])
    ax.set_xlabel("Mean absolute ATE error")
    ax.set_ylabel("")
    ax.set_title("Causal utility by generator and estimator")
    _finish(fig, path)


def plot_causal_rmse(summary: pd.DataFrame, path: str | Path) -> None:
    """Show truth-relative ATE RMSE for reference and synthetic analyses."""
    if summary.empty:
        return
    s = summary.copy()
    s["label"] = s["scenario"] + " | " + s["generator"] + " | " + s["estimator"]
    s = s.sort_values(["scenario", "estimator", "generator"])
    fig, ax = plt.subplots(figsize=(8, max(4.5, 0.28 * len(s))))
    ax.barh(s["label"], s["rmse"])
    ax.set_xlabel("ATE RMSE relative to known truth")
    ax.set_ylabel("")
    ax.set_title("Reference and synthetic causal estimation error")
    _finish(fig, path)


def plot_fidelity_vs_causal_error(
    fidelity: pd.DataFrame,
    estimates: pd.DataFrame,
    path: str | Path,
    estimator: str = "aipw",
) -> None:
    """Relate descriptive fidelity to synthesis-specific ATE distortion."""
    f = fidelity[fidelity["metric"] == "descriptive_fidelity_distance"].copy()
    e = estimates[
        (estimates["estimator"] == estimator)
        & (estimates["generator"] != "reference")
    ].copy()
    merged = f.merge(e, on=["scenario", "generator", "replicate"], how="inner")
    if merged.empty:
        return

    if "reference_estimate_distortion" in merged.columns:
        merged["causal_plot_error"] = merged["reference_estimate_distortion"].abs()
        y_label = f"Absolute synthetic-reference ATE distortion ({estimator.upper()})"
    else:
        merged["causal_plot_error"] = merged["absolute_error"]
        y_label = f"Absolute ATE error ({estimator.upper()})"

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for (scenario, generator), g in merged.groupby(["scenario", "generator"]):
        ax.scatter(
            g["value"],
            g["causal_plot_error"],
            alpha=0.75,
            label=f"{scenario} | {generator}",
        )
    ax.set_xlabel("Descriptive fidelity distance (lower is better)")
    ax.set_ylabel(y_label)
    ax.set_title("Does conventional fidelity predict causal utility?")
    ax.legend()
    _finish(fig, path)
