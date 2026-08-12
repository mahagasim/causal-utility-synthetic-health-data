"""Synthetic-data generators: transparent Gaussian copula and optional CTGAN."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import norm, rankdata

CATEGORICAL_COLUMNS = [
    "age_group",
    "female",
    "education",
    "income",
    "active",
    "diabetes",
    "general_health",
    "treatment",
]
CONTINUOUS_COLUMNS = ["bmi", "outcome"]


def _nearest_correlation(matrix: np.ndarray, floor: float = 1e-6) -> np.ndarray:
    a = (matrix + matrix.T) / 2
    values, vectors = np.linalg.eigh(a)
    values = np.maximum(values, floor)
    psd = (vectors * values) @ vectors.T
    d = np.sqrt(np.diag(psd))
    corr = psd / np.outer(d, d)
    np.fill_diagonal(corr, 1.0)
    return corr


@dataclass
class EmpiricalGaussianCopula:
    categorical_columns: list[str]
    continuous_columns: list[str]
    seed: int = 1

    def fit(self, data: pd.DataFrame) -> "EmpiricalGaussianCopula":
        cols = self.categorical_columns + self.continuous_columns
        missing = sorted(set(cols) - set(data.columns))
        if missing:
            raise ValueError(f"Missing columns for copula synthesis: {missing}")
        if data[cols].isna().any().any():
            raise ValueError("Gaussian-copula input must not contain missing values")
        self.columns_ = cols
        latent = np.empty((len(data), len(cols)), dtype=float)
        self.category_info_: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self.continuous_values_: dict[str, np.ndarray] = {}
        for j, col in enumerate(cols):
            values = data[col].to_numpy()
            if col in self.continuous_columns:
                ranks = rankdata(values, method="average")
                u = (ranks - 0.5) / len(values)
                latent[:, j] = norm.ppf(np.clip(u, 1e-6, 1 - 1e-6))
                self.continuous_values_[col] = np.sort(values.astype(float))
            else:
                categories, counts = np.unique(values, return_counts=True)
                probs = counts / counts.sum()
                upper = np.cumsum(probs)
                lower = np.r_[0.0, upper[:-1]]
                midpoint = np.clip(lower + probs / 2, 1e-6, 1 - 1e-6)
                z_mid = norm.ppf(midpoint)
                mapping = {cat: z for cat, z in zip(categories, z_mid)}
                latent[:, j] = np.array([mapping[v] for v in values], dtype=float)
                self.category_info_[col] = (categories, upper)
        self.correlation_ = _nearest_correlation(np.corrcoef(latent, rowvar=False))
        return self

    def sample(self, n: int, seed: int | None = None) -> pd.DataFrame:
        if not hasattr(self, "correlation_"):
            raise RuntimeError("fit must be called before sample")
        rng = np.random.default_rng(self.seed if seed is None else seed)
        z = rng.multivariate_normal(np.zeros(len(self.columns_)), self.correlation_, size=n)
        u = np.clip(norm.cdf(z), 1e-8, 1 - 1e-8)
        out: dict[str, np.ndarray] = {}
        for j, col in enumerate(self.columns_):
            if col in self.continuous_columns:
                values = self.continuous_values_[col]
                q = u[:, j] * (len(values) - 1)
                lo = np.floor(q).astype(int)
                hi = np.ceil(q).astype(int)
                weight = q - lo
                out[col] = values[lo] * (1 - weight) + values[hi] * weight
            else:
                categories, upper = self.category_info_[col]
                idx = np.searchsorted(upper, u[:, j], side="right")
                idx = np.clip(idx, 0, len(categories) - 1)
                out[col] = categories[idx]
        frame = pd.DataFrame(out, columns=self.columns_)
        for col in self.categorical_columns:
            if np.all(np.isclose(frame[col].astype(float), np.round(frame[col].astype(float)))):
                frame[col] = frame[col].astype(int)
        return frame


def synthesize_gaussian_copula(data: pd.DataFrame, seed: int) -> pd.DataFrame:
    model = EmpiricalGaussianCopula(CATEGORICAL_COLUMNS, CONTINUOUS_COLUMNS, seed=seed)
    return model.fit(data).sample(len(data), seed=seed + 1)


def synthesize_ctgan(data: pd.DataFrame, seed: int, epochs: int = 150, verbose: bool = False) -> pd.DataFrame:
    try:
        import torch
        from ctgan import CTGAN
    except ImportError as exc:
        raise RuntimeError("CTGAN is optional and not installed. Install with: pip install -e '.[ctgan]'") from exc
    np.random.seed(seed)
    torch.manual_seed(seed)
    work = data.copy()
    for col in CATEGORICAL_COLUMNS:
        work[col] = work[col].astype(int)
    model = CTGAN(epochs=epochs, verbose=verbose)
    if hasattr(model, "set_random_state"):
        model.set_random_state(seed)
    model.fit(work, discrete_columns=CATEGORICAL_COLUMNS)
    sampled = model.sample(len(work)).loc[:, work.columns].copy()
    for col in CATEGORICAL_COLUMNS:
        sampled[col] = pd.to_numeric(sampled[col], errors="coerce").round().astype("Int64")
    sampled[CONTINUOUS_COLUMNS] = sampled[CONTINUOUS_COLUMNS].apply(pd.to_numeric, errors="coerce")
    sampled = sampled.dropna().reset_index(drop=True)
    if len(sampled) < len(work):
        extra = model.sample(len(work) - len(sampled))
        for col in CATEGORICAL_COLUMNS:
            extra[col] = pd.to_numeric(extra[col], errors="coerce").round().astype("Int64")
        extra[CONTINUOUS_COLUMNS] = extra[CONTINUOUS_COLUMNS].apply(pd.to_numeric, errors="coerce")
        sampled = pd.concat([sampled, extra.dropna()], ignore_index=True).head(len(work))
    for col in CATEGORICAL_COLUMNS:
        sampled[col] = sampled[col].astype(int)
    return sampled.head(len(work)).reset_index(drop=True)


def synthesize(data: pd.DataFrame, method: str, seed: int, ctgan_epochs: int = 150) -> pd.DataFrame:
    if method == "gaussian_copula":
        return synthesize_gaussian_copula(data, seed=seed)
    if method == "ctgan":
        return synthesize_ctgan(data, seed=seed, epochs=ctgan_epochs)
    raise ValueError(f"Unknown synthesizer: {method}")
