from __future__ import annotations

import math

import numpy as np
from scipy.stats import t


def mean_ci95(values: list[float]) -> tuple[float | None, list[float] | None]:
    if not values:
        return None, None
    mean = float(np.mean(values))
    if len(values) < 2:
        return mean, None
    half = float(t.ppf(0.975, len(values) - 1) * np.std(values, ddof=1) / np.sqrt(len(values)))
    return mean, [mean - half, mean + half]


def wilson(successes: int, attempts: int) -> list[float] | None:
    if attempts == 0:
        return None
    z, p = 1.959963984540054, successes / attempts
    denominator = 1 + z * z / attempts
    center = (p + z * z / (2 * attempts)) / denominator
    half = z * math.sqrt(p * (1 - p) / attempts + z * z / (4 * attempts * attempts)) / denominator
    return [max(0., center - half), min(1., center + half)]


def median_ci95(values: list[float]) -> list[float] | None:
    if len(values) < 2:
        return None
    rng = np.random.default_rng(0)
    medians = np.median(rng.choice(values, size=(2000, len(values)), replace=True), axis=1)
    return np.quantile(medians, [0.025, 0.975]).tolist()
