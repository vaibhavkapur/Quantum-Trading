from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from .config import Config
from .features import WARMUP


@dataclass
class Split:
    fold: int
    train: np.ndarray
    calibration: np.ndarray
    validation: np.ndarray
    test: np.ndarray

    def to_dict(self):
        return {key: value.tolist() if isinstance(value, np.ndarray) else value
                for key, value in vars(self).items()}


def walk_forward(n: int, config: Config) -> list[Split]:
    """Expanding training; disjoint test windows; purge label endpoints at every boundary."""
    train_n = int(n * config.train_fraction)
    cal_n = int(n * config.calibration_fraction)
    val_n = int(n * config.validation_fraction)
    test_start = train_n + cal_n + val_n
    edges = np.linspace(test_start, n, config.folds + 1, dtype=int)
    result = []
    for fold in range(config.folds):
        start = edges[fold]
        boundaries = [0, start - cal_n - val_n, start - val_n, start, edges[fold + 1]]
        parts = []
        for j, (left, right) in enumerate(zip(boundaries[:-1], boundaries[1:])):
            lo = max(WARMUP, left + (config.embargo if j else 0))
            # A label ending exactly at the next window's start is purged too.
            part = np.arange(lo, right - config.horizon, dtype=int)
            if len(part) < 3:
                raise ValueError("windows too small after purge/embargo; increase data or reduce folds")
            parts.append(part)
        result.append(Split(fold, *parts))
    return result


def stratified_train_sample(indices: np.ndarray, y: np.ndarray, limit: int, seed: int) -> np.ndarray:
    """Only called on the already-purged training indices; retain chronological order."""
    if limit >= len(indices):
        return indices.copy()
    rng = np.random.default_rng(seed)
    groups = [indices[y[indices] == c] for c in np.unique(y[indices])]
    if limit < len(groups):
        raise ValueError("training budget must include each observed class")
    counts = np.array([len(group) for group in groups])
    ideal = limit * counts / counts.sum()
    quotas = np.maximum(1, np.floor(ideal).astype(int))
    while quotas.sum() > limit:
        candidates = np.where(quotas > 1, quotas - ideal, -np.inf)
        quotas[np.argmax(candidates)] -= 1
    while quotas.sum() < limit:
        candidates = np.where(quotas < counts, ideal - quotas, -np.inf)
        quotas[np.argmax(candidates)] += 1
    chosen = np.concatenate([rng.choice(group, count, replace=False) for group, count in zip(groups, quotas)])
    return np.sort(np.asarray(chosen, dtype=int))
