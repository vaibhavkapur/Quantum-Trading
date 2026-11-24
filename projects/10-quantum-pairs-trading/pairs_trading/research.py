from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import combinations
import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import coint


@dataclass(frozen=True)
class Pair:
    left: str
    right: str
    sectors: tuple[str, ...]
    alpha: float
    beta: float
    mean: float
    sigma: float
    half_life: float
    pvalue: float
    score: float
    capital_units: int = 1

    @property
    def name(self) -> str:
        return f"{self.left}/{self.right}"

    def to_dict(self) -> dict:
        return {"name": self.name, **asdict(self)}


@dataclass(frozen=True)
class Hedge:
    alpha: float
    beta: float
    mean: float
    sigma: float

    def zscore(self, left: float, right: float) -> float:
        return (left - self.alpha - self.beta * right - self.mean) / self.sigma


def fit_hedge(left: np.ndarray, right: np.ndarray) -> Hedge:
    y, x = np.asarray(left, float), np.asarray(right, float)
    if len(x) < 40 or x.shape != y.shape or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("hedge fit needs aligned finite arrays with at least 40 rows")
    if np.std(x) < 1e-9 or np.std(y) < 1e-9:
        raise ValueError("constant price series")
    alpha, beta = np.linalg.lstsq(np.column_stack((np.ones(len(x)), x)), y, rcond=None)[0]
    residual = y - alpha - beta * x
    sigma = float(np.std(residual, ddof=1))
    if sigma < 1e-8 or not 0.05 <= beta <= 20:
        raise ValueError("degenerate spread or unsupported hedge ratio")
    return Hedge(float(alpha), float(beta), float(np.mean(residual)), sigma)


def half_life(residual: np.ndarray) -> float:
    """Exact AR(1) decay half-life; oscillating/nonstationary fits are excluded."""
    residual = np.asarray(residual, float)
    if len(residual) < 3 or not np.isfinite(residual).all() or np.std(residual[:-1]) < 1e-10:
        return float("inf")
    phi = np.linalg.lstsq(np.column_stack((np.ones(len(residual) - 1), residual[:-1])), residual[1:], rcond=None)[0][1]
    return float(-np.log(2) / np.log(phi)) if 0 < phi < 1 else float("inf")


def discover_pairs(train: pd.DataFrame, sectors: dict, settings: dict, cost_bps: float) -> tuple[list[Pair], np.ndarray, list[dict]]:
    candidates, diagnostics, residuals = [], [], {}
    for left, right in combinations(sorted(train.columns), 2):
        row = {"left": left, "right": right, "train_end": train.index[-1].isoformat()}
        if settings["same_sector_only"] and sectors[left] != sectors[right]:
            diagnostics.append({**row, "status": "cross_sector_excluded"})
            continue
        try:
            hedge = fit_hedge(train[left].to_numpy(), train[right].to_numpy())
            residual = train[left].to_numpy() - hedge.alpha - hedge.beta * train[right].to_numpy()
            with warnings.catch_warnings():
                warnings.simplefilter("error")
                _, pvalue, _ = coint(train[left], train[right], trend="c", maxlag=4, autolag="aic")
            life = half_life(residual)
            if not np.isfinite(pvalue) or not np.isfinite(life):
                raise ValueError("nonfinite cointegration or non-mean-reverting residual")
            accepted = pvalue <= settings["pvalue"] and settings["min_half_life"] <= life <= settings["max_half_life"]
            diagnostics.append({**row, "pvalue": float(pvalue), "half_life": life, "beta": hedge.beta, "status": "eligible" if accepted else "screened_out"})
            if accepted:
                # A declared train-only quality proxy, not a return forecast.
                score = -np.log10(max(float(pvalue), 1e-8)) / (1 + life / 10) - 2 * cost_bps / 1e4 * 252 / life
                pair = Pair(left, right, tuple(sorted({sectors[left], sectors[right]})), hedge.alpha, hedge.beta, hedge.mean, hedge.sigma, life, float(pvalue), float(score))
                candidates.append(pair)
                residuals[pair.name] = residual
        except (ValueError, np.linalg.LinAlgError, Warning) as exc:
            diagnostics.append({**row, "status": "invalid", "reason": str(exc)})
    candidates.sort(key=lambda p: (-p.score, p.name))
    candidates = candidates[:settings["max_candidates"]]
    names = {p.name for p in candidates}
    for row in diagnostics:
        if row["status"] == "eligible":
            row["status"] = "candidate" if f"{row['left']}/{row['right']}" in names else "candidate_cap"
    if len(candidates) > 1:
        correlation = np.abs(np.corrcoef(np.array([residuals[p.name] for p in candidates])))
    else:
        correlation = np.eye(len(candidates))
    return candidates, correlation, diagnostics


def walk_forward_splits(length: int, settings: dict) -> list[dict]:
    train, validation, test = (settings[x] for x in ("train", "validation", "test"))
    splits = []
    for start in range(0, length - train - validation - test + 1, test):
        splits.append({"fold": len(splits), "train_start": start, "train_end": start + train,
                       "validation_start": start + train, "validation_end": start + train + validation,
                       "test_start": start + train + validation, "test_end": start + train + validation + test})
    if not splits:
        raise ValueError("no complete walk-forward fold")
    return splits
