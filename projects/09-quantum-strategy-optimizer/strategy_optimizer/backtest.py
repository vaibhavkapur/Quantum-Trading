"""Sleeve accounting with explicit entry, turnover and liquidation costs."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Outcomes:
    gross: np.ndarray
    net: np.ndarray
    turnover: np.ndarray
    costs: np.ndarray


def backtest(prices, positions, strategies, start, stop, cost_bps, max_selected):
    if not 0 <= start < stop <= len(prices):
        raise ValueError("Invalid backtest interval")
    if max_selected < 1 or not np.isfinite(cost_bps) or cost_bps < 0:
        raise ValueError("Invalid capital allocation or costs")
    if positions.shape != (len(prices), len(strategies)) or not np.isin(positions, [-1, 0, 1]).all():
        raise ValueError("Position array must match data and strategy dimensions")
    asset_returns = prices.pct_change(fill_method=None).fillna(0)
    returns = np.column_stack([asset_returns[s.asset].to_numpy()[start:stop] for s in strategies])
    held = positions[start:stop] / max_selected
    turnover = np.abs(np.diff(held, axis=0, prepend=np.zeros((1, held.shape[1]))))
    turnover[-1] += np.abs(held[-1])  # Explicit liquidation at the window's final close.
    gross = returns * held
    costs = turnover * cost_bps / 10_000
    return Outcomes(gross, gross - costs, turnover, costs)


def ensemble(outcomes: Outcomes, selected):
    z = np.asarray(selected)
    if z.shape != (outcomes.net.shape[1],) or not np.isin(z, [0, 1]).all():
        raise ValueError("Selection must be a binary vector")
    return {name: getattr(outcomes, name) @ z for name in ("gross", "net", "turnover", "costs")}


def walk_forward(length, train, validation, test):
    if min(train, validation, test) < 3:
        raise ValueError("Each walk-forward window needs at least three observations")
    folds = []
    for start in range(0, length - train - validation - test + 1, test):
        train_end, val_end = start + train, start + train + validation
        folds.append({"fold": len(folds), "train": (start, train_end),
                      "validation": (train_end, val_end), "test": (val_end, val_end + test)})
    if not folds:
        raise ValueError("No complete walk-forward folds")
    return folds

