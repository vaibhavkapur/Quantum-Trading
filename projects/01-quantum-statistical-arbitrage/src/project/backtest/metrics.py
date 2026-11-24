"""Performance metrics on daily return series (P&L / initial capital)."""
from __future__ import annotations

import numpy as np

TRADING_DAYS = 252


def sharpe(returns: np.ndarray) -> float:
    r = np.asarray(returns, float)
    sd = r.std(ddof=1) if len(r) > 1 else 0.0
    return float(r.mean() / sd * np.sqrt(TRADING_DAYS)) if sd > 0 else 0.0


def annualized_return(returns: np.ndarray) -> float:
    r = np.asarray(returns, float)
    return float(r.mean() * TRADING_DAYS) if len(r) else 0.0


def max_drawdown(returns: np.ndarray) -> float:
    """Largest peak-to-trough loss as a fraction of initial capital (non-compounded)."""
    r = np.asarray(returns, float)
    if not len(r):
        return 0.0
    curve = np.concatenate([[0.0], np.cumsum(r)])
    return float(np.max(np.maximum.accumulate(curve) - curve))


def annualized_turnover(traded_notional: float, capital: float, n_bars: int) -> float:
    return float(traded_notional / capital * TRADING_DAYS / n_bars) if n_bars else 0.0


def bootstrap_ci(values: np.ndarray, n_boot: int = 2000, alpha: float = 0.05, seed: int = 0) -> tuple[float, float]:
    """Percentile bootstrap CI of the mean; (nan, nan) for fewer than two values."""
    v = np.asarray(values, float)
    v = v[np.isfinite(v)]
    if len(v) < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = v[rng.integers(0, len(v), size=(n_boot, len(v)))].mean(axis=1)
    return float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))
