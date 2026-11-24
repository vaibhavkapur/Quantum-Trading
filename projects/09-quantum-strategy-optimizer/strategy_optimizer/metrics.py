"""Out-of-sample metrics; conservative multiplicity and dependence handling."""
from __future__ import annotations

import numpy as np
from scipy.stats import norm


def sharpe(returns, annualization=252):
    values = np.asarray(returns, dtype=float)
    if len(values) < 2 or not np.isfinite(values).all():
        raise ValueError("Sharpe requires at least two finite observations")
    volatility = float(values.std(ddof=1))
    if volatility < 1e-14:
        return 0.0 if np.all(np.abs(values) < 1e-14) else None
    return float(values.mean() / volatility * np.sqrt(annualization))


def performance(returns, turnover=None, costs=None, annualization=252, trials=1):
    values = np.asarray(returns, dtype=float)
    if values.ndim != 1 or len(values) < 3 or not np.isfinite(values).all() or (values <= -1).any():
        raise ValueError("Need at least three finite returns greater than -100%")
    wealth = np.r_[1.0, np.cumprod(1 + values)]
    peak = np.maximum.accumulate(wealth)
    centered = values - values.mean()
    n = len(values)
    lag = min(n - 1, int(np.floor(4 * (n / 100) ** (2 / 9))))
    long_variance = float(centered @ centered / n)
    for k in range(1, lag + 1):
        long_variance += 2 * (1 - k / (lag + 1)) * float(centered[k:] @ centered[:-k] / n)
    standard_error = np.sqrt(max(long_variance, 0) / n)
    pvalue = float(norm.sf(values.mean() / standard_error)) if standard_error > 1e-14 else 1.0
    return {
        "observations": n, "sharpe": sharpe(values, annualization),
        "total_return": float(wealth[-1] - 1),
        "annualized_return": float(wealth[-1] ** (annualization / n) - 1),
        "max_drawdown": float(np.min(wealth / peak - 1)),
        "annualized_volatility": float(values.std(ddof=1) * np.sqrt(annualization)),
        "annualized_turnover": None if turnover is None else float(np.mean(turnover) * annualization),
        "total_cost_return": None if costs is None else float(np.sum(costs)),
        "positive_mean_hac_pvalue": pvalue, "hac_lags": lag,
        "bonferroni_pvalue": min(1.0, pvalue * max(1, trials)),
        "multiplicity_trials": max(1, trials),
    }


def block_bootstrap_sharpe(returns, annualization=252, samples=400, block_length=5, seed=2718):
    values = np.asarray(returns, dtype=float)
    if samples < 1 or block_length < 1:
        raise ValueError("Bootstrap settings must be positive")
    rng = np.random.default_rng(seed)
    estimates = []
    n = len(values)
    for _ in range(samples):
        starts = rng.integers(n, size=(n + block_length - 1) // block_length)
        indices = ((starts[:, None] + np.arange(block_length)) % n).ravel()[:n]
        estimate = sharpe(values[indices], annualization)
        if estimate is not None:
            estimates.append(estimate)
    return None if not estimates else np.quantile(estimates, [0.025, 0.975]).tolist()


def stability(selections, names):
    values = [np.asarray(z, dtype=bool) for z in selections]
    scores = []
    for left, right in zip(values, values[1:]):
        union = (left | right).sum()
        scores.append(1.0 if not union else float((left & right).sum() / union))
    return {"mean_adjacent_jaccard": float(np.mean(scores)) if scores else None,
            "selection_frequency": dict(zip(names, np.mean(values, axis=0).tolist())) if values else {}}

