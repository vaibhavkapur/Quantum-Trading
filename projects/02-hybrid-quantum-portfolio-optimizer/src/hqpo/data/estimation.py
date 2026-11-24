"""Point-in-time return and covariance estimation."""
from __future__ import annotations

import numpy as np
import pandas as pd


def simple_returns(prices: pd.DataFrame) -> pd.DataFrame:
    return prices.pct_change().iloc[1:]


def ledoit_wolf(X: np.ndarray) -> tuple[np.ndarray, float]:
    """Ledoit-Wolf (2004) shrinkage towards a scaled identity. Returns (cov, intensity)."""
    X = np.asarray(X, dtype=float)
    t, n = X.shape
    Xc = X - X.mean(axis=0)
    S = Xc.T @ Xc / t
    m = np.trace(S) / n
    d2 = np.sum((S - m * np.eye(n)) ** 2) / n
    if d2 <= 0:
        return S, 0.0
    b2_bar = sum(np.sum((np.outer(x, x) - S) ** 2) for x in Xc) / (n * t * t)
    b2 = min(b2_bar, d2)
    shrink = b2 / d2
    return shrink * m * np.eye(n) + (1 - shrink) * S, float(shrink)


def estimate(prices: pd.DataFrame, asof, cfg: dict) -> tuple[np.ndarray, np.ndarray, dict]:
    """Annualised ``(sigma, mu)`` using only prices with ``date <= asof``."""
    lookback = int(cfg["lookback_days"])
    ann = float(cfg.get("annualization", 252))
    window = prices.loc[:asof].iloc[-(lookback + 1):]
    if len(window) < max(20, prices.shape[1] + 2):
        raise ValueError(f"not enough history before {asof} ({len(window)} rows)")
    R = simple_returns(window).to_numpy()
    kind = cfg.get("covariance", "ledoit_wolf")
    if kind == "ledoit_wolf":
        cov, shrink = ledoit_wolf(R)
    elif kind == "sample":
        cov, shrink = np.cov(R, rowvar=False, ddof=1), 0.0
    else:
        raise ValueError(f"unknown covariance estimator {kind!r}")
    sigma = cov * ann

    mean = R.mean(axis=0) * ann
    how = cfg.get("expected_return", "shrunk_mean")
    if how == "zero":                      # minimum-variance configuration
        mu = np.zeros_like(mean)
    elif how == "mean":
        mu = mean
    elif how == "shrunk_mean":             # shrink towards the cross-sectional mean
        k = float(cfg.get("mean_shrinkage", 0.5))
        mu = (1 - k) * mean + k * mean.mean()
    elif how == "momentum":                # total return over the window, skipping the last 21 days
        skip = min(21, len(window) // 4)
        mu = (window.iloc[-1 - skip] / window.iloc[0] - 1).to_numpy() * ann / max(len(window) - skip, 1)
    else:
        raise ValueError(f"unknown expected_return model {how!r}")
    info = {"window_start": str(window.index[0].date()), "window_end": str(window.index[-1].date()),
            "n_obs": int(R.shape[0]), "shrinkage": shrink}
    return sigma, mu, info
