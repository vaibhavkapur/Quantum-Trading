"""Cointegration / stationarity statistics. Every function takes a window and fits only on it."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from statsmodels.tsa.stattools import adfuller

ADF_FLOOR = -6.0  # t-stats are clipped to [ADF_FLOOR, 0] and mapped to a [0, 1] strength


def adf_tstat(x: np.ndarray, lags: int = 1) -> float:
    """Augmented Dickey-Fuller t-statistic (constant, fixed lag order).

    Small closed-form implementation used in the O(N^2) pair scan; it matches
    ``statsmodels.adfuller(x, maxlag=lags, autolag=None, regression='c')``.
    """
    x = np.asarray(x, dtype=float)
    dx = np.diff(x)
    n = len(dx) - lags
    if n < lags + 5:
        return 0.0
    y = dx[lags:]
    cols = [x[lags:-1], np.ones(n)]
    for k in range(1, lags + 1):
        cols.append(dx[lags - k : len(dx) - k])
    X = np.column_stack(cols)
    xtx = X.T @ X
    try:
        xtx_inv = np.linalg.inv(xtx)
    except np.linalg.LinAlgError:
        return 0.0
    beta = xtx_inv @ (X.T @ y)
    resid = y - X @ beta
    dof = n - X.shape[1]
    if dof <= 0:
        return 0.0
    se = np.sqrt(max(resid @ resid / dof * xtx_inv[0, 0], 0.0))
    if not np.isfinite(se) or se == 0.0:
        return 0.0
    return float(beta[0] / se)


def adf_pvalue(x: np.ndarray, lags: int = 1) -> tuple[float, float]:
    """(t-stat, MacKinnon p-value). Valid for a series whose weights were NOT fit on ``x``."""
    x = np.asarray(x, dtype=float)
    if len(x) < 20 or np.std(x) == 0:
        return 0.0, 1.0
    stat, pvalue = adfuller(x, maxlag=lags, autolag=None, regression="c", result_object=False)[:2]
    return float(stat), float(pvalue)


def adf_strength(tstat: float) -> float:
    return float(np.clip(tstat, ADF_FLOOR, 0.0) / ADF_FLOOR)


def half_life(x: np.ndarray) -> float:
    """Mean-reversion half-life (bars) from an AR(1) fit; inf if not mean-reverting."""
    x = np.asarray(x, dtype=float)
    if len(x) < 5:
        return float("inf")
    lag, dx = x[:-1], np.diff(x)
    lag_c = lag - lag.mean()
    denom = lag_c @ lag_c
    if denom == 0:
        return float("inf")
    kappa = float(lag_c @ (dx - dx.mean()) / denom)
    if not (-1.0 < kappa < 0.0):
        return float("inf")
    return float(-np.log(2) / np.log1p(kappa))


def _ols(y: np.ndarray, x: np.ndarray) -> tuple[float, float, np.ndarray]:
    xc, yc = x - x.mean(), y - y.mean()
    var = xc @ xc
    slope = float(xc @ yc / var) if var > 0 else 0.0
    intercept = float(y.mean() - slope * x.mean())
    return slope, intercept, y - intercept - slope * x


@dataclass
class PairStats:
    strength: np.ndarray    # [0,1] residual-ADF strength on the full window
    stability: np.ndarray   # [0,1] sub-window strength x hedge-ratio stability
    resid_var: np.ndarray   # 1 - R^2 of the level regression
    half_life: np.ndarray   # bars (inf if not mean-reverting)
    hedge: np.ndarray       # slope of y_i on x_j for the chosen direction


def pair_stats(logp: np.ndarray, hl_cap: float = 60.0) -> PairStats:
    """Pairwise Engle-Granger style statistics on a (T, N) window of log prices."""
    T, N = logp.shape
    strength, stability = np.zeros((N, N)), np.zeros((N, N))
    resid_var, hl, hedge = np.ones((N, N)), np.full((N, N), np.inf), np.zeros((N, N))
    half = T // 2
    for i in range(N):
        for j in range(i + 1, N):
            best = None
            for a, b in ((i, j), (j, i)):
                slope, _, resid = _ols(logp[:, a], logp[:, b])
                t = adf_tstat(resid)
                if best is None or t < best[0]:
                    best = (t, a, b, slope, resid)
            t, a, b, slope, resid = best
            y, x = logp[:, a], logp[:, b]
            s1, _, r1 = _ols(y[:half], x[:half])
            s2, _, r2 = _ols(y[half:], x[half:])
            sub = min(adf_strength(adf_tstat(r1)), adf_strength(adf_tstat(r2)))
            drift = abs(s1 - s2) / (abs(s1) + abs(s2) + 1e-12)
            total = np.var(y)
            strength[i, j] = strength[j, i] = adf_strength(t)
            stability[i, j] = stability[j, i] = sub * (1.0 - min(drift * 2.0, 1.0))
            resid_var[i, j] = resid_var[j, i] = float(np.var(resid) / total) if total > 0 else 1.0
            hl[i, j] = hl[j, i] = half_life(resid)
            hedge[i, j] = hedge[j, i] = slope
    return PairStats(strength, stability, resid_var, hl, hedge)


def market_betas(logp: np.ndarray) -> np.ndarray:
    """Return betas against the equal-weight market return of the window."""
    rets = np.diff(logp, axis=0)
    mkt = rets.mean(axis=1)
    var = np.var(mkt)
    if var == 0:
        return np.ones(logp.shape[1])
    return np.array([np.cov(rets[:, i], mkt, bias=True)[0, 1] / var for i in range(logp.shape[1])])
