"""Continuous hedge ratios, estimated *after* the discrete basket is chosen."""
from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np

from .stats import adf_pvalue, adf_tstat, half_life


@dataclass
class HedgeFit:
    weights: np.ndarray       # dollar weights on log prices, sum |w| = 1
    method: str               # johansen | ols | ols(fallback)
    trace_stat: float         # Johansen trace statistic for rank 0 (nan for OLS)
    trace_crit95: float
    mean: float               # spread mean on the fit window
    std: float                # spread std on the fit window
    half_life: float
    in_sample_adf: float      # optimistic by construction - weights were fit on this window

    @property
    def net_exposure(self) -> float:
        return float(self.weights.sum())


def _normalise(w: np.ndarray) -> np.ndarray:
    w = np.asarray(w, dtype=float)
    gross = np.abs(w).sum()
    if not np.isfinite(gross) or gross == 0:
        raise ValueError("degenerate hedge weights")
    w = w / gross
    return w if w[np.argmax(np.abs(w))] > 0 else -w


def _ols_weights(logp: np.ndarray) -> np.ndarray:
    y, X = logp[:, 0], np.column_stack([logp[:, 1:], np.ones(len(logp))])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    return np.concatenate([[1.0], -coef[:-1]])


def fit_hedge(logp: np.ndarray, method: str = "johansen") -> HedgeFit:
    """Fit hedge ratios on a (T, K) window of log prices."""
    if logp.ndim != 2 or logp.shape[1] < 2:
        raise ValueError("need at least two assets to form a spread")
    trace, crit = float("nan"), float("nan")
    used = method
    if method == "johansen":
        try:
            from statsmodels.tsa.vector_ar.vecm import coint_johansen

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res = coint_johansen(logp, det_order=0, k_ar_diff=1)
            vec = res.evec[:, 0]
            if np.iscomplexobj(vec):
                if np.max(np.abs(vec.imag)) > 1e-8:
                    raise ValueError("complex cointegrating vector")
                vec = vec.real
            w = _normalise(vec)
            trace, crit = float(np.real(res.lr1[0])), float(res.cvt[0, 1])
        except Exception:  # singular / too-short windows
            w, used = _normalise(_ols_weights(logp)), "ols(fallback)"
    elif method == "ols":
        w = _normalise(_ols_weights(logp))
    else:
        raise ValueError(f"unknown hedge method: {method}")
    spread = logp @ w
    return HedgeFit(
        weights=w,
        method=used,
        trace_stat=trace,
        trace_crit95=crit,
        mean=float(spread.mean()),
        std=float(spread.std(ddof=1)),
        half_life=half_life(spread),
        in_sample_adf=adf_tstat(spread),
    )


def validation_gate(fit: HedgeFit, logp_val: np.ndarray, gates: dict) -> tuple[bool, dict]:
    """Out-of-sample check with frozen weights; standard ADF critical values apply here."""
    spread = logp_val @ fit.weights
    tstat, pvalue = adf_pvalue(spread)
    hl = half_life(spread)
    max_w = float(np.max(np.abs(fit.weights)))
    reasons = []
    if pvalue > gates["val_adf_pvalue"]:
        reasons.append("validation_adf")
    if not np.isfinite(hl) or hl > gates["max_half_life"]:
        reasons.append("half_life")
    if max_w > gates["max_asset_weight"]:
        reasons.append("asset_weight")
    if not np.isfinite(fit.std) or fit.std <= 0:
        reasons.append("degenerate_spread")
    info = {"val_adf": tstat, "val_adf_pvalue": pvalue, "val_half_life": hl, "max_abs_weight": max_w, "reasons": reasons}
    return (not reasons), info
