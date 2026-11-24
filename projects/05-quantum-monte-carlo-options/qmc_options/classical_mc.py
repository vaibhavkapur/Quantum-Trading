"""Vectorised classical Monte Carlo pricers with variance reduction."""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Dict, Tuple

import numpy as np
from scipy.stats import norm, qmc, t as student_t

from . import analytic
from .models import Market, Option, paths_from_normals

METHODS = ("plain", "antithetic", "control_variate", "qmc")
QMC_REPLICATES = 16


@dataclass
class Estimate:
    price: float
    stderr: float
    ci_low: float
    ci_high: float
    n_paths: int          # payoff evaluations actually used
    method: str
    wall_time_s: float = 0.0
    extra: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def _summarise(samples: np.ndarray, confidence: float, use_t: bool = False) -> Tuple[float, float, float, float]:
    n = samples.size
    mean = float(samples.mean())
    stderr = float(samples.std(ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
    q = student_t.ppf(0.5 + confidence / 2, n - 1) if use_t else norm.ppf(0.5 + confidence / 2)
    return mean, stderr, mean - q * stderr, mean + q * stderr


def discounted_payoffs(market: Market, option: Option, z: np.ndarray) -> np.ndarray:
    paths = paths_from_normals(market, option, z)
    return np.exp(-market.rate * option.maturity) * option.payoff(paths)


def _control(market: Market, option: Option, z: np.ndarray) -> Tuple[np.ndarray, float]:
    """A correlated quantity with known expectation, per path."""
    paths = paths_from_normals(market, option, z)
    df = np.exp(-market.rate * option.maturity)
    if option.kind in ("european_call", "european_put"):
        return df * paths[..., -1], market.spot          # discounted asset is a martingale
    if option.kind == "asian_call":
        geo = np.exp(np.log(paths).mean(axis=-1))
        expected = analytic.geometric_asian_call(market, option.strike, option.maturity, option.steps)
        return df * np.maximum(geo - option.strike, 0.0), expected
    vanilla = df * np.maximum(paths[..., -1] - option.strike, 0.0)
    return vanilla, analytic.black_scholes_price(market, option.strike, option.maturity, call=True)


def mc_price(market: Market, option: Option, n_paths: int, method: str = "plain",
             seed: int = 0, confidence: float = 0.95) -> Estimate:
    """Price ``option`` with ``n_paths`` payoff evaluations (rounded down to
    what the method can use: an even number, or a power of two per Sobol
    replicate)."""
    if method not in METHODS:
        raise ValueError(f"unknown method {method!r}; expected one of {METHODS}")
    if n_paths < 4:
        raise ValueError("n_paths must be >= 4")
    start = time.perf_counter()
    rng = np.random.default_rng(seed)
    extra: Dict[str, float] = {}
    use_t = False

    if method == "plain":
        samples = discounted_payoffs(market, option, rng.standard_normal((n_paths, option.steps)))
        used = n_paths
    elif method == "antithetic":
        z = rng.standard_normal((n_paths // 2, option.steps))
        samples = 0.5 * (discounted_payoffs(market, option, z) + discounted_payoffs(market, option, -z))
        used = 2 * samples.size
    elif method == "control_variate":
        z = rng.standard_normal((n_paths, option.steps))
        y = discounted_payoffs(market, option, z)
        c, c_mean = _control(market, option, z)
        var_c = c.var(ddof=1)
        beta = float(np.cov(y, c, ddof=1)[0, 1] / var_c) if var_c > 0 else 0.0
        samples = y - beta * (c - c_mean)
        used = n_paths
        extra["beta"] = beta
        extra["variance_ratio"] = float(samples.var(ddof=1) / y.var(ddof=1)) if y.var() > 0 else 1.0
    else:  # randomised quasi-Monte Carlo: independent scrambles give an honest error bar
        per_rep = max(n_paths // QMC_REPLICATES, 2)
        log2_points = int(np.floor(np.log2(per_rep)))
        means = np.empty(QMC_REPLICATES)
        for i in range(QMC_REPLICATES):
            sobol = qmc.Sobol(d=option.steps, scramble=True, seed=rng)
            u = np.clip(sobol.random_base2(log2_points), 1e-12, 1 - 1e-12)
            means[i] = discounted_payoffs(market, option, norm.ppf(u)).mean()
        samples = means
        used = QMC_REPLICATES * 2**log2_points
        use_t = True
        extra["replicates"] = QMC_REPLICATES

    price, stderr, lo, hi = _summarise(samples, confidence, use_t)
    return Estimate(price, stderr, lo, hi, used, method, time.perf_counter() - start, extra)


def price_from_paths(market: Market, option: Option, paths: np.ndarray, confidence: float = 0.95) -> Estimate:
    """Plain Monte Carlo over stored price paths (shape (n, option.steps))."""
    if paths.ndim != 2 or paths.shape[1] != option.steps:
        raise ValueError(f"expected paths of shape (n, {option.steps}), got {paths.shape}")
    start = time.perf_counter()
    samples = np.exp(-market.rate * option.maturity) * option.payoff(paths)
    price, stderr, lo, hi = _summarise(samples, confidence)
    return Estimate(price, stderr, lo, hi, samples.size, "dataset", time.perf_counter() - start)


def mc_greeks(market: Market, option: Option, n_paths: int, seed: int = 0,
              rel_bump: float = 0.01, confidence: float = 0.95) -> Dict[str, Estimate]:
    """Central finite-difference delta, gamma and vega. Every bumped
    revaluation reuses the same normals (common random numbers), so the
    standard error is that of the per-path difference, not of two prices."""
    start = time.perf_counter()
    z = np.random.default_rng(seed).standard_normal((n_paths, option.steps))
    h_s = rel_bump * market.spot
    h_v = rel_bump * market.vol
    base = discounted_payoffs(market, option, z)
    up = discounted_payoffs(market.bumped(spot=market.spot + h_s), option, z)
    dn = discounted_payoffs(market.bumped(spot=market.spot - h_s), option, z)
    v_up = discounted_payoffs(market.bumped(vol=market.vol + h_v), option, z)
    v_dn = discounted_payoffs(market.bumped(vol=market.vol - h_v), option, z)
    per_path = {
        "delta": (up - dn) / (2 * h_s),
        "gamma": (up - 2 * base + dn) / h_s**2,
        "vega": (v_up - v_dn) / (2 * h_v),
    }
    elapsed = time.perf_counter() - start
    out = {}
    for name, samples in per_path.items():
        value, stderr, lo, hi = _summarise(samples, confidence)
        out[name] = Estimate(value, stderr, lo, hi, n_paths, "finite_difference_crn", elapsed,
                             {"rel_bump": rel_bump})
    return out
