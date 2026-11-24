"""Closed-form reference prices used to validate the simulation engines."""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
from scipy.stats import norm

from .models import Market, Option


def _d1_d2(market: Market, strike: float, maturity: float):
    sig_t = market.vol * np.sqrt(maturity)
    d1 = (np.log(market.spot / strike) + (market.rate + 0.5 * market.vol**2) * maturity) / sig_t
    return d1, d1 - sig_t


def black_scholes_price(market: Market, strike: float, maturity: float, call: bool = True) -> float:
    d1, d2 = _d1_d2(market, strike, maturity)
    df = np.exp(-market.rate * maturity)
    if call:
        return float(market.spot * norm.cdf(d1) - strike * df * norm.cdf(d2))
    return float(strike * df * norm.cdf(-d2) - market.spot * norm.cdf(-d1))


def black_scholes_greeks(market: Market, strike: float, maturity: float, call: bool = True) -> Dict[str, float]:
    d1, d2 = _d1_d2(market, strike, maturity)
    df = np.exp(-market.rate * maturity)
    sqrt_t = np.sqrt(maturity)
    pdf = norm.pdf(d1)
    gamma = pdf / (market.spot * market.vol * sqrt_t)
    vega = market.spot * pdf * sqrt_t
    decay = -market.spot * pdf * market.vol / (2 * sqrt_t)
    if call:
        delta = norm.cdf(d1)
        theta = decay - market.rate * strike * df * norm.cdf(d2)
        rho = strike * maturity * df * norm.cdf(d2)
    else:
        delta = norm.cdf(d1) - 1.0
        theta = decay + market.rate * strike * df * norm.cdf(-d2)
        rho = -strike * maturity * df * norm.cdf(-d2)
    return {"delta": float(delta), "gamma": float(gamma), "vega": float(vega),
            "theta": float(theta), "rho": float(rho)}


def geometric_asian_call(market: Market, strike: float, maturity: float, steps: int) -> float:
    """Discretely monitored geometric-average call (dates T/n, 2T/n, ..., T).
    The log of the geometric mean is normal, so the price is Black-like."""
    n = steps
    dt = maturity / n
    mean = np.log(market.spot) + (market.rate - 0.5 * market.vol**2) * dt * (n + 1) / 2
    var = market.vol**2 * dt * (n + 1) * (2 * n + 1) / (6 * n)
    std = np.sqrt(var)
    d1 = (mean - np.log(strike) + var) / std
    d2 = d1 - std
    df = np.exp(-market.rate * maturity)
    return float(df * (np.exp(mean + 0.5 * var) * norm.cdf(d1) - strike * norm.cdf(d2)))


def barrier_up_out_call(market: Market, strike: float, maturity: float, barrier: float,
                        monitoring_steps: Optional[int] = None) -> float:
    """Continuously monitored up-and-out call (Reiner-Rubinstein, no rebate,
    barrier > strike). With ``monitoring_steps`` the barrier is shifted by the
    Broadie-Glasserman-Kou continuity correction to approximate discrete
    monitoring; that variant is an approximation, not an exact price."""
    if monitoring_steps:
        barrier = barrier * np.exp(0.5826 * market.vol * np.sqrt(maturity / monitoring_steps))
    s, r, v = market.spot, market.rate, market.vol
    if s >= barrier:
        return 0.0
    sig_t = v * np.sqrt(maturity)
    lam = (r + 0.5 * v**2) / v**2
    df = np.exp(-r * maturity)
    x1 = np.log(s / barrier) / sig_t + lam * sig_t
    y = np.log(barrier**2 / (s * strike)) / sig_t + lam * sig_t
    y1 = np.log(barrier / s) / sig_t + lam * sig_t
    up_in = (s * norm.cdf(x1) - strike * df * norm.cdf(x1 - sig_t)
             - s * (barrier / s) ** (2 * lam) * (norm.cdf(-y) - norm.cdf(-y1))
             + strike * df * (barrier / s) ** (2 * lam - 2)
             * (norm.cdf(-y + sig_t) - norm.cdf(-y1 + sig_t)))
    return float(black_scholes_price(market, strike, maturity, call=True) - up_in)


def reference_price(market: Market, option: Option) -> Optional[float]:
    """Exact closed form where one exists (European only), else None."""
    if option.kind == "european_call":
        return black_scholes_price(market, option.strike, option.maturity, call=True)
    if option.kind == "european_put":
        return black_scholes_price(market, option.strike, option.maturity, call=False)
    return None
