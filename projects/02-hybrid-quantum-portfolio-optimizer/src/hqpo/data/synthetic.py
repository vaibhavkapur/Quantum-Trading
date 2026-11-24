"""Synthetic data: a factor-model price panel and covariance instances with known optima."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..models.problem import Constraints, PortfolioProblem


def factor_covariance(n: int, n_sectors: int, rng: np.random.Generator):
    """Annualised covariance from a market + sector + idiosyncratic model. Returns (sigma, sectors)."""
    sectors = np.arange(n) % n_sectors
    beta = rng.uniform(0.6, 1.4, n)
    sector_load = rng.uniform(0.5, 1.2, n)
    idio = rng.uniform(0.12, 0.30, n)
    market_vol, sector_vol = 0.16, 0.10
    sigma = market_vol ** 2 * np.outer(beta, beta)
    same = sectors[:, None] == sectors[None, :]
    sigma = sigma + sector_vol ** 2 * np.outer(sector_load, sector_load) * same
    sigma = sigma + np.diag(idio ** 2)
    return sigma, [f"SEC{k}" for k in sectors]


def random_instance(n: int, constraints: Constraints, seed: int, n_sectors: int = 3,
                    risk_aversion: float = 4.0, turnover_penalty: float = 0.0) -> PortfolioProblem:
    """Random benchmark instance (optimum unknown a priori)."""
    rng = np.random.default_rng(seed)
    sigma, sectors = factor_covariance(n, n_sectors, rng)
    mu = rng.normal(0.07, 0.05, n)
    w_prev = None
    if turnover_penalty > 0 and constraints.mode == "long_only":
        w_prev = np.full(n, 1.0 / n)
    return PortfolioProblem(sigma, mu, constraints, risk_aversion=risk_aversion,
                            turnover_penalty=turnover_penalty, w_prev=w_prev, sectors=sectors,
                            name=f"random-n{n}-s{seed}")


def planted_instance(n: int, constraints: Constraints, seed: int, n_sectors: int = 3,
                     risk_aversion: float = 4.0, turnover_penalty: float = 0.0):
    """Instance whose exact optimum is known analytically.

    Draw a feasible lot vector ``z*`` (weights ``w*``) and set

        mu = 2 lam Sigma w* + 2 tau (w* - w_prev) + nu 1

    Then ``w*`` satisfies the KKT conditions of the convex problem "minimise f subject to
    ``sum(w) = const``", so it is the unique global minimiser over that whole hyperplane and
    hence over every subset containing it, including the discrete constrained set.
    Returns ``(problem, z_star)``.
    """
    from ..models.repair import repair

    rng = np.random.default_rng(seed)
    sigma, sectors = factor_covariance(n, n_sectors, rng)
    probe = PortfolioProblem(sigma, np.zeros(n), constraints, sectors=sectors)
    z_star = None
    for _ in range(200):
        z0 = np.zeros(probe.m, dtype=int)
        for c in probe.budget_constraints:
            members = np.flatnonzero(c.a != 0)
            k = constraints.cardinality or n
            pool = rng.choice(members, size=min(len(members), max(1, k // len(probe.budget_constraints))),
                              replace=False)
            for j in rng.choice(pool, size=constraints.lots):
                z0[j] += 1
        z, ok = repair(probe, z0)
        if ok:
            z_star = z
            break
    if z_star is None:
        raise ValueError("could not draw a feasible planted solution for these constraints")
    w_star = probe.weights(z_star)
    w_prev = rng.dirichlet(np.ones(n)) if (turnover_penalty > 0 and constraints.mode == "long_only") \
        else np.zeros(n)
    nu = 0.03
    mu = 2 * risk_aversion * sigma @ w_star + 2 * turnover_penalty * (w_star - w_prev) + nu
    problem = PortfolioProblem(sigma, mu, constraints, risk_aversion=risk_aversion,
                               turnover_penalty=turnover_penalty, w_prev=w_prev, sectors=sectors,
                               name=f"planted-n{n}-s{seed}")
    return problem, z_star


def generate_price_panel(n_assets: int = 6, n_sectors: int = 3, n_days: int = 1008,
                         start: str = "2019-01-02", seed: int = 7) -> tuple[pd.DataFrame, dict]:
    """Deterministic synthetic adjusted-close panel (business days) and a sector map.

    Daily log returns follow a market + sector + idiosyncratic factor model with a two-state
    volatility regime, so covariance estimates are noisy and time-varying like real data.
    """
    rng = np.random.default_rng(seed)
    sectors = np.arange(n_assets) % n_sectors
    beta = rng.uniform(0.6, 1.4, n_assets)
    sector_load = rng.uniform(0.5, 1.2, n_assets)
    idio = rng.uniform(0.12, 0.30, n_assets) / np.sqrt(252)
    drift = rng.uniform(0.02, 0.12, n_assets) / 252
    market_vol, sector_vol = 0.16 / np.sqrt(252), 0.10 / np.sqrt(252)

    regime = np.zeros(n_days, dtype=int)                 # 0 calm, 1 stressed
    for t in range(1, n_days):
        stay = 0.985 if regime[t - 1] == 0 else 0.94
        regime[t] = regime[t - 1] if rng.random() < stay else 1 - regime[t - 1]
    vol_mult = np.where(regime == 1, 2.0, 1.0)[:, None]

    f_m = rng.normal(0, market_vol, (n_days, 1))
    f_s = rng.normal(0, sector_vol, (n_days, n_sectors))
    eps = rng.normal(0, 1, (n_days, n_assets)) * idio
    logret = drift + vol_mult * (f_m * beta + f_s[:, sectors] * sector_load + eps)
    prices = 100.0 * np.exp(np.cumsum(logret, axis=0))
    tickers = [f"SYN{i:02d}" for i in range(n_assets)]
    index = pd.bdate_range(start=start, periods=n_days, name="date")
    panel = pd.DataFrame(np.round(prices, 4), index=index, columns=tickers)
    return panel, {t: f"SEC{k}" for t, k in zip(tickers, sectors)}
