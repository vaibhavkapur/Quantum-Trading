"""Synthetic finance fixtures; no market data or trading performance claims."""
from __future__ import annotations

import numpy as np
from scipy.stats import norm

from .schema import SCHEMA_VERSION, exact_reference, validate_instance

GENERATOR_VERSION = "synthetic-finance-v1"


def generate(track: str, size: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    if track == "portfolio":
        factors = rng.normal(0, 0.08, (size, size))
        covariance = factors.T @ factors / size + np.eye(size) * 0.01
        returns = rng.uniform(0.02, 0.15, size)
        risk_aversion = 2.0
        problem = {"track": track, "size": size, "cardinality": max(1, size // 2),
                   "linear": (risk_aversion * covariance.diagonal() - returns).tolist(),
                   "quadratic": np.triu(2 * risk_aversion * covariance, 1).tolist(), "conflicts": [],
                   "model": {"expected_returns": returns.tolist(), "covariance": covariance.tolist(),
                             "risk_aversion": risk_aversion, "units": "synthetic equal-notional selection score"}}
    elif track == "basket":
        # First k triangles are vertex-disjoint; remaining candidate cycles overlap.
        k = max(1, size // 3)
        cycles = [[3 * i, 3 * i + 1, 3 * i + 2] for i in range(k)]
        cycles += [rng.choice(3 * k, 3, replace=False).tolist() for _ in range(size - k)]
        conflicts = [[i, j] for i in range(size) for j in range(i + 1, size)
                     if set(cycles[i]) & set(cycles[j])]
        values = rng.uniform(0.01, 0.12, size)
        problem = {"track": track, "size": size, "cardinality": k,
                   "linear": (-values).tolist(), "quadratic": np.zeros((size, size)).tolist(),
                   "conflicts": conflicts,
                   "model": {"candidate_cycles": cycles, "net_synthetic_values": values.tolist(),
                             "units": "negative synthetic cycle/basket value"}}
    elif track == "estimation":
        # A finite, uniform quantile distribution: both methods receive exactly this grid.
        spot, strike, rate, maturity, volatility = 100.0, 100.0, 0.03, 1.0, 0.2
        z = norm.ppf((np.arange(size) + 0.5) / size)
        terminal = spot * np.exp((rate - volatility**2 / 2) * maturity + volatility * np.sqrt(maturity) * z)
        payoff = np.exp(-rate * maturity) * np.maximum(terminal - strike, 0)
        scale = max(float(payoff.max()), 1.0)
        problem = {"track": track, "size": size, "probabilities": (np.ones(size) / size).tolist(),
                   "payoffs": (payoff / scale).tolist(), "scale": scale,
                   "model": {"spot": spot, "strike": strike, "rate": rate, "maturity": maturity,
                             "volatility": volatility, "terminal_prices": terminal.tolist(),
                             "units": "discounted finite-grid call payoff; not a continuous-model price"}}
    else:
        raise ValueError(f"unknown track: {track}")
    if track == "estimation":
        reference = {"value": float(np.dot(problem["probabilities"], problem["payoffs"]) * problem["scale"]),
                     "method": "finite_distribution_expectation", "solution": None}
    else:
        solution, value = exact_reference(problem)
        reference = {"value": value, "method": "complete_feasible_enumeration", "solution": solution}
    instance = {"schema_version": SCHEMA_VERSION, "instance_id": f"{track}-n{size}-s{seed}-v1",
                "problem": problem, "reference": reference,
                "provenance": {"generator": GENERATOR_VERSION, "seed": seed, "license": "CC0-1.0",
                               "source": "synthetic", "timestamps": "not applicable; no market observations"}}
    validate_instance(instance)
    return instance
