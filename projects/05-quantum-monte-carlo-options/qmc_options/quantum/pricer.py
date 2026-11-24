"""End-to-end quantum-simulated pricing with stage-by-stage timing."""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np

from ..models import Market, Option
from . import amplitude_estimation as ae
from .discretize import DiscretizedProblem, discretize
from .encoding import build_a_operator, build_grover_operator
from .resources import circuit_resources, estimation_resources

ESTIMATORS = ("iqae", "mlae", "direct_sampling")


@dataclass
class PreparedProblem:
    """Everything that depends on the contract but not on the estimator or
    seed: the lookup table, the circuits, their unitaries and gate counts."""

    problem: DiscretizedProblem
    oracle: ae.CircuitOracle
    a_resources: Dict[str, int]
    q_resources: Dict[str, int]
    timings: Dict[str, float]


def prepare(market: Market, option: Option, qubits_per_step: int, truncation: float = 3.0,
            with_resources: bool = True) -> PreparedProblem:
    timings: Dict[str, float] = {}
    t = time.perf_counter()
    problem = discretize(market, option, qubits_per_step, truncation)
    timings["discretize_s"] = time.perf_counter() - t

    t = time.perf_counter()
    a_op = build_a_operator(problem)
    q_op = build_grover_operator(a_op)
    timings["build_circuits_s"] = time.perf_counter() - t

    t = time.perf_counter()
    oracle = ae.CircuitOracle(a_op, q_op)
    timings["simulate_circuits_s"] = time.perf_counter() - t

    t = time.perf_counter()
    a_res = circuit_resources(a_op) if with_resources else {}
    q_res = circuit_resources(q_op) if with_resources else {}
    timings["transpile_s"] = time.perf_counter() - t
    return PreparedProblem(problem, oracle, a_res, q_res, timings)


def estimate_price(prepared: PreparedProblem, estimator: str = "iqae", seed: int = 0,
                   alpha: float = 0.05, **options: Any) -> Dict[str, Any]:
    """Run one estimator. ``options``: iqae -> epsilon, shots; mlae ->
    max_power_exponent, shots; direct_sampling -> shots."""
    if estimator not in ESTIMATORS:
        raise ValueError(f"unknown estimator {estimator!r}; expected one of {ESTIMATORS}")
    problem = prepared.problem
    rng = np.random.default_rng(seed)
    start = time.perf_counter()
    if problem.payoff_max == 0.0:
        # Nothing to estimate: every discretised path pays zero.
        result = ae.AEResult(estimator, 0.0, 0.0, 0.0, 0, 0, 0, 0)
    elif estimator == "iqae":
        result = ae.iqae(prepared.oracle, rng, alpha=alpha, **options)
    elif estimator == "mlae":
        result = ae.mlae(prepared.oracle, rng, alpha=alpha, **options)
    else:
        result = ae.direct_sampling(prepared.oracle, rng=rng, alpha=alpha, **options)
    elapsed = time.perf_counter() - start

    scale = problem.scale
    out: Dict[str, Any] = {
        "estimator": estimator,
        "seed": seed,
        "price": scale * result.estimate,
        "ci_low": scale * result.ci_low,
        "ci_high": scale * result.ci_high,
        "amplitude_estimate": result.estimate,
        "amplitude_exact": problem.amplitude,
        "discretized_exact_price": problem.exact_price,
        "price_scale": scale,
        "shots": result.shots,
        "oracle_queries": result.oracle_queries,
        "grover_iterations": result.grover_iterations,
        "max_grover_power": result.max_grover_power,
        "estimation_time_s": elapsed,
    }
    if prepared.a_resources:
        out["resources"] = estimation_resources(prepared.a_resources, prepared.q_resources,
                                                result.max_grover_power, result.grover_iterations,
                                                result.shots)
    return out


def quantum_price(market: Market, option: Option, qubits_per_step: int, estimator: str = "iqae",
                  seed: int = 0, truncation: float = 3.0, alpha: float = 0.05,
                  with_resources: bool = False, **options: Any) -> Dict[str, Any]:
    """Convenience wrapper: prepare + estimate in one call."""
    prepared = prepare(market, option, qubits_per_step, truncation, with_resources)
    out = estimate_price(prepared, estimator, seed, alpha, **options)
    out["timings"] = prepared.timings
    return out


def quantum_delta(market: Market, option: Option, qubits_per_step: int, rel_bump: float = 0.02,
                  seed: int = 0, truncation: float = 3.0, alpha: float = 0.05,
                  epsilon: float = 0.002, shots: int = 100,
                  exact: Optional[bool] = False) -> Dict[str, float]:
    """Central finite-difference delta from two IQAE prices. The two
    estimates are independent, so their interval half-widths add and are
    divided by 2h: the uncertainty grows as the bump shrinks."""
    h = rel_bump * market.spot
    legs = []
    for i, spot in enumerate((market.spot + h, market.spot - h)):
        legs.append(quantum_price(market.bumped(spot=spot), option, qubits_per_step, "iqae",
                                  seed=seed + i, truncation=truncation, alpha=alpha / 2,
                                  epsilon=epsilon, shots=shots))
    up, dn = legs
    return {
        "delta": (up["price"] - dn["price"]) / (2 * h),
        "ci_low": (up["ci_low"] - dn["ci_high"]) / (2 * h),
        "ci_high": (up["ci_high"] - dn["ci_low"]) / (2 * h),
        "discretized_exact_delta": (up["discretized_exact_price"] - dn["discretized_exact_price"]) / (2 * h),
        "oracle_queries": up["oracle_queries"] + dn["oracle_queries"],
        "rel_bump": rel_bump,
    }
