"""Solver registry with uniform invocation and a documented fallback path."""
from __future__ import annotations

from typing import Callable

from ..research.qubo import BasketProblem
from . import classical
from .base import Budget, SolverResult

SolverFn = Callable[..., SolverResult]

SOLVERS: dict[str, SolverFn] = {
    "brute_force": classical.brute_force,
    "greedy": classical.greedy,
    "cpsat": classical.cpsat,
    "random_search": classical.random_search,
    "simulated_annealing": classical.simulated_annealing,
}
STOCHASTIC = {"random_search", "simulated_annealing", "qaoa"}


def _qaoa(problem: BasketProblem, budget: Budget, seed: int | None = None, **kwargs) -> SolverResult:
    from ..quantum.qaoa import qaoa

    return qaoa(problem, budget, seed, **kwargs)


SOLVERS["qaoa"] = _qaoa


def run_solver(name: str, problem: BasketProblem, budget: Budget, seed: int | None = None, **kwargs) -> SolverResult:
    """Run a solver; on any failure fall back to greedy and say so in ``status``.

    The result is always labelled with the requested solver name so a failed
    run stays visible in the report instead of silently disappearing.
    """
    if name not in SOLVERS:
        raise KeyError(f"unknown solver '{name}'; available: {sorted(SOLVERS)}")
    try:
        result = SOLVERS[name](problem, budget, seed, **kwargs)
        if not problem.is_feasible(result.z):
            raise RuntimeError("solver returned an infeasible selection")
        return result
    except Exception as exc:  # noqa: BLE001 - any adapter failure must degrade, not crash the study
        fallback = classical.greedy(problem, budget, seed)
        fallback.solver = name
        fallback.status = f"fallback:{type(exc).__name__}: {exc}"[:200]
        fallback.meta = {"fallback_solver": "greedy"}
        return fallback
