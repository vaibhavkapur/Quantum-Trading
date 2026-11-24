"""Candidate generation -> exact-cent repair -> simulated orders -> audit data."""
from __future__ import annotations

from dataclasses import asdict
from importlib.metadata import PackageNotFoundError, version
import platform
from time import perf_counter

from .encoding import encode
from .model import OBJECTIVE_VERSION, Portfolio, Problem, ResourceLimitError, digest
from .quantum import qaoa
from .solvers import Budget, SolverFailure, anneal, convex_round, greedy, repair, solve_milp

SOLVERS = {"classical": solve_milp, "greedy": greedy, "convex-round": convex_round,
           "anneal": anneal, "quantum-sim": qaoa}


def versions() -> dict:
    result = {"python": platform.python_version(), "platform": platform.platform(), "implementation": "0.1.0"}
    for package in ("numpy", "scipy", "qiskit", "fastapi"):
        try:
            result[package] = version(package)
        except PackageNotFoundError:
            result[package] = "unavailable"
    return result


def plan(portfolio: Portfolio, solver: str = "classical", budget: Budget | None = None, *, fallback: str | None = None) -> dict:
    if solver not in SOLVERS or fallback not in (None, "classical"):
        raise ValueError("unknown solver or fallback")
    budget = budget or Budget()
    start = perf_counter()
    problem = Problem(portfolio)
    encoding = encode(problem)
    encoding_s = perf_counter() - start
    t = perf_counter()
    actual_solver, fallback_reason = solver, None
    try:
        candidate = SOLVERS[solver](encoding, budget)
    except (ImportError, ResourceLimitError, SolverFailure) as exc:
        if fallback is None or solver == fallback:
            raise
        actual_solver, fallback_reason = fallback, f"{type(exc).__name__}: {exc}"
        candidate = SOLVERS[fallback](encoding, budget)
    solve_s = perf_counter() - t
    t = perf_counter()
    indices, repair_info = repair(encoding, candidate, budget)
    repair_s = perf_counter() - t
    t = perf_counter()
    report = problem.evaluate(indices)
    if not report["feasible"] or report["conservation_residual_cents"] != 0:
        raise SolverFailure("risk gate rejected the proposed orders")
    decoding_s = perf_counter() - t
    identity = {"portfolio": portfolio.to_dict(), "solver_requested": solver, "solver_used": actual_solver,
                "budget": asdict(budget), "objective_version": OBJECTIVE_VERSION, "implementation_version": "0.1.0",
                "delta_lots": [r["delta_lots"] for r in report["holdings"]]}
    input_hash = digest(portfolio.to_dict())
    return {"schema_version": 1, "recommendation_id": digest(identity), "input_hash": input_hash,
            "solver_requested": solver, "solver_used": actual_solver, "fallback_reason": fallback_reason,
            "status": candidate.status, "budget": asdict(budget), "evaluations": candidate.evaluations,
            "portfolio": portfolio.to_dict(), "report": report, "repair": repair_info,
            "solver_metadata": candidate.metadata, "environment": versions(),
            "timings": {"encoding_s": encoding_s, "solver_s": solve_s, "repair_s": repair_s,
                        "decoding_s": decoding_s, "planning_total_s": perf_counter() - start}}

