"""Solver contract shared by classical and quantum adapters."""
from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field

import numpy as np

from ..research.qubo import BasketProblem


@dataclass(frozen=True)
class Budget:
    """Identical budget handed to every solver.

    ``max_evals`` caps how many candidate bitstrings a stochastic solver may
    score; ``time_limit_s`` caps wall time for exact solvers. Both the budget
    and what was actually consumed are reported.
    """

    max_evals: int
    time_limit_s: float


@dataclass
class SolverResult:
    solver: str
    z: np.ndarray                 # final feasible selection (after repair)
    objective: float              # base objective of z (no penalty terms)
    status: str                   # ok | optimal | feasible | fallback:<reason>
    n_evals: int = 0              # candidate bitstrings scored classically
    n_shots: int = 0              # circuit shots (quantum only)
    raw_feasible_rate: float = float("nan")  # share of raw candidates feasible before repair
    repaired: bool = False        # whether the returned z needed repair
    raw_best_objective: float = float("nan")  # best candidate feasible WITHOUT repair (nan if none)
    timings: dict = field(default_factory=dict)  # seconds per stage
    meta: dict = field(default_factory=dict)

    @property
    def wall_time(self) -> float:
        return float(sum(self.timings.values()))


class Timer:
    def __init__(self) -> None:
        self.t: dict[str, float] = {}

    @contextmanager
    def stage(self, name: str):
        start = time.perf_counter()
        try:
            yield
        finally:
            self.t[name] = self.t.get(name, 0.0) + time.perf_counter() - start


def best_of_candidates(problem: BasketProblem, candidates: np.ndarray, timer: Timer) -> tuple[np.ndarray, float, float, bool, float]:
    """Shared decode path: score raw candidates, repair, return the best feasible one.

    Returns (z, objective, raw_feasible_rate, best_needed_repair,
    raw_best_objective). Every stochastic solver uses this exact routine, so
    repair is never an advantage one solver has over another. Because greedy
    repair is itself a strong optimiser, ``raw_best_objective`` (best candidate
    that was feasible as sampled) is reported too: it isolates sampler quality.
    """
    with timer.stage("decode_repair"):
        cache: dict[bytes, tuple[np.ndarray, float, bool]] = {}
        feasible = 0
        best = None
        raw_best = float("nan")
        for raw in candidates:
            key = raw.tobytes()
            if key not in cache:
                ok = problem.is_feasible(raw)
                z = raw.astype(int) if ok else problem.repair(raw)
                cache[key] = (z, problem.base_objective(z), ok)
            z, val, ok = cache[key]
            feasible += ok
            if ok and not (val >= raw_best):
                raw_best = val
            if best is None or val < best[1] - 1e-12:
                best = (z, val, not ok)
    return best[0], best[1], feasible / max(len(candidates), 1), best[2], raw_best
