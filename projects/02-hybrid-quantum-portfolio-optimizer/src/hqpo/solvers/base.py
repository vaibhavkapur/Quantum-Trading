"""Common solver interface, budget and result record."""
from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field

import numpy as np

from ..models.problem import PortfolioProblem
from ..models.repair import repair


@dataclass(frozen=True)
class Budget:
    """Equal computation budget handed to every solver (both limits are reported)."""

    wall_time_s: float = 10.0
    max_evals: int = 100_000


class SolverUnavailable(RuntimeError):
    """Raised when an optional backend is missing or the instance exceeds a hard limit."""


@dataclass
class SolveResult:
    solver: str
    status: str = "ok"                 # ok | infeasible | failed | skipped
    z: np.ndarray | None = None
    weights: np.ndarray | None = None
    objective: float = float("nan")    # post-repair objective (nan if no feasible portfolio)
    feasible: bool = False             # post-repair feasibility
    raw_objective: float = float("nan")  # objective of the solver's own answer, pre-repair
    raw_violation: int = -1            # total constraint violation pre-repair
    repaired: bool = False             # True if repair changed the answer
    evals: int = 0                     # objective / circuit evaluations ("optimizer calls")
    wall_time_s: float = 0.0           # end-to-end, including model build, decode and repair
    timings: dict = field(default_factory=dict)
    meta: dict = field(default_factory=dict)
    message: str = ""

    def record(self) -> dict:
        """Flat, JSON/CSV-friendly record."""
        d = asdict(self)
        d.pop("z"), d.pop("weights")
        timings, meta = d.pop("timings"), d.pop("meta")
        d.update({f"t_{k}": v for k, v in timings.items()})
        d.update({k: v for k, v in meta.items() if np.isscalar(v) or v is None})
        return d


class Timer:
    def __init__(self):
        self.t0 = time.perf_counter()
        self.parts: dict[str, float] = {}

    @contextmanager
    def section(self, name: str):
        t = time.perf_counter()
        try:
            yield
        finally:
            self.parts[name] = self.parts.get(name, 0.0) + time.perf_counter() - t

    @property
    def elapsed(self) -> float:
        return time.perf_counter() - self.t0


class Solver:
    """Subclasses implement ``_solve`` and return ``(z_raw, evals, meta)``."""

    name = "base"
    stochastic = False

    def __init__(self, **params):
        self.params = params

    def _solve(self, problem: PortfolioProblem, budget: Budget, rng: np.random.Generator,
               timer: Timer) -> tuple[np.ndarray | None, int, dict]:
        raise NotImplementedError

    def solve(self, problem: PortfolioProblem, budget: Budget | None = None,
              seed: int | None = None) -> SolveResult:
        budget = budget or Budget()
        rng = np.random.default_rng(seed)
        timer = Timer()
        res = SolveResult(solver=self.name)
        try:
            z_raw, evals, meta = self._solve(problem, budget, rng, timer)
        except SolverUnavailable as exc:
            res.status, res.message = "skipped", str(exc)
            res.wall_time_s, res.timings = timer.elapsed, timer.parts
            return res
        except Exception as exc:  # recorded as a failed run, never silently dropped
            res.status, res.message = "failed", f"{type(exc).__name__}: {exc}"
            res.wall_time_s, res.timings = timer.elapsed, timer.parts
            return res
        res.evals, res.meta = int(evals), meta
        if z_raw is None:
            res.status = "infeasible"
            res.message = meta.get("message", "solver returned no solution")
        else:
            finalize(problem, z_raw, res, timer)
        res.wall_time_s, res.timings = timer.elapsed, dict(timer.parts)
        return res


def finalize(problem: PortfolioProblem, z_raw, res: SolveResult, timer: Timer | None = None):
    """Score the raw answer, repair it if needed and fill the pre/post-repair fields."""
    z_raw = np.asarray(z_raw, dtype=int)
    res.raw_objective = problem.objective(z_raw)
    res.raw_violation = problem.total_violation(z_raw)
    if res.raw_violation == 0:
        z, ok = z_raw, True
    else:
        if timer is not None:
            with timer.section("repair"):
                z, ok = repair(problem, z_raw)
        else:
            z, ok = repair(problem, z_raw)
        res.repaired = True
    if ok:
        res.z, res.weights = z, problem.weights(z)
        res.objective, res.feasible, res.status = problem.objective(z), True, "ok"
    else:
        res.status, res.message = "infeasible", "repair failed to reach a feasible portfolio"
    return res
