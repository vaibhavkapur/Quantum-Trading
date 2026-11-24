"""Solver registry.  Every solver receives the same problem object and budget."""
from .base import Budget, SolveResult, Solver, SolverUnavailable
from .classical import BruteForce, CpSat, CvxpyRelaxRound, Greedy, Miqp, SimulatedAnnealing
from .quantum import Qaoa, QaoaNoisy, QuboAnnealer, RandomQubo

SOLVERS = {cls.name: cls for cls in (BruteForce, CvxpyRelaxRound, Miqp, CpSat, Greedy,
                                     SimulatedAnnealing, Qaoa, QaoaNoisy, QuboAnnealer, RandomQubo)}

# what `--solver <group>` expands to when the config does not override it
DEFAULT_GROUPS = {
    "classical": ["cvxpy-relax", "miqp", "greedy", "sa"],
    "quantum-sim": ["qaoa"],
    "anneal-sim": ["qubo-sa"],
}
# solvers that consume the shared QUBO penalty settings
QUBO_SOLVERS = {"qaoa", "qaoa-noisy", "qubo-sa", "random-qubo"}


def preload() -> float:
    """Import the heavy optional backends up front so import cost is not billed to a solve.

    Returns the import time in seconds (recorded in run manifests)."""
    import importlib
    import time

    t = time.perf_counter()
    for mod in ("cvxpy", "clarabel", "pyscipopt", "ortools.sat.python.cp_model", "qiskit",
                "qiskit.transpiler", "qiskit_aer", "dwave.samplers", "dimod"):
        try:
            importlib.import_module(mod)
        except ImportError:
            pass
    return time.perf_counter() - t


def get_solver(name: str, **params) -> Solver:
    if name not in SOLVERS:
        raise KeyError(f"unknown solver {name!r}; available: {sorted(SOLVERS)}")
    return SOLVERS[name](**params)


__all__ = ["Budget", "SolveResult", "Solver", "SolverUnavailable", "SOLVERS", "DEFAULT_GROUPS",
           "QUBO_SOLVERS", "get_solver", "preload"]
