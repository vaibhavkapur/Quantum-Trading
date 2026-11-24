"""Uniform solver interface: name -> solve(problem, rng, warm, deadline)."""
from __future__ import annotations

from . import classical, quantum

SNAPSHOT_SOLVERS = ("best_price", "pro_rata", "greedy", "exact", "cpsat", "sa", "random_repair", "qaoa")
LIVE_SOLVERS = ("greedy_live",)
ALIASES = {
    "classical": ["best_price", "pro_rata", "greedy", "greedy_live", "exact", "cpsat"],
    "quantum-sim": ["qaoa", "sa", "random_repair"],
}
KIND = {"qaoa": "quantum simulator (noiseless statevector)", "sa": "classical QUBO heuristic",
        "random_repair": "control: uniform bitstrings + repair", "exact": "classical exact (enumeration)",
        "cpsat": "classical exact (CP-SAT)", "greedy": "classical heuristic",
        "best_price": "classical heuristic", "pro_rata": "classical heuristic",
        "greedy_live": "classical heuristic computed on the hot path (no snapshot)"}


def expand(name: str):
    if name in ALIASES:
        return list(ALIASES[name])
    if name in SNAPSHOT_SOLVERS or name in LIVE_SOLVERS:
        return [name]
    raise ValueError(f"unknown solver '{name}'; choose from "
                     f"{sorted(ALIASES) + list(SNAPSHOT_SOLVERS) + list(LIVE_SOLVERS)}")


def warmup(name: str) -> None:
    """Import heavy dependencies up front so they are not billed to the first solve."""
    if name == "cpsat":
        from ortools.sat.python import cp_model  # noqa: F401


def solve(name: str, problem, rng, solver_cfg: dict, warm=None, deadline=None):
    """Returns (units per venue, info dict). Raises on timeout / unsupported size."""
    if problem.target_units == 0:
        return tuple([0] * problem.V), {"evals": 0}
    opts = dict(solver_cfg.get(name, {}))
    if name == "best_price":
        return classical.best_price_first(problem), {"evals": 0}
    if name == "pro_rata":
        return classical.pro_rata(problem), {"evals": 0}
    if name == "greedy":
        return classical.greedy(problem), {"evals": 0}
    if name == "exact":
        return classical.exact(problem), {"evals": 0}
    if name == "cpsat":
        return classical.cpsat(problem, scale=opts.get("scale", 10000), deadline=deadline), {"evals": 0}
    if name == "sa":
        return quantum.simulated_annealing(problem, rng, deadline=deadline, **opts)
    if name == "random_repair":
        return quantum.random_repair(problem, rng, **opts)
    if name == "qaoa":
        if not opts.get("warm_start", True):
            warm = None
        return quantum.qaoa(problem, rng, warm=warm, deadline=deadline, **opts)
    raise ValueError(f"unknown solver '{name}'")
