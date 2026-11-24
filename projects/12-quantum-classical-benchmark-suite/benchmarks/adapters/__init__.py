"""Adapters share the same problem schema; no adapter receives reference answers."""

DISCRETE = {"exact", "milp", "greedy", "annealing", "qaoa"}
ESTIMATION = {"exact-expectation", "mc", "mc-antithetic", "qae"}


def compatible(solver: str, track: str) -> bool:
    return solver in (ESTIMATION if track == "estimation" else DISCRETE)


def get_adapter(name: str):
    if name in {"exact", "milp", "greedy", "annealing"}:
        from .classical import ClassicalAdapter
        return ClassicalAdapter(name)
    if name == "qaoa":
        from .quantum import QAOAAdapter
        return QAOAAdapter()
    if name in ESTIMATION:
        from .estimation import EstimationAdapter
        return EstimationAdapter(name)
    raise ValueError(f"unknown solver {name!r}")
