"""Random allocation instances for tests and the static solver benchmark."""
from __future__ import annotations

from .problem import CostModel, build_problem


def random_instance(rng, V=3, max_units=3, target=None, unit_lots=None, cross=True):
    """Instance with fixed charges (non-convex) and fade coupling (non-separable).

    With `unit_lots` set, the target is max_units * unit_lots and caps are
    generous, so the QUBO uses close to V * max_units qubits.
    Returns (coarse problem, same instance at one-lot granularity).
    """
    if unit_lots:
        target = max_units * unit_lots
    target = target or int(rng.integers(2, 3 * max_units))
    ladders, caps = [], []
    for _ in range(V):
        base = int(rng.integers(0, 2))
        n_levels = int(rng.integers(1, 4))
        hi = max(6, target) if unit_lots else 6
        lad = tuple((100 + base + i, int(rng.integers(1, hi))) for i in range(n_levels))
        depth = sum(s for _, s in lad)
        ladders.append(lad)
        caps.append(max(depth, target) if unit_lots else int(rng.integers(1, depth + 1)))
    fade = [[0.0] * V for _ in range(V)]
    if cross:
        for a in range(V):
            for b in range(V):
                if a != b and rng.random() < 0.5:
                    fade[a][b] = float(rng.uniform(0.2, 1.0))
    model = CostModel(fee=tuple(rng.uniform(-0.1, 0.4, V)), fixed=tuple(rng.uniform(0, 1.5, V)),
                      impact=tuple(rng.uniform(0, 0.05, V)), fill_prob=tuple(rng.uniform(0.7, 1.0, V)),
                      fade=tuple(tuple(r) for r in fade), nonfill_penalty=2.0)
    return (build_problem(1, target, ladders, caps, 100, model, max_units),
            build_problem(1, target, ladders, caps, 100, model, None))
