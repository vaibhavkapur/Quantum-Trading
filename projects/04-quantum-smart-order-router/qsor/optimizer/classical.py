"""Classical solvers: heuristics, exhaustive enumeration and CP-SAT."""
from __future__ import annotations

import time

import numpy as np

from .problem import AllocationProblem, repair


class SolveTimeout(Exception):
    pass


def best_price_first(p: AllocationProblem):
    """Fill venues in order of displayed price, ignoring fees, fade and fixed costs."""
    order = sorted((v for v in range(p.V) if p.n_units[v]),
                   key=lambda v: (p.best_px[v] * p.side, v))
    x, left = [0] * p.V, p.target_units
    for v in order:
        x[v] = min(p.n_units[v], left)
        left -= x[v]
    return tuple(x)


def pro_rata(p: AllocationProblem):
    """Split proportionally to displayed depth (largest remainder, capped)."""
    w = [p.depth[v] if p.n_units[v] else 0 for v in range(p.V)]
    tot = sum(w)
    if tot == 0:
        return tuple([0] * p.V)
    x = [min(p.n_units[v], p.target_units * w[v] // tot) for v in range(p.V)]
    order = sorted(range(p.V), key=lambda v: (-((p.target_units * w[v]) % tot), -w[v], v))
    left = p.target_units - sum(x)
    while left > 0:
        for v in order:
            if left and x[v] < p.n_units[v]:
                x[v] += 1
                left -= 1
    return tuple(x)


def greedy(p: AllocationProblem):
    """Liquidity-aware greedy: add one unit at a time where total cost rises least."""
    return repair(p, [0] * p.V)


def exact(p: AllocationProblem):
    """Exhaustive enumeration (ground truth for small instances)."""
    shape = tuple(n + 1 for n in p.n_units)
    if int(np.prod(shape, dtype=np.int64)) > 5_000_000:
        raise ValueError("instance too large for enumeration; use cpsat")
    grids = np.indices(shape).reshape(p.V, -1)
    grids = grids[:, grids.sum(axis=0) == p.target_units]
    cost = np.zeros(grids.shape[1])
    for v in range(p.V):
        cost += np.asarray(p.tables[v])[grids[v]]
    for a in range(p.V):
        for b in range(a + 1, p.V):
            if p.cross[a][b]:
                cost += p.cross[a][b] * grids[a] * grids[b]
    return tuple(int(k) for k in grids[:, int(np.argmin(cost))])


def cpsat(p: AllocationProblem, scale: int = 10000, deadline=None):
    """Exact integer program with OR-Tools CP-SAT (costs scaled to integers)."""
    from ortools.sat.python import cp_model

    m = cp_model.CpModel()
    xs = [m.NewIntVar(0, n, f"x{v}") for v, n in enumerate(p.n_units)]
    m.Add(sum(xs) == p.target_units)
    terms = []
    for v, row in enumerate(p.tables):
        if len(row) == 1:
            continue
        ints = [int(round(c * scale)) for c in row]
        c = m.NewIntVar(min(ints), max(ints), f"c{v}")
        m.AddElement(xs[v], ints, c)
        terms.append(c)
    for a in range(p.V):
        for b in range(a + 1, p.V):
            k = int(round(p.cross[a][b] * scale))
            if k and p.n_units[a] and p.n_units[b]:
                prod = m.NewIntVar(0, p.n_units[a] * p.n_units[b], f"p{a}_{b}")
                m.AddMultiplicationEquality(prod, [xs[a], xs[b]])
                terms.append(k * prod)
    m.Minimize(sum(terms))
    solver = cp_model.CpSolver()
    solver.parameters.num_workers = 1
    solver.parameters.random_seed = 0
    if deadline is not None:
        left = deadline - time.perf_counter()
        if left <= 0:
            raise SolveTimeout("no time left for CP-SAT")
        solver.parameters.max_time_in_seconds = left
    status = solver.Solve(m)
    if status != cp_model.OPTIMAL:
        raise SolveTimeout(f"CP-SAT ended with status {solver.StatusName(status)}")
    return tuple(int(solver.Value(x)) for x in xs)
