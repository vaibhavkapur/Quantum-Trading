"""One-hot QUBO encoding of the allocation problem.

Binary variable y[v,k] = 1 means "venue v receives exactly k units"
(k = 1..n_units[v]; all zero means nothing is sent to v).

    E(y) = sum c[v,k] y[v,k]                               (venue cost tables, exact)
         + sum_{a<b} cross[a][b] k k' y[a,k] y[b,k']       (fade coupling, exact)
         + A * sum_v sum_{k<k'} y[v,k] y[v,k']             (at most one k per venue)
         + A * (sum k y[v,k] - target)^2                   (conservation)

For feasible y, E(y) equals the problem objective exactly. A is chosen above
the spread between the largest feasible and smallest unconstrained objective,
so every infeasible assignment has higher energy than every feasible one.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import List, Tuple

import numpy as np

from .problem import AllocationProblem, repair


@dataclass
class Qubo:
    n: int
    lin: np.ndarray       # shape (n,)
    quad: np.ndarray      # strictly upper-triangular (n, n)
    offset: float
    var_map: List[Tuple[int, int]]  # variable -> (venue, units)
    penalty: float

    def energy(self, y) -> float:
        y = np.asarray(y, dtype=float)
        return float(self.lin @ y + y @ self.quad @ y + self.offset)

    def energies(self) -> np.ndarray:
        """Energy of every basis state; state index z has variable i = bit i of z."""
        B = _bits(self.n)
        return B @ self.lin + np.einsum("ij,jk,ik->i", B, self.quad, B) + self.offset


@lru_cache(maxsize=8)
def _bits(n: int) -> np.ndarray:
    z = np.arange(1 << n, dtype=np.int64)[:, None]
    return ((z >> np.arange(n)) & 1).astype(np.float64)


def build_qubo(p: AllocationProblem, penalty_scale: float = 1.0) -> Qubo:
    var_map = [(v, k) for v in range(p.V) for k in range(1, p.n_units[v] + 1)]
    n = len(var_map)
    lin, quad = np.zeros(n), np.zeros((n, n))
    hi = sum(max(row) for row in p.tables) + sum(
        p.cross[a][b] * p.n_units[a] * p.n_units[b] for a in range(p.V) for b in range(a + 1, p.V))
    lo = sum(min(row) for row in p.tables)
    A = penalty_scale * (hi - lo + 1.0)
    T = p.target_units
    for i, (v, k) in enumerate(var_map):
        lin[i] = p.tables[v][k] + A * (k * k - 2 * T * k)
        for j in range(i + 1, n):
            w, kk = var_map[j]
            q = 2 * A * k * kk
            q += A if w == v else p.cross[v][w] * k * kk
            quad[i, j] = q
    return Qubo(n, lin, quad, A * T * T, var_map, A)


def decode(p: AllocationProblem, qubo: Qubo, z: int):
    """Bitstring -> (repaired units per venue, was the raw assignment feasible)."""
    x, ok = [0] * p.V, True
    for i, (v, k) in enumerate(qubo.var_map):
        if (z >> i) & 1:
            if x[v]:
                ok = False
            x[v] = max(x[v], k)
    ok = ok and sum(x) == p.target_units
    return (tuple(x) if ok else repair(p, x)), ok


def best_of(p: AllocationProblem, qubo: Qubo, states):
    """Lowest-cost repaired allocation among sampled basis states."""
    best_x, best_c, n_feasible = None, float("inf"), 0
    for z in sorted(set(int(s) for s in states)):
        x, ok = decode(p, qubo, z)
        n_feasible += ok
        c = p.cost(x)
        if c < best_c - 1e-12:
            best_x, best_c = x, c
    return best_x, n_feasible
