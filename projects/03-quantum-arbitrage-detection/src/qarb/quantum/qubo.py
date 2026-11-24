"""QUBO for valid fixed-length cycles.

Position encoding: x[v, p] = 1 iff currency v sits at position p of the cycle.

    objective   sum_p sum_{u != v} w~[u, v] x[u, p] x[v, p+1]      (cyclic in p)
    one-hot     A * sum_p (sum_v x[v, p] - 1)^2                    exactly one currency per position
    uniqueness  A * sum_v sum_{p < p'} x[v, p] x[v, p']            no currency twice

Weights are normalized to w~ = w / max|w| so |w~| <= 1; missing edges get the
positive weight L, which makes any cycle through a missing edge non-negative.

Anchored variant: position 0 is fixed to a given currency, removing the L-fold
rotational degeneracy and shrinking the register from N*L to (N-1)*(L-1)
variables. The anchored problem only sees cycles through the anchor, so it is
solved once per anchor.

Penalty: with |w~| <= 1, A = 2L + 4 provably makes the QUBO minimum a valid
cycle of minimum weight (see `auto_penalty`). Smaller A gives a flatter, more
QAOA-friendly landscape but loses the guarantee; `argmin_is_valid_optimum`
checks a given instance by brute force.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ..graph.cycles import Cycle, canonical, cycle_weight


def auto_penalty(L: int) -> float:
    """Sufficient penalty for normalized weights.

    Let m_p be the number of currencies at position p. Edge terms are bounded
    below by -sum_p m_p m_{p+1} >= -sum_p m_p^2, so any assignment with some
    m_p != 1 has energy >= -(L-1) + (A-4), and one with all m_p = 1 but a
    repeated currency has energy >= -L + A. Every valid cycle of real edges has
    energy <= L. A = 2L + 4 therefore puts all invalid assignments above all
    valid ones.
    """
    return 2.0 * L + 4.0


@dataclass
class CycleQubo:
    Q: np.ndarray  # upper triangular; diagonal holds linear terms
    offset: float
    L: int
    n_nodes: int
    anchor: int | None
    free_nodes: list[int]  # node index for each "row" of the encoding
    positions: list[int]  # cycle position for each "column" of the encoding
    scale: float
    penalty: float

    @property
    def n_vars(self) -> int:
        return self.Q.shape[0]

    def var(self, row: int, col: int) -> int:
        return col * len(self.free_nodes) + row

    def energy(self, x: np.ndarray) -> float:
        x = np.asarray(x, dtype=float)
        return float(x @ self.Q @ x + self.offset)

    def encode(self, cycle: Cycle) -> np.ndarray:
        """Bit vector for a cycle (must start at the anchor if anchored)."""
        x = np.zeros(self.n_vars, dtype=np.int8)
        seq = list(cycle)
        if self.anchor is not None:
            k = seq.index(self.anchor)
            seq = seq[k:] + seq[:k]
        for p, v in enumerate(seq):
            if p in self.positions:
                x[self.var(self.free_nodes.index(v), self.positions.index(p))] = 1
        return x

    def assignment(self, x: np.ndarray) -> list[list[int]]:
        """Nodes selected at each encoded position."""
        m = len(self.free_nodes)
        return [[self.free_nodes[r] for r in range(m) if x[c * m + r]] for c in range(len(self.positions))]

    def decode(self, x: np.ndarray, W: np.ndarray) -> Cycle | None:
        """Canonical cycle if x satisfies every constraint and uses only real edges."""
        chosen = self.assignment(x)
        if any(len(c) != 1 for c in chosen):
            return None
        seq = [c[0] for c in chosen]
        if self.anchor is not None:
            seq = [self.anchor] + seq
        if len(set(seq)) != len(seq):
            return None
        cyc = canonical(tuple(seq))
        return cyc if math.isfinite(cycle_weight(W, cyc)) else None

    def repair(self, x: np.ndarray, W: np.ndarray) -> Cycle | None:
        """Greedy repair of an infeasible sample.

        Positions are filled left to right, preferring currencies the sample
        selected there, otherwise the unused currency with the cheapest
        incoming edge. Returns None if no real-edge cycle results.
        """
        chosen = self.assignment(x)
        seq = [self.anchor] if self.anchor is not None else []
        for sel in chosen:
            options = [v for v in sel if v not in seq] or [v for v in self.free_nodes if v not in seq]
            if not options:
                return None
            if seq:
                options.sort(key=lambda v: (W[seq[-1], v], v))
            seq.append(options[0])
        cyc = canonical(tuple(seq))
        return cyc if math.isfinite(cycle_weight(W, cyc)) else None


def build_cycle_qubo(
    W: np.ndarray, L: int, anchor: int | None = None, penalty: float | str = "auto"
) -> CycleQubo:
    n = W.shape[0]
    if L < 2 or L > n:
        raise ValueError(f"cycle length {L} invalid for {n} nodes")
    finite = np.isfinite(W)
    np.fill_diagonal(finite, False)
    scale = float(np.max(np.abs(W[finite]))) if finite.any() else 1.0
    scale = scale or 1.0
    Wn = np.where(finite, W / scale, float(L))
    A = auto_penalty(L) if penalty == "auto" else float(penalty)

    if anchor is None:
        free_nodes = list(range(n))
        positions = list(range(L))
    else:
        free_nodes = [v for v in range(n) if v != anchor]
        positions = list(range(1, L))
    m, P = len(free_nodes), len(positions)
    nv = m * P
    Q = np.zeros((nv, nv))
    offset = 0.0

    def var(r: int, c: int) -> int:
        return c * m + r

    def add(i: int, j: int, val: float) -> None:
        if i == j:
            Q[i, i] += val
        else:
            Q[min(i, j), max(i, j)] += val

    # objective
    if anchor is None:
        for c in range(P):
            c2 = (c + 1) % P
            for r1, u in enumerate(free_nodes):
                for r2, v in enumerate(free_nodes):
                    if u != v:
                        add(var(r1, c), var(r2, c2), Wn[u, v])
    else:
        for r, v in enumerate(free_nodes):
            add(var(r, 0), var(r, 0), Wn[anchor, v])
            add(var(r, P - 1), var(r, P - 1), Wn[v, anchor])
        for c in range(P - 1):
            for r1, u in enumerate(free_nodes):
                for r2, v in enumerate(free_nodes):
                    if u != v:
                        add(var(r1, c), var(r2, c + 1), Wn[u, v])
    # one-hot per position: A (sum_v x - 1)^2
    for c in range(P):
        offset += A
        for r1 in range(m):
            add(var(r1, c), var(r1, c), -A)
            for r2 in range(r1 + 1, m):
                add(var(r1, c), var(r2, c), 2 * A)
    # uniqueness per currency
    for r in range(m):
        for c1 in range(P):
            for c2 in range(c1 + 1, P):
                add(var(r, c1), var(r, c2), A)
    return CycleQubo(Q, offset, L, n, anchor, free_nodes, positions, scale, A)


def all_bitstrings(n: int) -> np.ndarray:
    """(2^n, n) matrix; row k is the bits of k with variable i at bit i (LSB first)."""
    return ((np.arange(2**n, dtype=np.int64)[:, None] >> np.arange(n)) & 1).astype(np.int8)


def energy_vector(qubo: CycleQubo, chunk: int = 1 << 14) -> np.ndarray:
    """Energy of every assignment, indexed by integer with variable i at bit i."""
    n = qubo.n_vars
    E = np.empty(2**n)
    idx = np.arange(n)
    for lo in range(0, 2**n, chunk):
        hi = min(lo + chunk, 2**n)
        bits = ((np.arange(lo, hi, dtype=np.int64)[:, None] >> idx) & 1).astype(float)
        E[lo:hi] = np.einsum("ki,ij,kj->k", bits, qubo.Q, bits) + qubo.offset
    return E


def index_to_bits(k: int, n: int) -> np.ndarray:
    return np.array([(k >> i) & 1 for i in range(n)], dtype=np.int8)


def to_ising(Q: np.ndarray, offset: float = 0.0) -> tuple[np.ndarray, np.ndarray, float]:
    """x = (1 - z) / 2. Returns (h, J upper triangular, constant)."""
    n = Q.shape[0]
    lin = np.diag(Q).copy()
    quad = np.triu(Q, 1)
    sym = quad + quad.T
    h = -lin / 2 - sym.sum(axis=1) / 4
    J = quad / 4
    const = offset + lin.sum() / 2 + quad.sum() / 4
    return h, J, const


def best_valid_cycle(qubo: CycleQubo, W: np.ndarray, E: np.ndarray | None = None) -> tuple[Cycle | None, float]:
    """Brute force over the register: the minimum-energy assignment that decodes to a cycle."""
    E = energy_vector(qubo) if E is None else E
    for k in np.argsort(E, kind="stable"):
        cyc = qubo.decode(index_to_bits(int(k), qubo.n_vars), W)
        if cyc is not None:
            return cyc, float(E[k])
    return None, math.inf


def argmin_is_valid_optimum(qubo: CycleQubo, W: np.ndarray, E: np.ndarray | None = None) -> bool:
    """True iff the global QUBO minimum decodes to a valid cycle (formulation is exact here)."""
    E = energy_vector(qubo) if E is None else E
    return qubo.decode(index_to_bits(int(np.argmin(E)), qubo.n_vars), W) is not None
