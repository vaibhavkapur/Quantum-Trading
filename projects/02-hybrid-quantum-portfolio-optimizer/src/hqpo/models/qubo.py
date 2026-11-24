"""QUBO construction for the integer-lot portfolio problem.

Encoding
--------
Each integer lot variable ``z_j in [0, ub_j]`` uses the *bounded-coefficient* binary
encoding ``z_j = sum_b c_b x_b`` with ``c = [1, 2, 4, ..., rest]`` and ``sum(c) = ub_j``.
The position limit is therefore satisfied by construction and costs no penalty term.

Penalties (single strength ``P``)
---------------------------------
* budget equalities          ``P (a.z - L)^2``
* sector caps (inequalities) ``P (a.z + s - hi)^2`` with a bounded binary slack ``s``
* cardinality                indicator qubit ``y_i`` per asset, ``P x_b (1 - y_i)`` for every
                             bit of the asset, and ``P (sum_i y_i - C)^2``.  ``y_i = 1`` with
                             an empty position is allowed, so the equality encodes
                             "at most C names" without slack qubits.
* long/short exclusivity     ``P x_b x_c`` for every bit pair of ``(p_i, q_i)``

Constraints that cannot bind are presolved away, so they consume no qubits.

Qubit growth (documented, and measured by ``qubit_breakdown``)::

    n_qubits = legs * n * bits(K)                       position bits, K = floor(w_max / lot)
             + n                  if cardinality binds   indicator bits
             + sum_s bits(range_s)                       sector slack bits
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .problem import PortfolioProblem


def bounded_coefficients(ub: int) -> list[int]:
    """Binary-expansion coefficients whose subset sums are exactly ``{0, ..., ub}``."""
    if ub < 0:
        raise ValueError("ub must be non-negative")
    coefs, power, remaining = [], 1, int(ub)
    while remaining > 0:
        c = min(power, remaining)
        coefs.append(c)
        remaining -= c
        power *= 2
    return coefs


def encode_int(value: int, ub: int) -> list[int]:
    """Bits of ``value`` under the bounded-coefficient encoding of ``[0, ub]``."""
    coefs = bounded_coefficients(ub)
    if not 0 <= value <= ub:
        raise ValueError(f"{value} outside encodable range [0, {ub}]")
    bits = [0] * len(coefs)
    if coefs and value >= coefs[-1] and coefs[-1] != 2 ** (len(coefs) - 1):
        bits[-1] = 1                      # use the remainder coefficient first
        value -= coefs[-1]
    for b in range(len(coefs)):
        if coefs[b] == 2 ** b and (value >> b) & 1:
            bits[b] = 1
            value -= coefs[b]
    if value != 0:                        # only reachable if the remainder bit is still free
        bits[-1] = 1
        value -= coefs[-1]
    assert value == 0
    return bits


def n_bits(ub: int) -> int:
    return len(bounded_coefficients(ub))


def gradient_bound(problem: PortfolioProblem) -> float:
    """Upper bound on the objective change from moving one lot anywhere in the box.

    ``|df/dz_j| <= |b_j| + 2 sum_k |A_jk| ub_k``; one unit of constraint violation costs at
    least ``P``, so ``P`` of this order makes violating a constraint unprofitable.
    """
    g = np.abs(problem.b) + 2.0 * np.abs(problem.A) @ problem.ub
    return float(g.max())


def penalty_strength(problem: PortfolioProblem, method: str = "gradient_bound", scale: float = 1.0,
                     value: float | None = None) -> float:
    if method == "fixed":
        if value is None:
            raise ValueError("penalty method 'fixed' needs a value")
        return float(value) * scale
    if method == "gradient_bound":
        return scale * gradient_bound(problem)
    if method == "range":
        # crude objective range over the box: always sufficient, usually far too large
        ub = problem.ub.astype(float)
        rng = ub @ np.abs(problem.A) @ ub + np.abs(problem.b) @ ub
        return scale * float(rng)
    raise ValueError(f"unknown penalty method {method!r}")


@dataclass
class QuboModel:
    """``E(x) = sum_i Q_ii x_i + sum_{i<j} Q_ij x_i x_j + offset`` (Q upper triangular)."""

    Q: np.ndarray
    offset: float
    T: np.ndarray               # (m, n_qubits) integer map: z = T x
    penalty: float
    labels: list[str]
    kinds: list[str]            # 'position' | 'indicator' | 'slack'
    problem: PortfolioProblem = field(repr=False, default=None)
    _slack_info: list = field(default_factory=list, repr=False)

    @property
    def n_qubits(self) -> int:
        return self.Q.shape[0]

    def qubit_breakdown(self) -> dict[str, int]:
        out = {"position": 0, "indicator": 0, "slack": 0}
        for k in self.kinds:
            out[k] += 1
        out["total"] = self.n_qubits
        return out

    def decode(self, x) -> np.ndarray:
        return (self.T @ np.asarray(x, dtype=int)).astype(int)

    def encode(self, z) -> np.ndarray:
        """A bitstring with zero penalty for a *feasible* ``z`` (slack/indicators filled in)."""
        z = np.asarray(z, dtype=int)
        p = self.problem
        x = np.zeros(self.n_qubits, dtype=int)
        col = 0
        for j in range(p.m):
            bits = encode_int(int(z[j]), int(p.ub[j]))
            x[col: col + len(bits)] = bits
            col += len(bits)
        ind = [q for q, k in enumerate(self.kinds) if k == "indicator"]
        if ind:
            held = p.held_mask(z)
            spare = p.constraints.cardinality - int(held.sum())
            for i, q in enumerate(ind):
                if held[i]:
                    x[q] = 1
                elif spare > 0:
                    x[q] = 1
                    spare -= 1
        for cons, cols, coefs in self._slack_info:
            x[cols] = encode_int(cons.hi - int(cons.a @ z), cons.hi - cons.lo)
        return x

    def energy(self, x) -> float:
        x = np.asarray(x, dtype=float)
        return float(x @ self.Q @ x + self.offset)

    def energies_all(self) -> np.ndarray:
        """Energy of every bitstring; index bit ``j`` (little-endian) is ``x_j``. O(n 2^n)."""
        n = self.n_qubits
        if n > 26:
            raise ValueError(f"refusing to enumerate 2^{n} states")
        E = np.array([self.offset])
        for k in range(n):
            lin = np.zeros(1)
            for j in range(k):
                lin = np.concatenate([lin, lin + self.Q[j, k]])
            E = np.concatenate([E, E + self.Q[k, k] + lin])
        return E

    def index_to_bits(self, idx) -> np.ndarray:
        idx = np.asarray(idx, dtype=np.int64)
        return ((idx[..., None] >> np.arange(self.n_qubits)) & 1).astype(int)

    def symmetric(self) -> tuple[np.ndarray, np.ndarray]:
        """(linear, symmetric zero-diagonal coupling) with ``E = lin.x + x' J x / 1 + offset``
        where ``J_ij = J_ji = Q_ij / 2`` so that ``x' J x = sum_{i<j} Q_ij x_i x_j``."""
        lin = np.diag(self.Q).copy()
        off = np.triu(self.Q, 1)
        return lin, (off + off.T) / 2.0

    def to_ising(self) -> tuple[np.ndarray, np.ndarray, float]:
        """``E = sum_i h_i z_i + sum_{i<j} J_ij z_i z_j + const`` with ``x = (1 - z) / 2``."""
        off = np.triu(self.Q, 1)
        J = off / 4.0
        sym = off + off.T
        h = -np.diag(self.Q) / 2.0 - sym.sum(axis=1) / 4.0
        const = self.offset + np.diag(self.Q).sum() / 2.0 + off.sum() / 4.0
        return h, J, float(const)


def build_qubo(problem: PortfolioProblem, penalty: float | None = None, *,
               method: str = "gradient_bound", scale: float = 1.0,
               value: float | None = None) -> QuboModel:
    p = problem
    P = penalty_strength(p, method, scale, value) if penalty is None else float(penalty)

    # ---- register qubits
    cols_T: list[np.ndarray] = []   # column of T per qubit
    labels: list[str] = []
    kinds: list[str] = []
    var_bits: list[list[int]] = []  # qubit indices per integer variable
    for j in range(p.m):
        bits = []
        for b, c in enumerate(bounded_coefficients(int(p.ub[j]))):
            col = np.zeros(p.m, dtype=int)
            col[j] = c
            bits.append(len(cols_T))
            cols_T.append(col)
            side = ""
            if p.market_neutral:
                side = "L:" if j < p.n else "S:"
            labels.append(f"{side}{p.assets[j % p.n]}[{c}]")
            kinds.append("position")
        var_bits.append(bits)

    ind_bits: list[int] = []
    if p.cardinality_binding:
        for i in range(p.n):
            ind_bits.append(len(cols_T))
            cols_T.append(np.zeros(p.m, dtype=int))
            labels.append(f"held:{p.assets[i]}")
            kinds.append("indicator")

    slack_info = []
    for cons in p.sector_constraints:
        coefs = bounded_coefficients(cons.hi - cons.lo)
        cols = []
        for c in coefs:
            cols.append(len(cols_T))
            cols_T.append(np.zeros(p.m, dtype=int))
            labels.append(f"slack:{cons.name}[{c}]")
            kinds.append("slack")
        slack_info.append((cons, cols, coefs))

    nq = len(cols_T)
    T = np.stack(cols_T, axis=1) if nq else np.zeros((p.m, 0), dtype=int)

    # ---- objective: z'Az + b'z with z = T x  (symmetric accumulator, folded at the end)
    Tf = T.astype(float)
    Qs = Tf.T @ p.A @ Tf
    lin = Tf.T @ p.b
    offset = p.c0

    def add_squared(v: np.ndarray, rhs: float):
        nonlocal Qs, lin, offset
        Qs = Qs + P * np.outer(v, v)
        lin = lin - 2.0 * P * rhs * v
        offset += P * rhs * rhs

    for cons in p.budget_constraints:
        add_squared(cons.a.astype(float) @ Tf, cons.lo)

    for cons, cols, coefs in slack_info:
        v = cons.a.astype(float) @ Tf
        v[cols] = coefs
        add_squared(v, cons.hi)

    if ind_bits:
        v = np.zeros(nq)
        v[ind_bits] = 1.0
        add_squared(v, p.constraints.cardinality)
        for i, group in enumerate(p.groups):
            y = ind_bits[i]
            for j in group:
                for b in var_bits[j]:
                    lin[b] += P                 # P * x_b
                    Qs[b, y] -= P / 2.0         # -P * x_b * y   (split symmetrically)
                    Qs[y, b] -= P / 2.0

    for (j1, j2) in p.exclusive_pairs:
        for b1 in var_bits[j1]:
            for b2 in var_bits[j2]:
                Qs[b1, b2] += P / 2.0
                Qs[b2, b1] += P / 2.0

    # ---- fold symmetric form into upper triangular with x_i^2 = x_i
    Q = np.triu(Qs, 1) * 2.0
    Q[np.diag_indices(nq)] = np.diag(Qs) + lin
    model = QuboModel(Q=Q, offset=float(offset), T=T, penalty=P, labels=labels, kinds=kinds,
                      problem=p)
    model._slack_info = slack_info
    return model


def qubit_count(problem: PortfolioProblem) -> dict[str, int]:
    """Qubit requirement without building the (dense) QUBO matrix."""
    p = problem
    position = int(sum(n_bits(int(u)) for u in p.ub))
    indicator = p.n if p.cardinality_binding else 0
    slack = int(sum(n_bits(c.hi - c.lo) for c in p.sector_constraints))
    return {"position": position, "indicator": indicator, "slack": slack,
            "total": position + indicator + slack}
