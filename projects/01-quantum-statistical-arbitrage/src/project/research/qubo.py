"""Binary basket-selection problem and its QUBO / Ising encodings.

Decision variables: ``z_i in {0,1}`` (asset i is in the basket).

Base objective (minimise), a *surrogate* for "low residual variance and stable
cointegration" - cointegration itself is not a quadratic function of z and is
tested afterwards on the selected basket::

    f(z) = sum_i a_i z_i + sum_{i<j} C_ij z_i z_j

Constraints: ``sum_i z_i = K`` and at most ``max_per_sector`` assets per sector.

QUBO: ``E(z) = f(z) + A (sum_i z_i - K)^2 [+ A sum_{same sector} z_i z_j if max_per_sector == 1]``.
Cardinality is encoded exactly. A sector cap of 1 is encoded exactly; a cap >= 2
would need slack qubits, so it is enforced by feasibility checks and repair
instead (the soft same-sector cost inside C still discourages concentration).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations

import numpy as np

from .stats import PairStats

MAX_ENUM_BITS = 22


@dataclass
class BasketProblem:
    symbols: list[str]
    linear: np.ndarray            # a, shape (N,)
    quad: np.ndarray              # C, symmetric (N, N) with zero diagonal
    sectors: np.ndarray           # integer sector id per asset
    basket_size: int
    max_per_sector: int
    penalty_scale: float = 1.5
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.linear = np.asarray(self.linear, dtype=float)
        self.quad = np.asarray(self.quad, dtype=float)
        self.sectors = np.asarray(self.sectors, dtype=int)
        n = len(self.symbols)
        if self.linear.shape != (n,) or self.quad.shape != (n, n) or self.sectors.shape != (n,):
            raise ValueError("inconsistent problem dimensions")
        if not np.allclose(self.quad, self.quad.T) or np.any(np.diag(self.quad) != 0):
            raise ValueError("quad must be symmetric with a zero diagonal")
        if not np.all(np.isfinite(self.linear)) or not np.all(np.isfinite(self.quad)):
            raise ValueError("objective coefficients must be finite")
        if not (1 <= self.basket_size <= n):
            raise ValueError(f"basket_size {self.basket_size} is not in [1, {n}]")
        if not self.has_feasible_solution():
            raise ValueError("constraints are infeasible: sector caps cannot accommodate basket_size")

    # ------------------------------------------------------------------ basics
    @property
    def n(self) -> int:
        return len(self.symbols)

    def has_feasible_solution(self) -> bool:
        counts = np.bincount(self.sectors)
        return int(np.minimum(counts, self.max_per_sector).sum()) >= self.basket_size

    def base_objective(self, z: np.ndarray) -> float:
        z = np.asarray(z, dtype=float)
        return float(self.linear @ z + 0.5 * z @ self.quad @ z)

    def violations(self, z: np.ndarray) -> dict:
        z = np.asarray(z, dtype=int)
        counts = np.bincount(self.sectors[z == 1], minlength=self.sectors.max() + 1)
        return {
            "cardinality": int(z.sum() - self.basket_size),
            "sector_excess": int(np.maximum(counts - self.max_per_sector, 0).sum()),
        }

    def is_feasible(self, z: np.ndarray) -> bool:
        v = self.violations(z)
        return v["cardinality"] == 0 and v["sector_excess"] == 0

    def decode(self, z: np.ndarray) -> list[str]:
        return [s for s, bit in zip(self.symbols, z) if bit]

    def encode(self, basket: list[str]) -> np.ndarray:
        idx = {s: i for i, s in enumerate(self.symbols)}
        z = np.zeros(self.n, dtype=int)
        for s in basket:
            z[idx[s]] = 1
        return z

    # ---------------------------------------------------------------- encodings
    def penalty(self) -> float:
        """A bound on the objective change from adding/removing one asset near a K-basket."""
        k = max(self.basket_size - 1, 1)
        top = np.sort(np.abs(self.quad), axis=1)[:, -k:].sum(axis=1)
        return float(self.penalty_scale * max(np.max(np.abs(self.linear) + top), 1e-9))

    def to_qubo(self) -> tuple[np.ndarray, float]:
        """Upper-triangular Q and offset with ``E(z) = z^T Q z + offset``."""
        n, A, K = self.n, self.penalty(), self.basket_size
        Q = np.triu(self.quad, 1).copy()
        Q[np.diag_indices(n)] = self.linear
        # A (sum z - K)^2 = A[(1 - 2K) sum z_i + 2 sum_{i<j} z_i z_j] + A K^2
        Q[np.diag_indices(n)] += A * (1 - 2 * K)
        Q += 2 * A * np.triu(np.ones((n, n)), 1)
        if self.max_per_sector == 1:
            same = np.triu(self.sectors[:, None] == self.sectors[None, :], 1)
            Q += A * same
        return Q, float(A * K * K)

    def qubo_energy(self, z: np.ndarray) -> float:
        Q, offset = self.to_qubo()
        z = np.asarray(z, dtype=float)
        return float(z @ Q @ z + offset)

    def to_ising(self) -> tuple[np.ndarray, np.ndarray, float]:
        """(h, J, offset) with ``E = sum h_i s_i + sum_{i<j} J_ij s_i s_j + offset``, z = (1 - s)/2."""
        Q, offset = self.to_qubo()
        n = self.n
        J = np.triu(Q, 1) / 4.0
        diag = np.diag(Q)
        off = np.triu(Q, 1)
        row = off.sum(axis=1) + off.sum(axis=0)
        h = -diag / 2.0 - row / 4.0
        const = offset + diag.sum() / 2.0 + off.sum() / 4.0
        return h, J, float(const)

    def all_bitstrings(self) -> np.ndarray:
        """(2^N, N) matrix; row k holds bit q of k in column q (little-endian, as in Qiskit)."""
        if self.n > MAX_ENUM_BITS:
            raise ValueError(f"refusing to enumerate 2^{self.n} states")
        k = np.arange(1 << self.n, dtype=np.int64)
        return ((k[:, None] >> np.arange(self.n)) & 1).astype(np.int8)

    def all_qubo_energies(self) -> np.ndarray:
        Q, offset = self.to_qubo()
        Z = self.all_bitstrings()
        out = np.empty(len(Z))
        step = 1 << 16
        for s in range(0, len(Z), step):
            blk = Z[s : s + step].astype(float)
            out[s : s + step] = np.einsum("ki,ij,kj->k", blk, Q, blk) + offset
        return out

    # --------------------------------------------------------- exact / repair
    def feasible_baskets(self):
        for combo in combinations(range(self.n), self.basket_size):
            counts = np.bincount(self.sectors[list(combo)])
            if counts.max() <= self.max_per_sector:
                yield combo

    def brute_force(self) -> tuple[np.ndarray, float, int]:
        """Exact optimum by enumerating feasible baskets: (z, objective, n_evaluated)."""
        best, best_val, count = None, np.inf, 0
        for combo in self.feasible_baskets():
            idx = list(combo)
            val = self.linear[idx].sum() + np.triu(self.quad[np.ix_(idx, idx)], 1).sum()
            count += 1
            if val < best_val - 1e-12:
                best, best_val = combo, val
        z = np.zeros(self.n, dtype=int)
        z[list(best)] = 1
        return z, float(best_val), count

    def repair(self, z: np.ndarray) -> np.ndarray:
        """Deterministic greedy repair to the nearest good feasible basket.

        Drops assets (sector excess first, then cardinality excess) choosing the
        removal that lowers the objective most, then adds the best feasible
        assets until the basket has exactly K members.
        """
        z = np.asarray(z, dtype=int).copy()
        cap = self.max_per_sector

        def marginal(i: int) -> float:
            return float(self.linear[i] + self.quad[i] @ z - self.quad[i, i] * z[i])

        while True:
            counts = np.bincount(self.sectors[z == 1], minlength=self.sectors.max() + 1)
            over = [i for i in np.flatnonzero(z) if counts[self.sectors[i]] > cap]
            if not over:
                break
            z[max(over, key=marginal)] = 0
        while z.sum() > self.basket_size:
            z[max(np.flatnonzero(z), key=marginal)] = 0
        while z.sum() < self.basket_size:
            counts = np.bincount(self.sectors[z == 1], minlength=self.sectors.max() + 1)
            cands = [i for i in np.flatnonzero(z == 0) if counts[self.sectors[i]] < cap]
            if not cands:
                # Dead end (rare): free a slot by dropping the worst member and retry.
                z[max(np.flatnonzero(z), key=marginal)] = 0
                continue
            z[min(cands, key=marginal)] = 1
        return z


def build_problem(
    symbols: list[str],
    stats: PairStats,
    betas: np.ndarray,
    sectors: np.ndarray,
    spread_bps: np.ndarray,
    selection_cfg: dict,
    hl_cap: float = 60.0,
) -> BasketProblem:
    """Assemble objective coefficients from train-window statistics."""
    w = selection_cfg["weights"]
    n = len(symbols)
    hl_norm = np.minimum(np.where(np.isfinite(stats.half_life), stats.half_life, hl_cap) / hl_cap, 1.0)
    beta_gap = (betas[:, None] - betas[None, :]) ** 2
    same_sector = (sectors[:, None] == sectors[None, :]).astype(float)
    C = (
        -w["coint"] * stats.strength
        - w["stability"] * stats.stability
        + w["resid_var"] * np.minimum(stats.resid_var, 1.0)
        + w["half_life"] * hl_norm
        + w["beta_dispersion"] * np.minimum(beta_gap, 1.0)
        + w["sector"] * same_sector
    )
    C = (C + C.T) / 2.0
    np.fill_diagonal(C, 0.0)
    scale = float(np.max(spread_bps)) if np.max(spread_bps) > 0 else 1.0
    a = w["cost"] * np.asarray(spread_bps, dtype=float) / scale
    return BasketProblem(
        symbols=list(symbols),
        linear=a,
        quad=C,
        sectors=np.asarray(sectors, dtype=int),
        basket_size=min(selection_cfg["basket_size"], n),
        max_per_sector=selection_cfg["max_per_sector"],
        penalty_scale=selection_cfg["penalty_scale"],
    )
