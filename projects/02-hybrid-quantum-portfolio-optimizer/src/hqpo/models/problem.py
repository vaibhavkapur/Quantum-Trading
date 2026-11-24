"""Discrete (integer-lot) portfolio problem shared by every solver.

All solvers optimise the *same* object, so objectives and constraints are identical by
construction (fair-comparison protocol, item 2).

Decision variables are integer lot counts ``z`` (length ``m``):

* ``long_only``      : ``m = n``,  ``w = lot * z``,        ``sum(z) = L``
* ``market_neutral`` : ``m = 2n``, ``z = (p, q)``, ``w = lot * (p - q)``,
  ``sum(p) = L`` and ``sum(q) = L`` (dollar neutral, gross exposure ``2 * leg_exposure``),
  with ``p_i * q_i = 0`` so an asset is never held long and short at once.

Objective (minimised)::

    f(w) = lam * w' Sigma w  -  mu' w  +  tau * ||w - w_prev||^2

The turnover term is the *quadratic* trading-cost proxy.  It is used because it is
QUBO-native (no auxiliary qubits) and therefore lets every solver, classical or quantum,
minimise literally the same function.  Realised L1 turnover and linear transaction costs
are measured out-of-sample by the backtester.

In lot space the objective is the quadratic form ``z' A z + b' z + c0``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property

import numpy as np

EPS = 1e-9
MODES = ("long_only", "market_neutral")


@dataclass(frozen=True)
class Constraints:
    mode: str = "long_only"
    lots: int = 8                      # L: lots per leg
    max_weight: float = 1.0            # per-name |weight| cap
    cardinality: int | None = None     # max number of names held
    sector_cap: float | None = None    # max |net weight| per sector
    leg_exposure: float | None = None  # weight per leg; default 1.0 long-only, 0.5 market-neutral

    def __post_init__(self):
        if self.mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {self.mode!r}")
        if self.lots < 1:
            raise ValueError("lots must be >= 1")
        if not 0 < self.max_weight:
            raise ValueError("max_weight must be positive")
        if self.cardinality is not None and self.cardinality < 1:
            raise ValueError("cardinality must be >= 1")

    @property
    def leg(self) -> float:
        if self.leg_exposure is not None:
            return float(self.leg_exposure)
        return 1.0 if self.mode == "long_only" else 0.5

    @property
    def lot_size(self) -> float:
        return self.leg / self.lots


@dataclass(frozen=True)
class LinearConstraint:
    """``lo <= a . z <= hi`` with integer data (``lo == hi`` for equalities)."""

    name: str
    a: np.ndarray
    lo: int
    hi: int

    @property
    def is_equality(self) -> bool:
        return self.lo == self.hi


@dataclass
class PortfolioProblem:
    sigma: np.ndarray
    mu: np.ndarray
    constraints: Constraints
    risk_aversion: float = 1.0
    turnover_penalty: float = 0.0
    w_prev: np.ndarray | None = None
    sectors: list | None = None
    assets: list[str] | None = None
    name: str = "problem"
    meta: dict = field(default_factory=dict)

    def __post_init__(self):
        self.sigma = np.asarray(self.sigma, dtype=float)
        self.mu = np.asarray(self.mu, dtype=float)
        n = self.mu.shape[0]
        if self.sigma.shape != (n, n):
            raise ValueError(f"sigma shape {self.sigma.shape} does not match mu length {n}")
        if not np.all(np.isfinite(self.sigma)) or not np.all(np.isfinite(self.mu)):
            raise ValueError("sigma and mu must be finite")
        if not np.allclose(self.sigma, self.sigma.T, atol=1e-10):
            raise ValueError("sigma must be symmetric")
        if np.linalg.eigvalsh(self.sigma).min() < -1e-8:
            raise ValueError("sigma must be positive semidefinite")
        self.w_prev = np.zeros(n) if self.w_prev is None else np.asarray(self.w_prev, dtype=float)
        if self.w_prev.shape != (n,):
            raise ValueError("w_prev has wrong length")
        if self.sectors is None:
            self.sectors = [0] * n
        if len(self.sectors) != n:
            raise ValueError("sectors has wrong length")
        if self.assets is None:
            self.assets = [f"A{i:03d}" for i in range(n)]
        if self.risk_aversion < 0 or self.turnover_penalty < 0:
            raise ValueError("risk_aversion and turnover_penalty must be non-negative")
        self._check_trivially_infeasible()

    # ------------------------------------------------------------------ structure
    @property
    def n(self) -> int:
        return self.mu.shape[0]

    @property
    def market_neutral(self) -> bool:
        return self.constraints.mode == "market_neutral"

    @property
    def m(self) -> int:
        return 2 * self.n if self.market_neutral else self.n

    @property
    def lots(self) -> int:
        return self.constraints.lots

    @property
    def lot_size(self) -> float:
        return self.constraints.lot_size

    @cached_property
    def ub(self) -> np.ndarray:
        """Per-variable upper bound in lots (position limit, enforced by the encoding)."""
        k = int(np.floor(self.constraints.max_weight / self.lot_size + EPS))
        return np.full(self.m, min(k, self.lots), dtype=int)

    @cached_property
    def S(self) -> np.ndarray:
        """Sign matrix: ``w = lot * S z``."""
        eye = np.eye(self.n)
        return np.hstack([eye, -eye]) if self.market_neutral else eye

    @cached_property
    def groups(self) -> list[list[int]]:
        """Variable indices belonging to each asset (for cardinality)."""
        if self.market_neutral:
            return [[i, self.n + i] for i in range(self.n)]
        return [[i] for i in range(self.n)]

    @cached_property
    def exclusive_pairs(self) -> list[tuple[int, int]]:
        return [(i, self.n + i) for i in range(self.n)] if self.market_neutral else []

    @cached_property
    def sector_ids(self) -> np.ndarray:
        labels = sorted(set(self.sectors), key=str)
        lookup = {s: k for k, s in enumerate(labels)}
        return np.array([lookup[s] for s in self.sectors], dtype=int)

    @cached_property
    def budget_constraints(self) -> list[LinearConstraint]:
        n, L = self.n, self.lots
        if self.market_neutral:
            a_long = np.concatenate([np.ones(n, int), np.zeros(n, int)])
            a_short = np.concatenate([np.zeros(n, int), np.ones(n, int)])
            return [LinearConstraint("budget_long", a_long, L, L),
                    LinearConstraint("budget_short", a_short, L, L)]
        return [LinearConstraint("budget", np.ones(n, int), L, L)]

    @cached_property
    def sector_constraints(self) -> list[LinearConstraint]:
        """Sector caps that can actually bind (non-binding ones are presolved away)."""
        cap = self.constraints.sector_cap
        if cap is None:
            return []
        u = int(np.floor(cap / self.lot_size + EPS))
        out = []
        for k in range(self.sector_ids.max() + 1):
            member = (self.sector_ids == k).astype(int)
            if self.market_neutral:
                a, lo, hi = np.concatenate([member, -member]), -u, u
            else:
                a, lo, hi = member, 0, u
            amax = int(np.sum(np.where(a > 0, a * self.ub, 0)))
            amin = int(np.sum(np.where(a < 0, a * self.ub, 0)))
            if not self.market_neutral:
                amax = min(amax, self.lots)
            if amax <= hi and amin >= lo:
                continue
            out.append(LinearConstraint(f"sector_{k}", a, max(lo, amin), min(hi, amax)))
        return out

    @property
    def max_holdable(self) -> int:
        legs = 2 if self.market_neutral else 1
        return min(self.n, legs * self.lots)

    @property
    def cardinality_binding(self) -> bool:
        c = self.constraints.cardinality
        return c is not None and c < self.max_holdable

    def _check_trivially_infeasible(self):
        k = int(self.ub[0])
        if k < 1:
            raise ValueError("max_weight is smaller than one lot: no asset can be held")
        if self.n * k < self.lots:
            raise ValueError("infeasible: position limits cannot absorb the budget")
        c = self.constraints.cardinality
        if c is not None:
            legs = 2 if self.market_neutral else 1
            if c * k < legs * self.lots:
                raise ValueError("infeasible: cardinality * max position < budget")

    # ------------------------------------------------------------------ objective
    @cached_property
    def _M(self) -> np.ndarray:
        return self.risk_aversion * self.sigma + self.turnover_penalty * np.eye(self.n)

    @cached_property
    def A(self) -> np.ndarray:
        return self.lot_size ** 2 * self.S.T @ self._M @ self.S

    @cached_property
    def b(self) -> np.ndarray:
        return self.lot_size * self.S.T @ (-self.mu - 2.0 * self.turnover_penalty * self.w_prev)

    @cached_property
    def c0(self) -> float:
        return float(self.turnover_penalty * self.w_prev @ self.w_prev)

    def weights(self, z) -> np.ndarray:
        return self.lot_size * (np.asarray(z, dtype=float) @ self.S.T)

    def objective_w(self, w) -> float:
        w = np.asarray(w, dtype=float)
        d = w - self.w_prev
        return float(self.risk_aversion * w @ self.sigma @ w - self.mu @ w
                     + self.turnover_penalty * d @ d)

    def objective(self, z) -> float:
        z = np.asarray(z, dtype=float)
        return float(z @ self.A @ z + self.b @ z + self.c0)

    def objective_batch(self, Z) -> np.ndarray:
        Z = np.asarray(Z, dtype=float)
        return np.einsum("ki,ij,kj->k", Z, self.A, Z) + Z @ self.b + self.c0

    # ------------------------------------------------------------------ feasibility
    def held_mask(self, z) -> np.ndarray:
        z = np.asarray(z)
        if self.market_neutral:
            return (z[: self.n] + z[self.n:]) > 0
        return z > 0

    def violations(self, z) -> dict[str, int]:
        """Integer violation amount per constraint family (all zero <=> feasible)."""
        z = np.asarray(z, dtype=int)
        out = {
            "bounds": int(np.sum(np.maximum(z - self.ub, 0)) + np.sum(np.maximum(-z, 0))),
            "budget": int(sum(abs(int(c.a @ z) - c.lo) for c in self.budget_constraints)),
            "sector": int(sum(max(int(c.a @ z) - c.hi, 0) + max(c.lo - int(c.a @ z), 0)
                              for c in self.sector_constraints)),
            "cardinality": 0,
            "overlap": 0,
        }
        if self.constraints.cardinality is not None:
            out["cardinality"] = max(int(self.held_mask(z).sum()) - self.constraints.cardinality, 0)
        if self.market_neutral:
            out["overlap"] = int(np.minimum(z[: self.n], z[self.n:]).sum())
        return out

    def ineq_violation(self, z) -> int:
        v = self.violations(z)
        return v["bounds"] + v["sector"] + v["cardinality"] + v["overlap"]

    def total_violation(self, z) -> int:
        return int(sum(self.violations(z).values()))

    def is_feasible(self, z) -> bool:
        return self.total_violation(z) == 0

    def total_violation_batch(self, Z) -> np.ndarray:
        Z = np.asarray(Z, dtype=np.int64)
        v = np.maximum(Z - self.ub, 0).sum(1) + np.maximum(-Z, 0).sum(1)
        for c in self.budget_constraints:
            v = v + np.abs(Z @ c.a - c.lo)
        for c in self.sector_constraints:
            s = Z @ c.a
            v = v + np.maximum(s - c.hi, 0) + np.maximum(c.lo - s, 0)
        if self.market_neutral:
            p, q = Z[:, : self.n], Z[:, self.n:]
            v = v + np.minimum(p, q).sum(1)
            held = ((p + q) > 0).sum(1)
        else:
            held = (Z > 0).sum(1)
        if self.constraints.cardinality is not None:
            v = v + np.maximum(held - self.constraints.cardinality, 0)
        return v

    # ------------------------------------------------------------------ reporting
    def describe(self, z) -> dict:
        w = self.weights(z)
        return {
            "risk": float(np.sqrt(max(w @ self.sigma @ w, 0.0))),
            "expected_return": float(self.mu @ w),
            "turnover_l1": float(np.abs(w - self.w_prev).sum()),
            "n_held": int(self.held_mask(z).sum()),
            "gross": float(np.abs(w).sum()),
            "net": float(w.sum()),
        }
