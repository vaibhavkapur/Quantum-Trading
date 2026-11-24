"""Faithful binary sleeve selection with an exact capacity penalty."""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np


@dataclass
class QUBO:
    # Upper triangular convention: E(x) = offset + x.T @ matrix @ x.
    matrix: np.ndarray
    offset: float
    base: np.ndarray
    capacity: int
    penalty: float

    @property
    def n_strategies(self):
        return self.base.shape[0]

    @property
    def n_bits(self):
        return self.matrix.shape[0]

    def bits(self, value):
        x = np.asarray(value, dtype=float)
        if x.shape != (self.n_bits,) or not np.isin(x, [0, 1]).all():
            raise ValueError(f"Expected {self.n_bits} binary values")
        return x

    def encode(self, selected):
        z = np.asarray(selected, dtype=int)
        if z.shape != (self.n_strategies,) or not np.array_equal(z, selected) or not np.isin(z, [0, 1]).all():
            raise ValueError("Invalid binary portfolio")
        slack = max(0, self.capacity - int(z.sum()))
        return np.r_[z, [(slack >> i) & 1 for i in range(self.n_bits - self.n_strategies)]]

    def feasible(self, value):
        x = self.bits(value)
        z, slack = x[:self.n_strategies], x[self.n_strategies:]
        return bool(z.sum() <= self.capacity and z.sum() + slack @ (2 ** np.arange(len(slack))) == self.capacity)

    def energy(self, value):
        x = self.bits(value)
        return float(self.offset + x @ self.matrix @ x)

    def objective(self, selected):
        z = np.asarray(selected, dtype=float)
        if z.shape != (self.n_strategies,) or not np.isin(z, [0, 1]).all():
            raise ValueError("Invalid portfolio")
        return float(z @ self.base @ z)

    def ising(self):
        diagonal = np.diag(self.matrix)
        pairs = np.triu(self.matrix, 1) / 4
        linear = -diagonal / 2 - pairs.sum(axis=0) - pairs.sum(axis=1)
        offset = self.offset + diagonal.sum() / 2 + pairs.sum()
        return float(offset), linear, pairs


def build_qubo(net_returns, capacity, risk_aversion, cardinality_cost, annualization=252):
    returns = np.asarray(net_returns, dtype=float)
    if returns.ndim != 2 or returns.shape[0] < 2 or not np.isfinite(returns).all():
        raise ValueError("Need at least two finite training observations per strategy")
    n = returns.shape[1]
    if type(capacity) is not int or not 1 <= capacity <= n:
        raise ValueError("Capacity must lie between one and the number of strategies")
    if any(not np.isfinite(x) or x < 0 for x in (risk_aversion, cardinality_cost)) or annualization <= 0:
        raise ValueError("Invalid objective parameters")
    mean = returns.mean(axis=0) * annualization
    covariance = np.atleast_2d(np.cov(returns, rowvar=False, ddof=1)) * annualization
    base = np.diag(-mean + cardinality_cost + risk_aversion * np.diag(covariance))
    base += np.triu(2 * risk_aversion * covariance, 1)
    # Any difference between two base energies is <= 2*sum(abs(base)).
    # A violation is an integer with square >=1, so it cannot beat feasible cash.
    penalty = 2 * float(np.abs(base).sum()) + 1.0
    slack_bits = capacity.bit_length()
    weights = np.r_[np.ones(n), 2 ** np.arange(slack_bits)]
    matrix = np.zeros((n + slack_bits, n + slack_bits))
    matrix[:n, :n] = base
    matrix += np.diag(penalty * (weights ** 2 - 2 * capacity * weights))
    matrix += np.triu(2 * penalty * np.outer(weights, weights), 1)
    return QUBO(matrix, penalty * capacity ** 2, base, capacity, penalty)


def feasible_portfolios(problem):
    for size in range(problem.capacity + 1):
        for indices in combinations(range(problem.n_strategies), size):
            z = np.zeros(problem.n_strategies, dtype=int)
            z[list(indices)] = 1
            yield z


def exact_reference(problem):
    """An explicitly separate verifier, never used to initialize a search solver."""
    candidates = list(feasible_portfolios(problem))
    values = [problem.objective(z) for z in candidates]
    best = int(np.argmin(values))
    return {"selected": candidates[best].tolist(), "objective": values[best],
            "evaluations": len(candidates), "kind": "exhaustive feasible portfolios"}


def milp_reference(problem):
    """Exact binary quadratic linearization, solved with SciPy's HiGHS MILP."""
    from scipy.optimize import Bounds, LinearConstraint, milp
    n = problem.n_strategies
    pairs = list(combinations(range(n), 2))
    c = np.r_[np.diag(problem.base), [problem.base[i, j] for i, j in pairs]]
    rows, lower, upper = [], [], []
    capacity = np.zeros(len(c))
    capacity[:n] = 1
    rows.append(capacity)
    lower.append(-np.inf)
    upper.append(problem.capacity)
    for k, (i, j) in enumerate(pairs, n):
        # w_ij <= z_i; w_ij <= z_j; w_ij >= z_i + z_j - 1.
        for coefficients, bound in [({k: 1, i: -1}, 0), ({k: 1, j: -1}, 0),
                                    ({i: 1, j: 1, k: -1}, 1)]:
            row = np.zeros(len(c))
            for index, coefficient in coefficients.items():
                row[index] = coefficient
            rows.append(row)
            lower.append(-np.inf)
            upper.append(bound)
    result = milp(c, integrality=np.r_[np.ones(n), np.zeros(len(pairs))],
                  bounds=Bounds(0, 1), constraints=LinearConstraint(np.array(rows), lower, upper),
                  options={"time_limit": 30.0, "mip_rel_gap": 0.0})
    if not result.success or result.x is None:
        raise RuntimeError(f"MILP reference did not certify optimality: {result.message}")
    z = np.rint(result.x[:n]).astype(int)
    return {"selected": z.tolist(), "objective": problem.objective(z),
            "kind": "HiGHS MILP with exact pair-product linearization",
            "mip_gap": float(result.mip_gap), "nodes": int(result.mip_node_count)}

