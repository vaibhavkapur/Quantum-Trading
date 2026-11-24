"""One-hot order encoding. Global cash/exposure inequalities stay in the risk gate.

The QUBO is explicitly a relaxation of those inequalities, not a claim that a
penalty magically enforces a budget. All solver outputs pass the same repair.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .model import Problem, ResourceLimitError, fraction_cents


@dataclass
class Encoding:
    problem: Problem
    blocks: list[list[int]]
    matrix: np.ndarray
    residual: np.ndarray
    cost: np.ndarray
    linear: np.ndarray
    quadratic: np.ndarray
    offset: float
    constraints: np.ndarray
    lower: np.ndarray
    upper: np.ndarray

    @property
    def size(self) -> int:
        return len(self.linear)

    def objective(self, bits: np.ndarray) -> float:
        return float(self.offset + self.linear @ bits + bits @ self.quadratic @ bits)

    def bits(self, indices: tuple[int, ...]) -> np.ndarray:
        bits = np.zeros(self.size)
        for block, index in zip(self.blocks, indices):
            bits[block[index]] = 1
        return bits

    def decode(self, bits: np.ndarray) -> tuple[int, ...]:
        # Invalid one-hot samples have a deterministic nearest-choice projection.
        return tuple(int(np.argmax(bits[block])) for block in self.blocks)


def encode(problem: Problem) -> Encoding:
    p = problem.portfolio
    blocks, k = [], 0
    for choices in problem.choices:
        blocks.append(list(range(k, k + len(choices))))
        k += len(choices)
    if k > 2048:
        raise ResourceLimitError("encoding supports at most 2048 order-choice variables")
    matrix = np.zeros((len(blocks) + 1, k))
    residual = np.array([-a.target_weight for a in p.assets] + [p.cash_cents / p.nav_cents - p.target_cash_weight])
    cost = np.zeros(k)
    cash = np.zeros(k)
    groups = {g: np.zeros(k) for g in p.group_caps}
    for i, (asset, block, choices) in enumerate(zip(p.assets, blocks, problem.choices)):
        for j, c in zip(block, choices):
            matrix[i, j] = c.value_cents / p.nav_cents
            matrix[-1, j] = c.cash_delta_cents / p.nav_cents
            cost[j] = (p.turnover_weight * c.notional_cents + p.cost_weight * (c.fee_cents + c.tax_cents)) / p.nav_cents
            cash[j] = c.cash_delta_cents
            if asset.group in groups:
                groups[asset.group][j] = c.value_cents
    q = matrix.T @ matrix
    linear = np.diag(q) + 2 * residual @ matrix + cost
    quadratic = np.triu(q, 1) * 2
    rows, lower, upper = [], [], []
    for block in blocks:
        row = np.zeros(k)
        row[block] = 1
        rows.append(row)
        lower.append(1)
        upper.append(1)
    rows.append(cash)
    lower.append(p.reserve_cents - p.cash_cents)
    upper.append(np.inf)
    for group, row in groups.items():
        rows.append(row)
        lower.append(-np.inf)
        upper.append(fraction_cents(p.nav_cents, p.group_caps[group]))
    return Encoding(problem, blocks, matrix, residual, cost, linear, quadratic, float(residual @ residual),
                    np.array(rows), np.array(lower), np.array(upper))


@dataclass
class QUBO:
    linear: np.ndarray
    quadratic: np.ndarray
    offset: float
    penalty: float

    def energy(self, bits: np.ndarray) -> float:
        return float(self.offset + self.linear @ bits + bits @ self.quadratic @ bits)

    def ising(self) -> tuple[float, np.ndarray, np.ndarray]:
        # x_i = (1 - Z_i) / 2. quadratic is strictly upper triangular.
        coupling = self.quadratic / 4
        fields = -self.linear / 2 - (self.quadratic.sum(axis=0) + self.quadratic.sum(axis=1)) / 4
        offset = self.offset + self.linear.sum() / 2 + self.quadratic.sum() / 4
        return float(offset), fields, coupling


def make_qubo(encoding: Encoding) -> QUBO:
    linear, quadratic = encoding.linear.copy(), encoding.quadratic.copy()
    # Any two binary assignments differ in raw objective by at most this L1
    # coefficient bound. One violated block incurs at least one penalty unit.
    penalty = 1 + float(np.abs(linear).sum() + np.abs(quadratic).sum())
    offset = encoding.offset
    for block in encoding.blocks:
        offset += penalty
        for position, i in enumerate(block):
            linear[i] -= penalty
            for j in block[position + 1:]:
                quadratic[i, j] += 2 * penalty
    return QUBO(linear, quadratic, offset, penalty)

