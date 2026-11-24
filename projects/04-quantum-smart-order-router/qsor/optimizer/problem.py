"""The route-allocation problem shared by every solver.

Decision: integer units ``x[v]`` routed to each venue (one unit = ``unit``
lots).  Objective, in tick-lots (ticks per share x lots):

    sum_v table[v][x_v]  +  sum_{a<b} cross[a][b] * x_a * x_b

``table[v][k]`` is the cost of sending k units to venue v: slippage from
walking the displayed ladder relative to the best far-touch price, taker fees,
a quadratic impact estimate, an expected non-fill penalty, and a fixed
per-order charge when k > 0.  The fixed charge makes tables non-convex, and
``cross`` (expected quote fade between venue pairs) makes the objective
non-separable, so greedy allocation is not optimal in general.

Constraints: ``0 <= x_v <= n_units[v]`` (displayed liquidity, participation
and venue caps) and ``sum_v x_v == target_units`` (conservation).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple


@dataclass(frozen=True)
class CostModel:
    """Router-side cost estimates per venue (all in ticks-per-share units)."""
    fee: Tuple[float, ...]         # taker fee per lot
    fixed: Tuple[float, ...]       # per-order charge
    impact: Tuple[float, ...]      # quadratic impact coefficient per lot^2
    fill_prob: Tuple[float, ...]   # calibrated IOC fill ratio
    fade: Tuple[Tuple[float, ...], ...]  # fade[a][b]: lots lost at b per lot filled first at a
    nonfill_penalty: float         # cost per lot expected to go unfilled


@dataclass(frozen=True)
class AllocationProblem:
    side: int
    target_lots: int
    unit: int
    n_units: Tuple[int, ...]
    target_units: int
    tables: Tuple[Tuple[float, ...], ...]
    cross: Tuple[Tuple[float, ...], ...]
    best_px: Tuple[Optional[int], ...]
    depth: Tuple[int, ...]

    @property
    def V(self) -> int:
        return len(self.n_units)

    def cost(self, x: Sequence[int]) -> float:
        c = 0.0
        for v, k in enumerate(x):
            c += self.tables[v][k]
        V = len(x)
        for a in range(V):
            if x[a]:
                row = self.cross[a]
                for b in range(a + 1, V):
                    if x[b]:
                        c += row[b] * x[a] * x[b]
        return c

    def is_feasible(self, x: Sequence[int]) -> bool:
        return (len(x) == self.V and all(0 <= k <= n for k, n in zip(x, self.n_units))
                and sum(x) == self.target_units)

    def lots(self, x: Sequence[int]) -> Tuple[int, ...]:
        return tuple(k * self.unit for k in x)


def build_problem(side: int, target_lots: int, ladders, caps, ref_px: int,
                  model: CostModel, max_units: Optional[int] = None) -> AllocationProblem:
    """Build a problem from per-venue ladders [(price, lots), ...] within the limit.

    ``caps[v]`` is the most lots venue v may receive. ``max_units`` bounds the
    number of allocation units (granularity); None means one lot per unit.
    """
    V = len(caps)
    unit = 1 if not max_units else max(1, -(-target_lots // max_units))
    want = -(-target_lots // unit)
    n_units, tables, depth, best = [], [], [], []
    lam = model.nonfill_penalty
    for v in range(V):
        lad = ladders[v] or ()
        depth.append(sum(s for _, s in lad))
        best.append(lad[0][0] if lad else None)
        n = min(caps[v] // unit, want) if lad else 0
        n_units.append(n)
        per_lot = model.fee[v] + lam * (1.0 - model.fill_prob[v])
        row = [0.0]
        for k in range(1, n + 1):
            x = k * unit
            left, walk = x, 0.0
            for p, s in lad:
                take = min(s, left)
                walk += (p - ref_px) * side * take
                left -= take
                if left == 0:
                    break
            if left:  # beyond displayed size inside the limit: price at the limit edge
                walk += (lad[-1][0] - ref_px) * side * left
            row.append(walk + per_lot * x + model.impact[v] * x * x + model.fixed[v])
        tables.append(tuple(row))
    cross = [[0.0] * V for _ in range(V)]
    for a in range(V):
        for b in range(a + 1, V):
            if n_units[a] and n_units[b]:
                k = lam * (model.fade[a][b] / max(1, depth[b]) + model.fade[b][a] / max(1, depth[a]))
                cross[a][b] = cross[b][a] = k * unit * unit
    return AllocationProblem(
        side=side, target_lots=target_lots, unit=unit, n_units=tuple(n_units),
        target_units=min(want, sum(n_units)), tables=tuple(tables),
        cross=tuple(tuple(r) for r in cross), best_px=tuple(best), depth=tuple(depth))


def repair(problem: AllocationProblem, x: Sequence[int]) -> Tuple[int, ...]:
    """Deterministically project x onto the feasible set with greedy unit moves."""
    x = [min(max(int(k), 0), n) for k, n in zip(x, problem.n_units)]
    total, target = sum(x), problem.target_units
    while total != target:
        step = 1 if total < target else -1
        base, best_v, best_d = problem.cost(x), -1, 0.0
        for v in range(problem.V):
            k = x[v] + step
            if 0 <= k <= problem.n_units[v]:
                x[v] = k
                d = problem.cost(x) - base
                x[v] = k - step
                if best_v < 0 or d < best_d - 1e-12:
                    best_v, best_d = v, d
        x[best_v] += step
        total += step
    return tuple(x)
