"""Feasibility repair and local search in lot space.

``repair`` maps any integer vector (e.g. a decoded QUBO sample that violates a penalty) to a
feasible portfolio with a deterministic greedy procedure:

1. net out long/short overlap and clip to position limits;
2. cardinality: keep the ``C`` largest positions, liquidate the rest;
3. remove lots until every remaining inequality (sector band) holds;
4. add / remove single lots (best marginal objective first) until each budget is met,
   never re-introducing an inequality violation.

It can fail (returns ``ok=False``) when the greedy gets stuck; callers count that as an
infeasible outcome rather than hiding it.
"""
from __future__ import annotations

import numpy as np

from .problem import PortfolioProblem


def _marginal(problem: PortfolioProblem, g: np.ndarray, j: int, step: int) -> float:
    """Objective change of ``z_j += step`` given ``g = A z``."""
    return float(2.0 * step * g[j] + problem.A[j, j] + step * problem.b[j])


def repair(problem: PortfolioProblem, z) -> tuple[np.ndarray, bool]:
    p = problem
    z = np.clip(np.asarray(z, dtype=int).copy(), 0, p.ub)

    # 1. overlap
    if p.market_neutral:
        n = p.n
        common = np.minimum(z[:n], z[n:])
        z[:n] -= common
        z[n:] -= common

    # 2. cardinality
    c = p.constraints.cardinality
    if c is not None:
        size = np.array([z[g].sum() for g in p.groups])
        held = np.flatnonzero(size > 0)
        if held.size > c:
            order = held[np.argsort(-size[held], kind="stable")]
            for i in order[c:]:
                z[p.groups[i]] = 0

    g = p.A @ z

    def apply(j: int, step: int):
        nonlocal g
        z[j] += step
        g = g + step * p.A[:, j]

    # 3. inequality violations: remove lots
    guard = int(z.sum()) + 1
    while guard > 0:
        guard -= 1
        v0 = p.ineq_violation(z)
        if v0 == 0:
            break
        best = None
        for j in np.flatnonzero(z > 0):
            z[j] -= 1
            v = p.ineq_violation(z)
            z[j] += 1
            if v < v0:
                key = (v, _marginal(p, g, j, -1))
                if best is None or key < best[0]:
                    best = (key, j)
        if best is None:
            return z, False
        apply(best[1], -1)

    # 4. budgets: greedy first, then a few seeded randomised retries if the greedy gets stuck
    z_fixed, ok = _fill_budgets(p, z, None)
    if not ok:
        rng = np.random.default_rng(0)
        for _ in range(30):
            z_try, ok = _fill_budgets(p, z, rng)
            if ok:
                z_fixed = z_try
                break
    return z_fixed, ok and p.is_feasible(z_fixed)


def _fill_budgets(p: PortfolioProblem, z0: np.ndarray, rng) -> tuple[np.ndarray, bool]:
    """Meet every budget equality with single-lot moves that keep the inequalities satisfied.

    ``rng=None`` is the deterministic best-marginal greedy; with an ``rng`` candidates are
    tried in random order.  When a leg cannot add a lot, one blocking position on another
    leg is liquidated (that leg is refilled on the next round).
    """
    z = z0.copy()
    g = p.A @ z
    card = p.constraints.cardinality
    kmax = int(p.ub.max())
    budgets = p.budget_constraints
    members = [np.flatnonzero(c.a != 0) for c in budgets]

    def apply(j: int, step: int):
        nonlocal g
        z[j] += step
        g = g + step * p.A[:, j]

    def names_reserved(t: int) -> int:
        """Names the other legs still need to open (keeps room under the cardinality cap)."""
        need = 0
        for u, other in enumerate(budgets):
            if u == t:
                continue
            deficit = other.lo - int(z[members[u]].sum())
            spare = int(np.sum((p.ub[members[u]] - z[members[u]])[z[members[u]] > 0]))
            need += max(0, -(-max(deficit - spare, 0) // kmax))
        return need

    def step_leg(t: int, step: int) -> bool:
        cand = [j for j in members[t] if (z[j] < p.ub[j] if step > 0 else z[j] > 0)]
        if rng is None:
            cand.sort(key=lambda j: _marginal(p, g, j, step))
        else:
            rng.shuffle(cand)
        held = p.held_mask(z)
        room = None if card is None else card - int(held.sum()) - names_reserved(t)
        for j in cand:
            if step > 0 and room is not None and room <= 0 and not held[j % p.n]:
                continue
            z[j] += step
            ok = p.ineq_violation(z) == 0
            z[j] -= step
            if ok:
                apply(j, step)
                return True
        return False

    frees = 0
    for _ in range(6 * len(budgets) + 6):
        for t, cons in enumerate(budgets):                 # trim every surplus first
            while int(cons.a @ z) > cons.lo:
                if not step_leg(t, -1):
                    return z, False
        done = True
        for t, cons in enumerate(budgets):
            while int(cons.a @ z) < cons.lo:
                if step_leg(t, +1):
                    continue
                done = False
                # stuck: liquidate one position held on another leg, then start a new round
                other = [j for u in range(len(budgets)) if u != t for j in members[u] if z[j] > 0]
                if not other or frees > 4 * len(budgets) + p.lots:
                    return z, False
                j = other[int(rng.integers(len(other)))] if rng is not None else \
                    min(other, key=lambda j: (z[j], j))
                apply(j, -int(z[j]))
                frees += 1
                break
        if done and all(int(c.a @ z) == c.lo for c in budgets):
            return z, True
    return z, False


def local_search(problem: PortfolioProblem, z, max_passes: int = 50,
                 deadline: float | None = None) -> tuple[np.ndarray, int]:
    """First-improvement lot shifts ``i -> j`` (same leg) that keep feasibility.

    Returns the improved vector and the number of candidate moves evaluated.
    """
    import time

    p = problem
    z = np.asarray(z, dtype=int).copy()
    g = p.A @ z
    evals = 0
    legs = [np.flatnonzero(c.a != 0) for c in p.budget_constraints]
    for _ in range(max_passes):
        improved = False
        for members in legs:
            for i in members:
                if z[i] == 0:
                    continue
                for q in dict.fromkeys((1, int(z[i]))):      # one lot, then the whole position
                    if z[i] < q:
                        break
                    deltas = (q * (2.0 * (g[members] - g[i]) + p.b[members] - p.b[i])
                              + q * q * (p.A[members, members] + p.A[i, i] - 2.0 * p.A[i, members]))
                    evals += len(members)
                    moved = False
                    for k in np.argsort(deltas):
                        j = members[k]
                        if deltas[k] >= -1e-12:
                            break
                        if j == i or z[j] + q > p.ub[j]:
                            continue
                        z[i] -= q
                        z[j] += q
                        if p.ineq_violation(z) == 0:
                            g = g + q * (p.A[:, j] - p.A[:, i])
                            improved = moved = True
                            break
                        z[i] += q
                        z[j] -= q
                    if moved:
                        break
                if deadline is not None and time.perf_counter() > deadline:
                    return z, evals
        if not improved:
            break
    return z, evals
