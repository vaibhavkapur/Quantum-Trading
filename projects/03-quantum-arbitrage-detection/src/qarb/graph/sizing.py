"""Liquidity-aware sizing of conversion cycles.

Profit as a function of the starting amount is concave and piecewise linear
(a composition of concave increasing ladders), so the optimum is found exactly
by pushing size through the cycle while the marginal round-trip rate exceeds 1.

Cross-venue cycles are modelled as *pre-funded*: every leg trades simultaneously
against inventory already held at that leg's venue, so each leg's input is
capped by the balance of its source currency at its venue. Nothing is
transferred between venues inside a cycle.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass

from .build import Edge, EdgeKey, MarketGraph

Inventory = dict[tuple[str, str], float]  # (venue, currency) -> balance


@dataclass
class Plan:
    cycle: tuple[str, ...]  # currencies in trade order, starting at the funding currency
    edges: list[Edge]
    amounts_in: list[float]
    amounts_out: list[float]
    limit_rates: list[float]  # worst raw rate accepted per leg
    profit: float  # in cycle[0]
    profit_num: float  # in numeraire
    notional_num: float

    @property
    def venues(self) -> tuple[str, ...]:
        return tuple(e.venue for e in self.edges)


def optimal_size(
    ladders: list[list[tuple[float, float]]],
    fees: list[float],
    caps: list[float],
) -> tuple[list[float], list[float], list[float]]:
    """Exact profit-maximizing size for a cycle.

    ladders[k]: levels (rate, cap_in) of leg k; fees[k]: taker fee on leg k's
    output; caps[k]: maximum total input for leg k (inventory / capital).
    Returns per-leg (amounts_in, amounts_out, worst raw rate used). All zeros
    if the cycle is not profitable at any size.
    """
    L = len(ladders)
    used = [0.0] * L
    got = [0.0] * L
    worst = [0.0] * L
    if any(not lad for lad in ladders) or any(c <= 0 for c in caps):
        return used, got, worst
    idx = [0] * L
    rem = [lad[0][1] for lad in ladders]
    cap_rem = list(caps)
    while True:
        rates = [ladders[k][idx[k]][0] * (1 - fees[k]) for k in range(L)]
        if math.prod(rates) <= 1.0:
            break
        g = [1.0] * L  # leg-k input per unit of starting amount
        for k in range(1, L):
            g[k] = g[k - 1] * rates[k - 1]
        dq = min(min(rem[k], cap_rem[k]) / g[k] for k in range(L))
        stop = False
        for k in range(L):
            a = dq * g[k]
            used[k] += a
            got[k] += a * rates[k]
            worst[k] = ladders[k][idx[k]][0]
            rem[k] -= a
            cap_rem[k] -= a
            if math.isfinite(caps[k]) and cap_rem[k] <= 1e-12 * caps[k]:
                stop = True
            if rem[k] <= 1e-12 * ladders[k][idx[k]][1]:
                idx[k] += 1
                if idx[k] >= len(ladders[k]):
                    stop = True
                else:
                    rem[k] = ladders[k][idx[k]][1]
        if stop:
            break
    return used, got, worst


def rotate_to_funding(cycle: tuple[str, ...], numeraire: str, funding: list[str]) -> tuple[str, ...]:
    """Start the cycle at the preferred funding currency so profit accrues there."""
    for cur in [numeraire, *funding]:
        if cur in cycle:
            k = cycle.index(cur)
            return cycle[k:] + cycle[:k]
    return cycle


def size_edges(
    graph: MarketGraph,
    cycle: tuple[str, ...],
    edges: list[Edge],
    max_notional: float,
    inventory: Inventory | None = None,
    consumed: dict[EdgeKey, float] | None = None,
) -> Plan | None:
    consumed = consumed or {}
    ladders = [e.trimmed(consumed.get(e.key, 0.0)) for e in edges]
    fees = [e.fee for e in edges]
    caps = [math.inf if inventory is None else inventory.get((e.venue, e.src), 0.0) for e in edges]
    caps[0] = min(caps[0], max_notional / graph.ref[cycle[0]])
    used, got, worst = optimal_size(ladders, fees, caps)
    if used[0] <= 0:
        return None
    profit = got[-1] - used[0]
    ref0 = graph.ref[cycle[0]]
    return Plan(cycle, list(edges), used, got, worst, profit, profit * ref0, used[0] * ref0)


def best_plan(
    graph: MarketGraph,
    cycle: tuple[str, ...],
    max_notional: float,
    venue_top_k: int | None = 2,
    max_venues_per_cycle: int | None = None,
    inventory: Inventory | None = None,
    consumed: dict[EdgeKey, float] | None = None,
    funding: list[str] | None = None,
) -> Plan | None:
    """Best venue assignment and size for a currency cycle, or None if unprofitable.

    Per leg the `venue_top_k` venues with the best top-of-book rate are tried
    (None = all venues); every combination is sized exactly.
    """
    consumed = consumed or {}
    cycle = rotate_to_funding(cycle, graph.numeraire, funding or [])
    legs: list[list[Edge]] = []
    for i, src in enumerate(cycle):
        dst = cycle[(i + 1) % len(cycle)]
        es = graph.edges.get((src, dst))
        if not es:
            return None
        es = sorted(es, key=lambda e: (-e.top_rate(consumed.get(e.key, 0.0)), e.venue))
        legs.append(es if venue_top_k is None else es[:venue_top_k])
    best: Plan | None = None
    for combo in itertools.product(*legs):
        if max_venues_per_cycle is not None and len({e.venue for e in combo}) > max_venues_per_cycle:
            continue
        if math.prod(e.top_rate(consumed.get(e.key, 0.0)) for e in combo) <= 1.0:
            continue
        plan = size_edges(graph, cycle, list(combo), max_notional, inventory, consumed)
        if plan is not None and (best is None or plan.profit_num > best.profit_num):
            best = plan
    return best
