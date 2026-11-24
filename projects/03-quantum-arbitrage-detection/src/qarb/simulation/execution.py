"""Execution simulator: IOC legs against the true books after a latency.

All legs of a plan are sent at once against pre-funded inventory. Each leg is
an immediate-or-cancel order limited at the worst rate the plan accepted (less
a slippage tolerance), so a leg whose liquidity moved fills partially or not at
all while the other legs still fill. That legging risk shows up as residual
inventory, marked at reference mids in the realised P&L.
"""

from __future__ import annotations

from ..graph.build import EdgeKey, MarketGraph
from ..graph.sizing import Inventory, Plan


def execute_plan(
    plan: Plan,
    true_graph: MarketGraph,
    inventory: Inventory,
    consumed: dict[EdgeKey, float],
    slippage_bps: float = 0.0,
) -> dict:
    """Execute against `true_graph`, mutating `inventory` and `consumed`."""
    deltas: dict[str, float] = {}
    fill_ratios = []
    fees_num = 0.0
    for edge, amount, limit in zip(plan.edges, plan.amounts_in, plan.limit_rates):
        live = next((e for e in true_graph.edges.get((edge.src, edge.dst), []) if e.key == edge.key), None)
        want = min(amount, inventory.get((edge.venue, edge.src), 0.0))
        out = filled = 0.0
        if live is not None and want > 0:
            out, filled, _ = live.convert(
                want, min_rate=limit * (1 - slippage_bps * 1e-4), consumed=consumed.get(edge.key, 0.0)
            )
            consumed[edge.key] = consumed.get(edge.key, 0.0) + filled
            inventory[(edge.venue, edge.src)] = inventory.get((edge.venue, edge.src), 0.0) - filled
            inventory[(edge.venue, edge.dst)] = inventory.get((edge.venue, edge.dst), 0.0) + out
            fees_num += out / (1 - live.fee) * live.fee * true_graph.ref.get(edge.dst, 0.0)
        deltas[edge.src] = deltas.get(edge.src, 0.0) - filled
        deltas[edge.dst] = deltas.get(edge.dst, 0.0) + out
        fill_ratios.append(filled / amount if amount > 0 else 0.0)
    start = plan.cycle[0]
    pnl = sum(d * true_graph.ref.get(c, 0.0) for c, d in deltas.items())
    residual = sum(abs(d) * true_graph.ref.get(c, 0.0) for c, d in deltas.items() if c != start)
    return {
        "realised_pnl": pnl,
        "fill_ratio": min(fill_ratios) if fill_ratios else 0.0,
        "leg_fill_ratios": fill_ratios,
        "residual_exposure": residual,
        "fees_paid": fees_num,
    }
