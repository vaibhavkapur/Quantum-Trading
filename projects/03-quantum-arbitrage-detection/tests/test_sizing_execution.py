import math

import pytest

from qarb.graph.build import build_graph
from qarb.graph.sizing import best_plan, optimal_size, size_edges
from qarb.simulation.execution import execute_plan

from conftest import book


def test_optimal_size_single_level_analytic():
    # 2% round trip, 100 units of depth on the first leg: trade all of it.
    used, got, worst = optimal_size([[(1.0, 100.0)], [(1.02, 1000.0)]], [0.0, 0.0], [math.inf, math.inf])
    assert used == pytest.approx([100.0, 100.0]) and got == pytest.approx([100.0, 102.0])
    assert worst == [1.0, 1.02]


def test_optimal_size_stops_when_marginal_rate_turns_unprofitable():
    # level 1: +2%, level 2: -1% -> only the first 50 units are worth trading
    ladders = [[(1.0, 50.0), (0.97, 50.0)], [(1.02, 1000.0)]]
    used, got, _ = optimal_size(ladders, [0.0, 0.0], [math.inf, math.inf])
    assert used[0] == pytest.approx(50.0) and got[-1] - used[0] == pytest.approx(1.0)


def test_optimal_size_is_the_argmax():
    ladders = [[(1.0, 30.0), (0.999, 40.0), (0.99, 50.0)], [(1.004, 45.0), (1.002, 100.0)], [(1.0, 1e9)]]
    fees = [0.0005, 0.0005, 0.0]
    used, got, _ = optimal_size(ladders, fees, [math.inf] * 3)
    best = got[-1] - used[0]

    def profit(q):
        amt = q
        for lad, fee in zip(ladders, fees):
            out, rem = 0.0, amt
            for rate, cap in lad:
                take = min(rem, cap)
                out += take * rate
                rem -= take
            if rem > 1e-9:
                return -math.inf
            amt = out * (1 - fee)
        return amt - q

    assert best == pytest.approx(profit(used[0]))
    assert best >= max(profit(q / 10) for q in range(0, 1200)) - 1e-9


def test_fees_and_caps_bind():
    ladders = [[(1.0, 100.0)], [(1.02, 1000.0)]]
    assert optimal_size(ladders, [0.01, 0.011], [math.inf] * 2)[0] == [0.0, 0.0]  # fees eat the edge
    used, _, _ = optimal_size(ladders, [0.0, 0.0], [40.0, math.inf])
    assert used[0] == pytest.approx(40.0)  # capital cap
    used, _, _ = optimal_size(ladders, [0.0, 0.0], [math.inf, 25.0])
    assert used == pytest.approx([25.0, 25.0])  # inventory cap on leg 2
    assert optimal_size(ladders, [0.0, 0.0], [math.inf, 0.0])[0] == [0.0, 0.0]
    assert optimal_size([[], [(1.02, 1.0)]], [0.0, 0.0], [math.inf] * 2)[0] == [0.0, 0.0]


def test_best_plan_on_triangle(triangle_books):
    g = build_graph(triangle_books, now=0)
    plan = best_plan(g, ("BTC", "ETH", "USD"), max_notional=1e9)
    assert plan.cycle == ("USD", "BTC", "ETH")  # rotated to the numeraire
    assert plan.venues == ("v1", "v1", "v2")
    assert plan.amounts_in[0] == pytest.approx(100.0) and plan.profit == pytest.approx(2.0)
    assert plan.profit_num == pytest.approx(2.0)
    capped = best_plan(g, ("BTC", "ETH", "USD"), max_notional=50.0)
    assert capped.amounts_in[0] == pytest.approx(50.0) and capped.profit == pytest.approx(1.0)
    assert best_plan(g, ("USD", "ETH", "BTC"), max_notional=1e9) is None  # reverse direction loses
    inv = {("v1", "USD"): 30.0, ("v1", "BTC"): 10.0, ("v2", "ETH"): 1e6}
    assert best_plan(g, ("BTC", "ETH", "USD"), 1e9, inventory=inv).amounts_in[0] == pytest.approx(30.0)
    assert best_plan(g, ("BTC", "ETH", "USD"), 1e9, max_venues_per_cycle=1) is None


def test_best_plan_picks_better_venue():
    books = [
        book("a", "BTC", "USD", 99.0, 100.0), book("b", "BTC", "USD", 101.0, 101.5), book("c", "BTC", "USD", 100.4, 100.6),
    ]
    g = build_graph(books, now=0)
    plan = best_plan(g, ("BTC", "USD"), 1e9, venue_top_k=None)
    assert plan.venues == ("a", "b") and plan.profit == pytest.approx(10.0)  # buy 10 at 100, sell at 101


def test_execution_full_fill_matches_plan(triangle_books):
    g = build_graph(triangle_books, now=0)
    plan = best_plan(g, ("USD", "BTC", "ETH"), 1e9)
    inv = {(e.venue, e.src): 1e6 for e in plan.edges}
    fill = execute_plan(plan, g, inv, {})
    assert fill["fill_ratio"] == pytest.approx(1.0)
    assert fill["realised_pnl"] == pytest.approx(plan.profit_num)
    assert fill["residual_exposure"] == pytest.approx(0.0, abs=1e-9)
    assert inv[("v1", "USD")] == pytest.approx(1e6 - 100.0) and inv[("v2", "USD")] == pytest.approx(102.0)


def test_execution_partial_fill_when_quote_is_gone(triangle_books):
    g = build_graph(triangle_books, now=0)
    plan = best_plan(g, ("USD", "BTC", "ETH"), 1e9)
    moved = [book("v1", "BTC", "USD", 103.0, 104.0, size=1.0)] + triangle_books[1:]  # ask ran away
    g2 = build_graph(moved, now=1)
    inv = {(e.venue, e.src): 1e6 for e in plan.edges}
    fill = execute_plan(plan, g2, inv, {})
    assert fill["leg_fill_ratios"][0] == 0.0 and fill["leg_fill_ratios"][1] == pytest.approx(1.0)
    assert fill["fill_ratio"] == 0.0 and fill["residual_exposure"] > 0
    # liquidity consumed by an earlier fill is not available twice
    inv = {(e.venue, e.src): 1e6 for e in plan.edges}
    consumed = {}
    first = execute_plan(plan, g, inv, consumed)
    second = execute_plan(plan, g, inv, consumed)
    assert first["fill_ratio"] == pytest.approx(1.0) and second["leg_fill_ratios"][0] == 0.0


def test_size_edges_respects_consumed(triangle_books):
    g = build_graph(triangle_books, now=0)
    plan = best_plan(g, ("USD", "BTC", "ETH"), 1e9)
    half = {plan.edges[0].key: 50.0}
    again = size_edges(g, plan.cycle, plan.edges, 1e9, consumed=half)
    assert again.amounts_in[0] == pytest.approx(50.0)
