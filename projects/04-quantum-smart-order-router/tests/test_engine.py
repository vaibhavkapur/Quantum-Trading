import numpy as np
import pytest

from conftest import flat_dataset, make_cfg
from qsor.core import ASK, BID, BUY, MS, SELL
from qsor.experiment import cost_model, simulate
from qsor.router.policy import PolicyStore, make_snapshot
from qsor.router.router import SnapshotRouter
from qsor.sim.engine import Engine, Parent
from qsor.sim.tca import parent_rows, summarize


def run(ds, cfg, parents, weights=((1, 1), (1, 1)), fix=0):
    store = PolicyStore()
    store.publish(make_snapshot(1, 0, 0, "fixed", (2, 4), weights, weights))
    router = SnapshotRouter(store, max_age_ns=10**18)
    eng = Engine(cfg, ds, (0, ds.T), router, None, store, seed=1, parents=parents, fix_log_limit=fix).run()
    rows = parent_rows(eng)
    return eng, rows, summarize(eng, rows)


def parent(pid=0, side=BUY, qty=4, t_ms=100, horizon_ms=200):
    return Parent(pid, side, qty, t_ms * MS, (t_ms + horizon_ms) * MS)


def test_simple_sweep_fills_at_touch_with_correct_tca():
    ds, cfg = flat_dataset(), make_cfg(2)
    eng, rows, m = run(ds, cfg, [parent(qty=4)], fix=50)
    r = rows[0]
    assert r["filled_lots"] == 4 and r["children"] == 2 and r["completed"]
    # bought 400 shares one tick (0.01) above mid: slippage 4.00; taker fee 0.002 * 400 = 0.80
    assert r["slippage"] == pytest.approx(4.0) and r["fees"] == pytest.approx(0.8)
    assert r["is_bps"] == pytest.approx((4.0 + 0.8) / (100.0 * 400) * 1e4)
    assert r["price_improvement"] == 0.0 and m["fill_rate"] == 1.0
    assert m["o2a_ms_p50"] == pytest.approx(2.0)            # 1 ms each way, no jitter
    assert eng.fix_log[0].startswith("35=D|11=1|100=V0|54=1|38=200|40=2|44=100.02|59=3|")
    assert any("150=2" in line and "32=200" in line and "31=100.01" in line for line in eng.fix_log)


def test_sell_parent_mirrors_buy():
    ds, cfg = flat_dataset(), make_cfg(2)
    _, rows, _ = run(ds, cfg, [parent(side=SELL, qty=4)])
    assert rows[0]["slippage"] == pytest.approx(4.0) and rows[0]["filled_lots"] == 4


def test_reject_triggers_cooldown_and_failover():
    ds, cfg = flat_dataset(), make_cfg(2)
    ds.status[10:60, 0] = 1                     # venue 0 down from t=100ms
    cfg["execution"]["heartbeat_ms"] = 50       # router has not noticed yet
    eng, rows, m = run(ds, cfg, [parent(qty=4, t_ms=101)])
    assert m["reject_reasons"] == {"VENUE_DOWN": 1}
    assert rows[0]["filled_lots"] == 4 and rows[0]["rejects"] == 1
    fills_by_venue = {f[1] for f in eng.parents[0].fills}
    assert fills_by_venue == {1}                # second sweep went entirely to the healthy venue


def test_throttle_reject_is_counted():
    ds, cfg = flat_dataset(size=9), make_cfg(2, throttle_max_orders=1, throttle_window_ms=500)
    parents = [parent(0, qty=2, t_ms=100), parent(1, qty=2, t_ms=110)]
    _, rows, m = run(ds, cfg, parents)
    assert m["reject_reasons"].get("THROTTLE", 0) >= 1
    assert all(r["completed"] for r in rows)


def test_residual_rests_passively_then_final_sweep_at_deadline():
    ds, cfg = flat_dataset(size=1, flow=0), make_cfg(2)
    cfg["execution"].update(max_sweeps=1, participation=1.0, limit_ticks=0, final_limit_ticks=2)
    eng, rows, m = run(ds, cfg, [parent(qty=6)])
    p = eng.parents[0]
    assert m["passive_orders"] == 1 and m["cancels"] == 1 and m["queue_error_samples"] == 1
    assert rows[0]["completed"] and p.t_done >= p.deadline
    # sweep 1 took the two touch lots; depleted liquidity partly replenished before the final sweep
    assert 2 < rows[0]["filled_lots"] <= 6
    assert rows[0]["filled_lots"] + p.remaining == 6


def test_passive_order_fills_from_aggressor_flow_as_maker():
    ds, cfg = flat_dataset(size=1, flow=1), make_cfg(2)
    cfg["execution"].update(max_sweeps=1, participation=1.0, limit_ticks=0)
    eng, rows, m = run(ds, cfg, [parent(qty=4)])
    assert rows[0]["filled_lots"] == 4 and rows[0]["passive_lots"] == 2
    assert m["cancels"] == 0 and m["passive_fill_share"] == pytest.approx(0.5)
    maker = [f for f in eng.parents[0].fills if f[4] == "M"]
    assert all(f[2] == 9999 for f in maker)     # bought at the bid
    assert rows[0]["fees"] == pytest.approx(0.002 * 200 - 0.001 * 200)


def test_passive_order_is_repriced_with_cancel_replace_when_touch_moves_away():
    ds, cfg = flat_dataset(size=1, flow=0), make_cfg(2)
    ds.px[BID][15:] += 1
    ds.px[ASK][15:] += 1                        # market ticks up at t=150ms
    ds.mid[15:] += 1
    cfg["execution"].update(max_sweeps=1, participation=1.0, limit_ticks=0, reprice_ms=20)
    eng, _, m = run(ds, cfg, [parent(qty=6)], fix=200)
    assert m["replaces"] >= 1 and m["queue_error_samples"] >= 2
    assert any(line.startswith("35=G|") for line in eng.fix_log)
    assert any("150=5" in line for line in eng.fix_log)


def test_quote_fade_reduces_fills_on_the_slower_venue():
    def filled_on_slow(fade):
        ds = flat_dataset(size=4)
        cfg = make_cfg(2)
        cfg["venues"][0].update(latency_ms=0.2)
        cfg["venues"][1].update(latency_ms=3.0, fade_coeff=fade, fade_delay_ms=1.0)
        cfg["execution"].update(max_sweeps=1, participation=1.0, limit_ticks=0)
        eng, _, _ = run(ds, cfg, [parent(qty=8, horizon_ms=50)], weights=((1, 1), (1, 1)))
        return sum(f[3] for f in eng.parents[0].fills if f[1] == 1 and f[0] < 110 * MS)
    assert filled_on_slow(0.0) == 4
    assert filled_on_slow(1.0) == 0


def test_invariants_and_determinism_on_generated_market():
    from qsor.exchange_sim.data import Dataset, _splits, generate_arrays
    cfg = make_cfg(3, fade_coeff=0.5, outage_rate_per_min=20.0, outage_ms=200, throttle_max_orders=3,
                   latency_jitter_ms=0.3, depth_lots=3.0)
    cfg["data"].update(steps=3000, seed=5)
    cfg["solvers"]["qaoa"].update(maxiter=10, shots=64)
    ds = Dataset.from_arrays(generate_arrays(cfg), cfg["data"]["dt_ms"], _splits(cfg))
    model = cost_model(cfg, [1.0] * 3, 2.0)
    results = {}
    for solver in ("exact", "cpsat", "qaoa", "greedy_live"):
        eng, rows, m = simulate(cfg, ds, "test", solver, 3, model)
        eng2, rows2, m2 = simulate(cfg, ds, "test", solver, 3, model)
        assert rows == rows2, solver                         # bit-for-bit repeatable
        assert m["incomplete_parents"] == 0 and len(rows) > 50
        for p, r in zip(eng.parents, rows):
            assert 0 <= r["filled_lots"] <= r["qty_lots"] and p.remaining == r["qty_lots"] - r["filled_lots"]
            assert p.open == 0 and p.passive is None
            assert sum(f[3] for f in p.fills) == r["filled_lots"] and all(f[3] > 0 for f in p.fills)
        assert m["rejects"] > 0
        results[solver] = m
    assert results["exact"]["is_bps"] == pytest.approx(results["cpsat"]["is_bps"], abs=0.05)
    assert results["greedy_live"]["decision_us_p50"] > results["exact"]["decision_us_p50"]


def test_stale_policy_is_detected_in_simulation():
    from qsor.exchange_sim.data import Dataset, _splits, generate_arrays
    cfg = make_cfg(2)
    cfg["data"].update(steps=2000, seed=9)
    cfg["optimizer"].update(refresh_ms=400, publish_delay_ms=50, max_policy_age_ms=100)
    ds = Dataset.from_arrays(generate_arrays(cfg), cfg["data"]["dt_ms"], _splits(cfg))
    _, _, m = simulate(cfg, ds, "test", "greedy", 1, cost_model(cfg, [1.0, 1.0], 1.0))
    assert 0.3 < m["stale_policy_frequency"] < 1.0


def test_measured_delay_skips_refresh_when_solver_overruns(monkeypatch):
    from qsor.exchange_sim.data import Dataset, _splits, generate_arrays
    from qsor.optimizer import service
    cfg = make_cfg(2)
    cfg["data"].update(steps=1000, seed=9)
    cfg["optimizer"].update(refresh_ms=100, delay_model="measured")
    ds = Dataset.from_arrays(generate_arrays(cfg), cfg["data"]["dt_ms"], _splits(cfg))
    real = service.PolicyOptimizer.compute
    monkeypatch.setattr(service.PolicyOptimizer, "compute", lambda self, b, t: (real(self, b, t)[0], 0.25))
    _, _, m = simulate(cfg, ds, "test", "greedy", 1, cost_model(cfg, [1.0, 1.0], 1.0))
    assert m["skipped_refreshes"] > 0 and m["policies_published"] < m["solver_refreshes"] + m["skipped_refreshes"]
