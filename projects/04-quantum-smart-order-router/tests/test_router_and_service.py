import dataclasses
import time

import numpy as np
import pytest

from conftest import make_cfg
from qsor.core import ASK, BID, BUY, MS, SELL
from qsor.experiment import cost_model
from qsor.optimizer.service import AsyncPolicyService, PolicyOptimizer, market_ladders
from qsor.router.policy import PolicyStore, make_snapshot
from qsor.router.router import LiveRouter, SnapshotRouter, best_price_alloc


def snap(version, buy, sell=None, t=0, buckets=(2, 4)):
    return make_snapshot(version, t, t, "test", buckets, buy, sell or buy)


def test_snapshot_is_immutable_and_versions_never_go_back():
    store = PolicyStore()
    s1, s2 = snap(1, [(1, 1), (2, 2)]), snap(2, [(2, 0), (4, 0)])
    assert store.publish(s2) and not store.publish(s1) and store.current is s2
    with pytest.raises(dataclasses.FrozenInstanceError):
        s2.version = 9
    with pytest.raises(TypeError):
        s2.entries[0][0].weights[0] = 5


def test_bucket_lookup_uses_nearest_bucket():
    s = snap(1, [(2, 0), (0, 4)], [(0, 2), (4, 0)])
    assert s.lookup(BUY, 1).weights == (2, 0) and s.lookup(BUY, 3).weights == (2, 0)
    assert s.lookup(BUY, 4).weights == (0, 4) and s.lookup(BUY, 50).weights == (0, 4)
    assert s.lookup(SELL, 4).weights == (4, 0)


def test_decide_scales_weights_and_respects_caps():
    store = PolicyStore()
    store.publish(snap(1, [(1, 1), (3, 1)]))
    r = SnapshotRouter(store, max_age_ns=10 * MS)
    assert r.decide(BUY, 4, [9, 9], [100, 100], 0) == [3, 1]
    assert r.decide(BUY, 8, [9, 9], [100, 100], 0) == [6, 2]      # scaled beyond the largest bucket
    assert r.decide(BUY, 4, [2, 9], [100, 100], 0) == [2, 2]      # cap on venue 0 spills to venue 1
    assert r.decide(BUY, 4, [0, 9], [None, 100], 0) == [0, 4]     # venue 0 down: failover
    assert r.decide(BUY, 4, [1, 1], [100, 100], 0) == [1, 1]      # not enough liquidity anywhere
    assert r.stale == 0 and r.decisions == 5


def test_missing_or_stale_policy_falls_back_to_best_price():
    store = PolicyStore()
    r = SnapshotRouter(store, max_age_ns=10 * MS)
    assert r.decide(BUY, 4, [3, 3], [101, 100], 0) == [1, 3] and r.stale == 1
    store.publish(snap(1, [(2, 0), (4, 0)]))
    assert r.decide(BUY, 4, [3, 3], [101, 100], 5 * MS) == [3, 1] and r.stale == 1
    assert r.decide(BUY, 4, [3, 3], [101, 100], 11 * MS) == [1, 3] and r.stale == 2
    assert best_price_alloc(SELL, 4, [3, 3], [101, 100]) == [3, 1]  # sells prefer the higher bid


def test_decide_conserves_quantity_on_random_inputs():
    rng = np.random.default_rng(0)
    store = PolicyStore()
    r = SnapshotRouter(store, max_age_ns=10**18)
    for i in range(300):
        V = int(rng.integers(1, 6))
        store.publish(snap(i + 1, [tuple(rng.integers(0, 4, V))] * 2))
        caps = [int(c) for c in rng.integers(0, 6, V)]
        qty = int(rng.integers(1, 12))
        alloc = r.decide(BUY, qty, caps, [100] * V, 0)
        assert all(0 <= a <= c for a, c in zip(alloc, caps))
        assert sum(alloc) == min(qty, sum(caps))


def books(sizes):
    return {BUY: [([101, 102], list(s)) for s in sizes], SELL: [([99, 98], list(s)) for s in sizes]}


def optimizer(solver, **opt):
    cfg = make_cfg(2)
    cfg["optimizer"].update(opt)
    cfg["solvers"]["qaoa"].update(maxiter=15, shots=64)
    model = cost_model(cfg, [1.0, 1.0], 1.0)
    return PolicyOptimizer(solver, model, cfg, [20, 20], seed=3), model


def test_market_ladders_applies_limit_participation_and_availability():
    lad, caps, best, far = market_ladders(BUY, [([101, 102, 103], [4, 4, 4]), ([102, 103, 104], [0, 6, 6]), None],
                                          1, 0.5, [20, 20, 20])
    assert far == 101 and best == [101, 103, None]
    assert lad[0] == ((101, 4), (102, 4)) and caps == [4, 0, 0]   # venue 1 only has size beyond the limit
    _, caps, _, _ = market_ladders(BUY, [([101], [5])], 1, 0.5, [2])
    assert caps == [2]                                            # venue child cap
    assert market_ladders(SELL, [None, None], 1, 0.5, [9, 9])[3] is None


@pytest.mark.parametrize("solver", ["greedy", "exact", "cpsat", "qaoa", "sa", "random_repair"])
def test_optimizer_adapters_produce_feasible_policies(solver):
    opt, _ = optimizer(solver)
    s, wall = opt.compute(books([(3, 3), (3, 3)]), 123)
    assert s.version == 1 and s.source == solver and s.created_ns == 123 and wall > 0
    for side in (0, 1):
        for bucket, e in zip(s.buckets, s.entries[side]):
            assert e.total * max(1, -(-bucket // 4)) == bucket
    assert opt.stats["fallbacks"] == 0 and opt.stats["problems"] == 4


def test_timeout_falls_back_to_classical():
    opt, _ = optimizer("qaoa", timeout_ms=0)
    s, _ = opt.compute(books([(3, 3), (3, 3)]), 0)
    assert s.source == "qaoa+fallback"
    assert opt.stats["fallbacks"] == 4 and opt.stats["fallback_reasons"] == {"SolveTimeout": 4}
    ref, _ = optimizer("greedy")
    assert s.entries == ref.compute(books([(3, 3), (3, 3)]), 0)[0].entries


def test_qubit_budget_and_solver_crash_fall_back(monkeypatch):
    opt, _ = optimizer("qaoa")
    opt.solver_cfg["qaoa"]["max_qubits"] = 1
    opt.compute(books([(3, 3), (3, 3)]), 0)
    assert opt.stats["fallback_reasons"] == {"TooManyQubits": 4}

    from qsor.optimizer import registry
    opt, _ = optimizer("exact")
    monkeypatch.setattr(registry, "solve", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    s, _ = opt.compute(books([(3, 3), (3, 3)]), 0)
    assert s.source == "exact+fallback" and opt.stats["fallback_reasons"] == {"RuntimeError": 4}


def test_infeasible_solver_output_is_rejected(monkeypatch):
    from qsor.optimizer import registry
    opt, _ = optimizer("exact")
    monkeypatch.setattr(registry, "solve", lambda name, p, *a, **k: (tuple([0] * p.V), {}))
    s, _ = opt.compute(books([(3, 3), (3, 3)]), 0)
    assert opt.stats["fallbacks"] == 4 and all(e.total > 0 for e in s.entries[0])


def test_empty_market_yields_empty_policy():
    opt, _ = optimizer("exact")
    s, _ = opt.compute({BUY: [None, None], SELL: [None, None]}, 0)
    assert all(e.total == 0 for side in s.entries for e in side)
    r = SnapshotRouter(PolicyStore(), 10)
    r.store.publish(s)
    assert r.decide(BUY, 4, [3, 3], [101, 100], 0) == [1, 3]   # zero-weight policy -> best price


def test_async_service_publishes_monotonic_snapshots_while_router_reads():
    opt, _ = optimizer("qaoa")
    store = PolicyStore()
    svc = AsyncPolicyService(opt, store, lambda: books([(3, 3), (3, 3)]), time.perf_counter_ns, 0.005)
    router = SnapshotRouter(store, max_age_ns=10**12)
    svc.start()
    seen, deadline = [], time.time() + 5
    while (store.current is None or store.current.version < 4) and time.time() < deadline:
        alloc = router.decide(BUY, 4, [3, 3], [101, 101], time.perf_counter_ns())
        assert sum(alloc) == 4
        seen.append(router.last_version)
    svc.stop()
    assert store.current.version >= 4 and svc.errors == 0
    versions = [v for v in seen if v >= 0]
    assert versions == sorted(versions)


def test_async_service_survives_failures_and_keeps_last_policy():
    opt, _ = optimizer("greedy")
    store = PolicyStore()
    calls = {"n": 0}

    def market():
        calls["n"] += 1
        if calls["n"] > 1:
            raise ConnectionError("market data gap")
        return books([(3, 3), (3, 3)])

    svc = AsyncPolicyService(opt, store, market, time.perf_counter_ns, 0.002)
    svc.start()
    time.sleep(0.1)
    svc.stop()
    assert store.current.version == 1 and svc.errors >= 1


def test_live_router_solves_on_the_hot_path():
    cfg = make_cfg(2)
    r = LiveRouter(cost_model(cfg, [1.0, 1.0], 1.0))
    ladders = [((101, 3),), ((101, 3), (102, 3))]
    alloc = r.decide(BUY, 5, [3, 6], [101, 101], 0, (ladders, 101))
    assert sum(alloc) == 5 and alloc[0] <= 3
