import copy
import csv
import json

import numpy as np
import pytest

from qarb import prepare_data, report, run
from qarb.adapters.synthetic import dataset_paths, generate_dataset, write_dataset
from qarb.config import load_config
from qarb.core.book import read_events
from qarb.graph.build import build_graph
from qarb.graph.cycles import enumerate_cycles
from qarb.metrics import summarize
from qarb.simulation.replay import EventStore, _truth, canonical_names, run_replay
from qarb.solvers import SearchContext, solve, solver_names

from conftest import make_config, random_weights


def test_dataset_is_deterministic_and_seed_dependent(cfg):
    a, ma = generate_dataset(cfg, 5)
    b, mb = generate_dataset(cfg, 5)
    c, _ = generate_dataset(cfg, 6)
    assert a == b and ma == mb and a != c
    kinds = [p["kind"] for p in ma["planted"]]
    assert kinds.count("profitable") == 3 and kinds.count("thin") == 1 and len(ma["outages"]) == 1
    assert [e.seq for e in a] == list(range(len(a)))
    assert all(x.recv_ts <= y.recv_ts for x, y in zip(a, a[1:]))  # arrival order
    assert any(x.exchange_ts > y.exchange_ts for x, y in zip(a, a[1:]))  # but asynchronous


def test_background_books_are_arbitrage_free(cfg):
    """With nothing planted, exhaustive search on the true books finds no executable cycle."""
    c = copy.deepcopy(cfg)
    c["data"]["planted"]["n_profitable"] = 0
    c["data"]["decoys"].update(n_thin=0, n_stale=0)
    events, man = generate_dataset(c, 1)
    store = EventStore(events, man["currencies"])
    for t in range(200, store.end_ts, 500):
        g = build_graph(store.true(t), t)
        assert _truth(g, c, None) == {}
        W = g.weight_matrix(0.0)
        assert all(w > 0 for _, w in enumerate_cycles(W, 2, 4)[0])


def test_planted_cycles_are_in_ground_truth_and_decoys_are_not(cfg):
    events, man = generate_dataset(cfg, 5)
    store = EventStore(events, man["currencies"])
    tick = cfg["data"]["tick_ms"]
    for p in man["planted"]:
        key = canonical_names(tuple(p["currencies"]))
        hits = []
        for t in range(p["start_ms"] + tick // 2, p["end_ms"], tick):
            truth = _truth(build_graph(store.true(t), t), cfg, None)
            hits.append(key in truth)
        assert all(hits) if p["kind"] == "profitable" else not any(hits), p["id"]


def test_observed_state_lags_true_state_and_ignores_out_of_order(cfg):
    events, man = generate_dataset(cfg, 5)
    store = EventStore(events, man["currencies"])
    o = man["outages"][0]
    t = o["start_ms"] + 2 * cfg["data"]["tick_ms"]
    obs = {b.key: b for b in store.observed(t)}
    true = {b.key: b for b in store.true(t)}
    lagging = [k for k in obs if k[0] == o["venue"]]
    assert lagging and all(obs[k].ts < o["start_ms"] <= true[k].ts for k in lagging)
    assert all(obs[k].ts <= true[k].ts for k in obs)
    final = {b.key: b.ts for b in store.observed(10**9)}
    assert final == {b.key: b.ts for b in store.true(10**9)}


@pytest.fixture
def replay(cfg):
    events, man = generate_dataset(cfg, 5)
    solvers = solver_names("all", cfg)
    return cfg, man, events, run_replay(cfg, 5, solvers, events, man)


def test_replay_metrics(replay):
    cfg, man, _, out = replay
    m = {n: summarize(s, out["truth_by_t"], out["decision_times"], man, cfg) for n, s in out["states"].items()}
    enum = m["enumeration"]
    assert enum["search_recall"] == 1.0  # exhaustive search is the reference on observed data
    assert enum["planted_recall"] >= 2 / 3 and enum["thin_decoys_flagged"] == 0
    assert enum["truth_episodes"] > 0 and enum["n_executions"] > 0
    for name, mm in m.items():
        assert 0 <= mm["search_recall"] <= 1 and mm["search_recall"] <= enum["search_recall"]
        assert mm["search_recall_unrepaired"] <= mm["search_recall"]
        assert mm["true_positives"] + mm["false_negatives"] == enum["true_positives"] + enum["false_negatives"]
        assert mm["objective_evals_mean"] > 0 and mm["search_ms_p95"] >= mm["search_ms_p50"] > 0
        assert mm["fallback_decisions"] == 0
    # the mid-price detector raises far more nominal signals than survive the cost gate
    assert m["naive_mid"]["rejected_by_cost_gate"] > m["naive_mid"]["true_positives"]
    assert m["qaoa"]["max_qubits"] == 6 and 0 < m["qaoa"]["valid_shot_fraction"] <= 1
    assert m["qaoa"]["formulation_exact_fraction"] == 1.0
    assert m["qaoa"]["p_ground_mean"] > m["qaoa"]["p_ground_uniform_mean"]


def test_replay_is_deterministic(replay):
    cfg, man, events, out = replay
    again = run_replay(cfg, 5, solver_names("all", cfg), events, man)
    for name, st in out["states"].items():
        assert st.opportunities == again["states"][name].opportunities
        assert st.detections == again["states"][name].detections
        assert st.inventory == again["states"][name].inventory


def test_stale_filter_prevents_trading_on_dead_quotes(cfg):
    events, man = generate_dataset(cfg, 5)
    res = {}
    for label, timeout in (("on", cfg["graph"]["stale_timeout_ms"]), ("off", None)):
        c = copy.deepcopy(cfg)
        c["graph"]["stale_timeout_ms"] = timeout
        out = run_replay(c, 5, ["enumeration"], events, man)
        res[label] = summarize(out["states"]["enumeration"], out["truth_by_t"], out["decision_times"], man, c)
    assert res["on"]["false_positives_after_costs"] == 0 and res["on"]["stale_decoys_flagged"] == 0
    assert res["off"]["false_positives_after_costs"] > 0 and res["off"]["stale_decoys_flagged"] == 1
    assert res["off"]["mean_fill_ratio"] < res["on"]["mean_fill_ratio"]
    assert res["off"]["residual_exposure"] > res["on"]["residual_exposure"]


def _ctx(cfg, W, seed=0):
    return SearchContext(W, W, 2, cfg["graph"]["max_cycle_len"], np.random.default_rng(seed), cfg)


def test_solver_adapters_agree_on_small_instance(cfg):
    W = random_weights(np.random.default_rng(2), 4, neg_edges=3)
    truth = set(solve("enumeration", _ctx(cfg, W)).cycles)
    assert truth
    for name in ("bellman_ford", "bounded_dp", "random_search", "qubo_sa", "qaoa"):
        res = solve(name, _ctx(cfg, W))
        assert set(res.cycles) <= truth and res.stats["status"] == "ok" and res.stats["search_ms"] > 0
    assert set(solve("random_search", _ctx(cfg, W)).cycles) == truth
    assert set(solve("qubo_sa", _ctx(cfg, W)).cycles) == truth


def test_quantum_solver_falls_back_when_register_too_large(cfg):
    W = random_weights(np.random.default_rng(2), 4, neg_edges=3)
    c = copy.deepcopy(cfg)
    c["qaoa"]["max_qubits"] = 2
    res = solve("qaoa", _ctx(c, W))
    assert res.stats["status"] == "fallback:bounded_dp" and "max_qubits" in res.stats["error"]
    assert set(res.cycles) == set(solve("bounded_dp", _ctx(c, W)).cycles)
    c["qaoa"]["fallback"] = None
    with pytest.raises(RuntimeError):
        solve("qaoa", _ctx(c, W))


def test_quantum_solver_falls_back_when_budget_too_small(cfg):
    c = copy.deepcopy(cfg)
    c["budget"]["max_objective_evals"] = 400
    res = solve("qaoa", _ctx(c, random_weights(np.random.default_rng(2), 4)))
    assert res.stats["status"].startswith("fallback") and "budget" in res.stats["error"]


def test_budget_caps_classical_search(cfg):
    c = copy.deepcopy(cfg)
    c["budget"]["max_objective_evals"] = 5
    c["graph"]["max_cycle_len"] = 4
    res = solve("enumeration", _ctx(c, random_weights(np.random.default_rng(2), 6)))
    assert res.stats["budget_exhausted"] and res.stats["objective_evals"] == 5


def test_unknown_solver(cfg):
    with pytest.raises(ValueError):
        solver_names("warp-drive", cfg)
    assert solver_names("classical", cfg)[0] == "enumeration" and solver_names("qaoa", cfg) == ["qaoa"]


def _write_config(tmp_path):
    import yaml

    cfg = make_config(tmp_path, seeds=[5, 6])
    cfg["name"] = "exp"
    cfg["solvers"] = {"classical": ["enumeration", "bellman_ford"], "quantum-sim": ["qaoa"]}
    path = tmp_path / "exp.yaml"
    path.write_text(yaml.safe_dump(cfg))
    return path


def test_cli_end_to_end(tmp_path, capsys):
    path = _write_config(tmp_path)
    assert prepare_data.main(["--config", str(path)]) == 0
    assert run.main(["--config", str(path), "--solver", "classical"]) == 0
    assert run.main(["--config", str(path), "--solver", "quantum-sim"]) == 0
    runs = tmp_path / "runs"
    for solver in ("enumeration", "bellman_ford", "qaoa"):
        for seed in (5, 6):
            d = runs / "exp" / solver / f"seed{seed}"
            man = json.loads((d / "manifest.json").read_text())
            assert man["status"] == "ok" and len(man["dataset"]["sha256"]) == 64 and man["versions"]["numpy"]
            assert json.loads((d / "metrics.json").read_text())["status"] == "ok"
            assert list(csv.DictReader(open(d / "ticks.csv")))
    qman = json.loads((runs / "exp" / "qaoa" / "seed5" / "manifest.json").read_text())
    assert {r["n_qubits"] for r in qman["quantum_resources"]} == {3, 6}
    out = tmp_path / "reports" / "comparison.html"
    assert report.main(["--runs", str(runs), "--output", str(out)]) == 0
    text = out.read_text()
    assert "enumeration" in text and "qaoa" in text and "±" in text and "quantum advantage" in text
    rows = list(csv.DictReader(open(out.with_suffix(".csv"))))
    assert {r["solver"] for r in rows} == {"enumeration", "bellman_ford", "qaoa"}
    assert all(r["seeds_ok"] == "2" and r["seeds_failed"] == "0" for r in rows)
    assert len(list(csv.DictReader(open(out.with_name("comparison_per_seed.csv"))))) == 6


def test_cli_records_failed_runs(tmp_path, capsys):
    path = _write_config(tmp_path)
    assert prepare_data.main(["--config", str(path)]) == 0
    cfg = load_config(path)
    ev, _ = dataset_paths(cfg, 6)
    ev.write_text(ev.read_text() + "\n")  # tamper with an immutable dataset
    assert run.main(["--config", str(path), "--solver", "enumeration"]) == 1
    runs = tmp_path / "runs"
    bad = json.loads((runs / "exp" / "enumeration" / "seed6" / "manifest.json").read_text())
    assert bad["status"] == "failed" and "sha256" in bad["error"]
    out = tmp_path / "r.html"
    assert report.main(["--runs", str(runs), "--output", str(out)]) == 0
    row = list(csv.DictReader(open(out.with_suffix(".csv"))))[0]
    assert row["seeds_ok"] == "1" and row["seeds_failed"] == "1" and "1 failed" in out.read_text()


def test_run_requires_prepared_data(tmp_path, capsys):
    path = _write_config(tmp_path)
    assert run.main(["--config", str(path), "--solver", "enumeration"]) == 1
    man = json.loads((tmp_path / "runs" / "exp" / "enumeration" / "seed5" / "manifest.json").read_text())
    assert "prepare_data" in man["error"]


def test_written_dataset_roundtrip(cfg):
    man = write_dataset(cfg, 5)
    ev_path, _ = dataset_paths(cfg, 5)
    assert list(read_events(ev_path)) == generate_dataset(cfg, 5)[0]
    assert man["n_events"] == sum(1 for _ in read_events(ev_path))


def test_config_validation(tmp_path):
    import yaml

    p = tmp_path / "bad.yaml"
    p.write_text(yaml.safe_dump({"numeraire": "JPY"}))
    with pytest.raises(ValueError):
        load_config(p)
    p.write_text(yaml.safe_dump({"data": {"noise_frac": 1.5}}))
    with pytest.raises(ValueError):
        load_config(p)
    p.write_text(yaml.safe_dump({"graph": {"probe_notional": 250}}))
    cfg = load_config(p)
    assert cfg["graph"]["probe_notional"] == 250 and cfg["graph"]["max_cycle_len"] == 4 and cfg["name"] == "bad"
