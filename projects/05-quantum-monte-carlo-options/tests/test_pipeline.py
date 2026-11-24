import json

import numpy as np
import pytest
import yaml

from qmc_options import analytic, prepare_data, report, run
from qmc_options.config import load_config
from qmc_options.models import Option
from qmc_options.quantum.pricer import estimate_price, prepare, quantum_delta, quantum_price


def test_quantum_price_brackets_the_discretized_price(market, call):
    for estimator, options in (("iqae", {"epsilon": 0.005}), ("mlae", {"max_power_exponent": 5}),
                               ("direct_sampling", {"shots": 5000})):
        out = quantum_price(market, call, 4, estimator, seed=2, truncation=4.0, **options)
        assert out["ci_low"] <= out["discretized_exact_price"] <= out["ci_high"]
        assert abs(out["price"] - analytic.black_scholes_price(market, 100.0, 1.0)) < 1.0
    with pytest.raises(ValueError):
        quantum_price(market, call, 4, "phase_estimation")


def test_same_seed_same_quantum_result(market, call):
    prepared = prepare(market, call, 3)
    a = estimate_price(prepared, "iqae", seed=5, epsilon=0.01)
    b = estimate_price(prepared, "iqae", seed=5, epsilon=0.01)
    assert a["price"] == b["price"] and a["oracle_queries"] == b["oracle_queries"]
    assert a["resources"]["qubits"] == 4
    assert a["resources"]["max_circuit_cx"] >= prepared.a_resources["cx"]


def test_option_that_never_pays_is_priced_at_zero_without_running_circuits(market):
    out = quantum_price(market, Option("european_call", 1e6, 1.0), 3, "iqae", epsilon=0.01)
    assert out["price"] == 0.0 and out["oracle_queries"] == 0


def test_quantum_delta_interval_contains_discretized_delta(market, call):
    out = quantum_delta(market, call, 4, rel_bump=0.05, epsilon=0.005, truncation=4.0)
    assert out["ci_low"] <= out["discretized_exact_delta"] <= out["ci_high"]
    assert 0.4 < out["delta"] < 0.9


def test_config_validation(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text(yaml.safe_dump({"market": {"spot": 1, "rate": 0, "vol": 0.1}}))
    with pytest.raises(ValueError, match="missing required section"):
        load_config(str(bad))


def test_dataset_generation_is_reproducible(tiny_config):
    config = load_config(str(tiny_config))
    first = prepare_data.generate(config)
    with np.load(first["path"]) as data:
        saved = {k: data[k].copy() for k in data.files}
    second = prepare_data.generate(config)
    assert first["content_sha256"] == second["content_sha256"]
    assert set(saved) == {"T1_steps1", "T1_steps2"}
    assert saved["T1_steps2"].shape == (2000, 2) and (saved["T1_steps2"] > 0).all()
    config["data"]["seed"] += 1
    assert prepare_data.generate(config)["content_sha256"] != first["content_sha256"]


def test_full_pipeline_through_the_cli(tiny_config, tmp_path, capsys):
    runs_dir, out = tmp_path / "runs", tmp_path / "reports" / "comparison.html"
    assert prepare_data.main(["--config", str(tiny_config)]) == 0
    for solver in run.SOLVERS:
        assert run.main(["--config", str(tiny_config), "--solver", solver, "--runs-dir", str(runs_dir)]) == 0
    assert report.main(["--runs", str(runs_dir), "--output", str(out), "--config", str(tiny_config)]) == 0

    classical = json.loads((runs_dir / "tiny-classical.json").read_text())
    quantum = json.loads((runs_dir / "tiny-quantum-sim.json").read_text())
    assert classical["dataset"]["status"] == "ok"
    assert len(classical["prices"]) == 3 * 4 * 2            # instruments x methods x seeds
    assert len(quantum["prices"]) == 3 * 3 * 2              # instruments x estimators x seeds
    assert classical["manifest"]["config_sha256"] == quantum["manifest"]["config_sha256"]
    for row in quantum["prices"]:
        assert row["ci_low"] <= row["price"] <= row["ci_high"]
        assert row["oracle_queries"] >= row["shots"] > 0
    call_ref = classical["references"]["call"]
    assert call_ref["kind"] == "analytic" and call_ref["price"] == pytest.approx(10.450584, abs=1e-6)

    page = out.read_text()
    for heading in ("Convergence at equal evaluation counts", "Classical prices", "Quantum-simulated prices",
                    "Circuit resources", "Extrapolated cost at a target error", "Provenance"):
        assert heading in page
    for name in ("classical_prices.csv", "quantum-sim_prices.csv", "quantum_scaling.csv", "figures/convergence.png"):
        assert (out.parent / name).stat().st_size > 0


def test_classical_run_survives_missing_or_corrupt_dataset(tiny_config):
    config = load_config(str(tiny_config))
    assert run.run_classical(config)["dataset"]["status"] == "missing"      # prepare_data never ran
    info = prepare_data.generate(config)
    with open(info["path"], "ab") as handle:
        handle.write(b"tampered")
    result = run.run_classical(config)
    assert result["dataset"]["status"] == "hash_mismatch"
    assert len(result["prices"]) > 0                                         # the rest still ran


def test_report_with_only_one_solver_and_with_no_runs(tiny_config, tmp_path):
    runs_dir = tmp_path / "runs"
    run.main(["--config", str(tiny_config), "--solver", "classical", "--runs-dir", str(runs_dir)])
    (runs_dir / "notes.json").write_text("{not json")                        # stray files are ignored
    out = tmp_path / "r" / "c.html"
    assert report.main(["--runs", str(runs_dir), "--output", str(out)]) == 0
    page = out.read_text()
    assert "No stored run for: quantum-sim" in page and "Quantum-simulated prices" not in page
    with pytest.raises(SystemExit):
        report.main(["--runs", str(tmp_path / "empty"), "--output", str(out)])


def test_cpp_fallback_when_binary_is_absent(market, call, monkeypatch, tmp_path):
    monkeypatch.setattr(run, "CPP_BINARY", tmp_path / "nope")
    assert run._cpp_throughput(market, call, 1000)["status"] == "not_built"
    broken = tmp_path / "broken"
    broken.write_text("#!/bin/sh\nexit 3\n")
    broken.chmod(0o755)
    monkeypatch.setattr(run, "CPP_BINARY", broken)
    assert run._cpp_throughput(market, call, 1000)["status"] == "failed"
