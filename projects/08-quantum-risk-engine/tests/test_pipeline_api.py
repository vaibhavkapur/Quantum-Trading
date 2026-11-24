import builtins
import json

from fastapi.testclient import TestClient
import numpy as np
import pytest

from api.app import app
from project.config import ExperimentConfig
from project.data import load_dataset, prepare_dataset
from project.report import build_report
from project.run import execute
from project.workflow import calculate


def test_immutable_dataset_and_integrity(config, tmp_path):
    manifest = prepare_dataset(config, tmp_path)
    content = manifest.read_bytes()
    assert prepare_dataset(config, tmp_path).read_bytes() == content
    returns, metadata = load_dataset(config, tmp_path)
    assert returns.shape == (80, 3)
    data_path = manifest.parent / metadata["file"]
    data_path.write_text(data_path.read_text() + "corrupted")
    with pytest.raises(ValueError, match="checksum"):
        load_dataset(config, tmp_path)


def test_classical_and_quantum_pipeline_report(config, tmp_path):
    pytest.importorskip("qiskit")
    prepare_dataset(config, tmp_path)
    for solver in ("classical", "quantum-sim"):
        path = execute(config, tmp_path, solver)
        result = json.loads(path.read_text())
        assert result["status"] == "ok"
        assert len(result["metrics"]) == (5 if solver == "classical" else 6)
        assert result["fit"]["training_range"] == [0, 40]
        assert result["backtests"][0]["coverage"]["observations"] == 40
        assert result["timings"]["end_to_end_seconds"] >= result["timings"]["compute_seconds"]
    report = build_report(tmp_path / "runs", tmp_path / "report.html")
    assert "quantum_mlae" in report.read_text()
    assert "http://" not in report.read_text()
    assert report.with_suffix(".csv").exists()


def test_explicit_missing_backend_fallback(config, tmp_path, monkeypatch):
    prepare_dataset(config, tmp_path)
    returns, _ = load_dataset(config, tmp_path)
    original_import = builtins.__import__
    def unavailable(name, *args, **kwargs):
        if name == "quantum.estimation":
            raise ImportError("backend deliberately unavailable")
        return original_import(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", unavailable)
    with pytest.raises(RuntimeError, match="unavailable"):
        calculate(config, returns, "quantum-sim")
    config.quantum.fallback = True
    result = calculate(config, returns, "quantum-sim")
    assert result["status"] == "fallback"
    assert result["solver_effective"] == "classical"
    assert result["quantum"] is None
    assert "unavailable" in result["fallback_reason"]


def test_fat_tailed_pipeline_and_config_validation(config, tmp_path):
    config.model.distribution = "student-t"
    prepare_dataset(config, tmp_path)
    returns, _ = load_dataset(config, tmp_path)
    result = calculate(config, returns)
    assert result["reference"]["distribution"] == "student-t"
    assert not any(row["method"] == "importance_sampling" for row in result["metrics"])
    bad = config.model_dump()
    bad["model"]["covariance"][0][0] = -1
    with pytest.raises(ValueError):
        ExperimentConfig.model_validate(bad)


def test_api_endpoints_and_validation(config):
    with TestClient(app) as client:
        assert client.get("/health").json()["status"] == "ok"
        assert "Quantum Risk Engine" in client.get("/").text
        assert client.get("/openapi.json").status_code == 200
        result = client.post("/api/risk/historical", json={"losses": [0, 1, 2, 3], "alpha": .5})
        assert result.json()["cvar"] == 2.5
        assert client.post("/api/risk/historical", json={"losses": [], "alpha": .99}).status_code == 422
        assert client.post("/api/risk/historical", json={"losses": [1, 2], "weights": [1]}).status_code == 422
        assert client.post("/api/risk/gaussian", json={"std_loss": -1}).status_code == 422
        result = client.post("/api/stress", json={"portfolio": config.portfolio.model_dump(), "shocks": config.stress})
        assert result.status_code == 200
        assert len(result.json()) == 3
        assert client.post("/api/backtest", json={"losses": [1, 2, 3], "window": 2}).status_code == 200
        result = client.post("/api/run", json={"config": config.model_dump(), "solver": "classical"})
        assert result.status_code == 200
        assert result.json()["status"] == "ok"
        config.quantum.qubits = 6
        assert client.post("/api/run", json={"config": config.model_dump()}).status_code == 422


def test_repeated_seed_benchmark_preserves_budgets(config, tmp_path):
    pytest.importorskip("qiskit")
    from benchmarks.run import run_benchmark
    prepare_dataset(config, tmp_path)
    result = run_benchmark(config, tmp_path)
    assert len(result["trials"]) == 4
    assert all(row["status"] == "ok" for row in result["trials"])
    assert len({row["oracle_queries"] for row in result["trials"]}) == 1
    assert {row["seed"] for row in result["trials"]} == {11, 23}
    assert all(row["successful"] == 2 for row in result["summary"])
    assert (tmp_path / "reports/benchmark.csv").exists()


def test_report_escapes_metadata_and_empty_runs(config, tmp_path):
    with pytest.raises(ValueError, match="no run"):
        build_report(tmp_path, tmp_path / "report.html")
    prepare_dataset(config, tmp_path)
    path = execute(config, tmp_path, "classical")
    record = json.loads(path.read_text())
    record["experiment"] = "<script>alert(1)</script>"
    path.write_text(json.dumps(record))
    output = build_report(tmp_path / "runs", tmp_path / "report.html")
    assert "<script>alert" not in output.read_text()
