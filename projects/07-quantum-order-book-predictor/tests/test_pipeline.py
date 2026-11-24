from dataclasses import replace
from pathlib import Path
import json
import subprocess
import sys
import numpy as np
import pytest
import yaml
from orderbook.data import prepare_data
from orderbook.experiment import run_experiment
from orderbook.reporting import write_report, summarize, seed_interval


def test_cli_contract_and_paired_experiment(tmp_path, config):
    cfg = replace(config, data_path=str(tmp_path / "book.csv"), runs_dir=str(tmp_path / "runs"))
    config_path = tmp_path / "ci.yaml"
    config_path.write_text(yaml.safe_dump(cfg.to_dict()))
    commands = [["project.prepare_data", "--config", str(config_path)],
                ["project.run", "--config", str(config_path), "--solver", "classical"],
                ["project.run", "--config", str(config_path), "--solver", "quantum-sim"],
                ["project.report", "--runs", cfg.runs_dir, "--output", str(tmp_path / "report.html")]]
    for command in commands:
        result = subprocess.run([sys.executable, "-m", *command], capture_output=True, text=True, timeout=120)
        assert result.returncode == 0, result.stdout + result.stderr
    manifests = [json.loads(p.read_text()) for p in Path(cfg.runs_dir).glob("*/manifest.json")]
    assert len(manifests) == 2
    assert manifests[0]["protocol_id"] == manifests[1]["protocol_id"]
    rows = [r for m in manifests for r in m["results"]]
    assert len(rows) == 6 and all(r["status"] == "ok" for r in rows)
    assert all(r["train_indices"] == rows[0]["train_indices"] for r in rows)
    assert all(r["latency"]["p99_ms"] > 0 for r in rows)
    assert all(r["model_bytes"] > 0 for r in rows)
    assert "Quantum ML" in (tmp_path / "report.html").read_text()
    assert (tmp_path / "report.csv").exists()
    with pytest.raises(ValueError, match="duplicate"):
        summarize([manifests[0], manifests[0]])


def test_failure_and_budget_are_recorded_and_reportable(tmp_path, config, monkeypatch):
    cfg = replace(config, data_path=str(tmp_path / "book.csv"), runs_dir=str(tmp_path / "runs"))
    prepare_data(cfg)
    def broken(name, **kwargs):
        if name == "majority":
            raise TimeoutError("injected budget exhaustion")
        raise ImportError("injected unavailable backend")
    monkeypatch.setattr("orderbook.experiment.make_model", broken)
    manifest = json.loads(run_experiment(cfg, "classical").read_text())
    assert manifest["status"] == "completed_with_failures"
    assert manifest["results"][0]["status"] == "budget_exceeded"
    assert all(r["status"] == "failed" for r in manifest["results"][1:])
    output = write_report(cfg.runs_dir, tmp_path / "failed.html")
    assert "injected unavailable backend" in output.read_text()


def test_confidence_interval_clusters_by_seed():
    rows = [{"seed": seed, "score": value} for seed, value in [(1, 1), (1, 3), (2, 4), (2, 6)]]
    mean, lo, hi = seed_interval(rows, lambda r: r["score"])
    assert mean == 3.5 and lo < mean < hi
    assert seed_interval(rows[:2], lambda r: r["score"]) == (2.0, None, None)
