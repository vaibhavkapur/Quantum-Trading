import copy
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from strategy_optimizer.config import file_hash
from strategy_optimizer.data import prepare_data
from strategy_optimizer.experiment import run_experiment
from strategy_optimizer.metrics import block_bootstrap_sharpe, performance, sharpe, stability
from strategy_optimizer.reporting import create_report
from strategy_optimizer.solvers import SOLVERS


def test_drawdown_includes_initial_capital_and_cash():
    result = performance([-.10, .02, .01], trials=7)
    assert result["max_drawdown"] == pytest.approx(-.10)
    assert result["total_return"] == pytest.approx(.9*1.02*1.01-1)
    assert result["bonferroni_pvalue"] == min(1, 7*result["positive_mean_hac_pvalue"])
    cash = performance(np.zeros(20))
    assert cash["sharpe"] == 0
    assert cash["bonferroni_pvalue"] == 1
    assert sharpe([.01]*20) is None
    with pytest.raises(ValueError):
        performance([-.1, -1, .2])


def test_bootstrap_and_stability_are_deterministic():
    returns = np.random.default_rng(1).normal(.001, .01, 50)
    assert block_bootstrap_sharpe(returns) == block_bootstrap_sharpe(returns)
    result = stability([[1, 0, 1], [0, 1, 1]], ["a", "b", "c"])
    assert result["mean_adjacent_jaccard"] == pytest.approx(1/3)
    assert result["selection_frequency"] == {"a": .5, "b": .5, "c": 1}


def test_end_to_end_all_adapters_and_report(config, tmp_path):
    prepare_data(config["data"])
    run_dir, manifest = run_experiment(config, list(SOLVERS))
    assert manifest["status"] == "complete"
    summaries = json.loads((run_dir / "summary.json").read_text())
    assert len(summaries) == 5
    for summary in summaries:
        assert summary["status"] == "ok"
        assert summary["observations"] == 24
        assert summary["evaluations"] <= 64
    daily = pd.read_csv(run_dir / "daily_returns.csv")
    np.testing.assert_allclose(daily.net, daily.gross - daily.costs)
    for case_file in (run_dir / "cases").glob("*.json"):
        case = json.loads(case_file.read_text())
        candidate = case["candidates"][case["selected_candidate"]]
        assert candidate["validation_sharpe"] == max(c["validation_sharpe"] for c in case["candidates"])
        assert sum(case["selected"]) <= config["optimization"]["max_selected"]
    report = create_report(config["output"]["runs"], tmp_path / "report.html")
    assert "Held-out performance" in report.read_text()
    assert len(pd.read_csv(report.with_suffix(".csv"))) == 5
    assert (tmp_path / "report-resources.csv").exists()


def test_future_test_changes_cannot_change_selection(config):
    path = prepare_data(config["data"])
    before_dir, _ = run_experiment(config, ["grid", "tpe"])
    prices = pd.read_csv(path)
    prices.iloc[88:, 1:] *= np.linspace(.5, 1.5, len(prices)-88)[:, None]
    prices.to_csv(path, index=False)
    manifest_path = path.with_suffix(".manifest.json")
    metadata = json.loads(manifest_path.read_text())
    metadata["sha256"] = file_hash(path)  # Re-sign a deliberately changed fixture.
    manifest_path.write_text(json.dumps(metadata))
    after_dir, _ = run_experiment(config, ["grid", "tpe"])
    for file in (before_dir / "cases").glob("*.json"):
        before = json.loads(file.read_text())
        after = json.loads((after_dir / "cases" / file.name).read_text())
        assert before["selected"] == after["selected"]
        assert before["risk_aversion"] == after["risk_aversion"]
        for left, right in zip(before["candidates"], after["candidates"]):
            assert left["validation_sharpe"] == pytest.approx(right["validation_sharpe"])
            np.testing.assert_allclose([r["energy"] for r in left["result"]["trace"]], [r["energy"] for r in right["result"]["trace"]], atol=1e-10)
    with pytest.raises(ValueError, match="Incompatible"):
        create_report(config["output"]["runs"], path.parent / "mixed.html")


def test_failed_quantum_runs_remain_visible(config, tmp_path):
    config["optimization"]["max_qubits"] = 1
    prepare_data(config["data"])
    run_dir, manifest = run_experiment(config, ["qaoa"])
    assert manifest["status"] == "complete_with_failures"
    summary = json.loads((run_dir / "summary.json").read_text())[0]
    assert summary["status"] == "failed"
    assert "sharpe" not in summary
    output = create_report(config["output"]["runs"], tmp_path / "failed.html")
    assert "above configured limit" in output.read_text()
    assert "No complete equity" in output.read_text()


def test_report_rejects_duplicate_runs_and_tampering(config, tmp_path):
    prepare_data(config["data"])
    run_dir, _ = run_experiment(config, ["grid"])
    summary = run_dir / "summary.json"
    original = summary.read_bytes()
    summary.write_bytes(original + b" ")
    with pytest.raises(ValueError, match="checksum"):
        create_report(config["output"]["runs"], tmp_path / "tamper.html")
    summary.write_bytes(original)
    run_experiment(config, ["grid"])
    with pytest.raises(ValueError, match="Duplicate"):
        create_report(config["output"]["runs"], tmp_path / "duplicate.html")


def test_pipeline_failure_is_persisted(config, tmp_path):
    with pytest.raises(FileNotFoundError):
        run_experiment(config, ["grid"])
    manifest = json.loads(next(Path(config["output"]["runs"]).rglob("manifest.json")).read_text())
    assert manifest["status"] == "failed"
    output = create_report(config["output"]["runs"], tmp_path / "failed.html")
    assert "FileNotFoundError" in output.read_text()


def test_cli_contract_with_config_outside_working_directory(config, tmp_path):
    config_path = tmp_path / "settings.yaml"
    config_path.write_text(yaml.safe_dump(config))
    root = Path(__file__).parents[1]
    environment = {**os.environ, "PYTHONPATH": str(root), "PYTHONDONTWRITEBYTECODE": "1"}
    def call(module, *args):
        return subprocess.run([sys.executable, "-m", module, *args], cwd=tmp_path, env=environment,
                              text=True, capture_output=True, timeout=120)
    prepared = call("project.prepare_data", "--config", str(config_path))
    assert prepared.returncode == 0, prepared.stderr
    run = call("project.run", "--config", str(config_path), "--solver", "classical")
    assert run.returncode == 0, run.stderr
    quantum = call("project.run", "--config", str(config_path), "--solver", "quantum-sim")
    assert quantum.returncode == 0, quantum.stderr
    report = call("project.report", "--runs", config["output"]["runs"], "--output", str(tmp_path / "cli.html"))
    assert report.returncode == 0, report.stderr
    assert (tmp_path / "cli.html").exists()

