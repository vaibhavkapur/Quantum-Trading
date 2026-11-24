import json
from pathlib import Path
import subprocess
import sys

import pandas as pd
import pytest

from pairs_trading.data import prepare_data
from pairs_trading.experiment import run_experiment
from pairs_trading.reporting import bootstrap_interval, generate_report


def test_all_solver_pipeline_and_report(config):
    prepare_data(config)
    path = run_experiment(config, "all")
    manifest = json.loads((path / "manifest.json").read_text())
    payload = json.loads((path / "run.json").read_text())
    assert manifest["status"] == "complete"
    assert manifest["successful_fold_count"] == 10
    assert not payload["failures"]
    assert len(payload["summaries"]) == 5
    assert all(len(r["validation"]) == 2 for r in payload["records"])
    assert all(r["train_end"] < r["test_start"] for r in payload["records"])
    assert all(r["metrics"]["max_asset_overlap"] <= 1 for r in payload["records"])
    output = Path(config["_root"]) / "reports/comparison.html"
    generate_report(path.parent, output)
    html = output.read_text()
    assert "Quantum simulation accounting" in html
    assert "95% percentile bootstrap" in html and "<svg" in html
    frame = pd.read_csv(output.with_name("comparison-benchmark.csv"))
    assert len(frame) == 5 and frame["seeds"].max() == 1
    # Repeated invocations must not masquerade as additional independent seeds.
    run_experiment(config, "quantum-sim")
    generate_report(path.parent, output)
    duplicate = pd.read_csv(output.with_name("comparison-benchmark.csv"))
    assert len(duplicate) == 5 and duplicate["seeds"].max() == 1


def test_failed_runs_are_saved_and_rendered(config):
    prepare_data(config)
    config["solver"]["max_qubits"] = 1
    path = run_experiment(config, "quantum-sim")
    manifest = json.loads((path / "manifest.json").read_text())
    payload = json.loads((path / "run.json").read_text())
    assert manifest["status"] == "completed_with_failures"
    assert len(payload["failures"]) == 1 and payload["summaries"] == []
    output = path / "failed.html"
    generate_report(path.parent, output)
    assert "All paths in this comparison failed" in output.read_text()
    assert "qubits" in output.read_text()


def test_fallback_is_not_reported_as_successful_quantum(config):
    prepare_data(config)
    config["solver"].update(max_qubits=1, fallback="greedy")
    path = run_experiment(config, "quantum-sim")
    output = path / "fallback.html"
    generate_report(path.parent, output)
    data = pd.read_csv(output.with_name("fallback-benchmark.csv"))
    assert data.iloc[0]["solver"] == "quantum-sim (fallback)"


def test_cli_contract_help_and_errors():
    root = Path(__file__).parents[1]
    for module in ("prepare_data", "run", "report"):
        result = subprocess.run([sys.executable, "-m", f"project.{module}", "--help"], cwd=root, capture_output=True, text=True)
        assert result.returncode == 0 and "usage:" in result.stdout
    with pytest.raises(ValueError, match="no run.json"):
        generate_report(root / "not-a-run-directory", root / "should-not-exist.html")


def test_seed_bootstrap_is_deterministic():
    assert bootstrap_interval([1, 2, 3]) == bootstrap_interval([1, 2, 3])
    assert bootstrap_interval([2, 2, 2]) == (2, 2)


def test_test_prices_cannot_change_selection_or_validation(config, monkeypatch):
    import pairs_trading.experiment as experiment
    from pairs_trading.data import load_data
    prepare_data(config)
    original_path = run_experiment(config, "top-k")
    original = json.loads((original_path / "run.json").read_text())["records"][0]
    prices, sectors, metadata = load_data(config)
    modified = prices.copy()
    modified.iloc[168:] *= 1.5
    monkeypatch.setattr(experiment, "load_data", lambda _: (modified, sectors, metadata))
    changed_path = run_experiment(config, "top-k")
    changed = json.loads((changed_path / "run.json").read_text())["records"][0]
    assert original["candidates"] == changed["candidates"]
    assert original["selected_pairs"] == changed["selected_pairs"]
    assert original["validation"] == changed["validation"]
    assert original["chosen_entry_z"] == changed["chosen_entry_z"]
