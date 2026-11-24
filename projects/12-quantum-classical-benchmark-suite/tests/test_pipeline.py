import copy
import hashlib
import json
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from benchmarks.instances.dataset import prepare
from experiments.runner import atomic_json, run_isolated, run_matrix
from experiments.worker import execute
from metrics.report import read_runs, render_report, summarize
from metrics.statistics import mean_ci95, wilson


def request_for(tmp_path, config, solver="exact"):
    manifest_path = prepare(config, tmp_path)
    manifest = json.loads(manifest_path.read_text())
    entry = next(e for e in manifest["instances"] if e["track"] == "portfolio")
    return {"instance_path": str(manifest_path.parent / entry["file"]), "instance_sha256": entry["sha256"],
            "solver": solver, "seed": 3, "budget": config["budget"], "algorithm": config["algorithm"],
            "success_tolerance": config["success_tolerance"]}


def test_partial_incumbent_and_failure_have_no_fallback(tmp_path, config):
    request = request_for(tmp_path, config)
    request["budget"] = request["budget"] | {"max_evaluations": 1}
    result = execute(request)
    assert result["status"] == "budget_exhausted"
    assert result["quality"]["feasible"]
    assert result["resources"]["objective_evaluations"] == 1
    result = execute(request | {"solver": "nonexistent"})
    assert result["status"] == "failed"
    assert not result["quality"]["success"]
    assert result["quality"]["objective"] is None
    assert "unknown solver" in result["error"]


def test_hard_timeout_terminates_worker(tmp_path, config):
    request = request_for(tmp_path, config, "qaoa")
    request["budget"] = request["budget"] | {"wall_seconds": 0.001}
    result = run_isolated(request)
    assert result["status"] == "timeout"
    assert not result["quality"]["success"]
    assert result["resources"] is None  # Never invent counters for a killed worker.
    assert result["timings"]["wall_seconds"] < 5


def test_integration_matrix_report_and_manifests(tmp_path, config):
    config["instances"] = [{"track": "portfolio", "sizes": [4]}, {"track": "estimation", "sizes": [4]}]
    config["solvers"] = ["milp", "qaoa", "mc-antithetic", "qae"]
    config["warmups"] = 1
    prepare(config, tmp_path)
    directory = run_matrix(config, tmp_path, "all")
    records, sources = read_runs(directory)
    assert len(records) == 12
    assert all(row["status"] == "ok" for row in records)
    assert all(row["timings"]["wall_seconds"] >= row["timings"]["worker_seconds"] for row in records)
    assert all(row["resources"]["peak_rss_bytes"] > 0 for row in records)
    summaries = summarize(records)
    assert len(summaries) == 4
    assert all(s["attempts"] == 2 for s in summaries)
    assert len(sources[0]["manifest"]["planned_runs"]) == 12
    report = render_report(directory, tmp_path / "report" / "comparison.html")
    assert "No claim of quantum advantage" in report.read_text()
    assert "plotly.js" in report.read_text()
    assert pq.read_table(report.with_suffix(".parquet")).num_rows == 12
    with (directory / "results.jsonl").open("a") as out:
        out.write("{}\n")
    with pytest.raises(ValueError, match="checksum"):
        read_runs(directory)


def test_failed_runs_stay_in_denominator_and_one_sample_has_no_ci():
    base = {"warmup": False, "protocol_sha256": "p", "instance_sha256": "i", "instance_id": "i", "track": "portfolio",
            "size": 4, "solver": "milp", "batch_id": "batch", "resources": None}
    rows = [base | {"seed": 1, "status": "ok", "timings": {"wall_seconds": 1.},
                    "quality": {"feasible": True, "absolute_gap": 0., "objective": -2., "success": True}},
            base | {"seed": 2, "status": "timeout", "timings": {"wall_seconds": 2.},
                    "quality": {"feasible": False, "success": False}}]
    summary = summarize(rows)[0]
    assert summary["success_probability"] == 0.5
    assert summary["quality_sample_count"] == 1
    assert summary["quality_mean_ci95"] is None
    assert summary["wall_p50_seconds"] == 1.5
    assert summary["status_counts"] == {"ok": 1, "timeout": 1}
    assert mean_ci95([]) == (None, None)
    assert wilson(0, 1)[1] > 0.5
    assert len(summarize(rows + [rows[0] | {"batch_id": "another"}])) == 2


def test_interrupted_planned_attempts_are_reported_missing(tmp_path):
    manifest = {"batch_id": "b", "protocol_sha256": "p", "status": "interrupted", "planned_runs": [
        {"instance_id": "i", "solver": "milp", "seed": 1, "warmup": False}],
        "dataset_manifest": {"instances": [{"instance_id": "i", "sha256": "i", "size": 4, "track": "portfolio"}]}}
    atomic_json(tmp_path / "manifest.json", manifest)
    (tmp_path / "results.jsonl").write_text("")
    rows, _ = read_runs(tmp_path)
    assert len(rows) == 1 and rows[0]["status"] == "missing"
    result = summarize(rows)[0]
    assert result["attempts"] == 1
    assert result["success_probability"] == 0
    assert result["wall_p50_seconds"] is None


def test_environment_manifest_supports_wheel_layout_without_repo_metadata(tmp_path, monkeypatch):
    from experiments import runner

    (tmp_path / "benchmarks").mkdir()
    (tmp_path / "benchmarks" / "__init__.py").write_text("__version__ = 'test'\n")
    monkeypatch.setattr(runner, "PROJECT_ROOT", tmp_path)
    environment = runner.environment()
    assert "benchmarks/__init__.py" in environment["source_files"]
    assert "pyproject.toml" not in environment["source_files"]
    assert environment["dependencies"]["numpy"] != "missing"
