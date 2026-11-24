from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import random
import subprocess
import sys
import tempfile
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from benchmarks.adapters import compatible
from benchmarks.instances.dataset import load_manifest
from benchmarks.instances.schema import canonical_json, digest

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def environment() -> dict:
    versions = {}
    for package in ("numpy", "scipy", "qiskit", "PyYAML", "plotly", "pyarrow"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "missing"
    source = {}
    for folder in ("benchmarks", "experiments", "metrics", "project"):
        for path in sorted((PROJECT_ROOT / folder).rglob("*.py")):
            source[str(path.relative_to(PROJECT_ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    for name in ("pyproject.toml", "requirements.txt"):
        # Wheel installs retain package sources, but need not include repository metadata.
        path = PROJECT_ROOT / name
        if path.is_file():
            source[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return {"python": sys.version, "platform": platform.platform(), "machine": platform.machine(),
            "cpu": platform.processor(), "logical_cpus": os.cpu_count(), "dependencies": versions,
            "source_files": source, "source_sha256": digest(source),
            "thread_environment": {"OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}}


def run_isolated(request: dict) -> dict:
    """Parent deadline includes process launch, imports, complete solve, validation and IPC."""
    started = time.perf_counter()
    env = os.environ.copy()
    env.update({"PYTHONHASHSEED": str(request["seed"]), "OMP_NUM_THREADS": "1",
                "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"})
    env["PYTHONPATH"] = str(PROJECT_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    result = None
    with tempfile.TemporaryDirectory(prefix="quantum-benchmark-") as temporary:
        request_path, response_path = Path(temporary) / "request.json", Path(temporary) / "response.json"
        request_path.write_text(canonical_json(request))
        process = subprocess.Popen([sys.executable, "-m", "experiments.worker", str(request_path), str(response_path)],
                                   cwd=PROJECT_ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        try:
            stdout, stderr = process.communicate(timeout=max(0.001, request["budget"]["wall_seconds"] - (time.perf_counter() - started)))
            if process.returncode != 0 or not response_path.exists():
                status, error = "failed", f"worker exit {process.returncode}: {stderr[-3000:]}"
            else:
                try:
                    result = json.loads(response_path.read_text())
                except (OSError, ValueError) as exc:
                    status, error = "failed", f"invalid worker response: {exc}"
        except subprocess.TimeoutExpired:
            process.kill()
            process.communicate()
            status, error = "timeout", "hard wall-time deadline exceeded; process terminated, no fallback"
        except BaseException:
            process.kill()
            process.communicate()
            raise
    elapsed = time.perf_counter() - started
    # Never accept a successful response that arrived after the end-to-end deadline.
    if elapsed > request["budget"]["wall_seconds"]:
        result = None
        status, error = "timeout", "end-to-end wall-time deadline exceeded"
    if result is None:
        result = {"status": status, "error": error, "outcome": {}, "quality": {"feasible": False, "success": False},
                  "resources": None, "timings": {"worker_seconds": None}}
    result["timings"]["wall_seconds"] = elapsed
    worker = result["timings"].get("worker_seconds")
    result["timings"]["startup_ipc_seconds"] = max(0., elapsed - worker) if worker is not None else None
    return result


def select_solvers(config: dict, selector: str) -> list[str]:
    groups = {"classical": {"exact", "milp", "greedy", "annealing", "exact-expectation", "mc", "mc-antithetic"},
              "quantum-sim": {"qaoa", "qae"}, "all": set(config["solvers"])}
    selected = [s for s in config["solvers"] if s in groups.get(selector, {selector})]
    if not selected:
        raise ValueError(f"solver selection {selector!r} is not present in config")
    return selected


def atomic_json(path: Path, value: dict) -> None:
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def run_matrix(config: dict, base: Path, selector: str) -> Path:
    campaign_start = time.perf_counter()
    manifest_path = base / config["dataset"] / "manifest.json"
    validation_start = time.perf_counter()
    dataset = load_manifest(manifest_path, config)
    dataset_validation_seconds = time.perf_counter() - validation_start
    solvers = select_solvers(config, selector)
    batch_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:10]
    destination = base / config["runs"] / batch_id
    destination.mkdir(parents=True, exist_ok=False)
    env = environment()
    # Excludes run/data locations and group selector; same protocol joins classical and simulator runs.
    protocol = {k: v for k, v in config.items() if k not in {"name", "dataset", "runs", "solvers"}}
    protocol.update(dataset_sha256=digest(dataset), source_sha256=env["source_sha256"],
                    environment_sha256=digest({k: v for k, v in env.items() if k != "source_files"}))
    protocol_hash = digest(protocol)
    jobs, warmups = [], []
    for entry in dataset["instances"]:
        for solver in solvers:
            if not compatible(solver, entry["track"]):
                continue
            for seed in config["seeds"]:
                jobs.append({"entry": entry, "solver": solver, "seed": seed, "warmup": False})
            for index in range(config["warmups"]):
                warmups.append({"entry": entry, "solver": solver, "seed": (config["seeds"][0] + index) % 2**32, "warmup": True})
    if not jobs:
        raise ValueError("selected solver has no compatible problem instances")
    random.Random(config["data_seed"]).shuffle(jobs)
    planned = warmups + jobs
    manifest = {"schema_version": 1, "batch_id": batch_id, "status": "running", "created_at": now(),
                "config": config, "protocol_sha256": protocol_hash, "dataset_manifest": dataset, "environment": env,
                "selected_solvers": solvers, "dataset_validation_seconds": dataset_validation_seconds,
                "protocol": {"comparison_budget": "hard end-to-end wall seconds; other work units reported separately",
                             "warmup_policy": "isolated discarded attempts before measurements; measurements include cold process startup",
                             "execution_order": "predeclared seeded shuffle after warmups",
                             "repair": "common deterministic linear-score repair, separately timed and labeled",
                             "fallback_policy": "none; errors and timeouts retained", "hardware": "CPU-only; no QPU submission",
                             "reported_wall_excludes": "campaign preflight, warmups, append-only result persistence, report generation; campaign overhead recorded"},
                "planned_runs": [{"instance_id": j["entry"]["instance_id"], "solver": j["solver"],
                                  "seed": j["seed"], "warmup": j["warmup"]} for j in planned]}
    atomic_json(destination / "manifest.json", manifest)
    records_path = destination / "results.jsonl"
    measured = 0
    try:
        with records_path.open("x") as output:
            for index, job in enumerate(planned):
                entry = job["entry"]
                request = {"instance_path": str(manifest_path.parent / entry["file"]),
                           "instance_sha256": entry["sha256"], "solver": job["solver"], "seed": job["seed"],
                           "algorithm": config["algorithm"], "budget": config["budget"],
                           "success_tolerance": config["success_tolerance"]}
                result = run_isolated(request)
                record = {"schema_version": 1, "run_id": f"{batch_id}-{index}", "batch_id": batch_id,
                          "protocol_sha256": protocol_hash, "dataset_sha256": digest(dataset), "instance_sha256": entry["sha256"],
                          "instance_id": entry["instance_id"], "track": entry["track"], "size": entry["size"],
                          "solver": job["solver"], "seed": job["seed"], "warmup": job["warmup"],
                          "budget": config["budget"], "completed_at": now(), **result}
                output.write(canonical_json(record) + "\n")
                output.flush()
                os.fsync(output.fileno())
                measured += not job["warmup"]
                print(f"[{index + 1}/{len(planned)}] {entry['instance_id']} {job['solver']} seed={job['seed']} "
                      f"{result['status']} {result['timings']['wall_seconds']:.3f}s"
                      + (" (warmup)" if job["warmup"] else ""), flush=True)
        manifest["status"] = "complete"
    except BaseException:
        manifest["status"] = "interrupted"
        raise
    finally:
        manifest.update(completed_at=now(), measured_runs=measured,
                        campaign_wall_seconds=time.perf_counter() - campaign_start)
        if records_path.exists():
            manifest["results_sha256"] = hashlib.sha256(records_path.read_bytes()).hexdigest()
        atomic_json(destination / "manifest.json", manifest)
    return destination
