from __future__ import annotations

import csv
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import platform
import sys
import time
import traceback
import uuid

from .config import digest
from .data import prepare, load_dataset
from .native import build, library
from .optimizer import PolicyService
from .policy import baseline
from .simulation import simulate

SOLVERS = ("fixed-spread", "inventory-skew", "avellaneda-stoikov", "classical", "quantum-sim")


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def write_csv(path, rows, fields=None):
    with Path(path).open("w", newline="") as file:
        names = fields or (list(rows[0]) if rows else [])
        writer = csv.DictWriter(file, fieldnames=names)
        writer.writeheader()
        writer.writerows(rows)


def source_digest():
    root = Path(__file__).parent
    files = sorted([*root.rglob("*.py"), *root.rglob("*.cpp")])
    hasher = hashlib.sha256()
    for path in files:
        hasher.update(str(path.relative_to(root)).encode())
        hasher.update(path.read_bytes())
    return hasher.hexdigest()


def run_experiment(config, solver, runs="runs", data_root="data", dataset=None):
    if solver not in SOLVERS:
        raise ValueError(f"unknown solver {solver}")
    started = time.perf_counter()
    dataset = Path(dataset) if dataset else prepare(config, data_root)
    begin = time.perf_counter()
    manifest, frames_by_case = load_dataset(dataset)
    load_seconds = time.perf_counter()-begin
    expected = {"schema": "mm-l2-v1", "data": config["data"], "seeds": config["seeds"], "scenarios": config["scenarios"]}
    if manifest["spec"] != expected:
        raise ValueError("dataset specification differs from config")
    begin = time.perf_counter()
    native_path, build_seconds = build()
    library()
    initialization_seconds = time.perf_counter()-begin
    compatibility = dict(config_id=digest(config), dataset_id=manifest["dataset_id"], source_id=source_digest())
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    directory = Path(runs) / f"{config['name']}-{solver}-{stamp}-{uuid.uuid4().hex[:8]}"
    directory.mkdir(parents=True)
    environment = dict(python=sys.version, platform=platform.platform(), machine=platform.machine(),
                       dependencies={p: version(p) for p in ("numpy", "scipy", "qiskit", "PyYAML")},
                       native_library=native_path.name, compiler_flags="-std=c++20 -O3 -DNDEBUG -shared -fPIC")
    experiment = dict(schema="mm-run-v1", created_utc=datetime.now(timezone.utc).isoformat(), solver=solver,
                      config=config, compatibility=compatibility, environment=environment, dataset=str(dataset.resolve()),
                      splits=manifest["splits"], data_loading_seconds=load_seconds, native_build_or_cache_seconds=build_seconds,
                      initialization_seconds=initialization_seconds,
                      activation="offline deterministic barriers; bounded background optimizer; quote path never calls QAOA",
                      trials=[dict(scenario=s, seed=seed, status="pending", error="trial has not completed")
                              for s, seed in sorted(frames_by_case)])
    write_json(directory / "manifest.json", experiment)
    train_end = manifest["splits"]["train"][1]
    val_end = manifest["splits"]["validation"][1]
    for trial_index, ((scenario, seed), frames) in enumerate(sorted(frames_by_case.items())):
        trial = directory / scenario / str(seed)
        trial.mkdir(parents=True)
        trial_start = time.perf_counter()
        try:
            train, validation, test = frames[:train_end], frames[train_end:val_end], frames[val_end:]
            records = []
            if solver in ("classical", "quantum-sim"):
                with PolicyService(config, solver, seed) as service:
                    initial = service.update(train, baseline("inventory-skew", config), validation[0]["timestamp_ns"])
                    val_result = simulate(validation, config, initial, history=train)
                    result = simulate(test, config, initial, controller=service, history=frames[:val_end])
                    records = service.records
            else:
                initial = baseline(solver, config)
                val_result = simulate(validation, config, initial, history=train)
                result = simulate(test, config, initial, history=frames[:val_end])
            metrics = result["metrics"]
            metrics.update(solver=solver, scenario=scenario, seed=seed, test_frames=len(test),
                           initial_policy=asdict(initial), policy_updates=len(records),
                           optimizer_fallbacks=sum(r["status"] != "ok" for r in records),
                           policy_update_seconds=sum(r.get("total_seconds", r.get("barrier_wait_seconds", 0)) for r in records),
                           optimizer_barrier_wait_seconds=sum(r.get("barrier_wait_seconds", 0) for r in records),
                           objective_evaluations=sum(r.get("objective_evaluations", 0) for r in records),
                           expectation_evaluations=sum(r.get("expectation_evaluations", 0) for r in records),
                           shots=sum(r.get("shots", 0) for r in records),
                           max_optimality_gap=max((r.get("optimality_gap", 0) for r in records), default=0),
                           end_to_end_seconds=time.perf_counter()-trial_start)
            write_csv(trial / "trajectory.csv", result["trajectory"])
            write_csv(trial / "fills.csv", result["fills"], fields=["timestamp_ns", "order_id", "side", "price", "quantity", "maker", "fee", "frame_index", "reference_mid_ticks", "spread_capture", "adverse_selection_cost"])
            write_json(trial / "metrics.json", metrics)
            write_json(trial / "validation.json", val_result["metrics"])
            write_json(trial / "policies.json", result["policies"])
            write_json(trial / "optimizer.json", records)
            labels = f'solver="{solver}",scenario="{scenario}",seed="{seed}"'
            values = {"net_pnl": metrics["net_pnl"], "quote_ack_p99_seconds": metrics["quote_ack_p99_ns"]*1e-9,
                      "inventory_max": metrics["max_abs_inventory"], "fallbacks_total": metrics["optimizer_fallbacks"],
                      "risk_halted": int(metrics["halted"])}
            (trial / "metrics.prom").write_text("".join(f"# TYPE mm_{k} gauge\nmm_{k}{{{labels}}} {v}\n" for k,v in values.items()))
            record = dict(scenario=scenario, seed=seed, status="ok", directory=str(trial.relative_to(directory)),
                          fallbacks=metrics["optimizer_fallbacks"])
            print(f"{solver:20s} {scenario:17s} seed={seed:3d} pnl={metrics['net_pnl']:+.4f} fallbacks={metrics['optimizer_fallbacks']}", flush=True)
        except Exception as exc:
            record = dict(scenario=scenario, seed=seed, status="failed", error=f"{type(exc).__name__}: {exc}", directory=str(trial.relative_to(directory)))
            (trial / "error.txt").write_text(traceback.format_exc())
            print(f"FAILED {solver}/{scenario}/{seed}: {exc}", flush=True)
        experiment["trials"][trial_index] = record
        write_json(directory / "manifest.json", experiment)
    experiment["full_pipeline_seconds"] = time.perf_counter()-started
    experiment["status"] = "failed" if any(t["status"]=="failed" for t in experiment["trials"]) else "complete"
    write_json(directory / "manifest.json", experiment)
    return directory, experiment
