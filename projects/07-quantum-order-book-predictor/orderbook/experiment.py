from __future__ import annotations

from datetime import datetime, timezone
from importlib.metadata import version, PackageNotFoundError
from itertools import product
from pathlib import Path
from time import perf_counter
import csv
import hashlib
import json
import platform
import sys
import uuid
import warnings
import numpy as np
from threadpoolctl import threadpool_limits

from .config import Config
from .data import load_book
from .features import features, labels, FEATURE_NAMES
from .splits import walk_forward, stratified_train_sample
from .preprocessing import Reducer
from .models import make_model
from .calibration import TemperatureCalibrator
from .evaluation import classification_metrics, backtest, select_threshold
from .serving import Predictor, profile_stream


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def write_csv(path, rows, fieldnames=None):
    with Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def source_hash():
    root = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    for path in sorted([*root.glob("orderbook/**/*.py"), *root.glob("project/*.py"), root / "pyproject.toml"]):
        digest.update(str(path.relative_to(root)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def environment():
    packages = {}
    for name in ("numpy", "scipy", "scikit-learn", "polars", "pennylane", "torch", "xgboost", "PyYAML"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = "not installed"
    return {"python": sys.version, "platform": platform.platform(), "machine": platform.machine(),
            "packages": packages, "blas_threads": 1, "torch_threads": 1, "xgboost_threads": 1}


def run_experiment(config: Config, solver: str, *, bootstrap_seconds=0.0) -> Path:
    config.validate()
    if solver not in {"classical", "quantum-sim"}:
        raise ValueError("solver must be classical or quantum-sim")
    started = perf_counter()
    begin = perf_counter()
    book, dataset_manifest = load_book(config.data_path, config.levels)
    if len(book) != config.n_snapshots:
        raise ValueError("n_snapshots must match the immutable dataset")
    load_seconds = perf_counter() - begin
    begin = perf_counter()
    x = features(book)
    y, label_end = labels(book, config.horizon, config.label_threshold_bps)
    splits = walk_forward(len(book), config)
    feature_seconds = perf_counter() - begin
    code_hash = source_hash()
    runtime = environment()
    protocol = hashlib.sha256((config.fingerprint + dataset_manifest["sha256"] + code_hash
                              + json.dumps(runtime, sort_keys=True)).encode()).hexdigest()[:16]
    run_id = f"{config.name}-{solver}-{uuid.uuid4().hex[:10]}"
    directory = Path(config.runs_dir) / run_id
    directory.mkdir(parents=True)
    for child in ("predictions", "trades", "latency", "models"):
        (directory / child).mkdir()
    manifest = {
        "schema_version": 1, "run_id": run_id, "protocol_id": protocol, "solver": solver,
        "created_utc": datetime.now(timezone.utc).isoformat(), "status": "running",
        "config": config.to_dict(), "config_sha256_prefix": config.fingerprint,
        "source_sha256": code_hash, "dataset": dataset_manifest, "environment": runtime,
        "features": FEATURE_NAMES, "class_order": ["down", "flat", "up"],
        "splits": [s.to_dict() for s in splits], "results": [],
        "shared_timing": {"bootstrap_seconds": bootstrap_seconds, "data_loading_seconds": load_seconds,
                          "features_labels_splits_seconds": feature_seconds},
        "budget": {"max_training_rows": config.train_samples, "fit_seconds_per_candidate": config.fit_budget_seconds,
                   "candidates_per_model": 1, "native_iteration_cap": config.max_iterations,
                   "enforcement": "cooperative checks plus post-fit rejection; native calls may overrun before rejection",
                   "comparison": "same rows, reduced inputs, horizon, cost model, wall-time allowance; iterations are not equivalent work"},
        "execution": {"unit_size": 1, "delay_snapshots": config.execution_delay,
                      "holding_exit": "signal index + horizon", "fees_per_side_bps": config.fee_bps,
                      "slippage_per_side_bps": config.slippage_bps,
                      "spread": "actual future touch quotes", "overlap": "at most one position"},
        "hardware": False, "notes": ["Synthetic research experiment; no evidence of live profitability or quantum advantage.",
            "Quantum kernel caches exact simulator states. Binomial shots are a CPU measurement model, not hardware queries.",
            "Compilation is eager QNode construction/execution; no QPU compilation, queue or communication.",
            "Training, calibration, threshold tuning and test windows are separate and purged."]}
    manifest_path = directory / "manifest.json"
    write_json(manifest_path, manifest)
    with threadpool_limits(limits=1):
        for split, seed, qubits in product(splits, config.seeds, config.qubits):
            sample = stratified_train_sample(split.train, y, config.train_samples, seed)
            begin = perf_counter()
            reducer = Reducer(qubits).fit(x[sample])
            transformed = {key: reducer.transform(x[index]) for key, index in
                           [("train", sample), ("calibration", split.calibration),
                            ("validation", split.validation), ("test", split.test)]}
            preprocess_seconds = perf_counter() - begin
            variants = [(name, "shared", None) for name in ("majority", "logistic", "xgboost", "mlp")] if solver == "classical" else list(product(
                ("quantum_kernel", "vqc"), config.encodings, config.shots))
            for name, encoding, shots in variants:
                key = f"{name}-q{qubits}-{encoding}-s{shots}-f{split.fold}-seed{seed}"
                row = {"id": key, "model": name, "qubits": qubits, "encoding": encoding,
                       "shots": shots, "fold": split.fold, "seed": seed, "status": "running",
                       "train_indices": sample.tolist(), "train_class_counts": np.bincount(y[sample], minlength=3).tolist(),
                       "timing": {"preprocessing_seconds": preprocess_seconds}, "warnings": []}
                trial_start = perf_counter()
                model = None
                try:
                    with warnings.catch_warnings(record=True) as caught:
                        warnings.simplefilter("always")
                        begin = perf_counter()
                        model = make_model(name, seed=seed, iterations=config.max_iterations, qubits=qubits,
                                           encoding=encoding, shots=shots, layers=config.circuit_layers)
                        row["timing"]["model_initialization_seconds"] = perf_counter() - begin
                        begin = perf_counter()
                        deadline = begin + config.fit_budget_seconds
                        try:
                            model.fit(transformed["train"], y[sample], deadline=deadline)
                        finally:
                            row["timing"]["fit_seconds"] = perf_counter() - begin
                        if perf_counter() > deadline:
                            raise TimeoutError("training completed beyond the predeclared fit allowance")
                        row["training_resources"] = model.stats.copy()
                        begin = perf_counter()
                        cal = TemperatureCalibrator().fit(model.predict_proba(transformed["calibration"]), y[split.calibration])
                        row["temperature"] = cal.temperature
                        row["timing"]["calibration_seconds"] = perf_counter() - begin
                        begin = perf_counter()
                        validation_p = cal.transform(model.predict_proba(transformed["validation"]))
                        threshold, trials = select_threshold(book, split.validation, validation_p, config)
                        row["threshold_trials"] = trials
                        row["timing"]["threshold_selection_seconds"] = perf_counter() - begin
                        begin = perf_counter()
                        raw_p = model.predict_proba(transformed["test"])
                        p = cal.transform(raw_p)
                        row["timing"]["batch_test_inference_seconds"] = perf_counter() - begin
                        row["uncalibrated_metrics"] = classification_metrics(y[split.test], raw_p)
                        row["metrics"] = classification_metrics(y[split.test], p)
                        row["trading"], trades = backtest(book, split.test, p, threshold, config)
                        row["test_resources"] = model.stats.copy()
                        predictor = Predictor(reducer, model, cal, threshold, config.levels)
                        begin = perf_counter()
                        row["latency"], timings = profile_stream(predictor, book, split.test, config.latency_samples)
                        row["timing"]["stream_profile_seconds"] = perf_counter() - begin
                        begin = perf_counter()
                        model_path = directory / "models" / f"{key}.pkl"
                        predictor.save(model_path)
                        row["model_bytes"] = model_path.stat().st_size
                        row["model_path"] = str(model_path.relative_to(directory))
                        predictions = [{"index": int(i), "timestamp_ms": int(book.timestamp_ms[i]),
                                        "label_end_index": int(label_end[i]), "label": int(y[i]),
                                        "p_down": float(prob[0]), "p_flat": float(prob[1]), "p_up": float(prob[2])}
                                       for i, prob in zip(split.test, p)]
                        write_csv(directory / "predictions" / f"{key}.csv", predictions)
                        write_csv(directory / "trades" / f"{key}.csv", trades, fieldnames=[
                            "signal_index", "entry_index", "exit_index", "entry_timestamp_ms", "exit_timestamp_ms",
                            "direction", "entry_price", "exit_price", "gross_bps", "spread_slippage_bps",
                            "fees_bps", "net_bps", "net_quote_currency"])
                        write_csv(directory / "latency" / f"{key}.csv", [{"latency_ms": t} for t in timings])
                        row["timing"]["artifact_writing_seconds"] = perf_counter() - begin
                        begin = perf_counter()
                        Predictor.load(model_path)
                        row["timing"]["artifact_loading_seconds"] = perf_counter() - begin
                        row["resources_including_profile"] = model.stats.copy()
                        row["status"] = "ok"
                    row["warnings"] = sorted(set(str(w.message) for w in caught))
                except Exception as error:
                    row["status"] = "budget_exceeded" if isinstance(error, TimeoutError) else "failed"
                    row["error"] = f"{type(error).__name__}: {error}"
                    if model is not None:
                        row["resources_at_failure"] = model.stats.copy()
                row["timing"]["trial_total_seconds"] = perf_counter() - trial_start + preprocess_seconds
                manifest["results"].append(row)
                write_json(manifest_path, manifest)
                print(json.dumps({"id": key, "status": row["status"], "error": row.get("error")}), flush=True)
    manifest["wall_seconds"] = perf_counter() - started + bootstrap_seconds
    manifest["status"] = "complete" if all(r["status"] == "ok" for r in manifest["results"]) else "completed_with_failures"
    write_json(manifest_path, manifest)
    return manifest_path
