"""Strict experiment configuration and reproducibility helpers."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import yaml


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value) -> None:
    def convert(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.generic):
            return obj.item()
        raise TypeError(type(obj).__name__)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=convert,
                               allow_nan=False) + "\n")


def _integer(value, name, minimum=1):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


def _nonnegative(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and nonnegative")


def validate(config: dict) -> None:
    required = {
        "data": {"path", "seed", "periods", "start", "assets"},
        "strategies": {"windows"},
        "backtest": {"annualization", "train_cost_bps", "evaluation_cost_bps"},
        "walk_forward": {"train", "validation", "test"},
        "optimization": {"max_selected", "risk_aversion", "cardinality_cost", "evaluations",
                         "seeds", "qaoa_layers", "shots_per_evaluation", "max_qubits"},
        "report": {"bootstrap_samples", "block_length"},
        "output": {"runs"},
    }
    if not isinstance(config, dict) or set(config) != set(required):
        raise ValueError(f"Configuration must contain exactly {sorted(required)}")
    for section, fields in required.items():
        if not isinstance(config[section], dict) or set(config[section]) != fields:
            raise ValueError(f"{section} must contain exactly {sorted(fields)}")
    for section, names in {
        "data": ["periods"], "backtest": ["annualization"],
        "walk_forward": ["train", "validation", "test"],
        "optimization": ["max_selected", "evaluations", "qaoa_layers", "shots_per_evaluation", "max_qubits"],
        "report": ["bootstrap_samples", "block_length"],
    }.items():
        for name in names:
            _integer(config[section][name], f"{section}.{name}")
    _integer(config["data"]["seed"], "data.seed", 0)
    assets = config["data"]["assets"]
    if (not isinstance(assets, list) or not assets or
            any(not isinstance(a, str) or not a or a == "date" for a in assets) or len(set(assets)) != len(assets)):
        raise ValueError("data.assets must be unique nonempty names other than 'date'")
    for section, field in [("strategies", "windows"), ("optimization", "seeds")]:
        values = config[section][field]
        if not isinstance(values, list) or not values:
            raise ValueError(f"{field} must be a nonempty list")
        for value in values:
            _integer(value, field, 2 if field == "windows" else 0)
        if len(set(values)) != len(values):
            raise ValueError(f"{field} must be unique")
    opt = config["optimization"]
    risks = opt["risk_aversion"]
    if not isinstance(risks, list) or not risks:
        raise ValueError("risk_aversion must be a nonempty list")
    for risk in risks:
        _nonnegative(risk, "risk_aversion")
    if len(set(risks)) != len(risks):
        raise ValueError("risk_aversion values must be unique")
    _nonnegative(opt["cardinality_cost"], "cardinality_cost")
    for name in ["train_cost_bps", "evaluation_cost_bps"]:
        _nonnegative(config["backtest"][name], name)
    n = len(assets) * len(config["strategies"]["windows"]) * 2
    if not 1 <= opt["max_selected"] <= n <= 12:
        raise ValueError("Require 1 <= max_selected <= strategy count <= 12 for exact verification")
    if opt["max_qubits"] > 16:
        raise ValueError("The local simulator has a hard 16-qubit memory guard")
    if opt["evaluations"] < 4 * opt["shots_per_evaluation"]:
        raise ValueError("Budget must allow at least four QAOA circuit evaluations")
    wf = config["walk_forward"]
    if min(wf.values()) < 3 or wf["train"] <= max(config["strategies"]["windows"]) + 2:
        raise ValueError("Walk-forward windows are too short for the strategy library")
    if config["data"]["periods"] < sum(wf.values()):
        raise ValueError("Not enough observations for a complete walk-forward fold")
    for section, name in [("data", "path"), ("data", "start"), ("output", "runs")]:
        if not isinstance(config[section][name], str) or not config[section][name]:
            raise ValueError(f"{section}.{name} must be a nonempty string")


def load_config(path) -> dict:
    path = Path(path).resolve()
    config = yaml.safe_load(path.read_text())
    validate(config)
    # All file paths are relative to the YAML file, never to the invoking cwd.
    for section, name in [("data", "path"), ("output", "runs")]:
        config[section][name] = str((path.parent / config[section][name]).resolve())
    return config

