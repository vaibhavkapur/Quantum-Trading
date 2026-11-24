from __future__ import annotations

import copy
import json
import math
from pathlib import Path

import yaml


def load_config(path: str | Path) -> dict:
    path = Path(path).resolve()
    with path.open() as stream:
        cfg = yaml.safe_load(stream)
    if not isinstance(cfg, dict):
        raise ValueError("configuration must be a mapping")
    cfg = copy.deepcopy(cfg)
    cfg["_root"] = str(path.parent.parent)
    cfg["_config_path"] = str(path)
    validate_config(cfg)
    return cfg


def resolve(cfg: dict, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else Path(cfg["_root"]) / path


def validate_config(cfg: dict) -> None:
    required = {"data", "research", "selection", "solver", "execution", "output", "experiment"}
    if not required <= cfg.keys():
        raise ValueError(f"missing configuration sections: {sorted(required - cfg.keys())}")
    for section, keys in {
        "data": ["periods", "sectors", "assets_per_sector"],
        "research": ["train", "validation", "test", "max_candidates"],
        "selection": ["max_pairs", "capital_budget", "asset_limit", "sector_limit"],
        "solver": ["max_evaluations", "qaoa_depth", "shots", "max_qubits"],
        "execution": ["max_holding", "refit_every", "lookback"],
    }.items():
        for key in keys:
            v = cfg[section].get(key)
            if type(v) is not int or v <= 0:
                raise ValueError(f"{section}.{key} must be a positive integer")
    for section in ("research", "selection", "solver", "execution"):
        for key, value in cfg[section].items():
            if isinstance(value, (int, float)) and not isinstance(value, bool) and not math.isfinite(value):
                raise ValueError(f"{section}.{key} must be finite")
    r, e, s = cfg["research"], cfg["execution"], cfg["solver"]
    if type(cfg["data"].get("seed")) is not int or cfg["data"]["seed"] < 0:
        raise ValueError("data.seed must be a nonnegative integer")
    if r["train"] < 40 or e["lookback"] < 40 or r["validation"] < 2 or r["test"] < 2:
        raise ValueError("train/lookback need >=40 rows; validation/test need >=2")
    if cfg["data"]["periods"] < r["train"] + r["validation"] + r["test"]:
        raise ValueError("dataset is too short for a complete train/validation/test fold")
    if not 0 < r["pvalue"] <= 1 or not 0 < r["min_half_life"] <= r["max_half_life"]:
        raise ValueError("invalid cointegration screening bounds")
    if cfg["selection"]["correlation_penalty"] < 0:
        raise ValueError("correlation penalty must be nonnegative")
    if not s["seeds"] or any(type(x) is not int or x < 0 for x in s["seeds"]):
        raise ValueError("seeds must be a nonempty list of nonnegative integers")
    if len(set(s["seeds"])) != len(s["seeds"]):
        raise ValueError("solver seeds must be unique")
    if s["max_evaluations"] < 2 * s["qaoa_depth"] + 2 or s["wall_seconds"] <= 0:
        raise ValueError("insufficient optimizer budget")
    if not 1 <= s["max_qubits"] <= 20:
        raise ValueError("local statevector max_qubits must be between 1 and 20")
    if s["backend"] not in ("numpy", "qiskit") or s["fallback"] not in ("none", "greedy"):
        raise ValueError("unsupported quantum backend or fallback")
    if e["initial_capital"] <= 0 or not 0 < e["max_gross"] <= 1 or not 0 < e["stop_loss_pct"] < 1:
        raise ValueError("invalid capital, gross exposure or stop-loss setting")
    if not e["entry_grid"] or not all(math.isfinite(z) and 0 <= e["exit_z"] < z < e["stop_z"] for z in e["entry_grid"]):
        raise ValueError("thresholds must satisfy 0 <= exit < entry < stop")
    for key in ("commission_bps", "slippage_bps", "borrow_rate", "financing_rate"):
        if e[key] < 0:
            raise ValueError(f"execution.{key} must be nonnegative")
    if any(not math.isfinite(x) or x < 0 for x in cfg["output"]["sensitivity_cost_multipliers"]):
        raise ValueError("sensitivity cost multipliers must be finite and nonnegative")
    offsets = cfg["output"]["sensitivity_entry_offsets"]
    if any(not math.isfinite(o) or any(not e["exit_z"] < z + o < e["stop_z"] for z in e["entry_grid"]) for o in offsets):
        raise ValueError("sensitivity entry offsets violate signal thresholds")


def public_config(cfg: dict) -> dict:
    return {k: v for k, v in cfg.items() if not k.startswith("_")}


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
