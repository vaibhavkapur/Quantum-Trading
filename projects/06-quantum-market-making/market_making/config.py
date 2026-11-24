from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
import yaml


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def merge(base, override):
    result = copy.deepcopy(base)
    for key, value in override.items():
        result[key] = merge(result[key], value) if isinstance(value, dict) and isinstance(result.get(key), dict) else value
    return result


def load_config(path, seen=None):
    path = Path(path).resolve()
    seen = set() if seen is None else seen
    if path in seen:
        raise ValueError("cyclic config extends")
    seen.add(path)
    with path.open() as file:
        config = yaml.safe_load(file)
    if not isinstance(config, dict):
        raise ValueError("config must be a mapping")
    parent = config.pop("extends", None)
    if parent:
        config = merge(load_config(path.parent / parent, seen), config)
    validate(config)
    return config


def validate(c):
    def positive(value, name, integer=False):
        if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or value <= 0 or (integer and not isinstance(value, int)):
            raise ValueError(f"{name} must be a positive {'integer' if integer else 'finite number'}")
    for section in ("data", "execution", "optimizer"):
        if not isinstance(c.get(section), dict):
            raise ValueError(f"missing {section} mapping")
    d, e, o = c["data"], c["execution"], c["optimizer"]
    for key in ("steps", "interval_ns", "initial_mid_ticks", "depth_levels", "depth_qty"):
        positive(d[key], key, True)
    if d["steps"] < 30 or d["steps"] > 1000000 or d["initial_mid_ticks"] < 100 or d["initial_mid_ticks"] > 999000000:
        raise ValueError("steps must be 30..1000000 and initial price 100..999000000 ticks")
    if not 0 < d["train_fraction"] < 1 or not 0 < d["validation_fraction"] < 1 or d["train_fraction"] + d["validation_fraction"] >= 0.95:
        raise ValueError("invalid chronological splits")
    if min(int(d["steps"] * d["train_fraction"]), int(d["steps"] * d["validation_fraction"])) < 5:
        raise ValueError("training and validation windows need at least five frames")
    for key in ("tick_value", "max_loss", "max_drawdown", "max_volatility_ticks"):
        positive(e[key], key)
    for key in ("quote_size", "max_inventory", "adverse_horizon"):
        positive(e[key], key, True)
    if e["quote_size"] > e["max_inventory"] or e["max_inventory"] > 1000000:
        raise ValueError("invalid inventory/quote size")
    for key in ("maker_fee_bps", "taker_fee_bps", "cancel_cost"):
        if not math.isfinite(e[key]) or (key == "cancel_cost" and e[key] < 0):
            raise ValueError(f"invalid {key}")
    for key in ("spreads", "skews", "inventory_bands"):
        if not o[key] or len(set(o[key])) != len(o[key]):
            raise ValueError(f"{key} must be nonempty and unique")
        for x in o[key]:
            if key == "skews":
                if not math.isfinite(x) or x < 0:
                    raise ValueError("skews must be finite and nonnegative")
            else:
                positive(x, key, True)
    if max(o["inventory_bands"]) > e["max_inventory"] or min(o["inventory_bands"]) < e["quote_size"]:
        raise ValueError("policy bands must be between quote size and hard inventory limit")
    if max(o["spreads"]) > 1000000 or max(o["skews"]) > 1000000:
        raise ValueError("policy offsets exceed native numeric bounds")
    for key in ("update_interval", "rolling_window", "max_evaluations", "qaoa_layers", "shots"):
        positive(o[key], key, True)
    positive(o["timeout_seconds"], "timeout_seconds")
    for key in ("inventory_penalty", "adverse_penalty"):
        if not math.isfinite(o[key]) or o[key] < 0:
            raise ValueError(f"invalid {key}")
    count = len(o["spreads"]) * len(o["skews"]) * len(o["inventory_bands"])
    if count > min(256, o["max_evaluations"]) or o["rolling_window"] < 5 or o["qaoa_layers"] > 4:
        raise ValueError("grid must fit objective budget and 8 qubits; window >=5; layers <=4")
    if not c["seeds"] or len(set(c["seeds"])) != len(c["seeds"]) or any(type(s) is not int or s < 0 for s in [d["seed"], *c["seeds"]]):
        raise ValueError("seeds must be unique nonnegative integers")
    allowed = {"normal", "volatile", "trend", "liquidity_crisis"}
    if not c["scenarios"] or len(set(c["scenarios"])) != len(c["scenarios"]) or set(c["scenarios"]) - allowed:
        raise ValueError("unknown or duplicate scenarios")
    if not isinstance(c["name"], str) or not c["name"] or any(x not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for x in c["name"]):
        raise ValueError("name must be a safe filename")
