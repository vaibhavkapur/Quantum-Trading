"""Configuration loading. Configs are plain YAML merged over defaults."""
from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass, fields
from pathlib import Path

import yaml

DEFAULTS = {
    "name": "experiment",
    "instrument": {"tick_size": 0.01, "lot_size": 100, "start_price": 100.0},
    "data": {
        "root": "data", "seed": 1, "steps": 20000, "dt_ms": 10, "levels": 4,
        "mid_move_prob": 0.02, "size_ar": 0.97, "size_sigma": 0.5, "depth_decay": 1.3,
        "splits": {"train": 0.5, "validation": 0.2, "test": 0.3},
    },
    "venues": [],
    "orders": {"rate_per_s": 2.0, "sizes_lots": [2, 4, 6, 8], "horizon_ms": 400, "warmup_ms": 500},
    "execution": {
        "limit_ticks": 1, "final_limit_ticks": 3, "participation": 0.6, "max_sweeps": 2,
        "reprice_ms": 100, "cooldown_ms": 200, "heartbeat_ms": 20, "replenish_halflife_ms": 200,
    },
    "optimizer": {
        "refresh_ms": 1000, "delay_model": "fixed", "publish_delay_ms": 50, "timeout_ms": 2000,
        "max_policy_age_ms": 3000, "buckets_lots": [2, 4, 6, 8], "max_units": 4,
        "nonfill_penalty_grid": [1.0, 2.0, 4.0], "calibration_probes": 400, "calibration_seed": 7,
    },
    "solvers": {
        "qaoa": {"layers": 2, "maxiter": 40, "shots": 256, "max_qubits": 16, "backend": "numpy",
                 "warm_start": True},
        "sa": {"sweeps": 60, "restarts": 2, "t_hot": 1.0, "t_cold": 0.02, "max_qubits": 24},
        "random_repair": {"shots": 256, "max_qubits": 24},
        "cpsat": {"scale": 10000},
    },
    "seeds": [11, 23, 37],
    "runs_root": "runs",
}


@dataclass(frozen=True)
class VenueCfg:
    name: str
    half_spread_ticks: int = 1
    widen_on_prob: float = 0.005   # per step: chance a side widens by one tick
    widen_off_prob: float = 0.02   # per step: chance a widened side returns
    depth_lots: float = 6.0        # mean displayed size at the touch
    taker_fee: float = 0.003       # currency per share (negative = rebate)
    maker_fee: float = -0.002
    order_fee: float = 0.0         # currency per accepted child order
    latency_ms: float = 1.0        # one-way order latency
    latency_jitter_ms: float = 0.1
    md_latency_ms: float = 1.0     # market-data delay to the router
    fade_coeff: float = 0.0        # lots withdrawn here per lot filled elsewhere
    fade_delay_ms: float = 1.0     # reaction time of this venue's quoters
    est_fade_coeff: float = 0.0    # router's estimate of fade_coeff
    impact_coeff: float = 0.0      # router's quadratic impact estimate (ticks per lot^2)
    max_child_lots: int = 20
    throttle_max_orders: int = 50
    throttle_window_ms: float = 100.0
    outage_rate_per_min: float = 0.0
    outage_ms: float = 500.0
    flow_rate: float = 20.0        # exogenous aggressor lots per second per side


def _merge(base, over):
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_config(path) -> dict:
    path = Path(path)
    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    parent = raw.pop("extends", None)
    if parent:  # single-level inheritance: lists (e.g. venues) are replaced, dicts merged
        with open(path.parent / parent) as f:
            raw = _merge(yaml.safe_load(f) or {}, raw)
    cfg = _merge(DEFAULTS, raw)
    cfg["_path"] = str(path)
    cfg["_base_dir"] = str(path.resolve().parent.parent)
    validate(cfg)
    return cfg


def validate(cfg: dict) -> None:
    if not cfg["venues"]:
        raise ValueError("config must define at least one venue")
    known = {f.name for f in fields(VenueCfg)}
    for v in cfg["venues"]:
        extra = set(v) - known
        if extra:
            raise ValueError(f"unknown venue keys: {sorted(extra)}")
    s = cfg["data"]["splits"]
    if abs(sum(s.values()) - 1.0) > 1e-9:
        raise ValueError("data.splits must sum to 1")
    if cfg["optimizer"]["delay_model"] not in ("fixed", "measured"):
        raise ValueError("optimizer.delay_model must be 'fixed' or 'measured'")


def venues(cfg: dict):
    return [VenueCfg(**v) for v in cfg["venues"]]


def _public(cfg: dict) -> dict:
    return {k: v for k, v in cfg.items() if not k.startswith("_")}


def config_hash(cfg: dict, keys=None) -> str:
    pub = _public(cfg)
    if keys is not None:
        pub = {k: pub[k] for k in keys}
    return hashlib.sha256(json.dumps(pub, sort_keys=True).encode()).hexdigest()[:10]


def public_config(cfg: dict) -> dict:
    return copy.deepcopy(_public(cfg))


def base_dir(cfg: dict) -> Path:
    return Path(cfg.get("_base_dir", "."))
