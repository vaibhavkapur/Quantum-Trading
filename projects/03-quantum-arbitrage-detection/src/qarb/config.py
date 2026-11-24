"""Configuration loading with defaults."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import yaml

DEFAULTS: dict = {
    "name": "run",
    "seeds": [1],
    "numeraire": "USD",
    "funding_currencies": ["USDT", "USDC", "EUR"],
    "runs_dir": "runs",
    "data": {
        "dir": "data",
        "currencies": ["USD", "USDT", "BTC", "ETH", "SOL"],
        "venues": [
            {"name": "alpha", "fee_bps": 5.0, "feed_delay_ms": [5, 30], "symbol_style": "dash"},
            {"name": "beta", "fee_bps": 7.5, "feed_delay_ms": [10, 50], "symbol_style": "concat_xbt"},
            {"name": "gamma", "fee_bps": 10.0, "feed_delay_ms": [15, 80], "symbol_style": "underscore_lower"},
        ],
        "market_coverage": 1.0,
        "stable_fee_mult": 0.5,
        "n_ticks": 60,
        "tick_ms": 250,
        "levels": 5,
        "half_spread_bps": 4.0,
        "level_step_bps": 3.0,
        "noise_frac": 0.6,
        "vol_bps_per_tick": 2.0,
        "depth_notional": 15000.0,
        "planted": {"n_profitable": 5, "lengths": [2, 3, 4], "edge_bps": [12.0, 35.0], "duration_ticks": [3, 8]},
        "decoys": {"n_thin": 2, "thin_notional": 25.0, "n_stale": 2, "stale_ticks": 8, "stale_jump_bps": 70.0},
    },
    "graph": {
        "probe_notional": 500.0,
        "stale_timeout_ms": 300,
        "min_cycle_len": 2,
        "max_cycle_len": 4,
        "allowed_venues": None,
        "max_venues_per_cycle": None,
    },
    "sizing": {
        "max_notional": 10000.0,
        "min_profit": 1.0,
        "venue_top_k": 2,
        "inventory_notional": 100000.0,
    },
    "execution": {
        "decision_interval_ms": 250,
        "decision_offset_ms": 200,
        "latency_ms": 40,
        "slippage_bps": 0.5,
    },
    "budget": {"max_objective_evals": 100000, "max_wall_ms": 20000},
    "solvers": {
        "classical": ["enumeration", "bellman_ford", "bounded_dp", "random_search", "qubo_sa", "naive_mid"],
        "quantum-sim": ["qaoa"],
    },
    "qubo": {"penalty": "auto", "anchored": True},
    "qaoa": {
        "layers": 2,
        "maxiter": 30,
        "shots_per_iter": 128,
        "final_shots": 512,
        "restarts": 1,
        "cvar_alpha": 0.3,
        "expectation": "sampled",
        "max_qubits": 16,
        "engine": "numpy",
        "fallback": "bounded_dp",
        "circuit_metrics": True,
    },
    "sa": {"reads": 32, "sweeps": 60},
}


def _merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def validate(cfg: dict) -> None:
    d, g = cfg["data"], cfg["graph"]
    if cfg["numeraire"] not in d["currencies"]:
        raise ValueError("numeraire must be one of data.currencies")
    if len(set(d["currencies"])) != len(d["currencies"]):
        raise ValueError("duplicate currencies")
    if not d["venues"]:
        raise ValueError("at least one venue is required")
    if not 0 <= d["noise_frac"] < 1:
        raise ValueError("data.noise_frac must be in [0, 1) to keep background books arbitrage-free")
    if not 2 <= g["min_cycle_len"] <= g["max_cycle_len"]:
        raise ValueError("need 2 <= graph.min_cycle_len <= graph.max_cycle_len")
    if not cfg["seeds"]:
        raise ValueError("at least one seed must be declared")


def load_config(path: str | Path, overrides: dict | None = None) -> dict:
    path = Path(path)
    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    cfg = _merge(_merge(DEFAULTS, raw), overrides or {})
    if "name" not in raw:
        cfg["name"] = path.stem
    validate(cfg)
    return cfg


def config_hash(cfg: dict) -> str:
    return hashlib.sha256(json.dumps(cfg, sort_keys=True, default=str).encode()).hexdigest()[:16]
