"""Configuration loading: YAML deep-merged over documented defaults."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

DEFAULTS: dict[str, Any] = {
    "name": "default",
    "data": {
        "source": "synthetic",
        "dir": "data/demo",
        "seed": 20240101,
        "n_days": 1700,
        "start": "2017-01-03",
        "path": None,          # for source: file
        "assets_path": None,   # for source: file
        "timezone": "UTC",     # timezone of naive timestamps in source files
        "l2": {"n_events": 400000, "seed": 7},
    },
    "walkforward": {"train_days": 378, "val_days": 126, "test_days": 126, "step_days": 126},
    "selection": {
        "basket_size": 4,
        "max_per_sector": 2,
        "penalty_scale": 1.5,
        "weights": {
            "coint": 1.0,
            "stability": 0.5,
            "resid_var": 0.5,
            "half_life": 0.25,
            "cost": 0.1,
            "beta_dispersion": 0.25,
            "sector": 0.1,
        },
    },
    "hedge": {"method": "johansen"},
    "gates": {"val_adf_pvalue": 0.10, "max_half_life": 60, "max_asset_weight": 0.70},
    "signal": {
        "entry_z": 2.0,
        "exit_z": 0.5,
        "stop_z": 4.0,
        "max_holding_halflives": 4.0,
        "cooldown_bars": 5,
    },
    "risk": {
        "capital": 1_000_000,
        "gross_leverage": 1.0,
        "max_asset_notional_frac": 0.70,
        "kill_switch_drawdown": 0.08,
    },
    "execution": {
        "latency_bars": 1,
        "fee_bps": 0.5,
        "book_levels": 5,
        "level_step": 0.5,
        "depth_frac_adv": 0.01,
        "beyond_book_bps": 25,
        "delist_haircut": 0.10,
    },
    "budget": {"max_evals": 64, "time_limit_s": 5.0},
    "solvers": {
        "classical": ["brute_force", "greedy", "cpsat", "random_search", "simulated_annealing"],
        "quantum": {"backend": "qiskit", "p": 2, "shots": 256, "maxiter": 40, "cvar_alpha": 0.5},
    },
    "seeds": [11, 23, 37, 41, 53],
}


def deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def load_config(path: str | Path) -> dict:
    path = Path(path)
    with path.open() as fh:
        user = yaml.safe_load(fh) or {}
    cfg = deep_merge(DEFAULTS, user)
    validate(cfg)
    cfg["_path"] = str(path)
    cfg["_root"] = str(project_root(path))
    return cfg


def project_root(config_path: Path) -> Path:
    """Relative paths in a config resolve against the directory holding configs/."""
    parent = config_path.resolve().parent
    return parent.parent if parent.name == "configs" else parent


def resolve(cfg: dict, rel: str) -> Path:
    p = Path(rel)
    return p if p.is_absolute() else Path(cfg.get("_root", ".")) / p


def validate(cfg: dict) -> None:
    sel, wf = cfg["selection"], cfg["walkforward"]
    if sel["basket_size"] < 2:
        raise ValueError("selection.basket_size must be >= 2")
    if sel["max_per_sector"] < 1:
        raise ValueError("selection.max_per_sector must be >= 1")
    for key in ("train_days", "val_days", "test_days", "step_days"):
        if wf[key] <= 0:
            raise ValueError(f"walkforward.{key} must be positive")
    if cfg["budget"]["max_evals"] < 1:
        raise ValueError("budget.max_evals must be >= 1")
    if not cfg["seeds"]:
        raise ValueError("at least one seed must be declared")
    sig = cfg["signal"]
    if not (sig["exit_z"] < sig["entry_z"] < sig["stop_z"]):
        raise ValueError("signal thresholds must satisfy exit_z < entry_z < stop_z")


def experiment_fingerprint(cfg: dict) -> str:
    """Hash of everything that must match for two runs to be comparable.

    Solver lists and solver hyper-parameters are excluded on purpose: runs with
    different solvers are exactly what the report compares.
    """
    keys = ["data", "walkforward", "selection", "hedge", "gates", "signal", "risk", "execution", "budget", "seeds"]
    payload = {k: cfg[k] for k in keys}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()[:16]
