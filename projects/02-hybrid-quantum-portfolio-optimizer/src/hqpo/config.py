"""YAML configuration loading with explicit defaults."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import yaml

from .models.problem import Constraints
from .solvers import DEFAULT_GROUPS, QUBO_SOLVERS, SOLVERS, Budget, get_solver

DEFAULTS = {
    "name": "run",
    "seeds": [11, 23, 37],
    "data": {"source": "synthetic", "path": "data/prices.csv",
             "synthetic": {"n_assets": 6, "n_sectors": 3, "n_days": 1008, "start": "2019-01-02",
                           "seed": 7}},
    "splits": {"train_end": None, "validation_end": None},
    "estimation": {"lookback_days": 252, "covariance": "ledoit_wolf",
                   "expected_return": "shrunk_mean", "mean_shrinkage": 0.5,
                   "annualization": 252},
    "portfolio": {"mode": "long_only", "lots": 4, "max_weight": 0.5, "cardinality": None,
                  "sector_cap": None, "leg_exposure": None, "risk_aversion": 4.0,
                  "turnover_penalty": 0.0},
    "backtest": {"rebalance_every_days": 63, "cost_bps": 10.0},
    "budget": {"wall_time_s": 5.0, "max_evals": 200_000},
    "penalty": {"method": "gradient_bound", "scale": 1.0},
    "solvers": copy.deepcopy(DEFAULT_GROUPS),
    "solver_params": {},
    "benchmark": {},
}


def _merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict) and k not in ("solvers",):
            out[k] = _merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_config(path: str | Path) -> dict:
    path = Path(path)
    with open(path) as fh:
        raw = yaml.safe_load(fh) or {}
    cfg = _merge(DEFAULTS, raw)
    cfg["solvers"] = {**DEFAULT_GROUPS, **(raw.get("solvers") or {})}
    cfg["_path"] = str(path)
    cfg["_root"] = str(path.resolve().parent.parent)   # configs/ lives one level below the root
    cfg["_hash"] = config_hash(raw)
    for group, names in cfg["solvers"].items():
        for n in names:
            if n not in SOLVERS:
                raise ValueError(f"config solver group {group!r} names unknown solver {n!r}")
    return cfg


def config_hash(raw: dict) -> str:
    return hashlib.sha256(json.dumps(raw, sort_keys=True, default=str).encode()).hexdigest()[:16]


def resolve(cfg: dict, rel: str | Path) -> Path:
    p = Path(rel)
    return p if p.is_absolute() else Path(cfg["_root"]) / p


def constraints_from(spec: dict) -> Constraints:
    keys = ("mode", "lots", "max_weight", "cardinality", "sector_cap", "leg_exposure")
    return Constraints(**{k: spec[k] for k in keys if k in spec and spec[k] is not None})


def budget_from(cfg: dict) -> Budget:
    return Budget(float(cfg["budget"]["wall_time_s"]), int(cfg["budget"]["max_evals"]))


def expand_solvers(cfg: dict, selector: str) -> list[str]:
    """`--solver` accepts a group name, a solver name, a comma list, or `all`."""
    if selector == "all":
        names = [n for group in cfg["solvers"].values() for n in group]
    else:
        names = []
        for part in selector.split(","):
            part = part.strip()
            if part in cfg["solvers"]:
                names += cfg["solvers"][part]
            elif part in SOLVERS:
                names.append(part)
            else:
                raise ValueError(f"unknown solver or group {part!r}; groups: "
                                 f"{sorted(cfg['solvers'])}, solvers: {sorted(SOLVERS)}")
    return list(dict.fromkeys(names))


def make_solver(cfg: dict, name: str):
    params = dict(cfg.get("solver_params", {}).get(name, {}) or {})
    if name in QUBO_SOLVERS:
        params.setdefault("penalty", cfg["penalty"])
    return get_solver(name, **params)
