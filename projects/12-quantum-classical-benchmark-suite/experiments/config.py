from __future__ import annotations

import math
from pathlib import Path

import yaml

from benchmarks.instances.schema import integer

SOLVERS = {"exact", "milp", "greedy", "annealing", "qaoa", "exact-expectation", "mc", "mc-antithetic", "qae"}
DEFAULT_BUDGET = {"wall_seconds": 30.0, "max_evaluations": 256, "max_shots": 4096,
                  "max_oracle_queries": 2048, "mip_nodes": 10000}
DEFAULT_ALGORITHM = {"qaoa_layers": 1, "qaoa_iterations": 24, "shots_per_circuit": 128,
                     "qae_powers": [0, 1, 2, 4], "repair": True}


def validate_config(raw: dict) -> dict:
    if not isinstance(raw, dict) or raw.get("schema_version") != 1:
        raise ValueError("config requires schema_version: 1")
    allowed = {"schema_version", "name", "dataset", "runs", "data_seed", "instances", "seeds",
               "warmups", "budget", "algorithm", "solvers", "success_tolerance"}
    if set(raw) - allowed:
        raise ValueError(f"unknown config keys: {sorted(set(raw) - allowed)}")
    config = dict(raw)
    for key in ("name", "dataset", "runs"):
        if not isinstance(config.get(key), str) or not config[key].strip():
            raise ValueError(f"{key} must be a nonempty string")
    integer(config.get("data_seed"), "data_seed", 0, 2**32 - 1)
    integer(config.get("warmups", 1), "warmups", 0, 10)
    config.setdefault("warmups", 1)
    seeds = config.get("seeds", [])
    if not isinstance(seeds, list) or not seeds or len(seeds) != len(set(seeds)):
        raise ValueError("seeds must be a nonempty list of unique predeclared seeds")
    for seed in seeds:
        integer(seed, "seed", 0, 2**32 - 1)
    instances = config.get("instances")
    if not isinstance(instances, list) or not instances:
        raise ValueError("instances must be a nonempty list")
    seen = set()
    for spec in instances:
        if not isinstance(spec, dict) or set(spec) != {"track", "sizes"}:
            raise ValueError("each instance specification needs only track and sizes")
        if spec["track"] not in {"portfolio", "basket", "estimation"} or not spec["sizes"]:
            raise ValueError("invalid track or empty sizes")
        for size in spec["sizes"]:
            integer(size, "size", 2, 256 if spec["track"] == "estimation" else 16)
            if spec["track"] == "estimation" and size & (size - 1):
                raise ValueError("estimation sizes must be powers of two")
            key = spec["track"], size
            if key in seen:
                raise ValueError("duplicate instance specifications")
            seen.add(key)
    budget = config.get("budget", {})
    if set(budget) - DEFAULT_BUDGET.keys():
        raise ValueError("unknown budget field")
    config["budget"] = DEFAULT_BUDGET | budget
    for key, value in config["budget"].items():
        if key == "wall_seconds":
            if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or value <= 0:
                raise ValueError("wall_seconds must be positive and finite")
        else:
            integer(value, key, 1)
    algorithm = config.get("algorithm", {})
    if set(algorithm) - DEFAULT_ALGORITHM.keys():
        raise ValueError("unknown algorithm field")
    config["algorithm"] = DEFAULT_ALGORITHM | algorithm
    a = config["algorithm"]
    integer(a["qaoa_layers"], "qaoa_layers", 1, 4)
    integer(a["qaoa_iterations"], "qaoa_iterations", 1, 10000)
    integer(a["shots_per_circuit"], "shots_per_circuit", 1)
    if type(a["repair"]) is not bool:
        raise ValueError("repair must be a boolean")
    powers = a["qae_powers"]
    if not isinstance(powers, list) or not powers or powers[0] != 0 or powers != sorted(set(powers)):
        raise ValueError("qae_powers must be increasing, unique, and start with zero")
    for power in powers:
        integer(power, "Grover power", 0, 32)
    solvers = config.get("solvers", sorted(SOLVERS))
    if not isinstance(solvers, list) or not solvers or len(solvers) != len(set(solvers)) or set(solvers) - SOLVERS:
        raise ValueError("solvers must be a unique nonempty list of known adapter names")
    config["solvers"] = solvers
    tolerance = config.get("success_tolerance", 0.05)
    if isinstance(tolerance, bool) or not isinstance(tolerance, (int, float)) or not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError("success_tolerance must be nonnegative and finite")
    config["success_tolerance"] = tolerance
    return config


def load_config(path: str | Path) -> tuple[dict, Path]:
    path = Path(path).resolve()
    config = validate_config(yaml.safe_load(path.read_text()))
    # The conventional configs/ directory is relative to the project, not the caller's cwd.
    base = path.parent.parent if path.parent.name == "configs" else path.parent
    return config, base
