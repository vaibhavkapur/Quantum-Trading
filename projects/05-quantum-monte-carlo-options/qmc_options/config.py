"""Config loading, reference prices and the run manifest."""
from __future__ import annotations

import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import yaml

from . import __version__, analytic, classical_mc
from .models import Market, Option


def load_config(path: str) -> Dict[str, Any]:
    with open(path) as handle:
        config = yaml.safe_load(handle)
    for key in ("market", "instruments", "seeds", "classical", "quantum", "data"):
        if key not in config:
            raise ValueError(f"config {path} is missing required section {key!r}")
    config["_path"] = str(path)
    config["_sha256"] = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    return config


def market_from(config: Dict[str, Any]) -> Market:
    return Market(**config["market"])


def instruments_from(config: Dict[str, Any]) -> List[Tuple[str, Option]]:
    out = []
    for spec in config["instruments"]:
        spec = dict(spec)
        out.append((spec.pop("name"), Option(**spec)))
    return out


def instrument_by_name(config: Dict[str, Any], name: str) -> Option:
    for candidate, option in instruments_from(config):
        if candidate == name:
            return option
    raise ValueError(f"instrument {name!r} is not defined in the config")


def reference_price(config: Dict[str, Any], market: Market, option: Option) -> Dict[str, Any]:
    """Closed form when it exists; otherwise a large randomised-QMC run whose
    own standard error is reported alongside it."""
    exact = analytic.reference_price(market, option)
    if exact is not None:
        return {"price": exact, "stderr": 0.0, "kind": "analytic"}
    ref = config.get("reference", {})
    est = classical_mc.mc_price(market, option, int(ref.get("n_paths", 2**22)), "qmc",
                                seed=int(ref.get("seed", 999)))
    return {"price": est.price, "stderr": est.stderr, "kind": f"rqmc_{est.n_paths}_paths"}


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def manifest(config: Dict[str, Any], solver: str) -> Dict[str, Any]:
    import qiskit
    import scipy
    return {
        "solver": solver,
        "config_path": config["_path"],
        "config_sha256": config["_sha256"],
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "seeds": config["seeds"],
        "package_version": __version__,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "processor": platform.processor() or platform.machine(),
        "versions": {"numpy": np.__version__, "scipy": scipy.__version__, "qiskit": qiskit.__version__},
        "hardware": "CPU only; quantum results are noiseless local simulation, not QPU runs",
    }


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    def default(obj):
        if isinstance(obj, np.generic):
            return obj.item()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        raise TypeError(f"cannot serialise {type(obj)}")

    path.write_text(json.dumps(payload, indent=1, default=default))
