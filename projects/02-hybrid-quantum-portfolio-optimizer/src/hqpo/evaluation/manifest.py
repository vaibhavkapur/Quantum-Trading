"""Machine-readable experiment manifest: what ran, on what, with which versions."""
from __future__ import annotations

import datetime as dt
import hashlib
import importlib.metadata as md
import json
import platform
from pathlib import Path

PACKAGES = ["numpy", "pandas", "scipy", "cvxpy", "clarabel", "ortools", "pyscipopt", "qiskit",
            "qiskit-aer", "qiskit-optimization", "dwave-samplers", "dimod", "plotly"]


def versions() -> dict:
    out = {}
    for p in PACKAGES:
        try:
            out[p] = md.version(p)
        except md.PackageNotFoundError:
            out[p] = None
    return out


def code_hash() -> str:
    """sha256 over the package sources, so a run can be tied to the code that produced it."""
    root = Path(__file__).resolve().parents[1]
    h = hashlib.sha256()
    for f in sorted(root.rglob("*.py")):
        h.update(f.relative_to(root).as_posix().encode())
        h.update(f.read_bytes())
    return h.hexdigest()[:16]


def build_manifest(cfg: dict, kind: str, **extra) -> dict:
    return {
        "kind": kind,
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "config_name": cfg["name"],
        "config_path": cfg.get("_path"),
        "config_hash": cfg.get("_hash"),
        "config": {k: v for k, v in cfg.items() if not k.startswith("_")},
        "code_hash": code_hash(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor() or platform.machine(),
        "packages": versions(),
        "hardware": "CPU only; all quantum results are classical simulation (no QPU)",
        **extra,
    }


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(obj, fh, indent=2, default=_default)
        fh.write("\n")


def _default(o):
    import numpy as np

    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return str(o)
