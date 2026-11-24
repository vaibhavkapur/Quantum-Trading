"""CLI: run the walk-forward study for one solver family and store an immutable run.

    python -m project.run --config configs/demo.yaml --solver classical
    python -m project.run --config configs/demo.yaml --solver quantum-sim
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

from .config import experiment_fingerprint, load_config, resolve
from .data.loader import load_dataset
from .research.walkforward import run_study

PACKAGES = ["numpy", "pandas", "scipy", "statsmodels", "pyarrow", "qiskit", "ortools"]


def _versions() -> dict:
    out = {}
    for pkg in PACKAGES:
        try:
            out[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            out[pkg] = None
    return out


def execute(cfg: dict, family: str, runs_dir: Path, log=print) -> Path:
    if family == "classical":
        solvers, kwargs = list(cfg["solvers"]["classical"]), {}
    elif family == "quantum-sim":
        solvers, kwargs = ["qaoa"], {"qaoa": dict(cfg["solvers"]["quantum"])}
    else:
        raise ValueError(f"unknown solver family: {family}")

    ds = load_dataset(resolve(cfg, cfg["data"]["dir"]))
    started = datetime.now(timezone.utc)
    t0 = time.perf_counter()
    results, returns = run_study(ds, cfg, solvers, kwargs, log=log)
    elapsed = time.perf_counter() - t0

    run_id = f"{cfg['name']}-{family}-{started.strftime('%Y%m%dT%H%M%SZ')}"
    out = runs_dir / run_id
    out.mkdir(parents=True, exist_ok=False)
    results.insert(0, "run_id", run_id)
    results.to_csv(out / "results.csv", index=False)
    results.to_parquet(out / "results.parquet", index=False)
    returns.to_parquet(out / "returns.parquet", index=False)
    manifest = {
        "run_id": run_id,
        "created_utc": started.isoformat(),
        "solver_family": family,
        "solvers": solvers,
        "solver_params": kwargs,
        "fingerprint": experiment_fingerprint(cfg),
        "config": {k: v for k, v in cfg.items() if not k.startswith("_")},
        "config_path": cfg.get("_path"),
        "dataset": {"version": ds.manifest.get("version"), "files": ds.manifest.get("files"), "license": ds.manifest.get("license")},
        "seeds": cfg["seeds"],
        "budget": cfg["budget"],
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "machine": platform.machine(),
            "packages": _versions(),
        },
        "quantum_hardware": "none - noiseless local statevector simulation" if family == "quantum-sim" else "n/a",
        "elapsed_s": round(elapsed, 3),
        "n_rows": int(len(results)),
        "n_failed": int(results["failed"].sum()),
        "files": ["results.csv", "results.parquet", "returns.parquet"],
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    log(f"run {run_id}: {len(results)} rows, {manifest['n_failed']} failed, {elapsed:.1f}s -> {out}")
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--solver", required=True, choices=["classical", "quantum-sim"])
    ap.add_argument("--runs", type=Path, default=None, help="output directory (default: <project>/runs)")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    execute(cfg, args.solver, args.runs or resolve(cfg, "runs"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
