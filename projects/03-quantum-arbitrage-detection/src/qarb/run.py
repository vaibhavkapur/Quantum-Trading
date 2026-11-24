"""Replay the prepared datasets with a solver group and store the runs.

    python -m qarb.run --config configs/demo.yaml --solver classical
    python -m qarb.run --config configs/demo.yaml --solver quantum-sim

`--solver` is a group from the config (`classical`, `quantum-sim`), `all`, or
a single solver name. One directory is written per (solver, seed):

    <runs_dir>/<config name>/<solver>/seed<seed>/{manifest.json,metrics.json,ticks.csv,opportunities.csv}
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import sys
import time
import traceback
from importlib import metadata
from pathlib import Path

from . import __version__
from .adapters.synthetic import dataset_paths, sha256_file
from .config import config_hash, load_config
from .core.book import read_events
from .metrics import summarize
from .simulation.replay import run_replay
from .solvers import solver_names


def _versions() -> dict:
    out = {"qarb": __version__, "python": platform.python_version(), "platform": platform.platform()}
    for pkg in ("numpy", "scipy", "networkx", "qiskit"):
        try:
            out[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            out[pkg] = None
    return out


def _write_csv(path: Path, rows: list[dict]) -> None:
    fields: list[str] = []
    for r in rows:
        for k in r:
            if k not in fields:
                fields.append(k)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def run_seed(cfg: dict, seed: int, solvers: list[str], runs_root: Path) -> list[dict]:
    """Run every solver on one seed. A failure is recorded, never silently dropped."""
    ev_path, man_path = dataset_paths(cfg, seed)
    base_manifest = {
        "config_name": cfg["name"], "config_hash": config_hash(cfg), "seed": seed, "versions": _versions(),
        "config": cfg, "hardware": "none (CPU simulation only)",
    }
    results = []
    out, error = None, None
    t0 = time.perf_counter()
    try:
        if not ev_path.exists() or not man_path.exists():
            raise FileNotFoundError(f"dataset for seed {seed} not found at {ev_path}; run qarb.prepare_data first")
        data_manifest = json.loads(man_path.read_text())
        digest = sha256_file(ev_path)
        if digest != data_manifest["events_sha256"]:
            raise ValueError(f"{ev_path} does not match the sha256 in its manifest (datasets are immutable)")
        base_manifest["dataset"] = {"events": str(ev_path), "sha256": digest}
        out = run_replay(cfg, seed, solvers, list(read_events(ev_path)), data_manifest)
    except Exception:
        error = traceback.format_exc()
    wall = time.perf_counter() - t0

    for name in solvers:
        run_dir = runs_root / cfg["name"] / name / f"seed{seed}"
        run_dir.mkdir(parents=True, exist_ok=True)
        manifest = dict(base_manifest, solver=name, wall_seconds_all_solvers=wall)
        if error is None:
            st = out["states"][name]
            metrics = summarize(st, out["truth_by_t"], out["decision_times"], data_manifest, cfg)
            metrics.update(seed=seed, config_name=cfg["name"], status="ok")
            manifest["status"] = "ok"
            manifest["quantum_resources"] = list(st.resources.values())
            _write_csv(run_dir / "ticks.csv", st.ticks)
            _write_csv(run_dir / "opportunities.csv", st.opportunities)
        else:
            metrics = {"solver": name, "seed": seed, "config_name": cfg["name"], "status": "failed"}
            manifest["status"] = "failed"
            manifest["error"] = error
        (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
        (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
        results.append(metrics)
    return results


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--solver", required=True, help="solver group (classical, quantum-sim, all) or solver name")
    ap.add_argument("--runs-dir", default=None, help="override runs_dir from the config")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    solvers = solver_names(args.solver, cfg)
    runs_root = Path(args.runs_dir or cfg["runs_dir"])
    failed = 0
    for seed in cfg["seeds"]:
        for m in run_seed(cfg, seed, solvers, runs_root):
            if m["status"] != "ok":
                failed += 1
                print(f"seed {seed} {m['solver']:<14} FAILED (see manifest.json)")
                continue

            def f(key: str, fmt: str = ".2f") -> str:
                return "n/a" if m.get(key) is None else format(m[key], fmt)

            print(
                f"seed {seed} {m['solver']:<14} planted {m['planted_detected']}/{m['planted_total']}  "
                f"search-recall {f('search_recall')}  truth-recall {f('truth_recall')}  FP {m['false_positives_after_costs']:.0f}  "
                f"pnl {m['realised_pnl']:9.2f}  search p50 {m['search_ms_p50']:8.3f} ms  "
                f"evals/decision {m['objective_evals_mean']:.0f}"
            )
    print(f"runs written under {runs_root / cfg['name']}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
