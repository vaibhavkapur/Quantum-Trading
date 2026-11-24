"""CLI: equal-budget solver benchmark on synthetic instances (scaling, penalties, frontier).

    python -m hqpo.benchmark --config configs/demo.yaml
"""
from __future__ import annotations

import argparse
import time

from .config import load_config, resolve
from .data.prices import load_dataset
from .evaluation import benchmark as bm
from .evaluation.backtest import build_problem
from .evaluation.manifest import build_manifest, write_json
from .solvers import preload


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", required=True)
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--only", choices=["grid", "penalty", "frontier", "qubits"], action="append",
                    help="run only the named part(s)")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    out = resolve(cfg, args.runs) / cfg["name"] / "benchmark"
    out.mkdir(parents=True, exist_ok=True)
    parts = args.only or ["grid", "penalty", "frontier", "qubits"]
    import_s = preload()
    t0 = time.perf_counter()

    if "qubits" in parts:
        bm.qubit_growth(cfg).to_csv(out / "qubit_growth.csv", index=False)
        print("qubit growth table written")
    if "grid" in parts:
        print("solver grid:")
        bm.run_solver_grid(cfg).to_csv(out / "results.csv", index=False)
    if "penalty" in parts:
        print("penalty sweep:")
        bm.run_penalty_sweep(cfg).to_csv(out / "penalty_sweep.csv", index=False)
    if "frontier" in parts and cfg["benchmark"].get("frontier"):
        print("efficient frontier (last signal date of the dataset, turnover term off):")
        prices, manifest = load_dataset(resolve(cfg, cfg["data"]["path"]))
        fcfg = {**cfg, "portfolio": {**cfg["portfolio"], "turnover_penalty": 0.0}}
        asof = prices.index[-1]
        bm.run_frontier(cfg, lambda lam: build_problem(
            fcfg, prices, manifest.get("sectors", {}), asof, risk_aversion=lam)
        ).to_csv(out / "frontier.csv", index=False)
    write_json(out / "manifest.json", build_manifest(
        cfg, "benchmark", parts=parts, library_import_s=import_s,
        elapsed_s=time.perf_counter() - t0))
    print(f"benchmark written to {out}  ({time.perf_counter() - t0:.0f}s)")


if __name__ == "__main__":
    main()
