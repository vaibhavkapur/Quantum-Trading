"""CLI: rolling backtest of one solver group under the config's constraints and budget.

    python -m hqpo.run --config configs/demo.yaml --solver classical
    python -m hqpo.run --config configs/demo.yaml --solver quantum-sim

`--solver` takes a group from the config (classical, quantum-sim, anneal-sim), a solver name,
a comma-separated list, or `all`.  Stochastic solvers run once per predeclared seed.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

from .config import expand_solvers, load_config, resolve
from .data.prices import load_dataset
from .evaluation.backtest import run_backtest
from .evaluation.manifest import build_manifest, write_json
from .solvers import SOLVERS, preload


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", required=True)
    ap.add_argument("--solver", default="classical")
    ap.add_argument("--runs", default="runs", help="output root (default: runs/)")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    prices, data_manifest = load_dataset(resolve(cfg, cfg["data"]["path"]))
    sectors = data_manifest.get("sectors", {})
    import_s = preload()
    root = resolve(cfg, args.runs) / cfg["name"] / "backtest"

    for name in expand_solvers(cfg, args.solver):
        seeds = cfg["seeds"] if SOLVERS[name].stochastic else [None]
        for seed in seeds:
            out = root / name / ("deterministic" if seed is None else f"seed-{seed}")
            t = time.perf_counter()
            res = run_backtest(cfg, prices, sectors, name, seed)
            out.mkdir(parents=True, exist_ok=True)
            res["daily"].to_csv(out / "daily.csv")
            res["rebalances"].to_csv(out / "rebalances.csv", index=False)
            res["weights"].to_csv(out / "weights.csv", index=False)
            write_json(out / "summary.json", res["summary"])
            write_json(out / "manifest.json", build_manifest(
                cfg, "backtest", solver=name, seed=seed, library_import_s=import_s,
                dataset={k: data_manifest[k] for k in ("file", "sha256", "source", "license")},
                elapsed_s=time.perf_counter() - t))
            s = res["summary"].get("test", res["summary"]["all"])
            print(f"{name:12s} {str(seed):>5s}  rebalances={s.get('n_rebalances', 0):2d} "
                  f"no_portfolio={s.get('n_no_portfolio', 0)} repaired={s.get('n_repaired', 0)} "
                  f"test_sharpe={s.get('sharpe', float('nan')):6.2f} "
                  f"solve_time={res['summary']['all'].get('total_wall_time_s', 0):6.1f}s -> "
                  f"{Path(out).relative_to(resolve(cfg, '.'))}")


if __name__ == "__main__":
    main()
