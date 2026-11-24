import argparse
import json
from time import perf_counter
from rebalancing.data import load_config, load_portfolio
from rebalancing.engine import SOLVERS, plan
from rebalancing.solvers import Budget
from rebalancing.storage import RunStore


def run_config(config_path: str, solver: str, *, seed: int | None = None, fallback: str | None = None) -> dict:
    start = perf_counter()
    config = load_config(config_path)
    portfolio = load_portfolio(config["dataset"])
    load_s = perf_counter() - start
    budget = {**config.get("budget", {})}
    if seed is not None:
        budget["seed"] = seed
    result = plan(portfolio, solver, Budget(**budget), fallback=fallback)
    result["timings"]["data_loading_s"] = load_s
    result["timings"]["pre_persistence_total_s"] = perf_counter() - start
    t = perf_counter()
    run_id = RunStore(config["runs"]).save(result)
    persistence_s = perf_counter() - t
    return {"run_id": run_id, "recommendation_id": result["recommendation_id"], "runs_directory": config["runs"],
            "solver_used": result["solver_used"], "fallback_reason": result["fallback_reason"], "status": result["status"],
            "objective": result["report"]["objective"], "feasible": result["report"]["feasible"],
            "fees_cents": result["report"]["fees_cents"], "taxes_cents": result["report"]["taxes_cents"],
            "cash_after_cents": result["report"]["cash_after_cents"], "orders": result["report"]["simulated_orders"],
            "timings": {**result["timings"], "persistence_s": persistence_s, "end_to_end_s": perf_counter() - start}}


def main():
    parser = argparse.ArgumentParser(description="Plan and audit simulated integer-lot orders")
    parser.add_argument("--config", default="configs/demo.yaml")
    parser.add_argument("--solver", choices=list(SOLVERS), default="classical")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--fallback", choices=["classical"], default=None)
    args = parser.parse_args()
    try:
        result = run_config(args.config, args.solver, seed=args.seed, fallback=args.fallback)
    except (ValueError, RuntimeError, ImportError, OSError) as exc:
        parser.exit(1, f"{type(exc).__name__}: {exc}\n")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()

