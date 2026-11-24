"""Run classical or quantum walk-forward strategy selection."""
import argparse
from time import perf_counter


def main():
    started = perf_counter()
    from strategy_optimizer.config import load_config
    from strategy_optimizer.experiment import run_experiment
    from strategy_optimizer.solvers import CLASSICAL, SOLVERS
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--solver", choices=["classical", "quantum-sim", "all", *SOLVERS], required=True)
    args = parser.parse_args()
    solvers = list(CLASSICAL) if args.solver == "classical" else ["qaoa"] if args.solver == "quantum-sim" else list(SOLVERS) if args.solver == "all" else [args.solver]
    try:
        run_dir, manifest = run_experiment(load_config(args.config), solvers, started)
    except (ValueError, OSError) as error:
        parser.exit(1, f"Experiment failed: {error}\n")
    print(f"{manifest['status']}: {run_dir}")
    return 0 if manifest["status"] == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())

