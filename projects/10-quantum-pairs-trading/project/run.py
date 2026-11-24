import argparse
import json

from pairs_trading.config import load_config
from pairs_trading.experiment import SOLVERS, run_experiment


def main():
    parser = argparse.ArgumentParser(description="Run reproducible, out-of-sample pair-selection benchmarks")
    parser.add_argument("--config", required=True)
    parser.add_argument("--solver", choices=list(SOLVERS) + ["top-k", "greedy", "milp", "annealing"], required=True)
    args = parser.parse_args()
    path = run_experiment(load_config(args.config), args.solver)
    manifest = json.loads((path / "manifest.json").read_text())
    print(f"{path} ({manifest['status']}; {manifest['successful_fold_count']} folds; {manifest['failure_count']} failures)")
    if manifest["failure_count"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
