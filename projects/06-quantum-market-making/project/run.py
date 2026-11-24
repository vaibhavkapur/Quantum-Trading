import argparse
from market_making.config import load_config
from market_making.experiment import SOLVERS, run_experiment


def main():
    parser = argparse.ArgumentParser(description="Run classical or QAOA market-making research on CPU")
    parser.add_argument("--config", default="configs/demo.yaml")
    parser.add_argument("--solver", choices=[*SOLVERS, "all"], default="classical")
    parser.add_argument("--runs", default="runs")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--dataset", help="Existing immutable dataset directory, validated against config")
    args = parser.parse_args()
    config = load_config(args.config)
    failed = False
    for solver in SOLVERS if args.solver == "all" else [args.solver]:
        path, manifest = run_experiment(config, solver, args.runs, args.data_root, args.dataset)
        failed |= manifest["status"] == "failed"
        print(path)
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
