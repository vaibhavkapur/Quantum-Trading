from time import perf_counter
START = perf_counter()
import argparse
import json
from orderbook.config import load_config
from orderbook.experiment import run_experiment


def main():
    parser = argparse.ArgumentParser(description="Run matched classical or quantum simulator experiments")
    parser.add_argument("--config", required=True)
    parser.add_argument("--solver", choices=["classical", "quantum-sim"], required=True)
    args = parser.parse_args()
    path = run_experiment(load_config(args.config), args.solver, bootstrap_seconds=perf_counter()-START)
    print(path)
    return 0 if json.loads(path.read_text())["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
