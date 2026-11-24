import argparse

from qsor.config import load_config
from qsor.experiment import run_experiment
from qsor.optimizer import registry


def main(argv=None):
    names = sorted(registry.ALIASES) + list(registry.SNAPSHOT_SOLVERS) + list(registry.LIVE_SOLVERS)
    ap = argparse.ArgumentParser(description="Run the router simulation for one solver or solver group.")
    ap.add_argument("--config", required=True)
    ap.add_argument("--solver", required=True, help="one of: " + ", ".join(names))
    ap.add_argument("--window", default="test", choices=["train", "validation", "test"])
    args = ap.parse_args(argv)
    registry.expand(args.solver)  # validate before doing any work
    root = run_experiment(load_config(args.config), args.solver, window=args.window)
    print(f"runs written to {root}")


if __name__ == "__main__":
    main()
