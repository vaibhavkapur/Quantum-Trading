import argparse

from experiments.config import load_config
from experiments.runner import run_matrix


def main():
    parser = argparse.ArgumentParser(description="Run isolated classical or quantum simulator benchmark matrices")
    parser.add_argument("--config", required=True)
    parser.add_argument("--solver", default="all", help="classical, quantum-sim, all, or an adapter name")
    args = parser.parse_args()
    try:
        config, base = load_config(args.config)
        print(run_matrix(config, base, args.solver))
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
