"""One-command CPU reproduction of data, solver comparisons and HTML reports."""
import argparse
from benchmarks.run import run_benchmarks
from rebalancing.data import load_config, prepare


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/demo.yaml")
    parser.add_argument("--output", default="benchmarks/results/reproduction")
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    prepare(load_config(args.config))
    run_benchmarks(args.config, args.output, quick=args.quick)


if __name__ == "__main__":
    main()

