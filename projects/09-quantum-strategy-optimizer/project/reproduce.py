"""One-command CPU reproduction: prepare data, run all methods, produce a report."""
import argparse
from time import perf_counter


def main():
    started = perf_counter()
    from strategy_optimizer.config import load_config
    from strategy_optimizer.data import prepare_data
    from strategy_optimizer.experiment import run_experiment
    from strategy_optimizer.reporting import create_report
    from strategy_optimizer.solvers import SOLVERS
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/ci.yaml")
    parser.add_argument("--output", default="reports/cpu-smoke.html")
    args = parser.parse_args()
    try:
        config = load_config(args.config)
        prepare_data(config["data"])
        run_dir, manifest = run_experiment(config, list(SOLVERS), started)
        output = create_report(run_dir, args.output)
    except (ValueError, OSError) as error:
        parser.exit(1, f"Reproduction failed: {error}\n")
    print(f"{manifest['status']}: {run_dir}\nReport: {output}")
    return 0 if manifest["status"] == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())

