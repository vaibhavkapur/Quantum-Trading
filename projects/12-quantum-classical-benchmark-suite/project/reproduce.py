"""One-command CPU reproduction, scoped to the newly created campaign."""
import argparse
from pathlib import Path

from benchmarks.instances.dataset import prepare
from experiments.config import load_config
from experiments.runner import run_matrix
from metrics.report import read_runs, render_report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/ci.yaml")
    parser.add_argument("--output", default="reports/reproduction.html")
    parser.add_argument("--solver", default="all")
    parser.add_argument("--strict", action="store_true", help="fail on worker errors/timeouts or incorrect exact/MILP answers")
    args = parser.parse_args()
    try:
        config, base = load_config(args.config)
        prepare(config, base)
        campaign = run_matrix(config, base, args.solver)
        output = render_report(campaign, Path(args.output))
        print(f"Campaign: {campaign}\nReport: {output.resolve()}")
        if args.strict:
            records, _ = read_runs(campaign)
            invalid = [r for r in records if r["status"] in {"failed", "timeout", "missing"}
                       or (r["solver"] in {"exact", "milp"} and not r["quality"]["success"])]
            if invalid:
                parser.exit(1, f"Strict verification failed for {len(invalid)} attempts; see the retained report.\n")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
