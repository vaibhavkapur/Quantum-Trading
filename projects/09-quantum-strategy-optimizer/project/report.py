"""Compare stored runs with identical data, constraints, code, and budgets."""
import argparse


def main():
    from strategy_optimizer.reporting import create_report
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    try:
        print(create_report(args.runs, args.output))
    except (ValueError, OSError) as error:
        parser.exit(1, f"Report failed: {error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

