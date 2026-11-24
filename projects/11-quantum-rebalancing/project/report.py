import argparse
from rebalancing.reporting import write_run_report


def main():
    parser = argparse.ArgumentParser(description="Render stored runs as a standalone HTML report")
    parser.add_argument("--runs", default="runs/")
    parser.add_argument("--output", default="reports/comparison.html")
    args = parser.parse_args()
    print(write_run_report(args.runs, args.output))


if __name__ == "__main__":
    main()

