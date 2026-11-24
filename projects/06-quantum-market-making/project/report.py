import argparse
from market_making.reporting import report


def main():
    parser = argparse.ArgumentParser(description="Build standalone HTML, CSV, JSON and SVG comparisons from recorded runs")
    parser.add_argument("--runs", default="runs")
    parser.add_argument("--output", default="reports/comparison.html")
    args = parser.parse_args()
    print(report(args.runs, args.output))


if __name__ == "__main__":
    main()
