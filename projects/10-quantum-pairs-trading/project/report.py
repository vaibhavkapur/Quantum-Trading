import argparse
from pathlib import Path

from pairs_trading.reporting import generate_report


def main():
    parser = argparse.ArgumentParser(description="Compare stored runs with full timing and failure disclosure")
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(generate_report(args.runs, args.output).resolve())


if __name__ == "__main__":
    main()
