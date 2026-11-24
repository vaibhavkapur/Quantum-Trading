import argparse
from orderbook.reporting import write_report


def main():
    parser = argparse.ArgumentParser(description="Render comparable stored runs to HTML, CSV and JSON")
    parser.add_argument("--runs", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(write_report(args.runs, args.output))


if __name__ == "__main__":
    main()
