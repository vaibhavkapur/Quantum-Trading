import argparse
from pathlib import Path

from metrics.report import render_report


def main():
    parser = argparse.ArgumentParser(description="Compare stored attempts without pooling unrelated objectives or budgets")
    parser.add_argument("--runs", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    try:
        print(render_report(Path(args.runs), Path(args.output)))
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
