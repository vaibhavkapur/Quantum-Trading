"""CLI: render stored runs into the comparison dashboard.

    python -m hqpo.report --runs runs/ --output reports/comparison.html
"""
from __future__ import annotations

import argparse
from pathlib import Path

from .evaluation.report import build_report


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--runs", default="runs/")
    ap.add_argument("--output", default="reports/comparison.html")
    args = ap.parse_args(argv)
    for kind, path in build_report(Path(args.runs), Path(args.output)).items():
        print(f"wrote {kind}: {path}")


if __name__ == "__main__":
    main()
