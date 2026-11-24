import argparse

from qsor.report import build


def main(argv=None):
    ap = argparse.ArgumentParser(description="Compare stored runs.")
    ap.add_argument("--runs", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args(argv)
    out = build(args.runs, args.output)
    print(f"report: {out}\ntable:  {out.with_suffix('.csv')}")


if __name__ == "__main__":
    main()
