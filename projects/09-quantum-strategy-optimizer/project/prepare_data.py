"""Generate a deterministic, documented sample dataset."""
import argparse


def main():
    from strategy_optimizer.config import load_config
    from strategy_optimizer.data import prepare_data
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--source-csv", help="Import local adjusted daily closes instead of generating prices")
    parser.add_argument("--source-name", help="Provider or dataset citation for imported data")
    parser.add_argument("--license-terms", help="License and redistribution restrictions for imported data")
    args = parser.parse_args()
    try:
        print(prepare_data(load_config(args.config)["data"], args.source_csv, args.source_name, args.license_terms))
    except (ValueError, OSError) as error:
        parser.exit(1, f"Data preparation failed: {error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

