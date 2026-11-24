import argparse
from market_making.config import load_config
from market_making.data import prepare


def main():
    parser = argparse.ArgumentParser(description="Generate immutable synthetic L2 datasets with checksums and chronological splits")
    parser.add_argument("--config", default="configs/demo.yaml")
    parser.add_argument("--output", default="data")
    args = parser.parse_args()
    print(prepare(load_config(args.config), args.output))


if __name__ == "__main__":
    main()
