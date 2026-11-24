import argparse
from orderbook.config import load_config
from orderbook.data import prepare_data


def main():
    parser = argparse.ArgumentParser(description="Generate immutable, documented synthetic L2 data")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    print(prepare_data(load_config(args.config)))


if __name__ == "__main__":
    main()
