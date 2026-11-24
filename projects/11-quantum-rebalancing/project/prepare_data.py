import argparse
import json
from rebalancing.data import load_config, prepare


def main():
    parser = argparse.ArgumentParser(description="Generate and hash an immutable synthetic portfolio")
    parser.add_argument("--config", default="configs/demo.yaml")
    args = parser.parse_args()
    print(json.dumps(prepare(load_config(args.config)), indent=2))


if __name__ == "__main__":
    main()

