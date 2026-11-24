import argparse

from .config import load_config
from .data import prepare_dataset


def main():
    parser = argparse.ArgumentParser(description="Generate an immutable seeded synthetic dataset")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config, root = load_config(args.config)
    print(prepare_dataset(config, root))


if __name__ == "__main__":
    main()
