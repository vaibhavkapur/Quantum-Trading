import argparse

from benchmarks.instances.dataset import prepare
from experiments.config import load_config


def main():
    parser = argparse.ArgumentParser(description="Generate immutable, exactly verified synthetic instances")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    try:
        config, base = load_config(args.config)
        print(prepare(config, base))
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
