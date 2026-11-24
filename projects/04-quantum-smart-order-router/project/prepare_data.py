import argparse

from qsor.config import load_config
from qsor.exchange_sim.data import Dataset, prepare


def main(argv=None):
    ap = argparse.ArgumentParser(description="Generate the versioned synthetic multi-venue L2 dataset.")
    ap.add_argument("--config", required=True)
    args = ap.parse_args(argv)
    path = prepare(load_config(args.config))
    ds = Dataset(path)
    print(f"dataset: {path}")
    print(f"  steps={ds.T} venues={ds.V} levels={ds.L} splits={ds.splits} sha256={ds.manifest['sha256'][:16]}…")


if __name__ == "__main__":
    main()
