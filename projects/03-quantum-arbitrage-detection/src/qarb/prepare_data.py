"""Generate the synthetic datasets declared by a config (one per seed).

    python -m qarb.prepare_data --config configs/demo.yaml
"""

from __future__ import annotations

import argparse
import sys

from .adapters.synthetic import dataset_paths, write_dataset
from .config import load_config


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    for seed in cfg["seeds"]:
        man = write_dataset(cfg, seed)
        kinds = [p["kind"] for p in man["planted"]]
        print(
            f"seed {seed}: {man['n_events']} events, {kinds.count('profitable')} profitable plants, "
            f"{kinds.count('thin')} thin decoys, {len(man['outages'])} stale decoys -> {dataset_paths(cfg, seed)[0]} "
            f"sha256={man['events_sha256'][:12]}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
