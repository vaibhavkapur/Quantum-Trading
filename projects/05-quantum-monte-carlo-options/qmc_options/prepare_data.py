"""Generate the synthetic GBM sample dataset declared in the config.

    python -m qmc_options.prepare_data --config configs/demo.yaml
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Any, Dict

import numpy as np

from .config import instruments_from, load_config, market_from, sha256_file, write_json
from .models import paths_from_normals


def manifest_path(data_path: Path) -> Path:
    return data_path.with_suffix(".manifest.json")


def generate(config: Dict[str, Any]) -> Dict[str, Any]:
    """One array of price paths per distinct monitoring grid (maturity, steps)."""
    market = market_from(config)
    spec = config["data"]
    rng = np.random.default_rng(int(spec["seed"]))
    arrays = {}
    for _, option in instruments_from(config):
        key = dataset_key(option.maturity, option.steps)
        if key not in arrays:
            z = rng.standard_normal((int(spec["n_paths"]), option.steps))
            arrays[key] = paths_from_normals(market, option, z)
    path = Path(spec["path"])
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **arrays)
    info = {
        "path": str(path),
        "sha256": sha256_file(path),                 # of the .npz container (includes zip timestamps)
        "content_sha256": content_hash(arrays),      # of the numbers only; identical on every regeneration
        "generator": "numpy.random.default_rng (PCG64), exact GBM transition",
        "seed": int(spec["seed"]),
        "n_paths": int(spec["n_paths"]),
        "market": config["market"],
        "arrays": {k: list(v.shape) for k, v in arrays.items()},
        "license": "synthetic; no third-party data",
        "config_sha256": config["_sha256"],
    }
    write_json(manifest_path(path), info)
    return info


def content_hash(arrays) -> str:
    digest = hashlib.sha256()
    for key in sorted(arrays):
        digest.update(key.encode())
        digest.update(np.ascontiguousarray(arrays[key]).tobytes())
    return digest.hexdigest()


def dataset_key(maturity: float, steps: int) -> str:
    return f"T{maturity:g}_steps{steps}".replace(".", "p")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True)
    args = parser.parse_args(argv)
    info = generate(load_config(args.config))
    print(f"wrote {info['path']} ({info['n_paths']} paths per grid, sha256 {info['sha256'][:12]}...)")
    for key, shape in info["arrays"].items():
        print(f"  {key}: shape {tuple(shape)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
