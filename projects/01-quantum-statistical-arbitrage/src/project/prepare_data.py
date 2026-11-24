"""CLI: generate (or import) the versioned dataset named by a config.

    python -m project.prepare_data --config configs/demo.yaml
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import pandas as pd

from .config import load_config, resolve
from .data import loader, synthetic


def prepare(cfg: dict, force: bool = False) -> dict:
    dcfg = cfg["data"]
    out_dir = resolve(cfg, dcfg["dir"])
    if (out_dir / "DATASET.json").exists():
        if not force:
            manifest = json.loads((out_dir / "DATASET.json").read_text())
            print(f"dataset already exists at {out_dir} (version {manifest.get('version')}); use --force to rebuild")
            return manifest
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    extra = {}
    if dcfg["source"] == "synthetic":
        ohlcv, assets = synthetic.generate_daily(dcfg["n_days"], dcfg["start"], dcfg["seed"])
        ohlcv = loader.normalize_ohlcv(ohlcv)
        l2, l2_info = synthetic.generate_l2(dcfg["l2"]["n_events"], dcfg["l2"]["seed"])
        l2.to_csv(out_dir / "l2_quotes.csv", index=False, header=False)
        extra["l2_quotes.csv"] = None
        meta = {
            "name": cfg["name"],
            "version": f"synthetic-v1-seed{dcfg['seed']}-n{dcfg['n_days']}",
            "source": "synthetic latent-factor generator (project.data.synthetic)",
            "license": "CC0 - generated data, no third-party rights",
            "generator": {"seed": dcfg["seed"], "n_days": dcfg["n_days"], "start": dcfg["start"]},
            "l2": {**l2_info, "seed": dcfg["l2"]["seed"], "columns": ["ts_ns", "sym", "side(0=bid,1=ask)", "level", "price", "size"]},
            "survivorship": "includes one late listing and one delisting; universes are built point-in-time",
        }
    elif dcfg["source"] == "file":
        if not dcfg.get("path"):
            raise ValueError("data.path is required when data.source is 'file'")
        src = resolve(cfg, dcfg["path"])
        raw = pd.read_parquet(src) if src.suffix == ".parquet" else pd.read_csv(src)
        ohlcv = loader.normalize_ohlcv(raw, dcfg.get("timezone", "UTC"))
        if dcfg.get("assets_path"):
            assets = pd.read_csv(resolve(cfg, dcfg["assets_path"]))
        else:
            assets = loader.default_assets(sorted(ohlcv["symbol"].unique()))
        meta = {
            "name": cfg["name"],
            "version": f"file-{loader.sha256_file(src)[:12]}",
            "source": str(src),
            "license": "USER-SUPPLIED - verify redistribution and market-data terms before sharing",
            "source_timezone": dcfg.get("timezone", "UTC"),
            "survivorship": "as supplied; include delisted symbols to avoid survivorship bias",
        }
    else:
        raise ValueError(f"unknown data.source: {dcfg['source']}")

    manifest = loader.write_dataset(out_dir, ohlcv, assets, meta, extra)
    print(f"wrote {manifest['n_rows']} rows / {manifest['n_symbols']} symbols to {out_dir} (version {manifest['version']})")
    return manifest


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--force", action="store_true", help="rebuild even if the dataset exists")
    args = ap.parse_args(argv)
    prepare(load_config(args.config), force=args.force)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
