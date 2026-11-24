from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .config import resolve, write_json


def digest(blob: bytes) -> str:
    return hashlib.sha256(blob).hexdigest()


def synthetic_prices(settings: dict) -> tuple[pd.DataFrame, dict[str, str]]:
    """Seeded common random walks plus stationary residuals, in price levels."""
    rng = np.random.default_rng(settings["seed"])
    n = settings["periods"]
    dates = pd.bdate_range(settings["start"], periods=n, tz="UTC") + pd.Timedelta(hours=21)
    prices, sectors = {}, {}
    for sector in range(settings["sectors"]):
        common = 100 + 15 * sector + np.cumsum(rng.normal(0.015, 0.85, n))
        for asset in range(settings["assets_per_sector"]):
            residual = np.zeros(n)
            noise = rng.normal(0, 0.65, n)
            for t in range(1, n):
                residual[t] = 0.82 * residual[t - 1] + noise[t]
            symbol = f"S{sector + 1}A{asset + 1}"
            prices[symbol] = 10 + (0.8 + 0.2 * asset) * common + residual
            sectors[symbol] = f"sector-{sector + 1}"
    frame = pd.DataFrame(prices, index=dates)
    frame.index.name = "timestamp"
    validate_prices(frame, sectors)
    return frame, sectors


def validate_prices(frame: pd.DataFrame, sectors: dict) -> None:
    if len(frame) < 40 or len(frame.columns) < 2:
        raise ValueError("prices need at least 40 observations and two assets")
    if not isinstance(frame.index, pd.DatetimeIndex) or frame.index.tz is None:
        raise ValueError("prices must have timezone-aware timestamps")
    if not frame.index.is_monotonic_increasing or frame.index.has_duplicates:
        raise ValueError("timestamps must be strictly increasing and unique")
    if frame.columns.has_duplicates or set(frame.columns) != set(sectors):
        raise ValueError("each unique price column needs a sector")
    values = frame.to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError("prices must be finite and positive; missing data is never forward-filled")
    if any(not isinstance(v, str) or not v for v in sectors.values()):
        raise ValueError("sector labels must be nonempty strings")


def prepare_data(cfg: dict) -> Path:
    prices, sectors = synthetic_prices(cfg["data"])
    blob = prices.to_csv(float_format="%.10f").encode()
    source = {
        "schema_version": 1, "generator": "common-walk-plus-ar1-v1",
        "settings": {k: v for k, v in cfg["data"].items() if k != "pointer"},
        "sectors": sectors, "prices_sha256": digest(blob),
        "source": "Generated locally; no real securities or external market data.",
        "license": "CC0-1.0 for generated sample data; no third-party market-data rights required.",
        "timestamp_normalization": "UTC 21:00 on synthetic business days; not an exchange holiday calendar.",
        "universe": "All synthetic assets exist throughout; no retrospective constituent selection.",
        "adjustments": "No corporate actions or dividends in this synthetic generator.",
    }
    metadata_blob = (json.dumps(source, sort_keys=True, indent=2) + "\n").encode()
    dataset_id = digest(blob + metadata_blob)
    folder = resolve(cfg, "data/versions") / dataset_id
    folder.mkdir(parents=True, exist_ok=True)
    for name, content in (("prices.csv", blob), ("metadata.json", metadata_blob)):
        target = folder / name
        if target.exists() and target.read_bytes() != content:
            raise ValueError(f"immutable dataset was modified: {target}")
        if not target.exists():
            target.write_bytes(content)
    pointer = resolve(cfg, cfg["data"]["pointer"])
    write_json(pointer, {"dataset_id": dataset_id, "directory": str(folder.relative_to(Path(cfg["_root"])))})
    return pointer


def load_data(cfg: dict) -> tuple[pd.DataFrame, dict, dict]:
    pointer_path = resolve(cfg, cfg["data"]["pointer"])
    if not pointer_path.exists():
        raise FileNotFoundError("dataset missing; run python -m project.prepare_data --config CONFIG first")
    pointer = json.loads(pointer_path.read_text())
    folder = resolve(cfg, pointer["directory"])
    blob, metadata_blob = (folder / "prices.csv").read_bytes(), (folder / "metadata.json").read_bytes()
    metadata = json.loads(metadata_blob)
    if digest(blob) != metadata["prices_sha256"] or digest(blob + metadata_blob) != pointer["dataset_id"]:
        raise ValueError("dataset content hash mismatch")
    requested = {k: v for k, v in cfg["data"].items() if k != "pointer"}
    if metadata["settings"] != requested:
        raise ValueError("dataset/config mismatch; regenerate the configured dataset")
    prices = pd.read_csv(io.BytesIO(blob), index_col=0)
    prices.index = pd.to_datetime(prices.index, utc=True)
    validate_prices(prices, metadata["sectors"])
    return prices, metadata["sectors"], {**metadata, **pointer}
