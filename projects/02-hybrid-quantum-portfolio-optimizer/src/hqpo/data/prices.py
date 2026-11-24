"""Price-panel I/O: normalisation, immutable versioning (sha256 manifest) and loading."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def normalize_prices(df: pd.DataFrame, max_ffill: int = 3) -> pd.DataFrame:
    """Timestamp normalisation for a wide adjusted-close panel.

    * index -> timezone-naive calendar dates (tz-aware stamps are converted to UTC first)
    * sorted, duplicate dates dropped (last observation wins)
    * gaps forward-filled for at most ``max_ffill`` sessions (never back-filled: no look-ahead)
    * assets without a complete history after that are dropped; non-positive prices rejected
    """
    out = df.copy()
    idx = pd.to_datetime(out.index, utc=True) if getattr(out.index, "tz", None) is not None \
        else pd.to_datetime(out.index)
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_convert("UTC").tz_localize(None)
    out.index = pd.DatetimeIndex(idx.normalize(), name="date")
    out = out[~out.index.duplicated(keep="last")].sort_index()
    out = out.apply(pd.to_numeric, errors="coerce").ffill(limit=max_ffill)
    out = out.dropna(axis=1, how="any")
    if out.shape[1] == 0:
        raise ValueError("no asset has a complete price history")
    if (out <= 0).any().any():
        raise ValueError("prices must be strictly positive")
    return out


def save_dataset(prices: pd.DataFrame, sectors: dict, path: str | Path, meta: dict) -> dict:
    """Write the panel as CSV plus ``<name>.manifest.json`` (hash, shape, provenance)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    prices.to_csv(path, float_format="%.4f", lineterminator="\n")
    manifest = {
        "file": path.name,
        "sha256": sha256_file(path),
        "rows": int(prices.shape[0]),
        "assets": list(prices.columns),
        "first_date": str(prices.index[0].date()),
        "last_date": str(prices.index[-1].date()),
        "sectors": {t: sectors.get(t, "UNKNOWN") for t in prices.columns},
        **meta,
    }
    with open(manifest_path(path), "w") as fh:
        json.dump(manifest, fh, indent=2)
        fh.write("\n")
    return manifest


def manifest_path(path: str | Path) -> Path:
    path = Path(path)
    return path.with_name(path.stem + ".manifest.json")


def load_dataset(path: str | Path, verify: bool = True) -> tuple[pd.DataFrame, dict]:
    """Load a prepared panel; refuse it if the file no longer matches its manifest hash."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found - run `python -m hqpo.prepare_data` first")
    with open(manifest_path(path)) as fh:
        manifest = json.load(fh)
    if verify and sha256_file(path) != manifest["sha256"]:
        raise ValueError(f"{path} does not match its manifest sha256 (dataset was modified)")
    prices = pd.read_csv(path, index_col="date", parse_dates=True)
    return prices, manifest
