"""Deterministic, immutable synthetic datasets with checked provenance."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .config import digest, file_hash, write_json

GENERATOR_VERSION = "regime-logreturns-v1"


def synthetic_data(spec: dict) -> pd.DataFrame:
    rng = np.random.default_rng(spec["seed"])
    n, m = spec["periods"], len(spec["assets"])
    # Synthetic business days, not an exchange holiday calendar.
    dates = pd.bdate_range(spec["start"], periods=n, tz="UTC")
    market = rng.normal(0, 0.006, n)
    innovations = rng.normal(0, 0.008, (n, m))
    returns = np.zeros((n, m))
    for t in range(1, n):
        regime = (t // 63) % 3
        phi = [0.20, -0.25, 0.0][regime]
        drift = [0.0005, -0.0001, 0.0002][regime]
        returns[t] = drift + phi * returns[t - 1] + market[t] + innovations[t]
    return pd.DataFrame(100 * np.exp(np.cumsum(returns, axis=0)), index=dates,
                        columns=spec["assets"]).rename_axis("date")


def _read_prices(path: Path, spec: dict) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if list(frame.columns) != ["date", *spec["assets"]]:
        raise ValueError("Dataset columns do not match the asset universe")
    dates = pd.to_datetime(frame.pop("date"), utc=True, errors="raise", format="ISO8601")
    frame.index = pd.DatetimeIndex(dates, name="date").normalize()
    values = frame.to_numpy(dtype=float)
    if (len(frame) != spec["periods"] or frame.index.has_duplicates or
            not frame.index.is_monotonic_increasing or frame.index.hasnans or
            not np.isfinite(values).all() or (values <= 0).any()):
        raise ValueError("Prices must be positive, finite, chronological, unique and complete")
    return frame.astype(float)


def prepare_data(spec: dict, source_csv=None, source_name=None, license_terms=None) -> Path:
    path = Path(spec["path"])
    manifest_path = path.with_suffix(".manifest.json")
    parameters = {k: v for k, v in spec.items() if k != "path"}
    source_id = digest({"version": GENERATOR_VERSION, "parameters": parameters})
    if source_csv is not None and (not source_name or not license_terms):
        raise ValueError("CSV import requires --source-name and --license-terms")
    if source_csv is None and (source_name or license_terms):
        raise ValueError("Source/license options require --source-csv")
    if path.exists() or manifest_path.exists():
        _, manifest = load_data(spec)
        if source_csv is not None and (manifest.get("source_sha256") != file_hash(Path(source_csv)) or
                                       manifest.get("source_name") != source_name or manifest.get("license") != license_terms):
            raise ValueError("Dataset is immutable; choose a new data.path for new parameters")
        return path
    imported = source_csv is not None
    source_hash = file_hash(Path(source_csv)) if imported else None
    if imported:
        frame = _read_prices(Path(source_csv), spec)
        source_id = digest({"version": "csv-import-v1", "parameters": parameters, "source_sha256": source_hash})
    else:
        frame = synthetic_data(spec)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation also prevents an accidental concurrent overwrite.
    with path.open("x") as handle:
        frame.to_csv(handle, float_format="%.12f", date_format="%Y-%m-%dT%H:%M:%SZ")
    write_json(manifest_path, {
        "schema_version": 1, "generator": "csv-import-v1" if imported else GENERATOR_VERSION, "source_id": source_id,
        "parameters": parameters, "sha256": file_hash(path), "rows": len(frame),
        "assets": list(frame.columns), "timezone": "UTC", "frequency": "daily UTC date labels",
        "kind": "imported" if imported else "synthetic", "source_name": source_name, "source_sha256": source_hash,
        "license": license_terms if imported else "CC0-1.0; generated synthetic prices, no vendor market data",
        "price_convention": "User-supplied adjusted/total-return closes" if imported else "Synthetic total-return close levels, no dividends or splits",
        "timestamp_convention": "Dates converted to UTC then normalized to midnight; labels are not execution timestamps",
    })
    return path


def load_data(spec: dict):
    path = Path(spec["path"])
    manifest = json.loads(path.with_suffix(".manifest.json").read_text())
    if manifest.get("schema_version") != 1 or manifest.get("sha256") != file_hash(path):
        raise ValueError("Dataset manifest or SHA-256 mismatch")
    if manifest.get("generator") not in (GENERATOR_VERSION, "csv-import-v1"):
        raise ValueError("Unsupported data generator/import format")
    identity = {"version": manifest["generator"], "parameters": {k: v for k, v in spec.items() if k != "path"}}
    if manifest["generator"] == "csv-import-v1":
        identity["source_sha256"] = manifest["source_sha256"]
    expected = digest(identity)
    if manifest.get("source_id") != expected:
        raise ValueError("Dataset parameters differ from config; use a new data.path")
    frame = _read_prices(path, spec)
    if len(frame) != manifest["rows"]:
        raise ValueError("Dataset row count differs from manifest")
    return frame, manifest

