"""Dataset IO: timestamp normalization, immutable versioned storage, point-in-time universe."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

OHLCV_COLUMNS = ["ts", "symbol", "open", "high", "low", "close", "volume"]


def normalize_timestamps(ts: pd.Series, source_tz: str = "UTC") -> pd.Series:
    """Return tz-aware UTC timestamps at nanosecond resolution.

    Naive timestamps are interpreted in ``source_tz`` (DST gaps/overlaps raise
    rather than being silently guessed); tz-aware ones are converted.
    """
    ts = pd.to_datetime(ts)
    if ts.dt.tz is None:
        ts = ts.dt.tz_localize(source_tz, ambiguous="raise", nonexistent="raise")
    return ts.dt.tz_convert("UTC").astype("datetime64[ns, UTC]")


def normalize_ohlcv(df: pd.DataFrame, source_tz: str = "UTC") -> pd.DataFrame:
    missing = [c for c in OHLCV_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"OHLCV input is missing columns: {missing}")
    out = df[OHLCV_COLUMNS].copy()
    out["ts"] = normalize_timestamps(out["ts"], source_tz)
    out["symbol"] = out["symbol"].astype(str)
    if out.duplicated(["ts", "symbol"]).any():
        raise ValueError("duplicate (ts, symbol) rows in OHLCV input")
    px = out[["open", "high", "low", "close"]]
    if (px <= 0).any().any() or px.isna().any().any():
        raise ValueError("non-positive or missing prices in OHLCV input")
    if (out["high"] < out[["open", "close"]].max(axis=1) - 1e-9).any() or (
        out["low"] > out[["open", "close"]].min(axis=1) + 1e-9
    ).any():
        raise ValueError("OHLC bars violate low <= open/close <= high")
    return out.sort_values(["ts", "symbol"]).reset_index(drop=True)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_dataset(out_dir: Path, ohlcv: pd.DataFrame, assets: pd.DataFrame, meta: dict, extra_files: dict | None = None) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    ohlcv.to_parquet(out_dir / "ohlcv.parquet", index=False)
    assets.to_csv(out_dir / "assets.csv", index=False)
    files = {"ohlcv.parquet": None, "assets.csv": None, **(extra_files or {})}
    manifest = {
        **meta,
        "n_rows": int(len(ohlcv)),
        "n_symbols": int(ohlcv["symbol"].nunique()),
        "first_ts": str(ohlcv["ts"].min()),
        "last_ts": str(ohlcv["ts"].max()),
        "timestamp_convention": "UTC, nanosecond resolution; daily bars stamped at session close",
        "files": {name: sha256_file(out_dir / name) for name in files},
    }
    (out_dir / "DATASET.json").write_text(json.dumps(manifest, indent=2, default=str))
    return manifest


@dataclass
class Dataset:
    close: pd.DataFrame   # dates x symbols, NaN when not listed
    open: pd.DataFrame
    volume: pd.DataFrame
    assets: pd.DataFrame  # indexed by symbol
    manifest: dict

    @property
    def dates(self) -> pd.DatetimeIndex:
        return self.close.index

    def universe(self, start: int, end: int) -> list[str]:
        """Point-in-time universe: symbols with a complete history on [start, end).

        Decided only from data available at ``end`` - an asset that delists
        later is still included, one that has not listed yet is not.
        """
        window = self.close.iloc[start:end]
        ok = window.notna().all(axis=0)
        return [s for s in self.close.columns if ok[s]]


def load_dataset(data_dir: Path, verify: bool = True) -> Dataset:
    data_dir = Path(data_dir)
    manifest_path = data_dir / "DATASET.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"{manifest_path} not found - run `python -m project.prepare_data --config <cfg>` first")
    manifest = json.loads(manifest_path.read_text())
    if verify:
        for name, digest in manifest["files"].items():
            actual = sha256_file(data_dir / name)
            if actual != digest:
                raise ValueError(f"dataset file {name} does not match its recorded sha256 (datasets are immutable)")
    ohlcv = pd.read_parquet(data_dir / "ohlcv.parquet")
    assets = pd.read_csv(data_dir / "assets.csv").set_index("symbol")
    wide = {c: ohlcv.pivot(index="ts", columns="symbol", values=c).sort_index() for c in ("close", "open", "volume")}
    symbols = [s for s in assets.index if s in wide["close"].columns]
    for sym in wide["close"].columns:
        if sym not in assets.index:
            raise ValueError(f"symbol {sym} has prices but no row in assets.csv")
    return Dataset(
        close=wide["close"][symbols],
        open=wide["open"][symbols],
        volume=wide["volume"][symbols],
        assets=assets.loc[symbols],
        manifest=manifest,
    )


def default_assets(symbols: list[str]) -> pd.DataFrame:
    """Fallback asset metadata for user files that ship none."""
    return pd.DataFrame(
        {
            "symbol": symbols,
            "sector": "UNKNOWN",
            "spread_bps": 5.0,
            "borrow_bps_annual": 50.0,
            "listed": pd.NaT,
            "delisted": pd.NaT,
        }
    )


def log_prices(close: pd.DataFrame) -> pd.DataFrame:
    return np.log(close)
