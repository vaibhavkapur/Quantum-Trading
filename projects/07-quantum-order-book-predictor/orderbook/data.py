from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import hashlib
import json
import numpy as np
import polars as pl

from .config import Config


@dataclass
class Book:
    timestamp_ms: np.ndarray
    bid_price: np.ndarray
    ask_price: np.ndarray
    bid_size: np.ndarray
    ask_size: np.ndarray

    def __len__(self):
        return len(self.timestamp_ms)

    @property
    def mid(self):
        return (self.bid_price[:, 0] + self.ask_price[:, 0]) / 2

    def take(self, indices) -> Book:
        return Book(*(getattr(self, name)[indices] for name in self.__dataclass_fields__))

    def validate(self) -> Book:
        n = len(self)
        if n < 1 or self.timestamp_ms.ndim != 1 or self.timestamp_ms.dtype.kind not in "iu":
            raise ValueError("timestamp_ms must be a nonempty integer UTC epoch-ms array")
        if np.any(self.timestamp_ms <= 0) or np.any(np.diff(self.timestamp_ms) <= 0):
            raise ValueError("timestamps must be strictly increasing and positive")
        arrays = [self.bid_price, self.ask_price, self.bid_size, self.ask_size]
        if any(x.ndim != 2 or x.shape[0] != n or x.shape[1] < 1 for x in arrays):
            raise ValueError("price/size arrays must have shape (snapshots, levels)")
        if any(x.shape != arrays[0].shape or not np.isfinite(x).all() for x in arrays):
            raise ValueError("book arrays must share shape and contain finite values")
        if np.any(self.bid_price <= 0) or np.any(self.ask_price <= 0):
            raise ValueError("prices must be positive")
        if np.any(self.bid_size < 0) or np.any(self.ask_size < 0):
            raise ValueError("sizes must be nonnegative")
        if np.any(self.bid_size[:, 0] <= 0) or np.any(self.ask_size[:, 0] <= 0):
            raise ValueError("best quotes must have positive size")
        if np.any(self.bid_price[:, 0] >= self.ask_price[:, 0]):
            raise ValueError("locked/crossed books are invalid")
        if np.any(np.diff(self.bid_price, axis=1) >= 0) or np.any(np.diff(self.ask_price, axis=1) <= 0):
            raise ValueError("levels must be strictly sorted away from the touch")
        return self


def generate_book(config: Config) -> Book:
    """Causal toy market: next price innovation depends on the preceding flow."""
    rng = np.random.default_rng(config.data_seed)
    n, levels = config.n_snapshots, config.levels
    ticks, flow = 10000, 0.0
    bp, ap, bq, aq = [np.zeros((n, levels)) for _ in range(4)]
    for i in range(n):
        ticks += int(np.rint(0.7 * flow + rng.normal(0, 0.85)))
        flow = np.clip(0.85 * flow + rng.normal(0, 0.55), -3, 3)
        half_spread = rng.choice([0.5, 1.0], p=[0.85, 0.15])
        bp[i] = (ticks - half_spread - np.arange(levels)) * 0.01
        ap[i] = (ticks + half_spread + np.arange(levels)) * 0.01
        bq[i] = np.exp(2 + 0.45 * flow + rng.normal(0, 0.3, levels))
        aq[i] = np.exp(2 - 0.45 * flow + rng.normal(0, 0.3, levels))
    timestamp = 1735689600000 + np.arange(n, dtype=np.int64) * config.interval_ms
    return Book(timestamp, bp, ap, bq, aq).validate()


def frame(book: Book) -> pl.DataFrame:
    columns = {"timestamp_ms": book.timestamp_ms}
    for name in ("bid_price", "ask_price", "bid_size", "ask_size"):
        values = getattr(book, name)
        columns.update({f"{name}_{level}": values[:, level] for level in range(values.shape[1])})
    return pl.DataFrame(columns)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare_data(config: Config) -> Path:
    path = Path(config.data_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    content = frame(generate_book(config)).write_csv().encode()
    digest = hashlib.sha256(content).hexdigest()
    manifest_path = path.with_suffix(".manifest.json")
    manifest = {
        "schema_version": 1, "generator_version": 1, "source": "synthetic causal L2 generator",
        "license": "CC0-1.0 (generated sample only)", "sha256": digest,
        "timestamp_normalization": "integer Unix milliseconds, UTC; no local-time conversion",
        "rows": config.n_snapshots, "levels": config.levels, "seed": config.data_seed,
        "interval_ms": config.interval_ms, "trades": "none; OFI is inferred from consecutive L2 quotes",
    }
    if path.exists() or manifest_path.exists():
        if not (path.exists() and manifest_path.exists()):
            raise ValueError("incomplete dataset; choose a new data_path")
        if sha256(path) != digest or json.loads(manifest_path.read_text()) != manifest:
            raise ValueError("immutable dataset differs; choose a new data_path instead of overwriting")
        return path
    path.write_bytes(content)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return path


def load_book(path: str | Path, levels: int, verify: bool = True) -> tuple[Book, dict]:
    path = Path(path)
    manifest_path = path.with_suffix(".manifest.json")
    if not path.exists():
        raise FileNotFoundError(f"{path} missing; run python -m project.prepare_data first")
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    if verify and (not manifest.get("license") or manifest.get("sha256") != sha256(path)):
        raise ValueError("dataset requires a matching SHA-256 manifest and documented license")
    data = pl.read_csv(path)
    if data["timestamp_ms"].dtype != pl.Int64:
        raise ValueError("timestamp_ms must be integer UTC epoch milliseconds")
    book = Book(data["timestamp_ms"].to_numpy(), *[
        data.select([f"{name}_{i}" for i in range(levels)]).to_numpy().astype(float)
        for name in ("bid_price", "ask_price", "bid_size", "ask_size")
    ]).validate()
    if verify and (manifest.get("rows") != len(book) or manifest.get("levels") != levels):
        raise ValueError("dataset dimensions do not match manifest/configuration")
    return book, manifest
