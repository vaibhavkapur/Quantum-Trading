"""Causal close-based strategy library and a content-addressed signal cache."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .config import digest, file_hash, write_json


@dataclass(frozen=True)
class Strategy:
    asset: str
    kind: str
    window: int

    @property
    def name(self):
        return f"{self.asset}:{self.kind}:{self.window}"


def library(assets, windows):
    return [Strategy(asset, kind, window) for asset in assets
            for kind in ("momentum", "mean_reversion") for window in windows]


def positions(prices: pd.DataFrame, strategies) -> np.ndarray:
    result = []
    for strategy in strategies:
        if strategy.window < 2 or strategy.asset not in prices:
            raise ValueError("Invalid strategy window or asset")
        series = prices[strategy.asset]
        if strategy.kind == "momentum":
            signal = np.sign(series / series.shift(strategy.window) - 1)
        elif strategy.kind == "mean_reversion":
            signal = -np.sign(series / series.rolling(strategy.window).mean() - 1)
        else:
            raise ValueError(f"Unknown strategy kind: {strategy.kind}")
        # Observe at close t-2, execute at close t-1, earn return ending at t.
        result.append(signal.shift(2).fillna(0).to_numpy(dtype=float))
    return np.column_stack(result)


def cached_positions(prices, strategies, data_hash: str, cache_dir: Path):
    key = digest({"data": data_hash, "strategies": [asdict(s) for s in strategies],
                  "implementation": file_hash(Path(__file__)), "execution_lag": 2})
    path, metadata = cache_dir / f"{key}.npy", cache_dir / f"{key}.json"
    if path.exists() and metadata.exists():
        info = json.loads(metadata.read_text())
        if info["sha256"] != file_hash(path):
            raise ValueError("Signal cache checksum mismatch")
        value = np.load(path, allow_pickle=False)
        if value.shape != (len(prices), len(strategies)) or not np.isin(value, [-1, 0, 1]).all():
            raise ValueError("Invalid signal cache")
        return value, True
    value = positions(prices, strategies)
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.save(path, value, allow_pickle=False)
    write_json(metadata, {"key": key, "sha256": file_hash(path)})
    return value, False

