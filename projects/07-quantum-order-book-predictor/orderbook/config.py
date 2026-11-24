from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
import hashlib
import json
import math
import yaml


@dataclass(frozen=True)
class Config:
    name: str = "demo"
    data_path: str = "data/demo.csv"
    runs_dir: str = "runs"
    n_snapshots: int = 900
    levels: int = 3
    interval_ms: int = 100
    data_seed: int = 31415
    horizon: int = 3
    label_threshold_bps: float = 0.5
    embargo: int = 3
    train_fraction: float = 0.45
    calibration_fraction: float = 0.12
    validation_fraction: float = 0.12
    folds: int = 2
    train_samples: int = 72
    seeds: list[int] = field(default_factory=lambda: [11, 23, 37])
    qubits: list[int] = field(default_factory=lambda: [2, 3])
    encodings: list[str] = field(default_factory=lambda: ["angle", "iqp"])
    shots: list[int | None] = field(default_factory=lambda: [None])
    circuit_layers: int = 2
    max_iterations: int = 30
    fit_budget_seconds: float = 30.0
    fee_bps: float = 0.5
    slippage_bps: float = 0.2
    execution_delay: int = 1
    thresholds: list[float] = field(default_factory=lambda: [0.4, 0.5, 0.6, 0.7, 1.0])
    latency_samples: int = 40

    def validate(self) -> Config:
        for key in ("n_snapshots", "levels", "interval_ms", "horizon", "folds",
                    "train_samples", "circuit_layers", "max_iterations", "latency_samples",
                    "execution_delay"):
            value = getattr(self, key)
            if type(value) is not int or value < 1:
                raise ValueError(f"{key} must be a positive integer")
        if type(self.embargo) is not int or self.embargo < 0:
            raise ValueError("embargo must be a nonnegative integer")
        if self.n_snapshots < 100 or not 1 <= self.levels <= 10:
            raise ValueError("need at least 100 snapshots and 1..10 levels")
        fractions = [self.train_fraction, self.calibration_fraction, self.validation_fraction]
        if not all(0 < x < 1 for x in fractions) or sum(fractions) >= 0.95:
            raise ValueError("split fractions must be positive and leave at least 5% for test")
        for key in ("label_threshold_bps", "fee_bps", "slippage_bps", "fit_budget_seconds"):
            value = getattr(self, key)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{key} must be finite and nonnegative")
        if self.fit_budget_seconds == 0:
            raise ValueError("fit_budget_seconds must be positive")
        if not self.seeds or any(type(s) is not int or s < 0 for s in [self.data_seed, *self.seeds]):
            raise ValueError("seeds must be nonnegative integers")
        if not self.qubits or any(type(q) is not int or not 1 <= q <= 6 for q in self.qubits):
            raise ValueError("qubits must be between 1 and 6 (CPU research limit)")
        if not self.encodings or not set(self.encodings) <= {"angle", "iqp"}:
            raise ValueError("encodings must be angle and/or iqp")
        if not self.shots or any(s is not None and (type(s) is not int or s < 1) for s in self.shots):
            raise ValueError("shots must contain null (exact) or positive integers")
        if not self.thresholds or any(not 1 / 3 <= t <= 1 for t in self.thresholds):
            raise ValueError("thresholds must be between 1/3 and 1")
        for key in ("seeds", "qubits", "encodings", "shots", "thresholds"):
            if len(set(getattr(self, key))) != len(getattr(self, key)):
                raise ValueError(f"{key} must not contain duplicates")
        if self.execution_delay >= self.horizon:
            raise ValueError("execution_delay must be smaller than label horizon")
        if self.train_samples < max(self.qubits) or self.train_samples < 3:
            raise ValueError("train_samples must cover PCA components and three classes")
        if not self.name or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in self.name):
            raise ValueError("name must contain only letters, digits, _ or -")
        return self

    def to_dict(self):
        return asdict(self)

    @property
    def fingerprint(self):
        return hashlib.sha256(json.dumps(self.to_dict(), sort_keys=True).encode()).hexdigest()[:12]


def load_config(path: str | Path) -> Config:
    with Path(path).open() as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict):
        raise ValueError("configuration must be a YAML mapping")
    unknown = set(raw) - {f.name for f in fields(Config)}
    if unknown:
        raise ValueError(f"unknown config keys: {sorted(unknown)}")
    return Config(**raw).validate()
