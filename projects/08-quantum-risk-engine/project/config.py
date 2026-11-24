import hashlib
import json
import re
from pathlib import Path
from typing import Dict, List, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from risk.models import Portfolio, validate_model


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class PortfolioConfig(StrictModel):
    assets: List[str] = Field(min_length=1, max_length=32)
    weights: List[float] = Field(min_length=1, max_length=32)
    value: float = Field(default=1_000_000, gt=0, le=1e15)

    def build(self):
        return Portfolio(**self.model_dump())

    @model_validator(mode="after")
    def valid_portfolio(self):
        self.build()
        if any(not re.fullmatch(r"[A-Za-z0-9_.-]{1,64}", asset) for asset in self.assets):
            raise ValueError("asset names must contain only letters, numbers, underscore, period or hyphen")
        return self


class ModelConfig(StrictModel):
    mean: List[float]
    covariance: List[List[float]]
    distribution: Literal["gaussian", "student-t"] = "gaussian"
    df: float = Field(default=5.0, gt=2)

    @model_validator(mode="after")
    def valid_covariance(self):
        validate_model(self.mean, self.covariance)
        return self


class DataConfig(StrictModel):
    directory: str = "data"
    observations: int = Field(default=1000, ge=20, le=20000)
    training_observations: int = Field(default=500, ge=10)
    start_date: str = "2020-01-01"

    @model_validator(mode="after")
    def valid_split(self):
        if self.training_observations >= self.observations:
            raise ValueError("training observations must be fewer than total observations")
        return self


class SimulationConfig(StrictModel):
    samples: int = Field(default=20000, ge=100, le=200000)
    importance_shift: float = Field(default=2.0, ge=0, le=10)


class QuantumConfig(StrictModel):
    qubits: int = Field(default=4, ge=1, le=6)
    shots: int = Field(default=128, ge=1, le=4096)
    powers: List[int] = Field(default_factory=lambda: [0, 1, 2, 4, 8], min_length=1, max_length=7)
    support_std: float = Field(default=5, gt=0, le=10)
    fallback: bool = False

    @model_validator(mode="after")
    def valid_schedule(self):
        if self.powers[0] != 0 or self.powers != sorted(set(self.powers)) or not all(0 <= k <= 32 for k in self.powers):
            raise ValueError("powers must start at zero and strictly increase, with maximum 32")
        return self


class BenchmarkConfig(StrictModel):
    seeds: List[int] = Field(default_factory=lambda: [11, 23, 37, 53, 71], min_length=2, max_length=50)
    powers: List[int] = Field(default_factory=lambda: [1, 2, 4, 8], min_length=1, max_length=6)
    shots: int = Field(default=64, ge=1, le=4096)

    @model_validator(mode="after")
    def valid_budgets(self):
        if len(set(self.seeds)) != len(self.seeds) or any(s < 0 for s in self.seeds):
            raise ValueError("benchmark seeds must be unique and nonnegative")
        if self.powers != sorted(set(self.powers)) or any(p not in [1, 2, 4, 8, 16, 32] for p in self.powers):
            raise ValueError("benchmark powers must be increasing distinct powers of two in [1,32]")
        return self


class ExperimentConfig(StrictModel):
    experiment: str = Field(default="demo", pattern=r"^[A-Za-z0-9_-]+$")
    seed: int = Field(default=42, ge=0, le=2**31-100)
    alpha: float = Field(default=0.99, gt=0, lt=1)
    portfolio: PortfolioConfig
    model: ModelConfig
    data: DataConfig = Field(default_factory=DataConfig)
    simulation: SimulationConfig = Field(default_factory=SimulationConfig)
    quantum: QuantumConfig = Field(default_factory=QuantumConfig)
    benchmark: BenchmarkConfig = Field(default_factory=BenchmarkConfig)
    stress: Dict[str, List[float]] = Field(default_factory=dict)

    @model_validator(mode="after")
    def valid_dimensions(self):
        if len(self.portfolio.assets) != len(self.model.mean):
            raise ValueError("portfolio and model dimensions differ")
        self.portfolio.build().stress(self.stress)
        return self


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def load_config(path):
    path = Path(path).resolve()
    return ExperimentConfig.model_validate(yaml.safe_load(path.read_text())), path.parent.parent
