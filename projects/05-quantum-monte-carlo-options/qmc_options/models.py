"""Market model and contract definitions (Black-Scholes / GBM)."""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Optional

import numpy as np

OPTION_KINDS = ("european_call", "european_put", "asian_call", "barrier_up_out_call")


@dataclass(frozen=True)
class Market:
    """Risk-neutral GBM parameters: dS = r S dt + sigma S dW."""

    spot: float
    rate: float
    vol: float

    def __post_init__(self) -> None:
        if self.spot <= 0:
            raise ValueError("spot must be positive")
        if self.vol <= 0:
            raise ValueError("vol must be positive")

    def bumped(self, **changes: float) -> "Market":
        return replace(self, **changes)


@dataclass(frozen=True)
class Option:
    """A contract. ``steps`` is the number of equally spaced monitoring dates
    used by path-dependent payoffs (arithmetic average / barrier check)."""

    kind: str
    strike: float
    maturity: float
    steps: int = 1
    barrier: Optional[float] = None

    def __post_init__(self) -> None:
        if self.kind not in OPTION_KINDS:
            raise ValueError(f"unknown option kind {self.kind!r}; expected one of {OPTION_KINDS}")
        if self.strike <= 0 or self.maturity <= 0:
            raise ValueError("strike and maturity must be positive")
        if self.steps < 1:
            raise ValueError("steps must be >= 1")
        if self.kind == "barrier_up_out_call" and (self.barrier is None or self.barrier <= self.strike):
            raise ValueError("barrier_up_out_call needs barrier > strike")

    @property
    def path_dependent(self) -> bool:
        return self.kind in ("asian_call", "barrier_up_out_call")

    def bumped(self, **changes: float) -> "Option":
        return replace(self, **changes)

    def to_dict(self) -> dict:
        return asdict(self)

    def payoff(self, paths: np.ndarray) -> np.ndarray:
        """Undiscounted payoff. ``paths`` has shape (..., steps) holding the
        asset price at each monitoring date (the last column is S_T)."""
        paths = np.asarray(paths, dtype=float)
        terminal = paths[..., -1]
        if self.kind == "european_call":
            return np.maximum(terminal - self.strike, 0.0)
        if self.kind == "european_put":
            return np.maximum(self.strike - terminal, 0.0)
        if self.kind == "asian_call":
            return np.maximum(paths.mean(axis=-1) - self.strike, 0.0)
        alive = paths.max(axis=-1) < self.barrier
        return np.where(alive, np.maximum(terminal - self.strike, 0.0), 0.0)


def paths_from_normals(market: Market, option: Option, z: np.ndarray) -> np.ndarray:
    """Exact GBM prices at the monitoring dates from standard normals of shape
    (..., steps)."""
    dt = option.maturity / option.steps
    increments = (market.rate - 0.5 * market.vol**2) * dt + market.vol * np.sqrt(dt) * z
    return market.spot * np.exp(np.cumsum(increments, axis=-1))
