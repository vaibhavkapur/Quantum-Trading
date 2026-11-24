from __future__ import annotations
from dataclasses import dataclass, replace
from itertools import product
import math


@dataclass(frozen=True)
class Policy:
    half_spread: float
    skew: float
    inventory_band: int
    version: int = 0
    source: str = "inventory-skew"

    def __post_init__(self):
        if not math.isfinite(self.half_spread) or not 1 <= self.half_spread <= 1000000 or not math.isfinite(self.skew) or not 0 <= self.skew <= 1000000:
            raise ValueError("invalid spread or skew")
        if type(self.inventory_band) is not int or self.inventory_band <= 0 or type(self.version) is not int or self.version < 0:
            raise ValueError("invalid policy band or version")

    def versioned(self, version, source=None):
        return replace(self, version=version, source=source or self.source)


def grid(config):
    o = config["optimizer"]
    return [Policy(float(s), float(k), b, source="grid") for s, k, b in product(o["spreads"], o["skews"], o["inventory_bands"])]


def baseline(name, config):
    limit = config["execution"]["max_inventory"]
    if name == "fixed-spread":
        return Policy(1.0, 0.0, limit, source=name)
    if name == "inventory-skew":
        return Policy(1.0, 0.25, limit, source=name)
    if name == "avellaneda-stoikov":
        return Policy(1.0, 0.25, limit, source=name)
    raise ValueError(f"unknown baseline {name}")


def adaptive_policy(policy, sigma, remaining_fraction):
    if policy.source != "avellaneda-stoikov":
        return policy
    # Dimensionless finite-horizon approximation in price ticks. No fitted market parameters.
    gamma, kappa = 0.15, 1.5
    tau = max(0.05, remaining_fraction)
    skew = gamma * sigma**2 * tau
    spread = max(1.0, math.log1p(gamma / kappa) / gamma + 0.5 * skew)
    return replace(policy, half_spread=spread, skew=skew)
