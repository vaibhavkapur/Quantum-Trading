from dataclasses import dataclass
from typing import Sequence

import numpy as np


def finite_vector(values, name="values"):
    a = np.asarray(values, dtype=float)
    if a.ndim != 1 or a.size == 0 or not np.isfinite(a).all():
        raise ValueError(f"{name} must be a nonempty finite vector")
    return a


def validate_model(mean, covariance):
    mean = finite_vector(mean, "mean")
    covariance = np.asarray(covariance, dtype=float)
    if covariance.shape != (mean.size, mean.size) or not np.isfinite(covariance).all():
        raise ValueError("covariance must be finite and match the mean dimension")
    if not np.allclose(covariance, covariance.T, rtol=1e-10, atol=1e-14):
        raise ValueError("covariance must be symmetric")
    eigenvalues = np.linalg.eigvalsh(covariance)
    tolerance = 1e-12 * max(float(np.max(np.abs(eigenvalues))), 1e-12)
    if eigenvalues.min() < -tolerance:
        raise ValueError("covariance must be positive semidefinite")
    return mean, covariance


@dataclass(frozen=True)
class Portfolio:
    assets: Sequence[str]
    weights: Sequence[float]
    value: float = 1_000_000.0

    def __post_init__(self):
        weights = finite_vector(self.weights, "weights").copy()
        if len(self.assets) != weights.size or len(set(self.assets)) != weights.size:
            raise ValueError("assets must be unique and match weights")
        if not all(isinstance(a, str) and a.strip() for a in self.assets):
            raise ValueError("assets must have nonempty names")
        if not np.isfinite(self.value) or self.value <= 0:
            raise ValueError("portfolio value must be finite and positive")
        weights.setflags(write=False)
        object.__setattr__(self, "assets", tuple(self.assets))
        object.__setattr__(self, "weights", weights)

    def losses(self, returns):
        returns = np.asarray(returns, dtype=float)
        if returns.ndim != 2 or returns.shape[1] != len(self.assets) or not np.isfinite(returns).all():
            raise ValueError("returns must be a finite observations × assets matrix")
        return -self.value * np.einsum("ij,j->i", returns, self.weights)

    def loss_moments(self, mean, covariance):
        mean, covariance = validate_model(mean, covariance)
        if mean.size != len(self.assets):
            raise ValueError("model dimension must match portfolio")
        return (float(-self.value * np.dot(mean, self.weights)),
                float(self.value * np.sqrt(max(np.einsum("i,ij,j->", self.weights, covariance, self.weights), 0.0))))

    def stress(self, shocks):
        """Absolute additive simple-return shocks; uninvested cash has zero return."""
        results = []
        for name, shock in shocks.items():
            vector = finite_vector(shock, "shock")
            if len(vector) != len(self.assets):
                raise ValueError("shock dimension must match portfolio")
            loss = float(self.losses(vector[None, :])[0])
            results.append({"name": name, "loss": loss, "loss_fraction": loss / self.value,
                            "asset_contributions": dict(zip(self.assets, (-self.value * self.weights * vector).tolist()))})
        return results
