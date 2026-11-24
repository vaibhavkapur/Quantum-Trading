"""Seeded one-period simple returns, including a shared-scale Student-t model."""
import numpy as np
from scipy.special import logsumexp

from risk.models import validate_model


def covariance_root(covariance):
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    return eigenvectors * np.sqrt(np.maximum(eigenvalues, 0))[None, :]


def correlated_returns(mean, covariance, count, seed, distribution="gaussian", df=5.0, antithetic=False):
    mean, covariance = validate_model(mean, covariance)
    if isinstance(count, bool) or not isinstance(count, (int, np.integer)) or count < 1:
        raise ValueError("count must be a positive integer")
    if distribution not in ("gaussian", "student-t"):
        raise ValueError("distribution must be gaussian or student-t")
    if distribution == "student-t" and (not np.isfinite(df) or df <= 2):
        raise ValueError("Student-t df must exceed 2 for finite covariance")
    rng = np.random.default_rng(seed)
    size = (count + 1) // 2 if antithetic else count
    z = rng.standard_normal((size, mean.size))
    if distribution == "student-t":
        z *= np.sqrt((df - 2) / rng.chisquare(df, size))[:, None]
    if antithetic:
        z = np.concatenate([z, -z], axis=0)[:count]
    # Explicit contraction avoids spurious floating-point flags from some macOS
    # Accelerate BLAS builds, without suppressing genuine numerical warnings.
    return mean + np.einsum("ij,kj->ik", z, covariance_root(covariance))


def importance_sample(portfolio, mean, covariance, count, seed, shift=2.0):
    """Gaussian exponential tilt toward positive loss, with self-normalized weights.

    Quantiles and ES use self-normalized likelihood ratios (finite-sample biased).
    The unnormalized weight mean is returned as an overlap diagnostic.
    """
    mean, covariance = validate_model(mean, covariance)
    if mean.size != len(portfolio.assets) or not np.isfinite(shift) or not 0 <= shift <= 10:
        raise ValueError("model dimension mismatch or shift outside [0, 10]")
    if not isinstance(count, (int, np.integer)) or isinstance(count, bool) or count < 1:
        raise ValueError("count must be a positive integer")
    root = covariance_root(covariance)
    exposure = -portfolio.value * np.einsum("ij,i->j", root, portfolio.weights)
    scale = np.linalg.norm(exposure)
    direction = np.zeros_like(exposure) if scale == 0 else shift * exposure / scale
    z = np.random.default_rng(seed).standard_normal((count, mean.size)) + direction
    log_weights = -np.einsum("ij,j->i", z, direction) + np.dot(direction, direction) / 2
    log_total = logsumexp(log_weights)
    weights = np.exp(log_weights - log_total)
    return mean + np.einsum("ij,kj->ik", z, root), weights, {"weight_mean": float(np.exp(log_total - np.log(count))),
                                      "effective_sample_size": float(1 / np.dot(weights, weights)),
                                      "shift": float(shift)}
