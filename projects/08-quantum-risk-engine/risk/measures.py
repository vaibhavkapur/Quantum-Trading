"""VaR is the left quantile; ES integrates exactly the worst 1-alpha mass.

For discrete losses, ES includes only the required fraction of an atom at VaR.
It is generally NOT E[L | L >= VaR], which overweights that atom.
"""
import numpy as np
from scipy.stats import norm, t

from .models import finite_vector


def validate_alpha(alpha):
    if not np.isfinite(alpha) or not 0 < alpha < 1:
        raise ValueError("alpha must be strictly between 0 and 1")


def empirical_risk(losses, alpha=0.99, weights=None):
    validate_alpha(alpha)
    losses = finite_vector(losses, "losses")
    if weights is None:
        weights = np.ones(losses.size)
    weights = finite_vector(weights, "probability weights")
    if weights.shape != losses.shape or np.any(weights < 0) or not 0 < weights.sum() < np.inf:
        raise ValueError("weights must match losses, be nonnegative, and have positive finite mass")
    positive = weights > 0
    x, w = losses[positive], weights[positive]
    order = np.argsort(x, kind="stable")
    x, p = x[order], w[order] / w.sum()
    # Comparing cumulative normalized weights can move a quantile by an entire
    # observation (e.g. ten 0.1 weights accumulate to 0.8999999999999999).
    # Uniform samples have an exact order-statistic definition; weighted inputs
    # compare unnormalized cumulative mass to the target mass.
    if np.all(w == w[0]):
        index = max(0, min(int(np.ceil(alpha * len(x))) - 1, len(x) - 1))
    else:
        index = min(int(np.searchsorted(np.cumsum(w[order]), alpha * w.sum(), side="left")), len(x) - 1)
    var = float(x[index])
    # Stable even when alpha is very close to one.
    es = var + float(np.dot(p, np.maximum(x - var, 0.0))) / (1.0 - alpha)
    return {"alpha": float(alpha), "var": var, "cvar": es, "expected_shortfall": es,
            "mean_loss": float(np.dot(p, x)), "tail_probability_strict": float(p[x > var].sum()),
            "effective_sample_size": float(1.0 / np.dot(p, p)), "observations": int(losses.size)}


def gaussian_risk(mean_loss, std_loss, alpha=0.99):
    validate_alpha(alpha)
    if not np.isfinite([mean_loss, std_loss]).all() or std_loss < 0:
        raise ValueError("loss mean must be finite and standard deviation nonnegative")
    z = norm.ppf(alpha)
    es = float(mean_loss + std_loss * norm.pdf(z) / (1 - alpha))
    return {"alpha": float(alpha), "var": float(mean_loss + std_loss * z),
            "cvar": es, "expected_shortfall": es, "mean_loss": float(mean_loss)}


def student_t_risk(mean_loss, std_loss, df, alpha=0.99):
    validate_alpha(alpha)
    if not np.isfinite([mean_loss, std_loss, df]).all() or std_loss < 0 or df <= 2:
        raise ValueError("Student-t requires finite moments, std >= 0 and df > 2")
    scale = std_loss * np.sqrt((df - 2) / df)
    z = t.ppf(alpha, df)
    es = float(mean_loss + scale * (df + z*z) / (df - 1) * t.pdf(z, df) / (1 - alpha))
    return {"alpha": float(alpha), "var": float(mean_loss + scale * z),
            "cvar": es, "expected_shortfall": es, "mean_loss": float(mean_loss)}
