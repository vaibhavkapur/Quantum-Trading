from dataclasses import dataclass

import numpy as np
from scipy.stats import norm, t

from risk.measures import empirical_risk
from risk.models import finite_vector


@dataclass(frozen=True)
class DiscreteLoss:
    losses: np.ndarray
    probabilities: np.ndarray
    clipped_tail_mass: float = 0.0

    def __post_init__(self):
        x = finite_vector(self.losses, "loss grid").copy()
        p = finite_vector(self.probabilities, "probabilities").copy()
        if x.size < 2 or x.size & (x.size - 1) or x.size > 64:
            raise ValueError("loss grid must have 2 to 64 power-of-two entries")
        if x.shape != p.shape or np.any(np.diff(x) < 0) or np.any(p < 0) or not np.isclose(p.sum(), 1, atol=1e-12, rtol=0):
            raise ValueError("losses must be sorted and probabilities nonnegative and sum to one")
        if not np.isfinite(self.clipped_tail_mass) or not 0 <= self.clipped_tail_mass <= 1:
            raise ValueError("invalid clipped tail mass")
        p /= p.sum()
        x.setflags(write=False)
        p.setflags(write=False)
        object.__setattr__(self, "losses", x)
        object.__setattr__(self, "probabilities", p)

    @property
    def qubits(self):
        return self.losses.size.bit_length() - 1

    def risk(self, alpha):
        return empirical_risk(self.losses, alpha, self.probabilities)

    def tail_probability(self, threshold):
        return float(self.probabilities[self.losses > threshold].sum())


def discretize_loss(mean_loss, std_loss, qubits=4, support_std=5.0, distribution="gaussian", df=5.0):
    if isinstance(qubits, bool) or not isinstance(qubits, int) or not 1 <= qubits <= 6:
        raise ValueError("qubits must be an integer from 1 to 6")
    if not np.isfinite([mean_loss, std_loss, support_std]).all() or std_loss < 0 or support_std <= 0:
        raise ValueError("invalid loss moments or support")
    if distribution not in ("gaussian", "student-t"):
        raise ValueError("unsupported distribution")
    if distribution == "student-t" and (not np.isfinite(df) or df <= 2):
        raise ValueError("df must exceed two")
    size = 2**qubits
    if std_loss == 0:
        return DiscreteLoss(np.full(size, mean_loss), np.full(size, 1/size))
    edges = np.linspace(mean_loss - support_std * std_loss, mean_loss + support_std * std_loss, size+1)
    centers = (edges[:-1] + edges[1:]) / 2
    rv = (norm(loc=mean_loss, scale=std_loss) if distribution == "gaussian" else
          t(df, loc=mean_loss, scale=std_loss * np.sqrt((df-2)/df)))
    cdf = rv.cdf(edges)
    clipped = float(cdf[0] + rv.sf(edges[-1]))
    cdf[0], cdf[-1] = 0, 1  # put overflow in edge bins; never silently condition on the support
    return DiscreteLoss(centers, np.diff(cdf), clipped)
