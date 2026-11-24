"""Map a GBM pricing problem onto a finite register of qubits.

Each monitoring step gets ``qubits_per_step`` qubits holding a truncated,
binned standard normal increment. The joint register therefore enumerates
2**(steps * qubits_per_step) price paths, each with a known probability and
payoff. That table defines both the circuit and its brute-force answer.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import norm

from ..models import Market, Option, paths_from_normals


def gaussian_grid(num_qubits: int, truncation: float = 3.0):
    """Bin midpoints and renormalised bin masses of N(0,1) on [-c, c]."""
    if num_qubits < 1:
        raise ValueError("need at least one qubit per step")
    edges = np.linspace(-truncation, truncation, 2**num_qubits + 1)
    probs = np.diff(norm.cdf(edges))
    return 0.5 * (edges[:-1] + edges[1:]), probs / probs.sum()


@dataclass(frozen=True)
class DiscretizedProblem:
    market: Market
    option: Option
    qubits_per_step: int
    steps: int
    step_probs: np.ndarray     # (2**qubits_per_step,) marginal of one step register
    probs: np.ndarray          # (2**n,) joint probability of each basis state
    payoffs: np.ndarray        # (2**n,) undiscounted payoff of each basis state
    payoff_max: float
    discount: float

    @property
    def num_state_qubits(self) -> int:
        return self.steps * self.qubits_per_step

    @property
    def normalized_payoffs(self) -> np.ndarray:
        return self.payoffs / self.payoff_max if self.payoff_max > 0 else np.zeros_like(self.payoffs)

    @property
    def amplitude(self) -> float:
        """Exact probability of measuring the objective qubit in |1>."""
        return float(self.probs @ self.normalized_payoffs)

    @property
    def scale(self) -> float:
        """price = scale * amplitude."""
        return self.discount * self.payoff_max

    @property
    def exact_price(self) -> float:
        """Brute-force price of the discretised model (not of the continuous one)."""
        return float(self.discount * (self.probs @ self.payoffs))


def discretize(market: Market, option: Option, qubits_per_step: int, truncation: float = 3.0) -> DiscretizedProblem:
    steps = option.steps if option.path_dependent else 1
    if steps * qubits_per_step > 12:
        raise ValueError("register above 12 state qubits is outside this dense-simulation prototype")
    z, p = gaussian_grid(qubits_per_step, truncation)
    size = z.size
    # Basis index i = sum_j i_j * size**j : step j lives on qubits [j*m, (j+1)*m).
    index = np.arange(size**steps)
    digits = np.stack([(index // size**j) % size for j in range(steps)], axis=-1)
    probs = p[digits].prod(axis=-1)
    single_step = option if option.path_dependent else option.bumped(steps=1)
    payoffs = single_step.payoff(paths_from_normals(market, single_step, z[digits]))
    return DiscretizedProblem(market, option, qubits_per_step, steps, p, probs, payoffs,
                              float(payoffs.max()), float(np.exp(-market.rate * option.maturity)))
