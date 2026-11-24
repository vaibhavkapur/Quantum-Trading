"""Classical simulated annealing on a QUBO.

This is a *classical* sampler. It stands in for an annealing-style solver of
the same formulation and scales well past what a statevector simulator can
hold; it is not evidence about quantum annealing hardware.
"""

from __future__ import annotations

import numpy as np


def simulated_annealing(
    Q: np.ndarray,
    rng: np.random.Generator,
    reads: int = 32,
    sweeps: int = 60,
    beta_range: tuple[float, float] = (0.1, 20.0),
) -> tuple[np.ndarray, np.ndarray]:
    """Single-flip Metropolis annealing, vectorized over independent reads.

    Returns (states [reads, n], energies without the QUBO offset).
    """
    n = Q.shape[0]
    lin = np.diag(Q).copy()
    S = np.triu(Q, 1)
    S = S + S.T
    x = rng.integers(0, 2, size=(reads, n)).astype(float)
    field = lin[None, :] + x @ S
    betas = np.geomspace(beta_range[0], beta_range[1], sweeps)
    for beta in betas:
        for i in rng.permutation(n):
            delta = (1 - 2 * x[:, i]) * field[:, i]
            accept = (delta <= 0) | (rng.random(reads) < np.exp(-beta * np.clip(delta, 0, None)))
            if accept.any():
                sign = 1 - 2 * x[accept, i]
                x[accept, i] += sign
                field[accept] += sign[:, None] * S[i][None, :]
    energies = x @ lin + np.einsum("ki,ij,kj->k", x, np.triu(Q, 1), x)
    return x.astype(np.int8), energies
