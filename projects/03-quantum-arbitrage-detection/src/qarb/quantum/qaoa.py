"""QAOA on a diagonal cost Hamiltonian, simulated exactly with NumPy.

The cost operator of a QUBO is diagonal in the computational basis, so one
QAOA layer is a phase multiplication by exp(-i gamma E) followed by RX(2 beta)
on every qubit. This is a noiseless statevector simulation: its run time grows
as 2^n and says nothing about hardware run time.

The optimizer only sees energies of sampled bitstrings (`shots_per_iter` per
objective evaluation) unless `expectation="exact"`, so the reported number of
circuit executions is what a shot-based device would have needed.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from scipy.optimize import minimize


def qaoa_probabilities(E_phase: np.ndarray, gammas: np.ndarray, betas: np.ndarray) -> np.ndarray:
    """Measurement distribution of the QAOA state for phase energies E_phase."""
    dim = E_phase.shape[0]
    n = dim.bit_length() - 1
    psi = np.full(dim, 1.0 / np.sqrt(dim), dtype=np.complex128)
    for gamma, beta in zip(gammas, betas):
        psi = psi * np.exp(-1j * gamma * E_phase)
        c, s = np.cos(beta), -1j * np.sin(beta)
        t = psi.reshape((2,) * n)
        for axis in range(n):
            t = c * t + s * np.flip(t, axis=axis)
        psi = t.reshape(dim)
    p = np.abs(psi) ** 2
    return p / p.sum()


def initial_parameters(layers: int, rng: np.random.Generator, jitter: float = 0.0) -> np.ndarray:
    """Linear-ramp schedule (gamma rising, beta falling), optionally perturbed."""
    k = (np.arange(layers) + 0.5) / layers
    x0 = np.concatenate([0.6 * k, 0.6 * (1 - k)])
    return x0 + jitter * rng.standard_normal(2 * layers) if jitter else x0


def cvar(energies: np.ndarray, alpha: float, weights: np.ndarray | None = None) -> float:
    """Mean of the lowest alpha-fraction (alpha=1 is the plain expectation)."""
    order = np.argsort(energies, kind="stable")
    e = energies[order]
    w = np.full(e.shape[0], 1.0 / e.shape[0]) if weights is None else weights[order]
    cum = np.cumsum(w)
    take = np.clip(alpha - (cum - w), 0.0, w)
    return float((take * e).sum() / alpha)


@dataclass
class QaoaResult:
    counts: dict[int, int]  # sampled basis-state index -> shots
    params: np.ndarray
    objective: float
    optimizer_iters: int
    circuit_executions: int
    p_ground: float  # exact probability of a minimum-energy state at the final parameters
    timings_ms: dict[str, float] = field(default_factory=dict)


def run_qaoa(
    E: np.ndarray,
    rng: np.random.Generator,
    layers: int = 2,
    maxiter: int = 30,
    shots_per_iter: int = 128,
    final_shots: int = 512,
    cvar_alpha: float = 0.3,
    restarts: int = 1,
    expectation: str = "sampled",
    probs_fn: Callable[[np.ndarray, np.ndarray], np.ndarray] | None = None,
) -> QaoaResult:
    """Optimize QAOA angles with COBYLA and sample the final state.

    E: energy of every basis state. `probs_fn(gammas, betas)` overrides the
    built-in simulator (used to drive the Qiskit circuit backend); it must use
    the same standardized phase energies (E - mean) / std.
    """
    spread = float(E.std()) or 1.0
    E_phase = (E - E.mean()) / spread
    E_obj = (E - E.min()) / spread
    simulate = probs_fn or (lambda g, b: qaoa_probabilities(E_phase, g, b))
    dim = E.shape[0]
    n_evals = 0

    def objective(theta: np.ndarray) -> float:
        nonlocal n_evals
        n_evals += 1
        p = simulate(theta[:layers], theta[layers:])
        if expectation == "exact":
            return cvar(E_obj, cvar_alpha, p)
        return cvar(E_obj[rng.choice(dim, size=shots_per_iter, p=p)], cvar_alpha)

    t0 = time.perf_counter()
    best_x, best_f = None, np.inf
    for r in range(max(1, restarts)):
        x0 = initial_parameters(layers, rng, jitter=0.0 if r == 0 else 0.2)
        res = minimize(objective, x0, method="COBYLA", options={"maxiter": maxiter, "rhobeg": 0.25})
        if res.fun < best_f:
            best_x, best_f = np.asarray(res.x), float(res.fun)
    t1 = time.perf_counter()
    p = simulate(best_x[:layers], best_x[layers:])
    samples = rng.choice(dim, size=final_shots, p=p)
    idx, cnt = np.unique(samples, return_counts=True)
    t2 = time.perf_counter()
    per_eval = shots_per_iter if expectation == "sampled" else 0
    return QaoaResult(
        counts={int(i): int(c) for i, c in zip(idx, cnt)},
        params=best_x,
        objective=best_f,
        optimizer_iters=n_evals,
        circuit_executions=n_evals * per_eval + final_shots,
        p_ground=float(p[np.isclose(E, E.min())].sum()),
        timings_ms={"optimize": (t1 - t0) * 1e3, "sample": (t2 - t1) * 1e3},
    )
