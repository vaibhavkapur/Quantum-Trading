"""QAOA for the basket-selection QUBO on a noiseless local simulator.

Two interchangeable backends produce the same state (cross-checked in tests):

* ``qiskit`` - builds the circuit gate by gate, transpiles it to a hardware-like
  basis (rz, sx, cx) to report depth / two-qubit counts, and simulates it with
  ``qiskit.quantum_info.Statevector``.
* ``numpy``  - closed-form simulation using the diagonal cost operator; used for
  the scaling benchmark because it is much faster.

Measurement is simulated by sampling ``shots`` bitstrings from the exact
distribution with a seeded generator, so shot noise is present but gate noise
is not. Nothing here ran on quantum hardware.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

from ..research.qubo import BasketProblem
from ..solvers.base import Budget, SolverResult, Timer, best_of_candidates

MAX_QUBITS = 20


def _normalised_ising(problem: BasketProblem) -> tuple[np.ndarray, np.ndarray, float]:
    """Ising coefficients scaled so the energy spectrum has unit variance under uniform sampling."""
    h, J, _ = problem.to_ising()
    scale = float(np.sqrt(np.sum(h**2) + np.sum(J**2)))
    scale = scale if scale > 0 else 1.0
    return h / scale, J / scale, scale


def ising_energies(h: np.ndarray, J: np.ndarray) -> np.ndarray:
    """Energy of every computational basis state (index bit q = qubit q)."""
    n = len(h)
    k = np.arange(1 << n, dtype=np.int64)
    s = 1.0 - 2.0 * ((k[:, None] >> np.arange(n)) & 1)
    return s @ h + np.einsum("ki,ij,kj->k", s, J, s)


def numpy_probabilities(energies: np.ndarray, n: int, gammas: np.ndarray, betas: np.ndarray) -> np.ndarray:
    psi = np.full(1 << n, 1.0 / np.sqrt(1 << n), dtype=complex)
    for gamma, beta in zip(gammas, betas):
        psi = psi * np.exp(-1j * gamma * energies)
        c, s = np.cos(beta), -1j * np.sin(beta)
        psi = psi.reshape([2] * n)
        for axis in range(n):
            a, b = np.take(psi, 0, axis=axis), np.take(psi, 1, axis=axis)
            psi = np.stack([c * a + s * b, s * a + c * b], axis=axis)
        psi = psi.reshape(-1)
    return np.abs(psi) ** 2


def build_circuit(h: np.ndarray, J: np.ndarray, p: int):
    """Parametrised QAOA circuit: H^n, then p layers of exp(-i g H_C) exp(-i b sum X)."""
    from qiskit import QuantumCircuit
    from qiskit.circuit import ParameterVector

    n = len(h)
    gammas, betas = ParameterVector("g", p), ParameterVector("b", p)
    qc = QuantumCircuit(n)
    qc.h(range(n))
    for layer in range(p):
        for i in range(n):
            if h[i] != 0:
                qc.rz(2 * gammas[layer] * float(h[i]), i)
        for i in range(n):
            for j in range(i + 1, n):
                if J[i, j] != 0:
                    qc.rzz(2 * gammas[layer] * float(J[i, j]), i, j)
        for i in range(n):
            qc.rx(2 * betas[layer], i)
    return qc, gammas, betas


class _QiskitBackend:
    def __init__(self, h: np.ndarray, J: np.ndarray, p: int, timer: Timer) -> None:
        from qiskit import transpile

        with timer.stage("compile"):
            self.logical, self.gammas, self.betas = build_circuit(h, J, p)
            self.circuit = transpile(self.logical, basis_gates=["rz", "sx", "cx"], optimization_level=1, seed_transpiler=0)
            ops = self.circuit.count_ops()
        self.stats = {
            "depth": int(self.circuit.depth()),
            "two_qubit_gates": int(ops.get("cx", 0)),
            "total_gates": int(sum(ops.values())),
        }

    def probabilities(self, gammas: np.ndarray, betas: np.ndarray) -> np.ndarray:
        from qiskit.quantum_info import Statevector

        bound = self.circuit.assign_parameters({self.gammas: list(gammas), self.betas: list(betas)})
        return np.asarray(Statevector.from_instruction(bound).probabilities())


class _NumpyBackend:
    def __init__(self, energies: np.ndarray, n: int) -> None:
        self.energies, self.n = energies, n
        self.stats = {"depth": None, "two_qubit_gates": None, "total_gates": None}

    def probabilities(self, gammas: np.ndarray, betas: np.ndarray) -> np.ndarray:
        return numpy_probabilities(self.energies, self.n, gammas, betas)


def _cvar(values: np.ndarray, alpha: float) -> float:
    if alpha >= 1.0:
        return float(values.mean())
    k = max(int(np.ceil(alpha * len(values))), 1)
    return float(np.sort(values)[:k].mean())


def qaoa(
    problem: BasketProblem,
    budget: Budget,
    seed: int | None = None,
    *,
    backend: str = "qiskit",
    p: int = 2,
    shots: int = 256,
    maxiter: int = 40,
    cvar_alpha: float = 0.5,
) -> SolverResult:
    """Optimise QAOA angles with COBYLA on shot estimates, then draw ``budget.max_evals`` final shots."""
    n = problem.n
    if n > MAX_QUBITS:
        raise ValueError(f"{n} qubits exceeds the local statevector limit ({MAX_QUBITS})")
    if p < 1 or shots < 1:
        raise ValueError("p and shots must be >= 1")
    timer = Timer()
    rng = np.random.default_rng(seed)

    with timer.stage("encode"):
        h, J, scale = _normalised_ising(problem)
        # Classical cost of simulating/reading out energies; on hardware each shot's
        # energy would be evaluated from Q instead of from this table.
        energies = ising_energies(h, J)
    if backend not in ("qiskit", "numpy"):
        raise ValueError(f"unknown QAOA backend: {backend}")
    sim = _QiskitBackend(h, J, p, timer) if backend == "qiskit" else _NumpyBackend(energies, n)

    state = {"shots": 0, "iters": 0}

    def cost(theta: np.ndarray) -> float:
        probs = sim.probabilities(theta[:p], theta[p:])
        probs = probs / probs.sum()
        idx = rng.choice(len(probs), size=shots, p=probs)
        state["shots"] += shots
        state["iters"] += 1
        return _cvar(energies[idx], cvar_alpha)

    with timer.stage("optimize"):
        frac = (np.arange(p) + 0.5) / p
        theta0 = np.concatenate([0.6 * frac, 0.6 * (1 - frac)])  # annealing-style ramp
        res = minimize(cost, theta0, method="COBYLA", options={"maxiter": maxiter, "rhobeg": 0.15})
        theta = res.x

    with timer.stage("sample"):
        probs = sim.probabilities(theta[:p], theta[p:])
        probs = probs / probs.sum()
        idx = rng.choice(len(probs), size=budget.max_evals, p=probs)
        state["shots"] += budget.max_evals
        cands = ((idx[:, None] >> np.arange(n)) & 1).astype(np.int8)

    z, val, rate, repaired, raw_best = best_of_candidates(problem, cands, timer)

    # Simulator-only diagnostics (need the full distribution; not available on hardware).
    bits = problem.all_bitstrings()
    card_ok = bits.sum(axis=1) == problem.basket_size
    p_cardinality = float(probs[card_ok].sum())
    z_opt, _, _ = problem.brute_force()
    opt_index = int((z_opt << np.arange(n)).sum())
    return SolverResult(
        "qaoa",
        z,
        val,
        "ok",
        n_evals=budget.max_evals,
        n_shots=state["shots"],
        raw_feasible_rate=rate,
        repaired=repaired,
        raw_best_objective=raw_best,
        timings=timer.t,
        meta={
            "backend": backend,
            "qubits": n,
            "p": p,
            "shots_per_iter": shots,
            "optimizer_iters": state["iters"],
            "cvar_alpha": cvar_alpha,
            "energy_scale": scale,
            "theta": [float(x) for x in theta],
            "p_optimal_state": float(probs[opt_index]),
            # Normalised Ising energy; uniform sampling scores exactly 0 on this scale.
            "expected_energy": float(probs @ energies),
            "p_cardinality_ok": p_cardinality,
            "uniform_p_optimal_state": 1.0 / (1 << n),
            **sim.stats,
            "hardware": "none (noiseless statevector simulation)",
        },
    )
