"""QUBO heuristics: QAOA (noiseless statevector), simulated annealing, random control.

All three share the same encoding, decoding and repair, so differences in
solution quality come from the sampler alone. `random_repair` draws uniform
bitstrings with the same shot budget as QAOA and is the control that shows how
much of QAOA's result is due to repair + best-of-samples rather than the circuit.
"""
from __future__ import annotations

import time

import numpy as np
from scipy.optimize import minimize

from .classical import SolveTimeout
from .problem import AllocationProblem
from .qubo import Qubo, best_of, build_qubo


class TooManyQubits(Exception):
    pass


def _encode(p: AllocationProblem, max_qubits: int) -> Qubo:
    qubo = build_qubo(p)
    if qubo.n > max_qubits:
        raise TooManyQubits(f"{qubo.n} qubits needed, budget is {max_qubits}")
    return qubo


def normalize(energies: np.ndarray):
    """Shift/scale so QAOA angles are comparable across instances."""
    mean, std = float(energies.mean()), float(energies.std())
    return mean, (std if std > 0 else 1.0)


def qaoa_state(en: np.ndarray, gammas, betas) -> np.ndarray:
    """Statevector after p QAOA layers with diagonal cost `en` and an X mixer.

    Qubit i is bit i of the basis-state index (same convention as Qiskit).
    """
    n = int(np.log2(en.size))
    psi = np.full(en.size, 1.0 / np.sqrt(en.size), dtype=np.complex128)
    shape = (2,) * n
    for g, b in zip(gammas, betas):
        psi = psi * np.exp(-1j * g * en)
        t = psi.reshape(shape)
        c, s = np.cos(b), -1j * np.sin(b)
        for ax in range(n):  # RX(2b) on every qubit: cos(b) I - i sin(b) X
            t = c * t + s * np.flip(t, axis=ax)
        psi = t.reshape(-1)
    return psi


def initial_angles(layers: int) -> np.ndarray:
    """Linear-ramp schedule: gamma increases, beta decreases across layers."""
    f = (np.arange(layers) + 0.5) / layers
    return np.concatenate([0.6 * f, 0.6 * (1 - f)])


def qaoa(p: AllocationProblem, rng, layers=2, maxiter=40, shots=256, max_qubits=16,
         backend="numpy", warm=None, deadline=None, **_):
    info = {"evals": 0, "shots": shots, "layers": layers}
    t0 = time.perf_counter()
    qubo = _encode(p, max_qubits)
    info["qubits"] = qubo.n
    if qubo.n == 0:
        return tuple([0] * p.V), info
    energies = qubo.energies()
    mean, std = normalize(energies)
    en = (energies - mean) / std
    t1 = time.perf_counter()

    if backend == "qiskit":
        from .qiskit_backend import probabilities as qk_probs
        probs_of = lambda th: qk_probs(qubo, mean, std, th[:layers], th[layers:])
    else:
        def probs_of(th):
            psi = qaoa_state(en, th[:layers], th[layers:])
            return psi.real ** 2 + psi.imag ** 2

    def expectation(th):
        if deadline is not None and time.perf_counter() > deadline:
            raise SolveTimeout("QAOA exceeded its wall-time budget")
        info["evals"] += 1
        return float(probs_of(th) @ en)

    x0 = np.asarray(warm, dtype=float) if warm is not None and len(warm) == 2 * layers else initial_angles(layers)
    res = minimize(expectation, x0, method="COBYLA", options={"maxiter": maxiter, "rhobeg": 0.2})
    theta = res.x if res.fun <= expectation(x0) else x0
    t2 = time.perf_counter()

    probs = probs_of(theta)
    probs = probs / probs.sum()
    states = rng.choice(probs.size, size=shots, p=probs)
    t3 = time.perf_counter()
    x, n_feasible = best_of(p, qubo, states)
    t4 = time.perf_counter()
    info.update(theta=[float(a) for a in theta], expectation=float(probs @ energies),
                feasible_shots=int(sum(1 for s in states if _is_feasible(p, qubo, int(s)))),
                unique_states=int(len(set(states.tolist()))), unique_feasible=int(n_feasible),
                t_encode=t1 - t0, t_optimize=t2 - t1, t_sample=t3 - t2, t_decode=t4 - t3)
    return x, info


def _is_feasible(p, qubo, z) -> bool:
    x = [0] * p.V
    for i, (v, k) in enumerate(qubo.var_map):
        if (z >> i) & 1:
            if x[v]:
                return False
            x[v] = k
    return sum(x) == p.target_units


def random_repair(p: AllocationProblem, rng, shots=256, max_qubits=24, **_):
    t0 = time.perf_counter()
    qubo = _encode(p, max_qubits)
    info = {"evals": 0, "shots": shots, "qubits": qubo.n}
    if qubo.n == 0:
        return tuple([0] * p.V), info
    states = rng.integers(0, 1 << qubo.n, size=shots)
    t1 = time.perf_counter()
    x, n_feasible = best_of(p, qubo, states)
    info.update(unique_feasible=int(n_feasible), t_encode=0.0, t_sample=t1 - t0,
                t_decode=time.perf_counter() - t1)
    return x, info


def simulated_annealing(p: AllocationProblem, rng, sweeps=60, restarts=2, t_hot=1.0, t_cold=0.02,
                        max_qubits=24, deadline=None, **_):
    """Single-flip Metropolis annealing on the QUBO (classical)."""
    t0 = time.perf_counter()
    qubo = _encode(p, max_qubits)
    info = {"evals": 0, "qubits": qubo.n, "sweeps": sweeps, "restarts": restarts}
    n = qubo.n
    if n == 0:
        return tuple([0] * p.V), info
    sym = qubo.quad + qubo.quad.T
    scale = float(np.abs(qubo.lin).mean() + np.abs(sym).mean()) or 1.0
    temps = np.geomspace(t_hot, t_cold, sweeps) * scale
    t1 = time.perf_counter()
    finals = []
    for _ in range(restarts):
        y = rng.integers(0, 2, n).astype(float)
        field = qubo.lin + sym @ y
        e = e_best = qubo.energy(y)
        y_best = y.copy()
        for T in temps:
            if deadline is not None and time.perf_counter() > deadline:
                raise SolveTimeout("annealing exceeded its wall-time budget")
            u = rng.random(n)
            for i in rng.permutation(n):
                d = (1.0 - 2.0 * y[i]) * field[i]
                info["evals"] += 1
                if d <= 0 or u[i] < np.exp(-d / T):
                    s = 1.0 - 2.0 * y[i]
                    y[i] += s
                    field += s * sym[i]
                    e += d
                    if e < e_best - 1e-12:
                        e_best, y_best = e, y.copy()
        finals.append(int(sum(1 << i for i in range(n) if y_best[i] > 0.5)))
    t2 = time.perf_counter()
    x, n_feasible = best_of(p, qubo, finals)
    info.update(unique_feasible=int(n_feasible), t_encode=t1 - t0, t_optimize=t2 - t1,
                t_decode=time.perf_counter() - t2)
    return x, info
