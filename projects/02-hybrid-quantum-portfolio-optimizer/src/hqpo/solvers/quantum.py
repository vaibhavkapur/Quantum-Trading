"""Quantum-style solvers that work on the QUBO: QAOA (simulated) and annealing (simulated).

Nothing here touches a QPU.  ``qaoa`` is an exact, noiseless statevector simulation with
finite-shot sampling; ``qaoa-noisy`` re-samples the optimised circuit under a depolarising
noise model in Qiskit Aer; ``qubo-sa`` is *classical* simulated annealing on the QUBO (the
standard software stand-in for a quantum annealer).  Results from all three must be
reported as simulation, never as hardware.
"""
from __future__ import annotations

import time

import numpy as np
from scipy.optimize import minimize

from ..models.qubo import QuboModel, build_qubo, qubit_count
from .base import Budget, Solver, SolverUnavailable


# --------------------------------------------------------------------------- QAOA kernel
def apply_mixer(psi: np.ndarray, n: int, beta: float) -> np.ndarray:
    """Apply ``exp(-i beta X)`` to every qubit of an ``n``-qubit state (in place)."""
    c, s = np.cos(beta), -1j * np.sin(beta)
    for k in range(n):
        v = psi.reshape(-1, 2, 1 << k)
        a0 = v[:, 0, :].copy()
        v[:, 0, :] = c * a0 + s * v[:, 1, :]
        v[:, 1, :] = s * a0 + c * v[:, 1, :]
    return psi


def qaoa_state(energies: np.ndarray, gammas, betas) -> np.ndarray:
    """QAOA statevector for a diagonal cost Hamiltonian given by ``energies``."""
    dim = energies.shape[0]
    n = dim.bit_length() - 1
    psi = np.full(dim, 1.0 / np.sqrt(dim), dtype=complex)
    for g, b in zip(gammas, betas):
        psi *= np.exp(-1j * g * energies)
        apply_mixer(psi, n, b)
    return psi


def cvar(probs: np.ndarray, energies: np.ndarray, order: np.ndarray, alpha: float) -> float:
    """Mean energy of the lowest-``alpha`` probability tail (``alpha = 1`` is the expectation)."""
    if alpha >= 1.0:
        return float(probs @ energies)
    p_sorted = probs[order]
    cum = np.cumsum(p_sorted)
    take = np.clip(alpha - (cum - p_sorted), 0.0, p_sorted)
    return float(take @ energies[order] / alpha)


def ramp_init(p: int, delta: float = 0.75) -> np.ndarray:
    """Linear-ramp (annealing-like) initial angles: gamma rises, beta falls."""
    k = (np.arange(p) + 0.5) / p
    return np.concatenate([delta * k, delta * (1.0 - k)])


def build_qaoa_circuit(h: np.ndarray, J: np.ndarray, gammas, betas, measure: bool = False):
    """Qiskit circuit for ``prod_l exp(-i beta_l sum X) exp(-i gamma_l H)`` on ``|+>^n``."""
    from qiskit import QuantumCircuit

    n = len(h)
    qc = QuantumCircuit(n)
    qc.h(range(n))
    for g, b in zip(gammas, betas):
        for i in range(n):
            if h[i] != 0.0:
                qc.rz(2.0 * g * h[i], i)
        for i in range(n):
            for j in range(i + 1, n):
                if J[i, j] != 0.0:
                    qc.rzz(2.0 * g * J[i, j], i, j)
        for i in range(n):
            qc.rx(2.0 * b, i)
    if measure:
        qc.measure_all()
    return qc


def circuit_metrics(h, J, p: int) -> dict:
    """Depth and gate counts after transpiling to a {rz, sx, x, cx} basis."""
    from qiskit import transpile
    from qiskit.transpiler import CouplingMap

    n = len(h)
    qc = build_qaoa_circuit(h, J, np.full(p, 0.37), np.full(p, 0.21))
    basis = ["rz", "sx", "x", "cx"]
    full = transpile(qc, basis_gates=basis, optimization_level=1, seed_transpiler=0)
    out = {"depth": int(full.depth()), "two_qubit_gates": int(full.count_ops().get("cx", 0))}
    if n >= 2:
        line = transpile(qc, basis_gates=basis, coupling_map=CouplingMap.from_line(n),
                         optimization_level=1, seed_transpiler=0)
        out["depth_line"] = int(line.depth())
        out["two_qubit_gates_line"] = int(line.count_ops().get("cx", 0))
    return out


def best_of_samples(qubo: QuboModel, indices: np.ndarray, counts: np.ndarray):
    """Pick the lowest-QUBO-energy sampled bitstring; also return sample feasibility stats."""
    bits = qubo.index_to_bits(indices)
    Z = bits @ qubo.T.T
    p = qubo.problem
    viol = p.total_violation_batch(Z)
    energy = np.einsum("ki,ij,kj->k", bits, qubo.Q, bits) + qubo.offset
    k = int(np.argmin(energy))
    feas_frac = float(counts[viol == 0].sum() / counts.sum())
    return Z[k].astype(int), feas_frac, Z, viol


class Qaoa(Solver):
    """QAOA with a transverse-field mixer, simulated exactly (noiseless) with shot sampling.

    Parameters: ``p`` layers, ``shots`` for the final sampling, ``cvar_alpha`` objective tail,
    ``expectation_shots`` (0 = exact expectation; >0 = finite-shot estimate each evaluation),
    ``max_qubits`` simulator limit, ``restarts``, ``circuit_metrics`` and optional ``noise``.
    """

    name = "qaoa"
    stochastic = True

    def _solve(self, problem, budget, rng, timer):
        prm = self.params
        p_layers = int(prm.get("p", 2))
        shots = int(prm.get("shots", 1024))
        alpha = float(prm.get("cvar_alpha", 0.25))
        est_shots = int(prm.get("expectation_shots", 0))
        max_qubits = int(prm.get("max_qubits", 14 if prm.get("noise") else 20))
        pen = prm.get("penalty", {})

        need = qubit_count(problem)["total"]
        if need > max_qubits:
            raise SolverUnavailable(
                f"{need} qubits needed, simulator limit is {max_qubits} (statevector 2^n)")

        with timer.section("model_build"):
            qubo = build_qubo(problem, method=pen.get("method", "gradient_bound"),
                              scale=pen.get("scale", 1.0), value=pen.get("value"))
            n = qubo.n_qubits
            h, J, const = qubo.to_ising()
        meta = {"n_qubits": n, **{f"qubits_{k}": v for k, v in qubo.qubit_breakdown().items()
                                  if k != "total"},
                "p": p_layers, "shots": shots, "cvar_alpha": alpha, "penalty": qubo.penalty,
                "backend": "numpy-statevector (noiseless)"}

        with timer.section("compile"):
            if prm.get("circuit_metrics", True):
                try:
                    meta.update(circuit_metrics(h, J, p_layers))
                except Exception as exc:  # metrics are diagnostics; never fail the solve
                    meta["circuit_metrics_error"] = f"{type(exc).__name__}: {exc}"

        with timer.section("state_prep"):
            E = qubo.energies_all()
            scale = float(np.abs(E - E.mean()).mean()) or 1.0
            En = (E - E.min()) / scale          # normalised so angles are O(1)
            order = np.argsort(En)

        evals = 0
        total_shots = 0
        deadline = timer.t0 + budget.wall_time_s * 0.9

        class _Stop(Exception):
            pass

        best = {"val": np.inf, "x": None}

        def fun(theta):
            nonlocal evals, total_shots
            if evals >= budget.max_evals or time.perf_counter() > deadline:
                raise _Stop
            evals += 1
            psi = qaoa_state(En, theta[:p_layers], theta[p_layers:])
            probs = np.abs(psi) ** 2
            if est_shots > 0:
                total_shots += est_shots
                probs = rng.multinomial(est_shots, probs / probs.sum()) / est_shots
            val = cvar(probs, En, order, alpha)
            if val < best["val"]:
                best["val"], best["x"] = val, np.array(theta, dtype=float)
            return val

        with timer.section("optimize"):
            restarts = int(prm.get("restarts", 2))
            maxiter = int(prm.get("maxiter", 150))
            for r in range(restarts):
                x0 = ramp_init(p_layers) if r == 0 else rng.uniform(0, np.pi / 2, 2 * p_layers)
                try:
                    minimize(fun, x0, method="COBYLA",
                             options={"maxiter": maxiter, "rhobeg": 0.3, "tol": 1e-4})
                except _Stop:
                    break
        theta = best["x"] if best["x"] is not None else ramp_init(p_layers)
        meta["optimizer"] = "COBYLA"
        meta["best_cost"] = float(best["val"])

        with timer.section("sampling"):
            psi = qaoa_state(En, theta[:p_layers], theta[p_layers:])
            probs = np.abs(psi) ** 2
            probs /= probs.sum()
            noise = prm.get("noise")
            if noise:
                indices, counts = self._sample_noisy(h, J, theta, p_layers, shots, noise, rng)
                meta["backend"] = (f"qiskit-aer depolarising noise p1={noise.get('p1', 0.001)}, "
                                   f"p2={noise.get('p2', 0.01)}, readout={noise.get('readout', 0.0)}")
            else:
                draw = rng.multinomial(shots, probs)
                indices = np.flatnonzero(draw)
                counts = draw[indices]
            total_shots += shots
        meta["total_shots"] = total_shots

        with timer.section("decode"):
            z, feas_frac, _, _ = best_of_samples(qubo, indices, counts)
            meta["sample_feasible_fraction"] = feas_frac
            meta["distinct_samples"] = int(len(indices))
            # simulator-only diagnostics from the exact distribution
            meta["p_ground_state"] = float(probs[En <= 1e-12].sum())
            meta["expected_energy_ratio"] = float(probs @ En / max(En.mean(), 1e-300))
        self._last = {"qubo": qubo, "theta": theta, "probs": probs}
        return z, evals, meta

    @staticmethod
    def _sample_noisy(h, J, theta, p_layers, shots, noise, rng):
        try:
            from qiskit import transpile
            from qiskit_aer import AerSimulator
            from qiskit_aer.noise import NoiseModel, ReadoutError, depolarizing_error
        except ImportError as exc:
            raise SolverUnavailable("qiskit-aer is required for noisy sampling") from exc
        p1, p2, ro = noise.get("p1", 0.001), noise.get("p2", 0.01), noise.get("readout", 0.0)
        model = NoiseModel()
        model.add_all_qubit_quantum_error(depolarizing_error(p1, 1), ["rz", "sx", "x"])
        model.add_all_qubit_quantum_error(depolarizing_error(p2, 2), ["cx"])
        if ro > 0:
            model.add_all_qubit_readout_error(ReadoutError([[1 - ro, ro], [ro, 1 - ro]]))
        sim = AerSimulator(noise_model=model, seed_simulator=int(rng.integers(2 ** 31 - 1)))
        qc = build_qaoa_circuit(h, J, theta[:p_layers], theta[p_layers:], measure=True)
        tqc = transpile(qc, basis_gates=["rz", "sx", "x", "cx"], optimization_level=1,
                        seed_transpiler=0)
        counts = sim.run(tqc, shots=shots).result().get_counts()
        idx = np.array([int(k, 2) for k in counts], dtype=np.int64)  # qiskit: qubit 0 is LSB
        return idx, np.array(list(counts.values()), dtype=np.int64)


class QaoaNoisy(Qaoa):
    """QAOA angles optimised noiselessly, final shots drawn from a noisy Aer simulation."""

    name = "qaoa-noisy"

    def __init__(self, **params):
        params.setdefault("noise", {"p1": 0.001, "p2": 0.01, "readout": 0.01})
        super().__init__(**params)


class RandomQubo(Solver):
    """Control: uniformly random bitstrings, same shot count and decoding as QAOA.

    Any solution quality QAOA shows beyond this baseline is attributable to the circuit;
    quality it shares with this baseline comes from "sample many, keep the best".
    """

    name = "random-qubo"
    stochastic = True

    def _solve(self, problem, budget, rng, timer):
        prm = self.params
        pen = prm.get("penalty", {})
        shots = int(prm.get("shots", 1024))
        with timer.section("model_build"):
            qubo = build_qubo(problem, method=pen.get("method", "gradient_bound"),
                              scale=pen.get("scale", 1.0), value=pen.get("value"))
        with timer.section("sampling"):
            bits = rng.integers(0, 2, size=(shots, qubo.n_qubits))
        with timer.section("decode"):
            energy = np.einsum("ki,ij,kj->k", bits, qubo.Q, bits) + qubo.offset
            Z = bits @ qubo.T.T
            feas = float(np.mean(problem.total_violation_batch(Z) == 0))
            z = Z[int(np.argmin(energy))].astype(int)
        return z, shots, {"n_qubits": qubo.n_qubits, "shots": shots, "penalty": qubo.penalty,
                          "sample_feasible_fraction": feas, "backend": "uniform random bitstrings"}


# --------------------------------------------------------------------------- annealing (simulated)
def _numpy_qubo_sa(qubo: QuboModel, rng, sweeps: int, deadline: float, max_evals: int,
                   reads: int):
    lin, Jm = qubo.symmetric()
    J2 = 2.0 * Jm                      # dE_i = (1 - 2 x_i) (lin_i + sum_j 2 J_ij x_j)
    n = qubo.n_qubits
    typical = float(np.abs(lin).mean() + np.abs(J2).sum(1).mean()) or 1.0
    t_hot, t_cold = 0.5 * typical, 1e-3 * typical
    temps = t_hot * (t_cold / t_hot) ** (np.arange(sweeps) / max(sweeps - 1, 1))
    best_x, best_e, evals, done = None, np.inf, 0, 0
    for _ in range(reads):
        x = rng.integers(0, 2, n)
        field = lin + J2 @ x
        e = qubo.energy(x)
        for t in temps:
            u = rng.random(n)
            for i in rng.permutation(n):
                d = (1 - 2 * x[i]) * field[i]
                if d <= 0 or u[i] < np.exp(-d / t):
                    x[i] ^= 1
                    field += J2[:, i] * (2 * x[i] - 1)
                    e += d
            evals += n
            if evals >= max_evals or time.perf_counter() > deadline:
                break
        if e < best_e:
            best_e, best_x = e, x.copy()
        done += 1
        if evals >= max_evals or time.perf_counter() > deadline:
            break
    return best_x, evals, done


class QuboAnnealer(Solver):
    """Simulated annealing on the QUBO: D-Wave Ocean's sampler if installed, else NumPy.

    This is the classical proxy for quantum annealing.  A real annealer (``DWaveSampler``)
    is deliberately not wired in: no QPU access was available to validate it.
    """

    name = "qubo-sa"
    stochastic = True

    def _solve(self, problem, budget, rng, timer):
        prm = self.params
        pen = prm.get("penalty", {})
        backend = prm.get("backend", "auto")
        sweeps = int(prm.get("sweeps", 400))
        with timer.section("model_build"):
            qubo = build_qubo(problem, method=pen.get("method", "gradient_bound"),
                              scale=pen.get("scale", 1.0), value=pen.get("value"))
        n = qubo.n_qubits
        meta = {"n_qubits": n, "penalty": qubo.penalty,
                **{f"qubits_{k}": v for k, v in qubo.qubit_breakdown().items() if k != "total"}}
        deadline = timer.t0 + budget.wall_time_s * 0.9
        sampler = None
        if backend in ("auto", "ocean"):
            try:
                from dwave.samplers import SimulatedAnnealingSampler
                sampler = SimulatedAnnealingSampler()
            except ImportError:
                if backend == "ocean":
                    raise SolverUnavailable("dwave-samplers is not installed")
                meta["fallback"] = "numpy"
                meta["fallback_reason"] = "dwave-samplers not installed"

        with timer.section("optimize"):
            if sampler is not None:
                import dimod

                meta["backend"] = "dwave-samplers SimulatedAnnealingSampler"
                bqm = dimod.BinaryQuadraticModel.from_numpy_matrix(qubo.Q) \
                    if hasattr(dimod.BinaryQuadraticModel, "from_numpy_matrix") else \
                    dimod.BinaryQuadraticModel(qubo.Q, "BINARY")
                best_x, best_e, evals, reads = None, np.inf, 0, 0
                cap = max(1, budget.max_evals // (sweeps * n))
                while True:
                    batch = max(1, min(int(prm.get("reads_per_batch", 10)), cap - reads))
                    ss = sampler.sample(bqm, num_reads=batch, num_sweeps=sweeps,
                                        seed=int(rng.integers(2 ** 31 - 1)))
                    evals += batch * sweeps * n
                    reads += batch
                    first = ss.first
                    if first.energy < best_e:
                        best_e = first.energy
                        best_x = np.array([first.sample[i] for i in range(n)], dtype=int)
                    if evals >= budget.max_evals or time.perf_counter() > deadline:
                        break
            else:
                meta["backend"] = "numpy single-flip Metropolis"
                best_x, evals, reads = _numpy_qubo_sa(
                    qubo, rng, sweeps, deadline, budget.max_evals, int(prm.get("reads", 10 ** 9)))
        meta["reads"], meta["sweeps"] = reads, sweeps
        with timer.section("decode"):
            z = qubo.decode(best_x)
        return z, evals, meta
