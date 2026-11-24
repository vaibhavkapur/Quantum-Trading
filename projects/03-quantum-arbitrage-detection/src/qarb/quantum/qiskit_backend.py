"""Qiskit circuit construction for the cycle QUBO.

Used for two things: (1) cross-checking the NumPy simulator against an
independently built gate-level circuit, and (2) reporting compiled circuit
resources (qubits, depth, two-qubit gates) for a generic CX/RZ/SX basis.
Nothing here talks to hardware.
"""

from __future__ import annotations

import time

import numpy as np

from .qubo import CycleQubo, energy_vector, to_ising


def _phase_ising(qubo: CycleQubo) -> tuple[np.ndarray, np.ndarray]:
    """Ising coefficients of the standardized phase energy (E - mean) / std."""
    E = energy_vector(qubo)
    spread = float(E.std()) or 1.0
    h, J, _ = to_ising(qubo.Q, qubo.offset)
    return h / spread, J / spread


def build_circuit(qubo: CycleQubo, layers: int, measure: bool = False):
    """Parametrized QAOA circuit: H^n, then per layer RZ/RZZ cost phases and RX mixers."""
    from qiskit import QuantumCircuit
    from qiskit.circuit import ParameterVector

    h, J = _phase_ising(qubo)
    n = qubo.n_vars
    gam = ParameterVector("gamma", layers)
    bet = ParameterVector("beta", layers)
    qc = QuantumCircuit(n)
    qc.h(range(n))
    for layer in range(layers):
        for i in range(n):
            if h[i] != 0:
                qc.rz(2 * gam[layer] * float(h[i]), i)
        for i in range(n):
            for j in range(i + 1, n):
                if J[i, j] != 0:
                    qc.rzz(2 * gam[layer] * float(J[i, j]), i, j)
        for i in range(n):
            qc.rx(2 * bet[layer], i)
    if measure:
        qc.measure_all()
    return qc, gam, bet


def probabilities_fn(qubo: CycleQubo, layers: int):
    """probs(gammas, betas) via Qiskit's Statevector, same convention as the NumPy engine."""
    from qiskit.quantum_info import Statevector

    qc, gam, bet = build_circuit(qubo, layers)

    def probs(gammas, betas) -> np.ndarray:
        bound = qc.assign_parameters({**dict(zip(gam, map(float, gammas))), **dict(zip(bet, map(float, betas)))})
        p = Statevector(bound).probabilities()
        return p / p.sum()

    return probs


def circuit_metrics(qubo: CycleQubo, layers: int) -> dict:
    """Resources after transpiling to a CX/RZ/SX/X basis with all-to-all connectivity."""
    from qiskit import transpile

    t0 = time.perf_counter()
    qc, _, _ = build_circuit(qubo, layers, measure=True)
    tqc = transpile(qc, basis_gates=["cx", "rz", "sx", "x"], optimization_level=1, seed_transpiler=0)
    ops = tqc.count_ops()
    return {
        "n_qubits": qubo.n_vars,
        "layers": layers,
        "cycle_len": qubo.L,
        "anchored": qubo.anchor is not None,
        "depth": int(tqc.depth()),
        "two_qubit_gates": int(ops.get("cx", 0)),
        "total_gates": int(sum(v for k, v in ops.items() if k not in ("measure", "barrier"))),
        "compile_ms": (time.perf_counter() - t0) * 1e3,
        "basis": "cx,rz,sx,x (all-to-all)",
    }
