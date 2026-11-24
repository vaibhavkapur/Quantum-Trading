"""QAOA circuit in Qiskit, used to cross-check the NumPy simulator.

The cost unitary exp(-i*gamma*(E - mean)/std) is diagonal, so with binary
variables it is exactly a product of phase gates (linear terms) and
controlled-phase gates (quadratic terms); the mean shift is a global phase.
"""
from __future__ import annotations

import numpy as np


def circuit(qubo, mean: float, std: float, gammas, betas):
    from qiskit import QuantumCircuit

    qc = QuantumCircuit(qubo.n)
    qc.h(range(qubo.n))
    for g, b in zip(gammas, betas):
        for i in range(qubo.n):
            if qubo.lin[i]:
                qc.p(-g * qubo.lin[i] / std, i)
            for j in range(i + 1, qubo.n):
                if qubo.quad[i, j]:
                    qc.cp(-g * qubo.quad[i, j] / std, i, j)
        qc.rx(2 * b, range(qubo.n))
    return qc


def probabilities(qubo, mean: float, std: float, gammas, betas) -> np.ndarray:
    from qiskit.quantum_info import Statevector

    return Statevector.from_instruction(circuit(qubo, mean, std, gammas, betas)).probabilities()
