"""Circuits: distribution loading, payoff rotation and the Grover operator."""
from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import UCRYGate

from .discretize import DiscretizedProblem


def distribution_loader(probs: np.ndarray) -> QuantumCircuit:
    """Prepare sum_i sqrt(p_i)|i> with a Grover-Rudolph tree: starting from the
    most significant qubit, each qubit is rotated by an angle that depends on
    the qubits above it (a uniformly controlled RY). Exact, but O(2**m) gates."""
    probs = np.asarray(probs, dtype=float)
    m = int(np.log2(probs.size))
    if 2**m != probs.size or m < 1:
        raise ValueError("probs must have a power-of-two length >= 2")
    qc = QuantumCircuit(m, name="P")
    for target in range(m - 1, -1, -1):
        # mass[c, b]: probability that the higher bits equal c and this bit equals b
        mass = probs.reshape(2 ** (m - target - 1), 2, 2**target).sum(axis=2)
        total = mass.sum(axis=1)
        ratio = np.divide(mass[:, 1], total, out=np.zeros_like(total), where=total > 0)
        angles = 2 * np.arcsin(np.sqrt(np.clip(ratio, 0.0, 1.0)))
        if target == m - 1:
            qc.ry(float(angles[0]), target)
        else:
            qc.append(UCRYGate(list(angles)), [target] + list(range(target + 1, m)))
    return qc


def build_state_preparation(problem: DiscretizedProblem) -> QuantumCircuit:
    """Load sqrt(p) on every step register. Steps are independent increments,
    so the joint distribution is a product of identical small loaders."""
    m = problem.qubits_per_step
    qc = QuantumCircuit(problem.num_state_qubits, name="P")
    loader = distribution_loader(problem.step_probs)
    for j in range(problem.steps):
        qc.compose(loader, range(j * m, (j + 1) * m), inplace=True)
    return qc


def build_a_operator(problem: DiscretizedProblem) -> QuantumCircuit:
    """A|0> = sum_i sqrt(p_i) |i> (sqrt(1-g_i)|0> + sqrt(g_i)|1>), g = payoff/max.

    The payoff is written by a uniformly controlled RY, i.e. an exact lookup
    table over all basis states. It is verifiable for tiny registers but costs
    O(2**n) gates; it is not a scalable arithmetic oracle.
    """
    n = problem.num_state_qubits
    qc = QuantumCircuit(n + 1, name="A")
    qc.compose(build_state_preparation(problem), range(n), inplace=True)
    angles = 2 * np.arcsin(np.sqrt(np.clip(problem.normalized_payoffs, 0.0, 1.0)))
    qc.append(UCRYGate(list(angles)), [n] + list(range(n)))
    return qc


def build_grover_operator(a_operator: QuantumCircuit) -> QuantumCircuit:
    """Q = -A S_0 A^dagger S_chi; the objective qubit is the last one."""
    total = a_operator.num_qubits
    objective = total - 1
    qc = QuantumCircuit(total, name="Q")
    qc.z(objective)                                   # S_chi: phase-flip the "payoff" states
    qc.compose(a_operator.inverse(), inplace=True)
    qc.x(range(total))                                # S_0: phase-flip |0...0>
    qc.h(objective)
    qc.mcx(list(range(objective)), objective)
    qc.h(objective)
    qc.x(range(total))
    qc.compose(a_operator, inplace=True)
    qc.global_phase += np.pi
    return qc
