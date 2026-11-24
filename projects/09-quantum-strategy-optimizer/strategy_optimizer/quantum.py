"""Gate-based QAOA on Qiskit's noiseless CPU statevector simulator."""
from __future__ import annotations

from time import perf_counter

import numpy as np


def qaoa_circuit(problem, layers):
    from qiskit import QuantumCircuit
    from qiskit.circuit import ParameterVector
    if type(layers) is not int or layers < 1:
        raise ValueError("QAOA needs at least one layer")
    _, h, pairs = problem.ising()
    scale = max(float(np.max(np.abs(h))), float(np.max(np.abs(pairs))), 1e-12)
    angles = ParameterVector("angle", 2 * layers)
    circuit = QuantumCircuit(problem.n_bits)
    for layer in range(layers):
        gamma, beta = angles[2 * layer], angles[2 * layer + 1]
        for i, value in enumerate(h):
            if value:
                circuit.rz(2 * gamma * float(value / scale), i)
        for i, j in zip(*np.nonzero(pairs)):
            circuit.rzz(2 * gamma * float(pairs[i, j] / scale), int(i), int(j))
        for i in range(problem.n_bits):
            circuit.rx(2 * beta, i)
    return circuit, angles, scale


def optimize_qaoa(problem, ledger, rng, layers=1, shots_per_evaluation=8, max_qubits=12):
    if problem.n_bits > min(max_qubits, 16):
        raise ValueError(f"QAOA requires {problem.n_bits} qubits, above configured limit {max_qubits}")
    if type(shots_per_evaluation) is not int or shots_per_evaluation < 1:
        raise ValueError("shots_per_evaluation must be positive")
    from qiskit import QuantumCircuit, transpile
    from qiskit.quantum_info import Statevector
    from scipy.optimize import minimize
    timing = {}
    started = perf_counter()
    circuit, angles, scale = qaoa_circuit(problem, layers)
    circuit = transpile(circuit, basis_gates=["rz", "rx", "rzz"], optimization_level=0, seed_transpiler=0)
    timing["compilation_s"] = perf_counter() - started
    started = perf_counter()
    preparation = QuantumCircuit(problem.n_bits)
    preparation.h(range(problem.n_bits))
    initial = Statevector.from_instruction(preparation)
    timing["state_preparation_s"] = perf_counter() - started
    timing["simulation_s"] = timing["sampling_decoding_s"] = 0.0
    calls = 0
    # Count each sampled bitstring (including repeats and infeasible samples) as
    # one objective evaluation. No full energy table or exhaustive decoded scan.
    target_calls = (ledger.budget + shots_per_evaluation - 1) // shots_per_evaluation

    def objective(theta):
        nonlocal calls
        remaining = ledger.budget - len(ledger.trace)
        if remaining <= 0:
            from .solvers import BudgetExhausted
            raise BudgetExhausted()
        started = perf_counter()
        bound = circuit.assign_parameters(dict(zip(angles, theta)))
        probabilities = initial.evolve(bound).probabilities()
        timing["simulation_s"] += perf_counter() - started
        started = perf_counter()
        count = min(shots_per_evaluation, remaining)
        outcomes = rng.choice(len(probabilities), size=count, p=probabilities / probabilities.sum())
        # Qiskit uses little-endian qubit indexing in statevector indices.
        bits = ((outcomes[:, None] >> np.arange(problem.n_bits)) & 1).astype(int)
        timing["sampling_decoding_s"] += perf_counter() - started
        calls += 1
        return float(np.mean([ledger.evaluate(x) for x in bits]) / scale)

    theta = rng.uniform(0, np.pi, 2 * layers)
    from .solvers import BudgetExhausted
    try:
        result = minimize(objective, theta, method="COBYLA",
                          options={"maxiter": target_calls, "rhobeg": 0.6, "tol": 1e-5})
        # COBYLA can terminate early; spend the remaining declared sample budget
        # at its returned angles. These are measurements, not a fallback search.
        while len(ledger.trace) < ledger.budget:
            objective(result.x)
    except BudgetExhausted:
        pass
    resources = {"backend": "qiskit-statevector", "hardware": False, "noise_model": None,
                 "qubits": problem.n_bits, "layers": layers, "shots": len(ledger.trace),
                 "circuit_evaluations": calls, "shots_per_evaluation": shots_per_evaluation,
                 "circuit_depth_with_preparation": circuit.depth() + 1,
                 "two_qubit_gates": int(circuit.count_ops().get("rzz", 0)),
                 "hamiltonian_scale": scale, "queue_s": 0.0, "qpu_cost": 0.0,
                 "statevector_bytes": int(initial.data.nbytes), "repair": "none"}
    return timing, resources

