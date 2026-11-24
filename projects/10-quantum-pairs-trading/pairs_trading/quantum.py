from __future__ import annotations

from time import perf_counter

import numpy as np
from scipy.optimize import minimize

from .selection import Problem, Qubo, SelectionResult, decode_samples, encode_qubo


def numpy_state(energies: np.ndarray, parameters: np.ndarray, depth: int) -> np.ndarray:
    state = np.ones(len(energies), complex) / np.sqrt(len(energies))
    n = (len(energies) - 1).bit_length()
    for layer in range(depth):
        state *= np.exp(-1j * parameters[layer] * energies)
        beta = parameters[depth + layer]
        for qubit in range(n):
            blocks = state.reshape(-1, 2, 1 << qubit)
            zero, one = blocks[:, 0].copy(), blocks[:, 1].copy()
            blocks[:, 0] = np.cos(beta) * zero - 1j * np.sin(beta) * one
            blocks[:, 1] = -1j * np.sin(beta) * zero + np.cos(beta) * one
    return state


def qiskit_template(qubo: Qubo, depth: int, scale: float):
    from qiskit import QuantumCircuit, transpile
    from qiskit.circuit import ParameterVector

    parameters = ParameterVector("theta", 2 * depth)
    circuit = QuantumCircuit(qubo.n)
    circuit.h(range(qubo.n))
    h = -np.diag(qubo.matrix) / 2
    h -= (np.triu(qubo.matrix, 1).sum(axis=0) + np.triu(qubo.matrix, 1).sum(axis=1)) / 4
    for layer in range(depth):
        for i in range(qubo.n):
            circuit.rz(2 * parameters[layer] * h[i] / scale, i)
        for i, j in zip(*np.nonzero(np.triu(qubo.matrix, 1))):
            circuit.rzz(parameters[layer] * qubo.matrix[i, j] / (2 * scale), int(i), int(j))
        for i in range(qubo.n):
            circuit.rx(2 * parameters[depth + layer], i)
    return transpile(circuit, basis_gates=["h", "rz", "rx", "cx"], optimization_level=1, seed_transpiler=0), parameters


class _BudgetReached(Exception):
    pass


def qaoa(problem: Problem, settings: dict, seed: int) -> SelectionResult:
    start = perf_counter()
    qubo = encode_qubo(problem, settings["max_qubits"])
    encoded = perf_counter()
    energies = qubo.energies()
    scale = max(1.0, float(np.ptp(energies)))
    normalized = (energies - energies.min()) / scale
    prepared = perf_counter()
    backend = settings["backend"]
    if backend == "qiskit":
        from qiskit.quantum_info import Statevector
        template, parameters = qiskit_template(qubo, settings["qaoa_depth"], scale)
    compiled = perf_counter()
    rng = np.random.default_rng(seed)
    initial = rng.uniform(-np.pi, np.pi, 2 * settings["qaoa_depth"])
    best_energy, best_probabilities, evaluations = float("inf"), None, 0
    simulation_seconds = 0.0

    def evaluate(theta):
        nonlocal best_energy, best_probabilities, evaluations, simulation_seconds
        if evaluations >= settings["max_evaluations"] or (evaluations and perf_counter() - start >= settings["wall_seconds"]):
            raise _BudgetReached
        tick = perf_counter()
        if backend == "qiskit":
            state = Statevector.from_instruction(template.assign_parameters(dict(zip(parameters, theta)))).data
        else:
            state = numpy_state(normalized, theta, settings["qaoa_depth"])
        probabilities = np.abs(state)**2
        simulation_seconds += perf_counter() - tick
        value = float(probabilities @ normalized)
        evaluations += 1
        if value < best_energy:
            best_energy, best_probabilities = value, probabilities
        return value

    exhausted = False
    try:
        optimized = minimize(evaluate, initial, method="COBYLA", options={"maxiter": settings["max_evaluations"], "rhobeg": 1.0, "tol": 1e-4})
        optimizer_status = str(optimized.message)
    except _BudgetReached:
        exhausted = True
        optimizer_status = "cooperative wall-time/evaluation budget reached"
    optimized_at = perf_counter()
    if best_probabilities is None:
        raise RuntimeError("QAOA produced no statevector")
    samples_int = rng.choice(len(energies), size=settings["shots"], p=best_probabilities / best_probabilities.sum())
    samples = ((samples_int[:, None] >> np.arange(qubo.n)) & 1).astype(int)
    sampled = perf_counter()
    best, decoded = decode_samples(problem, samples)
    details = {**decoded, "backend": backend, "hardware": "CPU noiseless statevector simulation", "noise": "none",
               "qubits": qubo.n, "decision_qubits": problem.n, "slack_qubits": qubo.n - problem.n,
               "depth": settings["qaoa_depth"], "shots": settings["shots"], "optimization_shots": 0,
               "expectation_method": "exact statevector; finite shots only for final sampling",
               "qubo_penalty": qubo.penalty, "budget_exhausted": exhausted, "optimizer_status": optimizer_status,
               "circuit_depth": template.depth() if backend == "qiskit" else None,
               "two_qubit_gates": template.count_ops().get("cx", 0) if backend == "qiskit" else None,
               "qpu_cost_usd": None, "queue_seconds": 0.0, "communication_seconds": 0.0}
    repair_seconds = details.pop("repair_decode_seconds")
    return SelectionResult("quantum-sim", best, problem.objective(best), evaluations=evaluations,
                           timings={"encoding_seconds": encoded - start, "state_preparation_seconds": prepared - encoded,
                                    "compilation_seconds": compiled - prepared, "optimizer_seconds": optimized_at - compiled,
                                    "simulation_seconds_nested_in_optimizer": simulation_seconds, "sampling_seconds": sampled - optimized_at,
                                    "repair_decode_seconds": repair_seconds, "total_seconds": perf_counter() - start}, details=details)
