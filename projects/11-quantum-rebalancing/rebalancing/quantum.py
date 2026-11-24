"""Small noiseless QAOA circuits executed by Qiskit's CPU statevector simulator."""
from __future__ import annotations

from time import perf_counter
import numpy as np
from scipy.optimize import minimize

from .encoding import Encoding, make_qubo
from .model import ResourceLimitError
from .solvers import Budget, Candidate


def qaoa(encoding: Encoding, budget: Budget) -> Candidate:
    if encoding.size > budget.max_qubits:
        raise ResourceLimitError(f"QAOA needs {encoding.size} qubits; configured limit is {budget.max_qubits}")
    start = perf_counter()
    from qiskit import QuantumCircuit, transpile
    from qiskit.circuit import ParameterVector
    from qiskit.quantum_info import Statevector

    timings = {"backend_initialization_s": perf_counter() - start}
    t = perf_counter()
    qubo = make_qubo(encoding)
    _, fields, coupling = qubo.ising()
    n = encoding.size
    bits = ((np.arange(1 << n, dtype=np.uint64)[:, None] >> np.arange(n, dtype=np.uint64)) & 1).astype(float)
    energies = qubo.offset + bits @ qubo.linear + np.einsum("bi,ij,bj->b", bits, qubo.quadratic, bits, optimize=True)
    energy_scale = max(1.0, float(energies.max() - energies.min()))
    timings["state_preparation_s"] = perf_counter() - t
    t = perf_counter()
    parameters = ParameterVector("theta", 2 * budget.qaoa_depth)
    circuit = QuantumCircuit(n)
    circuit.h(range(n))
    for layer in range(budget.qaoa_depth):
        gamma, beta = parameters[2 * layer], parameters[2 * layer + 1]
        for i, h in enumerate(fields):
            if h:
                circuit.rz(2 * gamma * h / energy_scale, i)
        for i in range(n):
            for j in range(i + 1, n):
                if coupling[i, j]:
                    circuit.rzz(2 * gamma * coupling[i, j] / energy_scale, i, j)
        circuit.rx(2 * beta, range(n))
    circuit = transpile(circuit, basis_gates=["h", "rx", "rz", "rzz"], optimization_level=1, seed_transpiler=budget.seed)
    timings["compilation_s"] = perf_counter() - t
    rng = np.random.default_rng(budget.seed)
    evaluations, statevector_time = 0, 0.0
    best_expectation, best_probabilities, best_parameters = np.inf, None, None
    optimization_start = perf_counter()

    class Stop(Exception):
        pass

    def objective(theta):
        nonlocal evaluations, statevector_time, best_expectation, best_probabilities, best_parameters
        if evaluations and (evaluations >= budget.max_evaluations or perf_counter() - start >= budget.time_limit_s):
            raise Stop()
        evaluation_start = perf_counter()
        bound = circuit.assign_parameters(dict(zip(parameters, theta)))
        probabilities = Statevector.from_instruction(bound).probabilities()
        statevector_time += perf_counter() - evaluation_start
        expectation = float(probabilities @ energies)
        evaluations += 1
        if expectation < best_expectation:
            best_expectation, best_probabilities, best_parameters = expectation, probabilities, list(map(float, theta))
        return expectation / energy_scale

    theta = rng.uniform(0, np.pi, len(parameters))
    # Evaluate once even for a one-evaluation budget; COBYLA otherwise has its own
    # minimum evaluation count. The wrapper enforces the requested ceiling.
    objective(theta)
    status = "evaluation_limit"
    if budget.max_evaluations > 1:
        try:
            result = minimize(objective, theta, method="COBYLA", options={
                "maxiter": max(budget.max_evaluations, len(parameters) + 2), "rhobeg": 0.8, "tol": 1e-5})
            status = "converged" if result.success else "evaluation_limit"
        except Stop:
            status = "evaluation_or_time_limit"
    timings["optimizer_s"] = perf_counter() - optimization_start
    timings["statevector_evolution_s"] = statevector_time  # nested inside optimizer_s
    t = perf_counter()
    samples = rng.choice(len(energies), size=budget.shots, p=best_probabilities / best_probabilities.sum())
    outcomes, counts = np.unique(samples, return_counts=True)
    timings["measurement_s"] = perf_counter() - t
    t = perf_counter()
    feasible, onehot_count, feasible_count = [], 0, 0
    for state, count in zip(outcomes, counts):
        x = bits[state]
        if all(int(x[block].sum()) == 1 for block in encoding.blocks):
            onehot_count += int(count)
            indices = encoding.decode(x)
            report = encoding.problem.evaluate(indices)
            if report["feasible"]:
                feasible_count += int(count)
                feasible.append((report["objective"], indices))
    if feasible:
        indices = min(feasible)[1]
        selection = "best fully feasible measured one-hot sample"
    else:
        state = min(map(int, outcomes), key=lambda s: (energies[s], s))
        indices = encoding.decode(bits[state])
        selection = "minimum-energy measured sample projected to one-hot; global risk gate follows"
    timings["decoding_s"] = perf_counter() - t
    timings.update({"queue_s": 0.0, "communication_s": 0.0, "quantum_total_s": perf_counter() - start})
    return Candidate(indices, status, evaluations, {
        "engine": "Qiskit Statevector QAOA", "execution_mode": "noiseless_cpu_simulation", "hardware": False,
        "noise_model": None, "shots": budget.shots, "optimizer_shots": 0,
        "optimizer_estimator": "exact statevector expectation (not a shot-based hardware estimator)",
        "qubits": n, "qaoa_depth": budget.qaoa_depth, "circuit_depth": circuit.depth(),
        "gates": {str(k): int(v) for k, v in circuit.count_ops().items()}, "penalty": qubo.penalty,
        "qubo_constraint_scope": "one-hot + local trade domains; cash and group caps enforced by postselection/repair",
        "best_expected_qubo_energy": best_expectation, "parameters": best_parameters,
        "measured_onehot_fraction": onehot_count / budget.shots, "measured_feasible_fraction": feasible_count / budget.shots,
        "unique_measured_states": len(outcomes), "sample_selection": selection, "qpu_cost": None,
        "top_counts": {format(int(s), f"0{n}b"): int(c) for s, c in sorted(zip(outcomes, counts), key=lambda x: (-x[1], x[0]))[:16]},
        "timings": timings})

