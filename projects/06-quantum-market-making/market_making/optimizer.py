"""Small, fully enumerated policy Hamiltonians and noiseless QAOA.

Encoding already evaluates the entire grid: this is a circuit experiment,
not an efficient quantum oracle or a speedup claim.
"""
from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
from dataclasses import asdict
import math
import time
import numpy as np
from scipy.optimize import minimize
from .policy import Policy, baseline, grid
from .simulation import simulate, policy_cost


class BudgetExceeded(TimeoutError):
    pass


def check_deadline(deadline):
    if time.perf_counter() > deadline:
        raise BudgetExceeded("optimizer wall-time budget exceeded")


def encode_costs(costs):
    costs = np.asarray(costs, dtype=float)
    if costs.ndim != 1 or not 1 <= len(costs) <= 256 or not np.all(np.isfinite(costs)):
        raise ValueError("expected 1..256 finite policy costs")
    qubits = max(1, math.ceil(math.log2(len(costs))))
    scale = max(float(np.ptp(costs)), 1e-12)
    energies = np.full(2**qubits, 2.0)
    energies[:len(costs)] = (costs - np.min(costs)) / scale
    return energies


def qaoa_circuit(energies, angles):
    from qiskit import QuantumCircuit
    from qiskit.circuit.library import DiagonalGate
    qubits = int(math.log2(len(energies)))
    if 2**qubits != len(energies) or not len(angles) or len(angles) % 2:
        raise ValueError("invalid Hamiltonian dimension or QAOA angles")
    circuit = QuantumCircuit(qubits)
    circuit.h(range(qubits))
    for gamma, beta in np.asarray(angles).reshape(-1, 2):
        circuit.append(DiagonalGate(np.exp(-1j * gamma * energies)), range(qubits))
        for qubit in range(qubits):
            circuit.rx(2*beta, qubit)
    return circuit


def qaoa_probabilities(energies, angles):
    from qiskit.quantum_info import Statevector
    return Statevector.from_instruction(qaoa_circuit(energies, angles)).probabilities()


def quantum_search(costs, seed, settings, deadline):
    start = time.perf_counter()
    from qiskit import transpile
    energies = encode_costs(costs)
    preparation_seconds = time.perf_counter() - start
    rng = np.random.default_rng(seed)
    best_angles = rng.uniform(0.1, 1.0, 2*settings["qaoa_layers"])
    best_expectation = float("inf")
    evaluations = 0

    def expectation(angles):
        nonlocal best_angles, best_expectation, evaluations
        check_deadline(deadline)
        if evaluations >= settings["max_evaluations"]:
            raise BudgetExceeded("QAOA expectation budget exceeded")
        value = float(qaoa_probabilities(energies, angles) @ energies)
        evaluations += 1
        if value < best_expectation:
            best_expectation, best_angles = value, np.array(angles)
        return value

    begin = time.perf_counter()
    minimize(expectation, best_angles, method="COBYLA", options={"maxiter": settings["max_evaluations"], "rhobeg": .7})
    iteration_seconds = time.perf_counter() - begin
    check_deadline(deadline)
    begin = time.perf_counter()
    circuit = transpile(qaoa_circuit(energies, best_angles), basis_gates=["h", "rx", "rz", "cx"], optimization_level=1, seed_transpiler=seed)
    compilation_seconds = time.perf_counter() - begin
    begin = time.perf_counter()
    probabilities = qaoa_probabilities(energies, best_angles)
    probabilities = np.maximum(probabilities, 0)
    probabilities /= probabilities.sum()
    samples = rng.choice(len(energies), size=settings["shots"], p=probabilities)
    sampling_seconds = time.perf_counter() - begin
    begin = time.perf_counter()
    valid = samples[samples < len(costs)]
    if not len(valid):
        raise RuntimeError("all measured indices were padding states")
    selected = int(min(set(valid.tolist()), key=lambda i: (costs[i], i)))
    decoding_seconds = time.perf_counter() - begin
    check_deadline(deadline)
    return selected, dict(
        backend="qiskit-statevector", hardware=False, noise="none", objective_expectations="exact statevector",
        qubits=circuit.num_qubits, depth=circuit.depth(), two_qubit_gates=int(circuit.count_ops().get("cx", 0)),
        expectation_evaluations=evaluations, shots=settings["shots"], invalid_shots=int(len(samples)-len(valid)),
        optimal_probability=float(sum(probabilities[i] for i,c in enumerate(costs) if np.isclose(c, min(costs), atol=1e-12, rtol=0))),
        state_preparation_seconds=preparation_seconds, compilation_seconds=compilation_seconds,
        optimizer_iteration_seconds=iteration_seconds, sampling_seconds=sampling_seconds, decoding_seconds=decoding_seconds,
        communication_seconds=0.0, queue_seconds=0.0, repair_seconds=0.0, qpu_cost=0.0,
    )


def optimize(frames, config, solver, seed, version, deadline=None):
    started = time.perf_counter()
    deadline = deadline or (started + config["optimizer"]["timeout_seconds"])
    candidates = grid(config)
    costs = []
    for candidate in candidates:
        check_deadline(deadline)
        result = simulate(frames, config, candidate)
        costs.append(policy_cost(result, config, len(frames)))
    objective_seconds = time.perf_counter() - started
    check_deadline(deadline)
    begin = time.perf_counter()
    if solver == "classical":
        index = min(range(len(costs)), key=lambda i: (costs[i], i))
        details = dict(backend="exhaustive-grid", hardware=False, shots=0, expectation_evaluations=0,
                       state_preparation_seconds=0.0, compilation_seconds=0.0, sampling_seconds=0.0,
                       communication_seconds=0.0, queue_seconds=0.0, repair_seconds=0.0, decoding_seconds=0.0,
                       optimizer_iteration_seconds=time.perf_counter()-begin)
    elif solver == "quantum-sim":
        index, details = quantum_search(costs, seed, config["optimizer"], deadline)
    else:
        raise ValueError(f"unknown solver {solver}")
    check_deadline(deadline)
    selected = candidates[index].versioned(version, solver)
    details.update(policy=asdict(selected), objective_evaluations=len(costs), objective_preparation_seconds=objective_seconds,
                   total_seconds=time.perf_counter()-started, selected_index=index, selected_cost=costs[index],
                   exact_cost=min(costs), optimality_gap=costs[index]-min(costs), costs=costs,
                   training_start_ns=frames[0]["timestamp_ns"], training_end_ns=frames[-1]["timestamp_ns"],
                   training_frames=len(frames), status="ok", seed=seed)
    return selected, details


class PolicyService:
    """Background optimizer with immutable publication and cooperative timeouts.

    request/poll never wait. Offline replay uses bounded, predeclared barriers
    for deterministic policy activation; waiting is excluded from quote latency
    but included in full-pipeline timing. Expired results are never published.
    """
    def __init__(self, config, solver, seed, worker=optimize):
        self.config, self.solver, self.seed, self.worker = config, solver, seed, worker
        # Initialize Qiskit's native circuit registry on the calling thread.
        # Importing it first on a short-lived worker can leave thread-local
        # extension state invalid when a subsequent worker constructs circuits.
        if solver == "quantum-sim":
            from qiskit import QuantumCircuit
            QuantumCircuit(1)
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="policy-optimizer")
        self.pending = None
        self.version = 0
        self.records = []
        self.expired = False

    def fallback(self, reason, timestamp):
        self.version += 1
        policy = baseline("inventory-skew", self.config).versioned(self.version, "fallback")
        self.records.append(dict(status="fallback", reason=reason, policy=asdict(policy), effective_timestamp_ns=timestamp))
        return policy

    def request(self, frames, timestamp):
        if not frames or frames[-1]["timestamp_ns"] >= timestamp:
            raise ValueError("optimizer input must strictly precede publication timestamp")
        if self.pending is not None:
            future, _, _, _ = self.pending
            if not future.done():
                return False
            if self.expired:
                self.pending = None
            else:
                return False
        self.expired = False
        version = self.version + 1
        deadline = time.perf_counter() + self.config["optimizer"]["timeout_seconds"]
        future = self.pool.submit(self.worker, tuple(frames), self.config, self.solver,
                                  self.seed + version, version, deadline)
        self.pending = (future, version, deadline, timestamp)
        return True

    def poll(self, timestamp, wait=False):
        if self.pending is None or self.expired:
            return None
        future, version, deadline, requested_at = self.pending
        if timestamp < requested_at:
            raise ValueError("cannot publish a policy before its request timestamp")
        if not wait and not future.done() and time.perf_counter() < deadline:
            return None
        started = time.perf_counter()
        try:
            if time.perf_counter() > deadline and not future.done():
                raise FutureTimeout("optimizer timed out")
            policy, details = future.result(timeout=max(0, deadline-time.perf_counter()) if wait else 0)
            if time.perf_counter() > deadline:
                raise FutureTimeout("optimizer result expired")
            if not isinstance(policy, Policy) or policy.version != version or version <= self.version or policy.inventory_band > self.config["execution"]["max_inventory"]:
                raise ValueError("invalid or stale optimizer policy")
            self.version = version
            details = dict(details, barrier_wait_seconds=time.perf_counter()-started,
                           requested_timestamp_ns=requested_at, effective_timestamp_ns=timestamp)
            self.records.append(details)
            self.pending = None
            return policy
        except Exception as exc:
            self.expired = True
            policy = self.fallback(f"{type(exc).__name__}: {exc}", timestamp)
            self.records[-1]["barrier_wait_seconds"] = time.perf_counter()-started
            return policy

    def update(self, frames, current, timestamp):
        if not self.request(frames, timestamp):
            return self.fallback("previous worker still busy", timestamp)
        return self.poll(timestamp, wait=True) or current

    def close(self):
        self.pool.shutdown(wait=True, cancel_futures=True)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
