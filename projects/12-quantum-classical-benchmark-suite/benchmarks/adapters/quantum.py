from __future__ import annotations

import math

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import Statevector
from scipy.optimize import minimize

from benchmarks.instances.schema import encode_qubo, feasible, objective, qubo_to_ising
from .base import Context


def qaoa_circuit(p: dict, layers: int):
    c, q, offset, penalty = encode_qubo(p)
    h, j, _ = qubo_to_ising(c, q, offset)
    scale = max(float(np.max(np.abs(h))), float(np.max(np.abs(j))), 1.0)
    gamma, beta = ParameterVector("gamma", layers), ParameterVector("beta", layers)
    circuit = QuantumCircuit(p["size"])
    circuit.h(range(p["size"]))
    for layer in range(layers):
        for i in range(p["size"]):
            circuit.rz(2 * gamma[layer] * h[i] / scale, i)
            for k in range(i + 1, p["size"]):
                if j[i, k] != 0:
                    circuit.rzz(2 * gamma[layer] * j[i, k] / scale, i, k)
        circuit.rx(2 * beta[layer], range(p["size"]))
    return circuit, list(gamma) + list(beta), (c, q, offset, penalty, scale)


class QAOAAdapter:
    def solve(self, p: dict, ctx: Context) -> dict:
        if p["size"] > 16:
            raise ValueError("CPU statevector QAOA is capped at 16 qubits")
        with ctx.stage("encoding"):
            circuit, params, (c, q, offset, penalty, scale) = qaoa_circuit(p, ctx.algorithm["qaoa_layers"])
        with ctx.stage("compilation"):
            compiled = transpile(circuit, basis_gates=["rz", "sx", "x", "cx"],
                                 optimization_level=0, seed_transpiler=ctx.seed)
        ctx.resources.update(logical_qubits=circuit.num_qubits, circuit_depth=compiled.depth(),
                             two_qubit_gates=compiled.count_ops().get("cx", 0),
                             noise_model="none; noiseless statevector with finite measurement shots",
                             hardware_kind="quantum_simulator_cpu", evaluation_unit="sampled_circuit_expectation")
        best_feasible, best_energy = math.inf, math.inf

        def cost(theta):
            nonlocal best_feasible, best_energy
            shots = min(ctx.algorithm["shots_per_circuit"], ctx.budget.max_shots - ctx.resources["shots"])
            if shots < 1:
                # Charge fails before doing any unbudgeted simulation.
                ctx.charge(shots=1)
            ctx.charge(evaluations=1, shots=shots)
            ctx.resources["optimizer_evaluations"] += 1
            bound = compiled.assign_parameters(dict(zip(params, theta)))
            probabilities = Statevector.from_instruction(bound).probabilities()
            counts = ctx.rng.multinomial(shots, probabilities / probabilities.sum())
            expectation = 0.0
            with ctx.stage("decoding"):
                for state in np.flatnonzero(counts):
                    x = ((int(state) >> np.arange(p["size"])) & 1).astype(int)
                    energy = float(c @ x + x @ q @ x + offset)
                    expectation += counts[state] * energy
                    if best_feasible == math.inf and energy < best_energy:
                        ctx.partial = {"candidate": x.tolist()}
                        best_energy = energy
                    if feasible(p, x) and (value := objective(p, x)) < best_feasible:
                        best_feasible = value
                        ctx.partial = {"candidate": x.tolist()}
            return expectation / shots / scale

        with ctx.stage("execution"):
            result = minimize(cost, ctx.rng.uniform(0, np.pi, len(params)), method="COBYLA",
                              options={"maxiter": max(len(params) + 2, ctx.algorithm["qaoa_iterations"]),
                                       "rhobeg": 0.5, "tol": 1e-3})
        return ctx.partial | {"termination": str(result.message), "penalty": penalty,
                              "optimizer_parameters": result.x.tolist(), "energy_scale": scale}
