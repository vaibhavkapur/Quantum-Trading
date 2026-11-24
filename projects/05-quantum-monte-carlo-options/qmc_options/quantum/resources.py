"""Gate-level resource counts for the A and Q circuits."""
from __future__ import annotations

from typing import Dict

from qiskit import QuantumCircuit, transpile

BASIS_GATES = ("cx", "rz", "sx", "x")


def circuit_resources(circuit: QuantumCircuit) -> Dict[str, int]:
    """Depth and gate counts after compiling to {CX, RZ, SX, X} with all-to-all
    connectivity. Routing to a real coupling map would add SWAPs on top."""
    compiled = transpile(circuit, basis_gates=list(BASIS_GATES), optimization_level=1, seed_transpiler=0)
    ops = compiled.count_ops()
    cx = int(ops.get("cx", 0))
    return {
        "qubits": circuit.num_qubits,
        "depth": int(compiled.depth()),
        "cx": cx,
        "cx_depth": int(compiled.depth(lambda inst: inst.operation.name == "cx")),
        "single_qubit_gates": int(sum(ops.values()) - cx),
        "rz": int(ops.get("rz", 0)),
    }


def estimation_resources(a_res: Dict[str, int], q_res: Dict[str, int], max_power: int,
                         grover_iterations: int, shots: int) -> Dict[str, int]:
    """Totals for a finished estimation run: every shot runs A once and each
    Grover iteration runs Q once."""
    return {
        "qubits": a_res["qubits"],
        "max_circuit_depth": a_res["depth"] + max_power * q_res["depth"],
        "max_circuit_cx": a_res["cx"] + max_power * q_res["cx"],
        "total_cx": shots * a_res["cx"] + grover_iterations * q_res["cx"],
        "total_shots": shots,
    }
