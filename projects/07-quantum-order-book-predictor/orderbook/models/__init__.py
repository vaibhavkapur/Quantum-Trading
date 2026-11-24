from .classical import ClassicalModel


def make_model(name, *, seed, iterations, qubits, encoding="angle", shots=None, layers=2):
    if name in {"majority", "logistic", "xgboost", "mlp"}:
        return ClassicalModel(name, seed, iterations)
    from .quantum import QuantumKernel, VariationalClassifier
    cls = {"quantum_kernel": QuantumKernel, "vqc": VariationalClassifier}[name]
    return cls(qubits, encoding, shots, layers, seed, iterations)
