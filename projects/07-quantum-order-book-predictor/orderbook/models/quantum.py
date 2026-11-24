from __future__ import annotations

from time import perf_counter
import numpy as np
import pennylane as qml
from scipy.special import softmax
from sklearn.svm import SVC
from .classical import align_probabilities


def encode(x, qubits, encoding, layers=1):
    """Broadcastable angle or IQP-style feature map, with data re-uploading."""
    for _ in range(layers):
        if encoding == "angle":
            for i in range(qubits):
                qml.RY(x[..., i], wires=i)
            for i in range(qubits - 1):
                qml.CNOT(wires=[i, i + 1])
        elif encoding == "iqp":
            for i in range(qubits):
                qml.Hadamard(wires=i)
                qml.RZ(x[..., i], wires=i)
            for i in range(qubits - 1):
                qml.IsingZZ(x[..., i] * x[..., i + 1] / np.pi, wires=[i, i + 1])
        else:
            raise ValueError(f"unknown encoding {encoding}")


class QuantumBase:
    def __init__(self, qubits, encoding, shots, layers, seed, iterations):
        if not 1 <= qubits <= 6 or layers < 1 or encoding not in {"angle", "iqp"}:
            raise ValueError("invalid circuit dimensions or encoding")
        if shots is not None and (type(shots) is not int or shots < 1):
            raise ValueError("shots must be null or a positive integer")
        self.qubits, self.encoding, self.shots = qubits, encoding, shots
        self.layers, self.seed, self.iterations = layers, seed, iterations
        self.rng = np.random.default_rng(seed)
        self.stats = {"optimizer_iterations": 0, "objective_evaluations": 0,
                      "circuit_executions": 0, "measurement_shots": 0,
                      "state_preparation_seconds": 0.0, "kernel_seconds": 0.0,
                      "repair_seconds": 0.0, "decode_seconds": 0.0,
                      "backend": "PennyLane default.qubit CPU simulator", "hardware": False,
                      "noise_model": "none", "queue_seconds": 0.0, "communication_seconds": 0.0,
                      "qpu_cost_usd": 0.0}
        start = perf_counter()
        self._build()
        self.stats["circuit_construction_seconds"] = perf_counter() - start

    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop("circuit", None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._build()

    def check_budget(self, deadline):
        if deadline is not None and perf_counter() >= deadline:
            raise TimeoutError("quantum training exceeded fit budget")

    def fallback(self, y):
        classes = np.unique(y)
        self.constant_class = int(classes[0]) if len(classes) == 1 else None
        self.stats["fallback"] = "single_class" if len(classes) == 1 else None
        return self.constant_class is not None

    def constant_probabilities(self, x):
        return align_probabilities(np.ones((len(x), 1)), [self.constant_class])


class QuantumKernel(QuantumBase):
    """Fidelity kernel. Finite shots use binomial overlap estimates on the CPU.

    Statevector access and cached training states are simulator optimizations, not
    a hardware latency estimate. Negative training eigenvalues are projected out.
    """
    def _build(self):
        device = qml.device("default.qubit", wires=self.qubits, seed=self.seed)

        @qml.qnode(device, interface=None)
        def circuit(x):
            encode(x, self.qubits, self.encoding, self.layers)
            return qml.state()
        self.circuit = circuit

    def states(self, x):
        start = perf_counter()
        states = np.asarray(self.circuit(np.asarray(x)))
        self.stats["state_preparation_seconds"] += perf_counter() - start
        self.stats["circuit_executions"] += len(x)
        return states.reshape(len(x), -1)

    def kernel(self, left, right, symmetric=False):
        start = perf_counter()
        matrix = np.clip(np.abs(left.conj() @ right.T) ** 2, 0, 1)
        if self.shots is not None:
            if symmetric:
                i, j = np.triu_indices(len(left), 1)
                draws = self.rng.binomial(self.shots, matrix[i, j]) / self.shots
                matrix[i, j] = matrix[j, i] = draws
                np.fill_diagonal(matrix, 1)
                pairs = len(i)
            else:
                matrix = self.rng.binomial(self.shots, matrix) / self.shots
                pairs = matrix.size
            self.stats["measurement_shots"] += pairs * self.shots
            self.stats["sampled_overlap_estimates"] = self.stats.get("sampled_overlap_estimates", 0) + pairs
        self.stats["kernel_seconds"] += perf_counter() - start
        return matrix

    def fit(self, x, y, deadline=None):
        if self.fallback(y):
            return self
        self.train_states = self.states(x)
        k = self.kernel(self.train_states, self.train_states, symmetric=True)
        start = perf_counter()
        values, vectors = np.linalg.eigh((k + k.T) / 2)
        positive = vectors[:, values > 1e-10]
        self.projector = positive @ positive.T
        k = (vectors * np.maximum(values, 0)) @ vectors.T
        self.stats["negative_eigenvalues"] = int(np.sum(values < -1e-10))
        self.stats["repair_seconds"] += perf_counter() - start
        self.check_budget(deadline)
        self.svm = SVC(C=1.0, kernel="precomputed", decision_function_shape="ovr",
                       max_iter=self.iterations * 100, random_state=self.seed).fit(k, y)
        self.stats["optimizer_iterations"] = int(self.svm.n_iter_.sum())
        self.stats["objective_evaluations"] = None  # native libsvm does not expose this
        self.stats["measurement_method"] = "exact fidelity" if self.shots is None else "binomial fidelity sampling"
        self.stats["encoding_gate_count"] = self.layers * (self.qubits * (2 if self.encoding == "iqp" else 1) + self.qubits - 1)
        self.check_budget(deadline)
        return self

    def predict_proba(self, x):
        if self.constant_class is not None:
            return self.constant_probabilities(x)
        k = self.kernel(self.states(x), self.train_states) @ self.projector
        start = perf_counter()
        scores = self.svm.decision_function(k)
        if scores.ndim == 1:
            scores = np.column_stack([-scores, scores])
        result = align_probabilities(softmax(scores, axis=1), self.svm.classes_)
        self.stats["decode_seconds"] += perf_counter() - start
        return result


class VariationalClassifier(QuantumBase):
    """Small parameterized circuit + linear readout trained jointly with SPSA."""
    def _build(self):
        device = qml.device("default.qubit", wires=self.qubits, seed=self.seed)

        @qml.qnode(device, interface=None)
        def circuit(x, weights):
            encode(x, self.qubits, self.encoding)
            for layer in range(self.layers):
                for i in range(self.qubits):
                    qml.RY(weights[layer, i, 0], wires=i)
                    qml.RZ(weights[layer, i, 1], wires=i)
                for i in range(self.qubits - 1):
                    qml.CNOT(wires=[i, i + 1])
            return qml.probs(wires=range(self.qubits))
        self.circuit = qml.set_shots(circuit, shots=self.shots)

    def unpack(self, parameters):
        count = self.layers * self.qubits * 2
        theta = parameters[:count].reshape(self.layers, self.qubits, 2)
        readout = parameters[count:-3].reshape(2**self.qubits, 3)
        return theta, readout, parameters[-3:]

    def forward(self, x, parameters):
        theta, readout, bias = self.unpack(parameters)
        start = perf_counter()
        states = np.asarray(self.circuit(np.asarray(x), theta)).reshape(len(x), -1)
        self.stats["state_preparation_seconds"] += perf_counter() - start
        self.stats["circuit_executions"] += len(x)
        self.stats["measurement_shots"] += len(x) * (self.shots or 0)
        start = perf_counter()
        result = softmax(states @ readout + bias, axis=1)
        self.stats["decode_seconds"] += perf_counter() - start
        return result

    def fit(self, x, y, deadline=None):
        if self.fallback(y):
            return self
        count = self.layers * self.qubits * 2 + 3 * 2**self.qubits + 3
        parameters = self.rng.normal(0, 0.2, count)
        parameters[-3:] = np.log((np.bincount(y, minlength=3) + 1) / (len(y) + 3))

        def loss(p):
            self.check_budget(deadline)
            probability = self.forward(x, p)
            self.stats["objective_evaluations"] += 1
            return -np.log(np.clip(probability[np.arange(len(y)), y], 1e-12, 1)).mean() + 1e-4 * np.square(p).sum()

        best_loss, best = loss(parameters), parameters.copy()
        for k in range(self.iterations):
            perturb = self.rng.choice([-1.0, 1.0], size=count)
            c, a = 0.1 / (k + 1)**0.101, 0.2 / (k + 1)**0.602
            gradient = (loss(parameters + c * perturb) - loss(parameters - c * perturb)) / (2 * c) * perturb
            parameters -= a * np.clip(gradient, -5, 5)
            value = loss(parameters)
            if value < best_loss:
                best_loss, best = value, parameters.copy()
            self.stats["optimizer_iterations"] = k + 1
        self.parameters = best
        self.stats["training_loss"] = float(best_loss)
        self.stats["measurement_method"] = "exact probabilities" if self.shots is None else "PennyLane finite-shot probabilities"
        self.stats["parameter_count"] = count
        self.stats["encoding_gate_count"] = self.qubits * (2 if self.encoding == "iqp" else 1) + self.qubits - 1
        self.stats["variational_gate_count"] = self.layers * (3 * self.qubits - 1)
        return self

    def predict_proba(self, x):
        return self.constant_probabilities(x) if self.constant_class is not None else self.forward(x, self.parameters)
