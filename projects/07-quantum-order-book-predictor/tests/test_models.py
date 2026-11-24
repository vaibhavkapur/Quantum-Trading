import pickle
import numpy as np
import pytest
from orderbook.models import make_model
from orderbook.models.quantum import QuantumKernel, VariationalClassifier
from orderbook.calibration import TemperatureCalibrator
from orderbook.evaluation import classification_metrics


def test_single_qubit_fidelity_matches_analytic_answer():
    model = QuantumKernel(1, "angle", None, 1, 11, 3)
    x = np.array([[0], [np.pi / 3], [np.pi]])
    states = model.states(x)
    k = model.kernel(states, states)
    expected = np.cos((x - x.T)/2)**2
    np.testing.assert_allclose(k, expected, atol=1e-12)


@pytest.mark.parametrize("encoding", ["angle", "iqp"])
@pytest.mark.parametrize("qubits", [1, 2, 3])
def test_state_normalization_psd_encoding(encoding, qubits):
    model = QuantumKernel(qubits, encoding, None, 2, 5, 3)
    x = np.random.default_rng(5).normal(size=(8, qubits))
    state = model.states(x)
    np.testing.assert_allclose(np.square(np.abs(state)).sum(1), 1)
    k = model.kernel(state, state)
    np.testing.assert_allclose(np.diag(k), 1)
    assert np.linalg.eigvalsh(k).min() > -1e-10


def test_seeded_shots_symmetry_and_accounting():
    x = np.random.default_rng(1).normal(size=(6, 2))
    matrices = []
    for _ in range(2):
        model = QuantumKernel(2, "iqp", 64, 2, 4, 3)
        s = model.states(x)
        k = model.kernel(s, s, symmetric=True)
        np.testing.assert_array_equal(k, k.T)
        assert model.stats["measurement_shots"] == 15 * 64
        matrices.append(k)
    np.testing.assert_array_equal(*matrices)


@pytest.mark.parametrize("name", ["majority", "logistic", "xgboost", "mlp", "quantum_kernel", "vqc"])
def test_adapters_probabilities_roundtrip_and_single_class(name):
    x = np.random.default_rng(2).normal(size=(12, 2))
    y = np.tile([0, 1, 2], 4)
    model = make_model(name, seed=11, iterations=100 if name == "logistic" else 3, qubits=2)
    model.fit(x, y)
    p = model.predict_proba(x[:3])
    assert p.shape == (3, 3) and np.isfinite(p).all()
    np.testing.assert_allclose(p.sum(1), 1)
    restored = pickle.loads(pickle.dumps(model))
    np.testing.assert_allclose(restored.predict_proba(x[:3]), p)
    model = make_model(name, seed=11, iterations=3, qubits=2).fit(x, np.full(len(x), 2))
    assert np.all(model.predict_proba(x).argmax(1) == 2)
    assert model.stats["fallback"] == "single_class"


def test_xgboost_missing_middle_class():
    x = np.random.default_rng(9).normal(size=(12, 2))
    model = make_model("xgboost", seed=11, iterations=3, qubits=2).fit(x, np.tile([0, 2], 6))
    assert np.all(model.predict_proba(x)[:, 1] < 1e-9)


def test_vqc_finite_shots_and_training_budget():
    x = np.random.default_rng(4).normal(size=(9, 2))
    model = VariationalClassifier(2, "angle", 64, 1, 8, 2).fit(x, np.tile([0, 1, 2], 3))
    assert model.stats["objective_evaluations"] == 7
    assert model.stats["measurement_shots"] == 7 * 9 * 64
    np.testing.assert_allclose(model.predict_proba(x).sum(1), 1)
    with pytest.raises(TimeoutError):
        VariationalClassifier(2, "angle", None, 1, 8, 2).fit(x, np.tile([0, 1, 2], 3), deadline=0)


def test_calibration_reduces_overconfidence_and_missing_class_metrics():
    p = np.tile([0.98, 0.01, 0.01], (30, 1))
    y = np.tile([0, 1, 2], 10)
    calibrator = TemperatureCalibrator().fit(p, y)
    assert calibrator.temperature > 1
    assert classification_metrics(y, calibrator.transform(p))["log_loss"] < classification_metrics(y, p)["log_loss"]
    result = classification_metrics(np.zeros(3, dtype=int), p[:3])
    assert result["balanced_accuracy"] == 1
    assert result["per_class"]["up"]["support"] == 0
