import itertools

import numpy as np
import pytest
from qiskit.quantum_info import Operator, Statevector

from qmc_options.models import Option, paths_from_normals
from qmc_options.quantum.amplitude_estimation import AnalyticOracle, CircuitOracle
from qmc_options.quantum.discretize import discretize, gaussian_grid
from qmc_options.quantum.encoding import (build_a_operator, build_grover_operator, build_state_preparation,
                                          distribution_loader)
from qmc_options.quantum.resources import circuit_resources


def test_gaussian_grid_is_a_symmetric_distribution():
    z, p = gaussian_grid(4, truncation=3.0)
    assert z.size == p.size == 16
    assert p.sum() == pytest.approx(1.0)
    assert np.allclose(p, p[::-1]) and np.allclose(z, -z[::-1])
    assert p @ z == pytest.approx(0.0, abs=1e-12)
    with pytest.raises(ValueError):
        gaussian_grid(0)


@pytest.mark.parametrize("qubits", [1, 2, 3, 4])
def test_loader_prepares_an_arbitrary_distribution(qubits):
    p = np.random.default_rng(qubits).dirichlet(np.ones(2**qubits))     # deliberately asymmetric
    state = Statevector(distribution_loader(p))
    assert np.allclose(state.probabilities(), p, atol=1e-12)
    assert np.allclose(state.data.imag, 0, atol=1e-12) and (state.data.real > -1e-12).all()


def test_loader_handles_zero_probability_branches():
    p = np.array([0.5, 0.0, 0.0, 0.0, 0.25, 0.25, 0.0, 0.0])
    assert np.allclose(Statevector(distribution_loader(p)).probabilities(), p, atol=1e-12)
    with pytest.raises(ValueError):
        distribution_loader(np.array([0.2, 0.3, 0.5]))


def test_discretized_price_matches_independent_enumeration(market):
    option = Option("barrier_up_out_call", 100.0, 1.0, steps=3, barrier=125.0)
    problem = discretize(market, option, qubits_per_step=2)
    z, p = gaussian_grid(2)
    total = 0.0
    for combo in itertools.product(range(4), repeat=3):          # brute force over all 64 paths
        path = paths_from_normals(market, option, z[list(combo)])
        total += np.prod(p[list(combo)]) * option.payoff(path)
    assert problem.exact_price == pytest.approx(np.exp(-0.05) * total, rel=1e-12)
    assert problem.probs.sum() == pytest.approx(1.0)
    assert problem.exact_price == pytest.approx(problem.scale * problem.amplitude)


@pytest.mark.parametrize("option,qubits", [
    (Option("european_call", 100.0, 1.0), 4),
    (Option("european_put", 105.0, 0.5), 3),
    (Option("asian_call", 100.0, 1.0, steps=2), 2),
    (Option("barrier_up_out_call", 100.0, 1.0, steps=2, barrier=125.0), 2),
])
def test_a_operator_encodes_probabilities_and_payoffs_exactly(market, option, qubits):
    problem = discretize(market, option, qubits)
    n = problem.num_state_qubits
    state = Statevector(build_a_operator(problem)).data
    # Objective qubit is the most significant bit: first half |0>, second half |1>.
    g = problem.normalized_payoffs
    assert np.allclose(state[: 2**n], np.sqrt(problem.probs * (1 - g)), atol=1e-10)
    assert np.allclose(state[2**n:], np.sqrt(problem.probs * g), atol=1e-10)
    assert np.sum(np.abs(state[2**n:]) ** 2) == pytest.approx(problem.amplitude, abs=1e-10)
    assert np.allclose(Statevector(build_state_preparation(problem)).probabilities(), problem.probs, atol=1e-10)


def test_grover_circuit_amplifies_as_predicted_and_modes_agree(market, call):
    problem = discretize(market, call, 3)
    a_op = build_a_operator(problem)
    unitary = CircuitOracle(a_op, mode="unitary")
    reflection = CircuitOracle(a_op, mode="reflection")
    ideal = AnalyticOracle(problem.amplitude)
    for k in (0, 1, 2, 3, 7, 20):
        assert unitary.probability(k) == pytest.approx(ideal.probability(k), abs=1e-9)
        assert reflection.probability(k) == pytest.approx(ideal.probability(k), abs=1e-9)
    assert unitary.probability(1) == pytest.approx(ideal.probability(1), abs=1e-9)   # rewinding the cache


def test_grover_circuit_equals_the_reflection_formula(market, call):
    problem = discretize(market, call, 2)
    a_op = build_a_operator(problem)
    q = Operator(build_grover_operator(a_op)).data
    psi = Statevector(a_op).data
    good = np.diag(np.where(np.arange(8) >= 4, -1.0, 1.0))
    expected = -(np.eye(8) - 2 * np.outer(psi, psi.conj())) @ good
    # They must agree on the invariant subspace spanned by |psi> and its good/bad parts.
    for vec in (psi, good @ psi):
        assert np.allclose(q @ vec, expected @ vec, atol=1e-9)


def test_register_size_guard_and_auto_mode(market):
    with pytest.raises(ValueError):
        discretize(market, Option("asian_call", 100.0, 1.0, steps=5), qubits_per_step=3)
    small = build_a_operator(discretize(market, Option("european_call", 100.0, 1.0), 2))
    assert CircuitOracle(small).mode == "unitary"
    with pytest.raises(ValueError):
        CircuitOracle(small, mode="tensor-network")


def test_resources_grow_with_register_size(market, call):
    small = circuit_resources(build_a_operator(discretize(market, call, 2)))
    large = circuit_resources(build_a_operator(discretize(market, call, 4)))
    assert small["qubits"] == 3 and large["qubits"] == 5
    assert 0 < small["cx"] < large["cx"] and small["depth"] < large["depth"]
