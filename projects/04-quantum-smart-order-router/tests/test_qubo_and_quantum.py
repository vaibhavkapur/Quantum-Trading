import itertools

import numpy as np
import pytest

from conftest import random_problem
from qsor.optimizer import classical, quantum, registry
from qsor.optimizer.qubo import build_qubo, decode


def bits_for(p, qubo, x):
    return sum(1 << i for i, (v, k) in enumerate(qubo.var_map) if x[v] == k)


@pytest.mark.parametrize("seed", range(8))
def test_energy_equals_cost_on_feasible_assignments(seed):
    p = random_problem(np.random.default_rng(seed), V=3, max_units=3)
    q = build_qubo(p)
    energies = q.energies()
    feasible_max = -np.inf
    for x in itertools.product(*[range(n + 1) for n in p.n_units]):
        if sum(x) == p.target_units:
            z = bits_for(p, q, x)
            assert energies[z] == pytest.approx(p.cost(x), abs=1e-7)
            assert q.energy([(z >> i) & 1 for i in range(q.n)]) == pytest.approx(p.cost(x), abs=1e-7)
            feasible_max = max(feasible_max, energies[z])
    # ground state is the exact optimum, and every infeasible state lies above every feasible one
    z_min = int(np.argmin(energies))
    x_min, ok = decode(p, q, z_min)
    assert ok and p.cost(x_min) == pytest.approx(p.cost(classical.exact(p)))
    feasible = {bits_for(p, q, x) for x in itertools.product(*[range(n + 1) for n in p.n_units])
                if sum(x) == p.target_units}
    infeasible = np.array([e for z, e in enumerate(energies) if z not in feasible])
    assert infeasible.size == 0 or infeasible.min() > feasible_max


def test_decode_repairs_infeasible_bitstrings():
    p = random_problem(np.random.default_rng(3), V=3, max_units=3)
    q = build_qubo(p)
    for z in (0, (1 << q.n) - 1, 0b101):
        x, _ = decode(p, q, z)
        assert p.is_feasible(x)


def test_single_qubit_qaoa_matches_analytic_formula():
    # E(y) = d*y: P(1) = (1 + sin(2 beta) sin(gamma d)) / 2
    d, g, b = 1.7, 0.4, 0.3
    psi = quantum.qaoa_state(np.array([0.0, d]), [g], [b])
    assert abs(psi[1]) ** 2 == pytest.approx(0.5 * (1 + np.sin(2 * b) * np.sin(g * d)))
    assert np.vdot(psi, psi).real == pytest.approx(1.0)


def test_numpy_simulator_matches_qiskit_statevector():
    pytest.importorskip("qiskit")
    from qsor.optimizer.qiskit_backend import probabilities

    p = random_problem(np.random.default_rng(11), V=3, max_units=2)
    q = build_qubo(p)
    energies = q.energies()
    mean, std = quantum.normalize(energies)
    gammas, betas = [0.31, 0.52], [0.44, 0.17]
    psi = quantum.qaoa_state((energies - mean) / std, gammas, betas)
    np.testing.assert_allclose(np.abs(psi) ** 2, probabilities(q, mean, std, gammas, betas), atol=1e-10)


@pytest.mark.parametrize("name", ["qaoa", "sa", "random_repair"])
def test_qubo_heuristics_return_feasible_and_are_seed_deterministic(name):
    p = random_problem(np.random.default_rng(21), V=3, max_units=3)
    cfg = {"qaoa": {"layers": 2, "maxiter": 25, "shots": 128}, "sa": {"sweeps": 30, "restarts": 2},
           "random_repair": {"shots": 64}}
    x1, info = registry.solve(name, p, np.random.default_rng(1), cfg)
    x2, _ = registry.solve(name, p, np.random.default_rng(1), cfg)
    assert x1 == x2 and p.is_feasible(x1)
    assert p.cost(x1) >= p.cost(classical.exact(p)) - 1e-9
    assert info["qubits"] == sum(p.n_units)


def test_qaoa_concentrates_probability_on_low_energy_states():
    p = random_problem(np.random.default_rng(8), V=2, max_units=3)
    q = build_qubo(p)
    energies = q.energies()
    _, info = quantum.qaoa(p, np.random.default_rng(0), layers=2, maxiter=60, shots=64)
    assert info["expectation"] < energies.mean()  # better than the uniform superposition


def test_qubit_budget_and_deadline_are_enforced():
    p = random_problem(np.random.default_rng(21), V=3, max_units=3)
    assert sum(p.n_units) > 3
    with pytest.raises(quantum.TooManyQubits):
        quantum.qaoa(p, np.random.default_rng(0), max_qubits=3)
    with pytest.raises(classical.SolveTimeout):
        quantum.qaoa(p, np.random.default_rng(0), deadline=0.0)
