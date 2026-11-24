import numpy as np
import pytest

from project.quantum import qaoa as q
from project.solvers.base import Budget, Timer
from project.solvers.registry import run_solver
from conftest import random_problem

BUDGET = Budget(max_evals=32, time_limit_s=5.0)


@pytest.mark.parametrize("p", [1, 2])
def test_qiskit_circuit_matches_closed_form_simulation(p):
    problem = random_problem(6, 1)
    h, J, _ = q._normalised_ising(problem)
    energies = q.ising_energies(h, J)
    rng = np.random.default_rng(0)
    gammas, betas = rng.uniform(0, 1, p), rng.uniform(0, 1, p)
    ours = q.numpy_probabilities(energies, problem.n, gammas, betas)
    theirs = q._QiskitBackend(h, J, p, Timer()).probabilities(gammas, betas)
    np.testing.assert_allclose(ours, theirs, atol=1e-9)
    assert ours.sum() == pytest.approx(1.0)


def test_zero_angles_give_the_uniform_distribution():
    problem = random_problem(5, 2)
    h, J, _ = q._normalised_ising(problem)
    probs = q.numpy_probabilities(q.ising_energies(h, J), 5, np.zeros(1), np.zeros(1))
    np.testing.assert_allclose(probs, 1 / 32)


def test_normalised_energies_are_an_affine_map_of_the_qubo():
    problem = random_problem(7, 3)
    h, J, scale = q._normalised_ising(problem)
    _, _, offset = problem.to_ising()
    np.testing.assert_allclose(q.ising_energies(h, J) * scale + offset, problem.all_qubo_energies(), atol=1e-9)
    assert q.ising_energies(h, J).mean() == pytest.approx(0.0, abs=1e-9)


@pytest.mark.parametrize("backend", ["numpy", "qiskit"])
def test_qaoa_returns_feasible_result_with_full_accounting(tiny_problem, backend):
    res = run_solver("qaoa", tiny_problem, BUDGET, seed=3, backend=backend, p=1, shots=64, maxiter=10)
    assert res.status == "ok"
    assert tiny_problem.is_feasible(res.z)
    assert res.n_evals == BUDGET.max_evals
    assert res.n_shots == res.meta["optimizer_iters"] * 64 + BUDGET.max_evals
    assert {"encode", "optimize", "sample", "decode_repair"} <= set(res.timings)
    if backend == "qiskit":
        assert "compile" in res.timings and res.meta["two_qubit_gates"] > 0 and res.meta["depth"] > 0


def test_qaoa_is_seed_deterministic(tiny_problem):
    kw = dict(backend="numpy", p=1, shots=64, maxiter=10)
    a = run_solver("qaoa", tiny_problem, BUDGET, seed=7, **kw)
    b = run_solver("qaoa", tiny_problem, BUDGET, seed=7, **kw)
    np.testing.assert_array_equal(a.z, b.z)
    assert a.meta["theta"] == b.meta["theta"]


def test_optimised_circuit_beats_uniform_sampling_in_expected_energy(tiny_problem):
    res = run_solver("qaoa", tiny_problem, BUDGET, seed=1, backend="numpy", p=2, shots=512, maxiter=40, cvar_alpha=1.0)
    assert res.meta["expected_energy"] < -0.1  # uniform sampling is exactly 0 on the normalised scale
    assert res.meta["p_optimal_state"] > res.meta["uniform_p_optimal_state"]


def test_too_many_qubits_degrades_to_fallback():
    problem = random_problem(q.MAX_QUBITS + 1, 0, k=3)
    res = run_solver("qaoa", problem, BUDGET, seed=1, backend="numpy")
    assert res.status.startswith("fallback:ValueError") and problem.is_feasible(res.z)


def test_bad_backend_degrades_to_fallback(tiny_problem):
    assert run_solver("qaoa", tiny_problem, BUDGET, seed=1, backend="ibm_real_device").status.startswith("fallback")
