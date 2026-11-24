import builtins
from itertools import product

import numpy as np
import pytest

from strategy_optimizer.quantum import qaoa_circuit
from strategy_optimizer.qubo import build_qubo, exact_reference, feasible_portfolios, milp_reference
from strategy_optimizer.solvers import SOLVERS, solve


@pytest.fixture
def problem():
    returns = np.array([[.03, -.01, .01], [.02, -.02, .02], [.01, -.01, .01], [.02, -.03, .02]])
    return build_qubo(returns, 2, 0.5, 0.01, annualization=1)


def test_qubo_matches_return_risk_and_ising_for_every_bitstring(problem):
    _, h, pairs = problem.ising()
    offset, _, _ = problem.ising()
    energies = []
    for bits in product([0, 1], repeat=problem.n_bits):
        x = np.array(bits)
        z = x[:problem.n_strategies]
        slack = x[problem.n_strategies:] @ 2 ** np.arange(problem.n_bits - problem.n_strategies)
        expected = problem.objective(z) + problem.penalty * (z.sum() + slack - problem.capacity) ** 2
        assert problem.energy(x) == pytest.approx(expected, abs=1e-12)
        spins = 1 - 2 * x
        assert offset + h @ spins + spins @ pairs @ spins == pytest.approx(expected, abs=1e-12)
        energies.append((expected, problem.feasible(x)))
    assert min(value for value, feasible in energies if not feasible) > min(value for value, feasible in energies if feasible)


def test_surrogate_is_exact_for_fixed_sleeve_returns():
    rng = np.random.default_rng(3)
    returns = rng.normal(.001, .01, (50, 4)) / 2
    problem = build_qubo(returns, 2, 1.7, .002)
    for z in feasible_portfolios(problem):
        series = returns @ z
        expected = -series.mean()*252 + 1.7*series.var(ddof=1)*252 + .002*z.sum()
        assert problem.objective(z) == pytest.approx(expected, abs=1e-12)


def test_known_optimum_and_milp_reference(problem):
    reference = exact_reference(problem)
    assert reference["selected"] == [1, 0, 1]
    assert milp_reference(problem)["objective"] == pytest.approx(reference["objective"])
    assert reference["evaluations"] == 7


def test_milp_handles_correlated_objectives():
    for seed in range(4):
        rng = np.random.default_rng(seed)
        problem = build_qubo(rng.normal(.001, .02, (25, 5)), 3, 4, .001)
        assert milp_reference(problem)["objective"] == pytest.approx(exact_reference(problem)["objective"], abs=1e-7)


@pytest.mark.parametrize("solver", SOLVERS)
def test_budget_feasibility_and_repeatability(problem, solver):
    first = solve(problem, solver, 48, 7, {"shots_per_evaluation": 4})
    second = solve(problem, solver, 48, 7, {"shots_per_evaluation": 4})
    assert first.status == "ok", first.error
    assert first.selected == second.selected
    assert first.trace == second.trace
    assert first.evaluations <= 48
    assert first.evaluations == len(first.trace)
    assert problem.feasible(problem.encode(first.selected))
    assert first.objective >= exact_reference(problem)["objective"] - 1e-10
    assert sum(first.timing.values()) == pytest.approx(first.elapsed_s, abs=1e-6)
    if solver == "grid":
        assert first.evaluations == 7
        assert first.objective == pytest.approx(exact_reference(problem)["objective"])
    if solver == "qaoa":
        assert first.resources["shots"] == 48
        assert first.resources["circuit_evaluations"] == 12


def test_cost_layer_matches_analytic_phase_and_little_endian(problem):
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Statevector
    circuit, angles, scale = qaoa_circuit(problem, 1)
    preparation = QuantumCircuit(problem.n_bits)
    preparation.h(range(problem.n_bits))
    gamma = 0.37
    actual = Statevector.from_instruction(preparation).evolve(circuit.assign_parameters({angles[0]: gamma, angles[1]: 0})).data
    energy = np.array([problem.energy([(index >> i) & 1 for i in range(problem.n_bits)]) for index in range(2**problem.n_bits)])
    expected = np.exp(-1j*gamma*energy/scale) / np.sqrt(2**problem.n_bits)
    np.testing.assert_allclose(actual/actual[0], expected/expected[0], atol=1e-12)


def test_quantum_limit_fails_without_classical_fallback(problem):
    result = solve(problem, "qaoa", 32, 7, {"max_qubits": 1})
    assert result.status == "failed"
    assert result.selected is None
    assert result.evaluations == 0
    assert "above configured limit" in result.error


def test_missing_optuna_is_recorded_not_replaced(problem, monkeypatch):
    original = builtins.__import__
    def importing(name, *args, **kwargs):
        if name == "optuna":
            raise ImportError("missing Optuna fixture")
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", importing)
    result = solve(problem, "tpe", 32, 7)
    assert result.status == "failed"
    assert result.evaluations == 0
    assert "missing Optuna" in result.error


def test_invalid_objective_inputs():
    for values in [np.ones(4), np.ones((1, 4)), np.full((4, 2), np.nan)]:
        with pytest.raises(ValueError):
            build_qubo(values, 1, 1, 0)
    with pytest.raises(ValueError):
        build_qubo(np.ones((4, 2)), 3, 1, 0)

