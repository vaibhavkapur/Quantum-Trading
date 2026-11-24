from dataclasses import replace
from itertools import product

import numpy as np
import pytest

from pairs_trading.quantum import numpy_state, qiskit_template
from pairs_trading.selection import Problem, bounded_weights, brute_force, encode_qubo, make_problem, repair, solve


def test_known_optimum_beats_pair_ranking(tiny_problem, config):
    exact = brute_force(tiny_problem)
    np.testing.assert_array_equal(exact.x, [0, 1, 1])
    assert exact.objective == -8
    ranked = solve(tiny_problem, "top-k", config["solver"], 0)
    assert ranked.objective == -5
    milp = solve(tiny_problem, "milp", config["solver"], 0)
    assert milp.objective == exact.objective
    assert milp.status == "optimal"


@pytest.mark.parametrize("weighted", [False, True])
def test_qubo_all_assignments_and_slack(tiny_problem, weighted):
    if weighted:
        tiny_problem.constraints.append(("weighted_capital", np.array([2, 3, 4]), 5))
        tiny_problem.quadratic[1, 2] = 0.7
    qubo = encode_qubo(tiny_problem)
    energies = qubo.energies()
    oracle = brute_force(tiny_problem)
    assert energies.min() == pytest.approx(oracle.objective)
    for decision in product([0, 1], repeat=tiny_problem.n):
        x = np.array(decision)
        values = [qubo.energy(np.concatenate((x, slack))) for slack in map(np.array, product([0, 1], repeat=qubo.n - tiny_problem.n))]
        if tiny_problem.feasible(x):
            assert min(values) == pytest.approx(tiny_problem.objective(x))
        else:
            assert min(values) >= tiny_problem.objective(x) + qubo.penalty - 1e-9
    mask = int(np.argmin(energies))
    best = np.array([(mask >> i) & 1 for i in range(tiny_problem.n)])
    assert tiny_problem.feasible(best)


def test_slack_represents_every_value_without_exceeding_bound():
    for bound in range(20):
        weights = bounded_weights(bound)
        values = {sum(w * b for w, b in zip(weights, bits)) for bits in product([0, 1], repeat=len(weights))}
        assert values == set(range(bound + 1))


def test_security_sector_and_capital_limits(tiny_problem, config):
    pairs = [replace(p, capital_units=i+1, sectors=("shared",)) for i, p in enumerate(tiny_problem.pairs)]
    problem = make_problem(pairs, np.eye(3), config["selection"])
    assert problem.feasible(np.array([1, 0, 0]))
    assert not problem.feasible(np.array([0, 0, 1]))  # capital units > budget
    assert not problem.feasible(np.array([1, 1, 0]))
    for bits in product([0, 1], repeat=3):
        assert problem.feasible(repair(problem, np.array(bits)))


@pytest.mark.parametrize("solver", ["top-k", "greedy", "milp", "annealing", "quantum-sim"])
def test_solvers_feasible_deterministic_and_within_eval_budget(tiny_problem, config, solver):
    first = solve(tiny_problem, solver, config["solver"], 4)
    second = solve(tiny_problem, solver, config["solver"], 4)
    assert tiny_problem.feasible(first.x)
    np.testing.assert_array_equal(first.x, second.x)
    assert first.objective == pytest.approx(tiny_problem.objective(first.x))
    if solver in ("annealing", "quantum-sim"):
        assert first.evaluations <= config["solver"]["max_evaluations"]


@pytest.mark.parametrize("depth", [1, 2])
def test_qiskit_matches_independent_numpy_amplitudes(tiny_problem, depth):
    pytest.importorskip("qiskit")
    from qiskit.quantum_info import Statevector
    qubo = encode_qubo(tiny_problem)
    energies = qubo.energies()
    scale = max(1, np.ptp(energies))
    theta = np.linspace(0.2, 1.7, depth * 2)
    template, params = qiskit_template(qubo, depth, scale)
    qiskit_state = Statevector.from_instruction(template.assign_parameters(dict(zip(params, theta)))).data
    numpy_result = numpy_state((energies - energies.min()) / scale, theta, depth)
    assert np.abs(np.vdot(qiskit_state, numpy_result)) == pytest.approx(1, abs=1e-12)


def test_quantum_qiskit_adapter(tiny_problem, config):
    pytest.importorskip("qiskit")
    config["solver"]["backend"] = "qiskit"
    result = solve(tiny_problem, "quantum-sim", config["solver"], 7)
    assert tiny_problem.feasible(result.x)
    assert result.details["shots"] == 128
    assert result.details["two_qubit_gates"] > 0


def test_qubit_cap_failure_and_explicit_fallback(tiny_problem, config):
    config["solver"]["max_qubits"] = 1
    with pytest.raises(ValueError, match="qubits"):
        solve(tiny_problem, "quantum-sim", config["solver"], 0)
    config["solver"]["fallback"] = "greedy"
    result = solve(tiny_problem, "quantum-sim", config["solver"], 0)
    assert result.status == "fallback_greedy"
    assert result.details["actual_solver"] == "greedy"


def test_missing_dependency_is_not_silent(tiny_problem, config, monkeypatch):
    import pairs_trading.quantum as quantum

    def unavailable(*args):
        raise ImportError("test dependency unavailable")
    monkeypatch.setattr(quantum, "qaoa", unavailable)
    with pytest.raises(ImportError, match="unavailable"):
        solve(tiny_problem, "quantum-sim", config["solver"], 0)
    config["solver"]["fallback"] = "greedy"
    assert solve(tiny_problem, "quantum-sim", config["solver"], 0).status == "fallback_greedy"


def test_zero_candidates_and_invalid_inputs(config):
    empty = Problem([], np.empty(0), np.empty((0, 0)), [])
    result = solve(empty, "quantum-sim", config["solver"], 0)
    assert result.status == "no_candidates" and result.objective == 0
    with pytest.raises(ValueError, match="dimension"):
        Problem([], np.ones(1), np.zeros((0, 0)), [])


def test_classical_budget_and_repair_accounting(tiny_problem, config):
    from pairs_trading.selection import decode_samples
    config["solver"]["max_evaluations"] = 1
    result = solve(tiny_problem, "greedy", config["solver"], 0)
    assert result.evaluations == 1 and result.details["budget_exhausted"]
    assert tiny_problem.feasible(result.x)
    _, accounting = decode_samples(tiny_problem, np.array([[1, 1, 1]]))
    assert accounting["repair_objective_evaluations"] >= 3
    assert accounting["decode_objective_evaluations"] == 1


def test_zero_bound_qubo(tiny_problem):
    tiny_problem.constraints.append(("disabled_asset", np.array([0, 1, 0]), 0))
    qubo = encode_qubo(tiny_problem)
    assert qubo.energies().min() == pytest.approx(brute_force(tiny_problem).objective)
