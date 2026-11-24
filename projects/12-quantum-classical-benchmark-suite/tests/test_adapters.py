import itertools

import numpy as np
import pytest
from qiskit.quantum_info import Statevector

from benchmarks.adapters import get_adapter
from benchmarks.adapters.base import Budget, BudgetExceeded, Context
from benchmarks.adapters.estimation import amplitude_circuits, maximum_likelihood
from benchmarks.adapters.quantum import qaoa_circuit
from benchmarks.instances.generators import generate
from benchmarks.instances.schema import feasible, objective
from experiments.config import DEFAULT_ALGORITHM, DEFAULT_BUDGET


@pytest.mark.parametrize("track", ["portfolio", "basket"])
@pytest.mark.parametrize("seed", [1, 7, 33])
@pytest.mark.parametrize("solver", ["exact", "milp"])
def test_strong_baselines_match_exhaustive_reference(track, seed, solver, context):
    instance = generate(track, 8, seed)
    outcome = get_adapter(solver).solve(instance["problem"], context)
    assert feasible(instance["problem"], outcome["candidate"])
    assert objective(instance["problem"], outcome["candidate"]) == pytest.approx(instance["reference"]["value"])
    assert outcome["proven_optimal"]
    if solver == "milp":
        assert context.resources["objective_evaluations"] is None
        assert not context.resources["evaluation_budget_enforced"]


@pytest.mark.parametrize("solver", ["greedy", "annealing", "qaoa"])
def test_stochastic_adapters_reproduce_outputs_and_respect_budget(solver):
    p = generate("portfolio", 4, 10)["problem"]
    contexts = [Context(Budget(**DEFAULT_BUDGET), 12, DEFAULT_ALGORITHM.copy()) for _ in range(2)]
    outputs = [get_adapter(solver).solve(p, ctx) for ctx in contexts]
    assert outputs[0] == outputs[1]
    assert feasible(p, outputs[0]["candidate"])
    for ctx in contexts:
        assert ctx.resources["objective_evaluations"] <= ctx.budget.max_evaluations
        assert ctx.resources["shots"] <= ctx.budget.max_shots
        if solver == "qaoa":
            assert ctx.resources["logical_qubits"] == 4
            assert ctx.resources["circuit_depth"] > 0
            assert ctx.resources["shots"] == ctx.resources["optimizer_evaluations"] * ctx.algorithm["shots_per_circuit"]


def test_qaoa_zero_mixer_implements_cost_phase():
    p = generate("portfolio", 4, 2)["problem"]
    circuit, parameters, (c, q, offset, _, scale) = qaoa_circuit(p, 1)
    gamma = 0.37
    actual = Statevector(circuit.assign_parameters(dict(zip(parameters, [gamma, 0.])))).data
    energies = []
    for state in range(2 ** p["size"]):
        x = (state >> np.arange(p["size"])) & 1
        energies.append(c @ x + x @ q @ x + offset)
    expected = np.exp(-1j * gamma * np.array(energies) / scale) / np.sqrt(len(energies))
    phase = actual[0] / expected[0]
    assert np.allclose(actual, phase * expected)


@pytest.mark.parametrize("payoffs", [[0., 0., 0., 0.], [1., 1., 1., 1.], [0., 0.2, 0.7, 1.]])
def test_actual_qae_circuits_match_amplification_identity(payoffs):
    p = {"size": 4, "probabilities": [0.1, 0.2, 0.3, 0.4], "payoffs": payoffs}
    amplitude = np.dot(p["probabilities"], payoffs)
    theta = np.arcsin(np.sqrt(amplitude))
    for power, circuit in zip([0, 1, 2, 4], amplitude_circuits(p, [0, 1, 2, 4])):
        actual = Statevector(circuit).probabilities()[4:].sum()
        assert actual == pytest.approx(np.sin((2 * power + 1) * theta)**2, abs=1e-10)


@pytest.mark.parametrize("amplitude", [0., 0.07, 0.4, 0.95, 1.])
def test_likelihood_fit_including_endpoints(amplitude):
    theta = np.arcsin(np.sqrt(amplitude))
    schedule = [{"power": k, "shots": 10000, "hits": round(10000 * np.sin((2*k + 1)*theta)**2)} for k in [0, 1, 2, 4]]
    estimate, interval = maximum_likelihood(schedule)
    assert estimate == pytest.approx(amplitude, abs=1e-3)
    assert interval[0] <= amplitude <= interval[1]


@pytest.mark.parametrize("solver", ["exact-expectation", "mc", "mc-antithetic", "qae"])
def test_estimators_use_same_distribution_and_account_queries(solver, context):
    instance = generate("estimation", 4, 7)
    result = get_adapter(solver).solve(instance["problem"], context)
    assert abs(result["estimate"] - instance["reference"]["value"]) < 0.8
    assert context.resources["oracle_queries"] <= context.budget.max_oracle_queries
    if solver == "exact-expectation":
        assert result["estimate"] == instance["reference"]["value"]
        assert context.resources["oracle_queries"] == 4
    if solver == "qae":
        assert context.resources["oracle_queries"] == sum(r["shots"] * (2 * r["power"] + 1) for r in result["schedule"])
        assert context.resources["shots"] == sum(r["shots"] for r in result["schedule"])
        assert context.resources["logical_qubits"] == 3


def test_antithetic_pair_variance_reduction(context):
    p = {"size": 2, "probabilities": [0.5, 0.5], "payoffs": [0., 1.], "scale": 1.}
    result = get_adapter("mc-antithetic").solve(p, context)
    assert result["estimate"] == 0.5
    assert result["estimate_ci95"] == [0.5, 0.5]
    assert result["independent_observations"] == context.budget.max_oracle_queries // 2


def test_budget_check_is_atomic(context):
    with pytest.raises(BudgetExceeded):
        context.charge(evaluations=1, shots=context.budget.max_shots + 1)
    assert context.resources["objective_evaluations"] == 0
    assert context.resources["shots"] == 0


def test_no_unbudgeted_qaoa_final_sampling():
    ctx = Context(Budget(**(DEFAULT_BUDGET | {"max_shots": 16})), 42, DEFAULT_ALGORITHM.copy())
    with pytest.raises(BudgetExceeded):
        get_adapter("qaoa").solve(generate("portfolio", 4, 1)["problem"], ctx)
    assert ctx.resources["shots"] == 16
    assert ctx.resources["optimizer_evaluations"] == 1
