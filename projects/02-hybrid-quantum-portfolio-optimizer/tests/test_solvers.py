"""Solver adapters: correctness on known optima, budgets, failure and fallback behaviour."""
import sys

import numpy as np
import pytest

from hqpo.data.synthetic import planted_instance
from hqpo.models import Constraints
from hqpo.solvers import SOLVERS, Budget, get_solver
from hqpo.solvers.classical import brute_force_optimum, solve_relaxation

from conftest import make

BUDGET = Budget(wall_time_s=2.0, max_evals=20_000)
EXACT = ["brute-force", "miqp", "cpsat"]


@pytest.mark.parametrize("name", EXACT)
def test_exact_solvers_match_brute_force(tiny, name):
    _, f, _ = brute_force_optimum(tiny)
    res = get_solver(name).solve(tiny, BUDGET, seed=0)
    assert res.status == "ok" and res.feasible and not res.repaired
    assert res.objective == pytest.approx(f, abs=1e-7)
    assert res.meta["optimal_proven"]


@pytest.mark.parametrize("name", sorted(SOLVERS))
def test_every_solver_returns_a_feasible_portfolio_or_says_why(tiny, name):
    res = get_solver(name).solve(tiny, BUDGET, seed=1)
    assert res.status in ("ok", "infeasible", "skipped")
    if res.status == "ok":
        assert tiny.is_feasible(res.z)
        assert res.objective == pytest.approx(tiny.objective(res.z))
        assert np.allclose(res.weights, tiny.weights(res.z))
        _, f, _ = brute_force_optimum(tiny)
        assert res.objective >= f - 1e-9                     # nobody beats the true optimum
    else:
        assert res.message
    if name != "qaoa-noisy":       # noisy Aer sampling cannot be interrupted; its time is reported
        assert res.wall_time_s < BUDGET.wall_time_s + 3.0
    rec = res.record()
    assert "z" not in rec and rec["solver"] == name


@pytest.mark.parametrize("name", ["cvxpy-relax", "miqp", "greedy", "sa", "qubo-sa", "qaoa"])
def test_solvers_recover_a_planted_optimum(name):
    p, z_star = planted_instance(5, Constraints(lots=4, max_weight=0.5, cardinality=3), seed=2)
    budget = Budget(2.0, 2_000_000) if name == "qubo-sa" else BUDGET
    res = get_solver(name, **({"p": 2, "shots": 2048} if name == "qaoa" else {})).solve(p, budget, seed=3)
    assert res.feasible
    assert res.objective == pytest.approx(p.objective(z_star), abs=1e-9)


def test_relaxation_is_a_lower_bound(tiny):
    _, f, _ = brute_force_optimum(tiny)
    _, bound, status = solve_relaxation(tiny)
    assert status == "optimal" and bound <= f + 1e-7


def test_stochastic_solvers_are_reproducible_by_seed():
    p = make(6, lots=4, max_weight=0.5, cardinality=3)
    for name, prm in [("qaoa", {"p": 1, "maxiter": 30, "restarts": 1, "circuit_metrics": False}),
                      ("random-qubo", {})]:
        a = get_solver(name, **prm).solve(p, Budget(30, 10_000), seed=5)
        b = get_solver(name, **prm).solve(p, Budget(30, 10_000), seed=5)
        assert np.array_equal(a.z, b.z) and a.evals == b.evals


def test_qaoa_reports_resources_and_respects_eval_budget():
    p = make(4, lots=4, max_weight=0.5, cardinality=3)
    res = get_solver("qaoa", p=2, shots=512, maxiter=500).solve(p, Budget(20, 25), seed=0)
    assert res.evals <= 25
    m = res.meta
    assert m["n_qubits"] == 12 and m["depth"] > 0 and m["two_qubit_gates"] > 0
    assert m["depth_line"] >= m["depth"] and m["total_shots"] == 512
    assert 0 <= m["sample_feasible_fraction"] <= 1
    assert {"model_build", "compile", "state_prep", "optimize", "sampling", "decode"} <= set(res.timings)


def test_qaoa_concentrates_on_feasible_states_more_than_random_sampling():
    p = make(5, lots=4, max_weight=0.5, seed=5)
    q = get_solver("qaoa", p=2, shots=4096, circuit_metrics=False).solve(p, Budget(20, 400), seed=0)
    r = get_solver("random-qubo", shots=4096).solve(p, Budget(20, 400), seed=0)
    assert q.meta["sample_feasible_fraction"] > 1.5 * r.meta["sample_feasible_fraction"]


def test_qaoa_skips_instances_beyond_the_simulator_limit():
    p = make(16, lots=16, max_weight=0.25, cardinality=8)
    res = get_solver("qaoa").solve(p, BUDGET, seed=0)
    assert res.status == "skipped" and "qubits needed" in res.message and not res.feasible


def test_noisy_qaoa_runs_or_skips_cleanly():
    p = make(4, lots=4, max_weight=0.5)
    res = get_solver("qaoa-noisy", p=1, shots=128, maxiter=20, restarts=1).solve(p, BUDGET, seed=0)
    if res.status == "skipped":
        assert "qiskit-aer" in res.message
    else:
        assert res.feasible and "depolarising" in res.meta["backend"]


def test_infeasible_raw_answer_is_repaired_and_both_objectives_reported():
    """A hopeless penalty makes the QUBO optimum infeasible; repair must fix it and say so."""
    p = make(5, lots=4, max_weight=0.5, cardinality=3, seed=5)
    res = get_solver("qubo-sa", penalty={"method": "fixed", "value": 1e-6}).solve(p, BUDGET, seed=0)
    assert res.raw_violation > 0 and res.repaired and res.feasible
    assert np.isfinite(res.raw_objective) and res.raw_objective != res.objective
    assert "repair" in res.timings


def test_miqp_falls_back_to_cpsat_when_scip_is_missing(monkeypatch):
    monkeypatch.setitem(sys.modules, "pyscipopt", None)       # simulate "not installed"
    p = make(5, lots=4, max_weight=0.5, cardinality=3)
    res = get_solver("miqp").solve(p, BUDGET, seed=0)
    _, f, _ = brute_force_optimum(p)
    assert res.feasible and res.meta["fallback"] == "cpsat" and "SCIP" in res.meta["fallback_reason"]
    assert res.objective == pytest.approx(f, abs=1e-7)
    forced = get_solver("miqp", backend="scip").solve(p, BUDGET, seed=0)
    assert forced.status == "skipped"                         # explicit backend: no silent fallback


def test_annealer_falls_back_to_numpy_when_ocean_is_missing(monkeypatch):
    monkeypatch.setitem(sys.modules, "dwave.samplers", None)
    p = make(4, lots=4, max_weight=0.5)
    res = get_solver("qubo-sa", sweeps=100, reads=5).solve(p, BUDGET, seed=0)
    assert res.feasible and res.meta["fallback"] == "numpy"
    assert get_solver("qubo-sa", backend="ocean").solve(p, BUDGET, seed=0).status == "skipped"


def test_solver_exceptions_become_failed_results():
    class Boom(SOLVERS["greedy"]):
        def _solve(self, *a, **k):
            raise RuntimeError("kaboom")

    res = Boom().solve(make(4, lots=4, max_weight=0.5), BUDGET)
    assert res.status == "failed" and "kaboom" in res.message and not res.feasible


def test_unknown_solver_name():
    with pytest.raises(KeyError):
        get_solver("quantum-magic")
