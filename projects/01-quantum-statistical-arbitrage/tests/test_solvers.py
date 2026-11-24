import numpy as np
import pytest

from project.solvers import registry
from project.solvers.base import Budget
from project.solvers.registry import run_solver
from conftest import random_problem

BUDGET = Budget(max_evals=40, time_limit_s=5.0)
ALL = ["brute_force", "greedy", "cpsat", "random_search", "simulated_annealing"]


@pytest.mark.parametrize("name", ["brute_force", "cpsat", "greedy"])
def test_exact_and_greedy_find_the_fixture_optimum(tiny_problem, name):
    res = run_solver(name, tiny_problem, BUDGET, seed=1)
    assert tiny_problem.decode(res.z) == ["A", "B", "C"]
    assert res.objective == pytest.approx(-2.9)
    assert not res.status.startswith("fallback")


@pytest.mark.parametrize("seed", range(6))
def test_cpsat_matches_brute_force_on_random_instances(seed):
    p = random_problem(11, seed, k=4, max_per_sector=2)
    assert run_solver("cpsat", p, BUDGET).objective == pytest.approx(p.brute_force()[1], abs=1e-5)
    assert run_solver("cpsat", p, BUDGET).status == "optimal"


@pytest.mark.parametrize("name", ALL)
def test_every_solver_returns_a_feasible_basket_never_better_than_optimal(name):
    p = random_problem(12, 3, k=4)
    res = run_solver(name, p, BUDGET, seed=5)
    assert p.is_feasible(res.z)
    assert res.objective == pytest.approx(p.base_objective(res.z))
    assert res.objective >= p.brute_force()[1] - 1e-9
    assert res.wall_time > 0


@pytest.mark.parametrize("name", ["random_search", "simulated_annealing"])
def test_stochastic_solvers_are_seed_deterministic_and_respect_the_budget(name):
    p = random_problem(12, 4, k=4)
    a, b = run_solver(name, p, BUDGET, seed=9), run_solver(name, p, BUDGET, seed=9)
    np.testing.assert_array_equal(a.z, b.z)
    assert a.n_evals == BUDGET.max_evals
    assert 0.0 <= a.raw_feasible_rate <= 1.0


def test_raw_best_is_never_better_than_repaired_best():
    p = random_problem(12, 6, k=4)
    res = run_solver("random_search", p, Budget(400, 1.0), seed=2)
    assert np.isnan(res.raw_best_objective) or res.raw_best_objective >= res.objective - 1e-12


def test_failing_solver_falls_back_to_greedy_and_is_flagged(tiny_problem, monkeypatch):
    def boom(problem, budget, seed=None):
        raise TimeoutError("backend unavailable")

    monkeypatch.setitem(registry.SOLVERS, "cpsat", boom)
    res = run_solver("cpsat", tiny_problem, BUDGET)
    assert res.solver == "cpsat"
    assert res.status.startswith("fallback:TimeoutError")
    assert tiny_problem.is_feasible(res.z)
    assert res.meta == {"fallback_solver": "greedy"}


def test_infeasible_solver_output_triggers_fallback(tiny_problem, monkeypatch):
    def cheat(problem, budget, seed=None):
        res = registry.classical.greedy(problem, budget, seed)
        res.z = np.ones(problem.n, dtype=int)
        return res

    monkeypatch.setitem(registry.SOLVERS, "greedy_bad", cheat)
    res = run_solver("greedy_bad", tiny_problem, BUDGET)
    assert res.status.startswith("fallback") and tiny_problem.is_feasible(res.z)


def test_unknown_solver_raises(tiny_problem):
    with pytest.raises(KeyError):
        run_solver("annealer-9000", tiny_problem, BUDGET)
