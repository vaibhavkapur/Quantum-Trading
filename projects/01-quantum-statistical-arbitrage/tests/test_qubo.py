import numpy as np
import pytest

from project.research.qubo import BasketProblem
from project.quantum.qaoa import ising_energies
from conftest import random_problem


def test_known_optimum_by_hand(tiny_problem):
    z, val, count = tiny_problem.brute_force()
    assert tiny_problem.decode(z) == ["A", "B", "C"]
    assert val == pytest.approx(-3.0 + 0.1)
    assert count == 8  # one asset from each of three two-asset sectors


def test_base_objective_counts_each_pair_once(tiny_problem):
    z = tiny_problem.encode(["D", "E", "F"])
    assert tiny_problem.base_objective(z) == pytest.approx(-1.6)
    assert tiny_problem.base_objective(np.zeros(6)) == 0.0


def test_qubo_equals_objective_plus_penalty_on_every_state(tiny_problem):
    A, K = tiny_problem.penalty(), tiny_problem.basket_size
    bits = tiny_problem.all_bitstrings()
    energies = tiny_problem.all_qubo_energies()
    same = tiny_problem.sectors[:, None] == tiny_problem.sectors[None, :]
    for z, e in zip(bits, energies):
        pairs = np.triu(np.outer(z, z) * same, 1).sum()  # max_per_sector == 1 -> exact pairwise term
        expected = tiny_problem.base_objective(z) + A * (z.sum() - K) ** 2 + A * pairs
        assert e == pytest.approx(expected)
        assert tiny_problem.qubo_energy(z) == pytest.approx(expected)


@pytest.mark.parametrize("seed", range(5))
def test_ising_matches_qubo(seed):
    p = random_problem(7, seed)
    h, J, offset = p.to_ising()
    np.testing.assert_allclose(ising_energies(h, J) + offset, p.all_qubo_energies(), atol=1e-9)


@pytest.mark.parametrize("seed", range(8))
def test_qubo_ground_state_is_the_constrained_optimum(seed):
    # Sector cap 1 is encoded exactly, so the unconstrained QUBO minimum must be feasible and optimal.
    p = random_problem(9, seed, k=3, max_per_sector=1)
    z = p.all_bitstrings()[np.argmin(p.all_qubo_energies())]
    assert p.is_feasible(z)
    assert p.base_objective(z) == pytest.approx(p.brute_force()[1])


@pytest.mark.parametrize("seed", range(8))
def test_cardinality_penalty_is_exact_without_sector_binding(seed):
    p = random_problem(8, seed, k=3, max_per_sector=8)
    z = p.all_bitstrings()[np.argmin(p.all_qubo_energies())]
    assert z.sum() == 3
    assert p.base_objective(z) == pytest.approx(p.brute_force()[1])


def test_repair_always_returns_a_feasible_basket():
    rng = np.random.default_rng(0)
    for seed in range(20):
        p = random_problem(10, seed, k=4, max_per_sector=2)
        for _ in range(30):
            z = p.repair(rng.integers(0, 2, p.n))
            assert p.is_feasible(z)
        assert p.is_feasible(p.repair(np.zeros(p.n, dtype=int)))
        assert p.is_feasible(p.repair(np.ones(p.n, dtype=int)))


def test_repair_leaves_feasible_input_unchanged(tiny_problem):
    z = tiny_problem.encode(["A", "E", "F"])
    np.testing.assert_array_equal(tiny_problem.repair(z), z)


def test_violations_report(tiny_problem):
    v = tiny_problem.violations(tiny_problem.encode(["A", "D", "B", "C"]))
    assert v == {"cardinality": 1, "sector_excess": 1}


def test_encode_decode_roundtrip(tiny_problem):
    assert tiny_problem.decode(tiny_problem.encode(["B", "D"])) == ["B", "D"]


def _kw(**over):
    base = dict(symbols=["a", "b", "c"], linear=np.zeros(3), quad=np.zeros((3, 3)), sectors=np.array([0, 0, 0]),
                basket_size=2, max_per_sector=2)
    base.update(over)
    return base


def test_rejects_bad_inputs():
    with pytest.raises(ValueError, match="infeasible"):
        BasketProblem(**_kw(max_per_sector=1))
    with pytest.raises(ValueError, match="symmetric"):
        BasketProblem(**_kw(quad=np.array([[0, 1, 0], [0, 0, 0], [0, 0, 0.0]])))
    with pytest.raises(ValueError, match="finite"):
        BasketProblem(**_kw(linear=np.array([0, np.nan, 0])))
    with pytest.raises(ValueError, match="basket_size"):
        BasketProblem(**_kw(basket_size=4))
    with pytest.raises(ValueError, match="dimensions"):
        BasketProblem(**_kw(linear=np.zeros(2)))
