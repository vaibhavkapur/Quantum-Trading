import itertools

import numpy as np
import pytest

from conftest import random_problem
from qsor.optimizer import classical
from qsor.optimizer.problem import CostModel, build_problem, repair


def model(V, **kw):
    base = dict(fee=(0.0,) * V, fixed=(0.0,) * V, impact=(0.0,) * V, fill_prob=(1.0,) * V,
                fade=tuple((0.0,) * V for _ in range(V)), nonfill_penalty=0.0)
    base.update(kw)
    return CostModel(**base)


def brute_force(p):
    best = None
    for x in itertools.product(*[range(n + 1) for n in p.n_units]):
        if sum(x) == p.target_units:
            c = p.cost(x)
            if best is None or c < best[0] - 1e-12:
                best = (c, x)
    return best


def test_table_values_by_hand():
    # venue 0: 2 lots @100, 3 @101; fee 0.3/lot, impact 0.1/lot^2, fixed 0.5, fill prob 0.9, penalty 2
    m = model(1, fee=(0.3,), fixed=(0.5,), impact=(0.1,), fill_prob=(0.9,), nonfill_penalty=2.0)
    p = build_problem(1, 3, [((100, 2), (101, 3))], [5], 100, m)
    # x=3: walk = 0*2 + 1*1 = 1; fee 0.9; impact 0.9; nonfill 2*0.1*3 = 0.6; fixed 0.5
    assert p.tables[0][0] == 0.0
    assert p.tables[0][3] == pytest.approx(1 + 0.9 + 0.9 + 0.6 + 0.5)
    assert p.tables[0][1] == pytest.approx(0 + 0.3 + 0.1 + 0.2 + 0.5)


def test_sell_side_slippage_sign():
    p = build_problem(-1, 2, [((100, 1), (99, 5))], [6], 100, model(1))
    assert p.tables[0][2] == pytest.approx(1.0)  # second lot sells one tick lower


def test_caps_units_and_target():
    lad = [((100, 10),), ((100, 10),), ()]
    p = build_problem(1, 8, lad, [3, 10, 0], 100, model(3), max_units=4)
    assert p.unit == 2 and p.n_units == (1, 4, 0) and p.target_units == 4
    p = build_problem(1, 8, lad, [3, 2, 0], 100, model(3), max_units=4)
    assert p.target_units == 2  # conservation target shrinks to what the caps allow
    assert p.lots((1, 1, 0)) == (2, 2, 0)


def test_cross_term_only_between_usable_venues():
    fade = ((0.0, 0.5, 0.5), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
    p = build_problem(1, 4, [((100, 4),), ((100, 4),), ()], [4, 4, 0], 100,
                      model(3, fade=fade, nonfill_penalty=2.0))
    assert p.cross[0][1] == pytest.approx(2.0 * 0.5 / 4) and p.cross[1][0] == p.cross[0][1]
    assert p.cross[0][2] == 0.0
    assert p.cost((2, 2, 0)) == pytest.approx(p.cross[0][1] * 4)


def test_known_optimum_where_greedy_fails():
    # Venue 0 is cheap per lot but has a large fixed charge; venue 1 has none.
    # Greedy never pays the fixed charge; the optimum routes everything to venue 0.
    m = model(2, fee=(0.0, 1.0), fixed=(2.5, 0.0))
    p = build_problem(1, 4, [((100, 4),), ((100, 4),)], [4, 4], 100, m)
    assert classical.greedy(p) == (0, 4) and p.cost((0, 4)) == pytest.approx(4.0)
    assert classical.exact(p) == (4, 0) and p.cost((4, 0)) == pytest.approx(2.5)
    assert classical.cpsat(p) == (4, 0)


@pytest.mark.parametrize("seed", range(12))
def test_exact_solvers_match_brute_force(seed):
    p = random_problem(np.random.default_rng(seed), V=3 + seed % 2)
    c_star, _ = brute_force(p)
    assert p.cost(classical.exact(p)) == pytest.approx(c_star)
    x = classical.cpsat(p)
    assert p.is_feasible(x)
    assert p.cost(x) == pytest.approx(c_star, abs=1e-3)  # CP-SAT works on costs rounded to 1e-4


@pytest.mark.parametrize("seed", range(12))
def test_heuristics_are_feasible_and_not_better_than_optimum(seed):
    p = random_problem(np.random.default_rng(100 + seed))
    c_star = p.cost(classical.exact(p))
    for solver in (classical.best_price_first, classical.pro_rata, classical.greedy):
        x = solver(p)
        assert p.is_feasible(x), solver.__name__
        assert p.cost(x) >= c_star - 1e-9


def test_repair_projects_onto_feasible_set():
    p = random_problem(np.random.default_rng(5))
    for x in ([9] * p.V, [0] * p.V, [-3] + [1] * (p.V - 1)):
        assert p.is_feasible(repair(p, x))


def test_empty_and_degenerate_inputs():
    p = build_problem(1, 4, [(), ()], [0, 0], 100, model(2))
    assert p.target_units == 0
    for solver in (classical.best_price_first, classical.pro_rata, classical.greedy, classical.exact):
        assert solver(p) == (0, 0)
    one = build_problem(1, 1, [((100, 1),)], [1], 100, model(1))
    assert classical.exact(one) == (1,) and classical.cpsat(one) == (1,)
