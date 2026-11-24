import numpy as np
import pytest

from hqpo.data.synthetic import planted_instance
from hqpo.models import Constraints, PortfolioProblem, local_search, repair
from hqpo.solvers.classical import brute_force_optimum, enumerate_lots

from conftest import make


def test_objective_in_lot_space_equals_weight_space(tiny, rng):
    for _ in range(20):
        z = rng.integers(0, tiny.ub + 1)
        assert tiny.objective(z) == pytest.approx(tiny.objective_w(tiny.weights(z)), abs=1e-12)
    Z = rng.integers(0, 3, size=(7, tiny.m))
    assert np.allclose(tiny.objective_batch(Z), [tiny.objective(z) for z in Z])


def test_hand_computed_objective():
    p = PortfolioProblem(np.diag([0.04, 0.09]), np.array([0.10, 0.20]),
                         Constraints(lots=2, max_weight=1.0), risk_aversion=2.0,
                         turnover_penalty=0.5, w_prev=np.array([1.0, 0.0]))
    # w = (0.5, 0.5): 2*(0.01+0.0225) - 0.15 + 0.5*(0.25+0.25)
    assert p.objective([1, 1]) == pytest.approx(0.065 - 0.15 + 0.25)
    assert p.weights([2, 0]).tolist() == [1.0, 0.0]


def test_constraint_violations_are_itemised():
    p = make(6, lots=4, max_weight=0.5, cardinality=2, sector_cap=0.5)   # sectors 0,1,2,0,1,2
    assert p.violations([2, 2, 0, 0, 0, 0]) == dict(bounds=0, budget=0, sector=0, cardinality=0, overlap=0)
    assert p.violations([3, 1, 0, 0, 0, 0])["bounds"] == 1               # 0.75 > max_weight
    assert p.violations([2, 1, 0, 0, 0, 0])["budget"] == 1
    assert p.violations([2, 0, 0, 2, 0, 0])["sector"] == 2               # sector 0 holds 100 %
    assert p.violations([2, 1, 1, 0, 0, 0])["cardinality"] == 1
    assert not p.is_feasible([2, 1, 1, 0, 0, 0])


def test_market_neutral_structure():
    p = make(3, lots=2, max_weight=0.5, mode="market_neutral")
    z = np.array([2, 0, 0, 0, 1, 1])
    w = p.weights(z)
    assert w.tolist() == [0.5, -0.25, -0.25] and w.sum() == 0 and np.abs(w).sum() == 1.0
    assert p.is_feasible(z)
    assert p.violations([1, 1, 0, 1, 1, 0])["overlap"] == 2


@pytest.mark.parametrize("kw, msg", [
    (dict(lots=4, max_weight=0.1), "smaller than one lot"),
    (dict(lots=4, max_weight=0.25, cardinality=2), "cardinality"),
])
def test_obviously_infeasible_configs_are_rejected(kw, msg):
    with pytest.raises(ValueError, match=msg):
        make(4, **kw)


@pytest.mark.parametrize("bad", ["nan", "asym", "indefinite", "shape"])
def test_input_edge_cases(bad):
    sigma, mu = np.eye(3) * 0.04, np.full(3, 0.05)
    if bad == "nan":
        mu = np.array([0.1, np.nan, 0.1])
    elif bad == "asym":
        sigma = sigma + np.triu(np.ones((3, 3)), 1) * 0.01
    elif bad == "indefinite":
        sigma = np.diag([0.04, 0.04, -0.01])
    else:
        mu = np.full(4, 0.05)
    with pytest.raises(ValueError):
        PortfolioProblem(sigma, mu, Constraints(lots=3))
    with pytest.raises(ValueError):
        Constraints(mode="short_only")


def test_enumeration_visits_exactly_the_budget_feasible_box():
    p = make(4, lots=4, max_weight=0.5)
    Z = np.vstack(list(enumerate_lots(p)))
    assert len(Z) == len({tuple(z) for z in Z}) == 19        # compositions of 4 into 4 parts <= 2
    assert np.all(Z.sum(1) == 4) and Z.max() <= 2


def test_repair_always_returns_feasible_or_reports_failure(tiny, rng):
    failures = 0
    for _ in range(200):
        z0 = rng.integers(0, tiny.ub + 2)                    # deliberately out of bounds sometimes
        z, ok = repair(tiny, z0)
        assert ok == tiny.is_feasible(z)
        failures += not ok
    assert failures <= 10


def test_repair_leaves_feasible_points_alone_and_is_deterministic(tiny):
    z, _, _ = brute_force_optimum(tiny)
    z2, ok = repair(tiny, z)
    assert ok and np.array_equal(z, z2)
    bad = np.zeros(tiny.m, dtype=int)
    assert np.array_equal(repair(tiny, bad)[0], repair(tiny, bad)[0])


def test_local_search_never_worsens_and_stays_feasible(tiny, rng):
    z, ok = repair(tiny, rng.integers(0, tiny.ub + 1))
    assert ok
    z2, _ = local_search(tiny, z)
    assert tiny.is_feasible(z2) and tiny.objective(z2) <= tiny.objective(z) + 1e-12


@pytest.mark.parametrize("kw", [
    dict(lots=4, max_weight=0.5, cardinality=3),
    dict(lots=6, max_weight=0.5, sector_cap=0.67),
    dict(lots=4, max_weight=0.5, cardinality=3, turnover_penalty=0.2),
    dict(lots=2, max_weight=0.5, mode="market_neutral"),
])
@pytest.mark.parametrize("seed", [1, 2, 3])
def test_planted_instance_has_the_known_optimum(kw, seed):
    """Synthetic covariance generator with an analytically known optimal solution."""
    kw = dict(kw)
    tau = kw.pop("turnover_penalty", 0.0)
    p, z_star = planted_instance(5, Constraints(**kw), seed=seed, turnover_penalty=tau)
    z, f, _ = brute_force_optimum(p)
    assert p.is_feasible(z_star)
    assert f == pytest.approx(p.objective(z_star), abs=1e-12)
    assert np.allclose(p.weights(z), p.weights(z_star))
