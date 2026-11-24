import numpy as np
import pytest

pytest.importorskip("qiskit")
from qiskit.quantum_info import Statevector

from quantum.distribution import DiscreteLoss, discretize_loss
from quantum.estimation import build_oracle, estimate_amplitude, estimate_risk


@pytest.mark.parametrize("objective", [[0, 1], [0.2, 0.8], [0, 0], [1, 1]])
def test_actual_grover_circuit_matches_analytic_rotation(objective):
    grid = DiscreteLoss(np.array([0., 1.]), np.array([.7, .3]))
    a, q = build_oracle(grid, objective)
    exact = np.dot(grid.probabilities, objective)
    theta = np.arcsin(np.sqrt(exact))
    state = Statevector(a)
    for k in range(5):
        actual = np.sum(np.abs(state.data[2:])**2)
        assert actual == pytest.approx(np.sin((2*k+1)*theta)**2, abs=1e-11)
        state = state.evolve(q)


def test_mlae_precision_resources_and_seed():
    grid = DiscreteLoss(np.array([0., 1.]), np.array([.7, .3]))
    result = estimate_amplitude(grid, [0, 1], [0, 1, 2, 4], 512, 17)
    assert result["estimate"] == pytest.approx(.3, abs=.01)
    assert result["ci95"][0] <= .3 <= result["ci95"][1]
    assert result["resources"]["oracle_queries"] == 512*(1+3+5+9)
    assert result["resources"]["shots"] == 2048
    assert result["resources"]["max_circuit_depth"] > 1
    repeat = estimate_amplitude(grid, [0, 1], [0, 1, 2, 4], 512, 17)
    assert repeat["schedule"] == result["schedule"]
    assert repeat["estimate"] == result["estimate"]


@pytest.mark.parametrize("objective,expected", [([0, 0], 0), ([1, 1], 1)])
def test_endpoint_amplitudes(objective, expected):
    grid = DiscreteLoss(np.array([0., 1.]), np.array([.7, .3]))
    assert estimate_amplitude(grid, objective, [0, 1, 2], 128, 7)["estimate"] == pytest.approx(expected, abs=1e-12)


def test_discrete_quantile_and_excess_oracle_against_bruteforce():
    grid = DiscreteLoss(np.array([0., 1., 2., 3.]), np.array([.1, .2, .3, .4]))
    result = estimate_risk(grid, .5, [0, 1, 2, 4], 512, 23)
    exact = grid.risk(.5)
    assert result["var"] == exact["var"] == 2
    assert result["cvar"] == pytest.approx(exact["cvar"], abs=.03)
    assert result["total_oracle_queries"] == sum(q["resources"]["oracle_queries"] for q in result["queries"])


def test_grid_tail_mass_and_constant_distribution():
    grid = discretize_loss(10, 2, 6, 5)
    assert grid.probabilities.sum() == pytest.approx(1)
    assert grid.clipped_tail_mass < 1e-6
    assert np.dot(grid.probabilities, grid.losses) == pytest.approx(10)
    heavy = discretize_loss(10, 2, 6, 5, "student-t", 5)
    assert heavy.clipped_tail_mass > grid.clipped_tail_mass
    constant = discretize_loss(-2, 0)
    assert constant.risk(.99)["var"] == -2


@pytest.mark.parametrize("probabilities", [[.1, .1], [-.1, 1.1], [np.nan, 1]])
def test_invalid_distribution(probabilities):
    with pytest.raises(ValueError):
        DiscreteLoss(np.array([0, 1]), np.array(probabilities))


def test_invalid_quantum_budget():
    grid = discretize_loss(0, 1, 1)
    with pytest.raises(ValueError):
        estimate_amplitude(grid, [0, 1], [1], 100)
    with pytest.raises(ValueError):
        estimate_amplitude(grid, [0, 1], [0, 33], 100)
