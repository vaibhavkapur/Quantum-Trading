import numpy as np
import pytest
from scipy.stats import norm

from risk import Portfolio, empirical_risk, gaussian_risk
from risk.measures import student_t_risk
from risk.models import validate_model


def test_known_quantile_and_fractional_tail_atom():
    result = empirical_risk([0, 1, 2, 3], 0.5)
    assert result["var"] == 1
    assert result["cvar"] == 2.5
    weighted = empirical_risk([0, 10], 0.90, [0.95, 0.05])
    assert weighted["var"] == 0
    assert weighted["cvar"] == pytest.approx(5)
    assert empirical_risk([0, 10], 0.99, [0.95, 0.05])["cvar"] == 10


def test_duplicates_zero_mass_and_constant_negative_losses():
    result = empirical_risk([1, 1, 3, 999], 0.5, [0.25, 0.25, 0.5, 0])
    assert result["var"] == 1
    assert result["cvar"] == 3
    assert empirical_risk([-7, -7], 0.99)["var"] == -7
    assert empirical_risk([0, 1], np.nextafter(1.0, 0))["cvar"] == 1
    assert empirical_risk(range(10), .9)["var"] == 8
    assert empirical_risk(range(10), .9, [.1]*10)["var"] == 8


@pytest.mark.parametrize("losses,alpha,weights", [([], .99, None), ([np.nan], .99, None), ([1], 1, None),
    ([1], 0, None), ([1], np.nan, None), ([1, 2], .99, [0, 0]), ([1, 2], .99, [-1, 2]),
    ([1], .99, [1, 2]), ([1], .99, [np.inf]), ([[1]], .99, None)])
def test_invalid_empirical_inputs(losses, alpha, weights):
    with pytest.raises(ValueError):
        empirical_risk(losses, alpha, weights)


def test_gaussian_and_student_references():
    result = gaussian_risk(10, 2, .99)
    assert result["var"] == pytest.approx(10 + 2*norm.ppf(.99))
    assert result["cvar"] == pytest.approx(10 + 2*norm.pdf(norm.ppf(.99))/.01)
    assert gaussian_risk(-2, 0)["cvar"] == -2
    assert student_t_risk(0, 1, 5)["cvar"] > gaussian_risk(0, 1)["cvar"]
    with pytest.raises(ValueError):
        gaussian_risk(0, -1)
    with pytest.raises(ValueError):
        student_t_risk(0, 1, 2)


def test_linear_portfolio_short_cash_and_stress():
    portfolio = Portfolio(["A", "B"], [1.2, -0.4], 100)
    assert portfolio.losses([[.1, -.2]])[0] == pytest.approx(-20)
    stress = portfolio.stress({"crash": [-.1, .2]})[0]
    assert stress["loss"] == pytest.approx(20)
    assert sum(stress["asset_contributions"].values()) == pytest.approx(stress["loss"])
    mu, sigma = portfolio.loss_moments([0, 0], [[.01, 0], [0, .01]])
    assert mu == 0
    assert sigma == pytest.approx(np.sqrt(160))


@pytest.mark.parametrize("covariance", [[[1, 2], [2, 1]], [[1, .1], [0, 1]], [[np.nan, 0], [0, 1]], [[1]]])
def test_invalid_covariance(covariance):
    with pytest.raises(ValueError):
        validate_model([0, 0], covariance)


def test_singular_covariance_allowed():
    _, covariance = validate_model([0, 0], [[1, 1], [1, 1]])
    assert covariance.shape == (2, 2)


@pytest.mark.parametrize("assets,weights,value", [(["A", "A"], [1, 1], 100), (["A"], [1, 2], 100),
    (["A"], [np.inf], 100), (["A"], [1], -1)])
def test_invalid_portfolio(assets, weights, value):
    with pytest.raises(ValueError):
        Portfolio(assets, weights, value)
