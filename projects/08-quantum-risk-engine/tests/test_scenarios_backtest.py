import numpy as np
import pytest
from scipy.stats import norm

from risk import Portfolio, empirical_risk
from risk.backtest import coverage_statistics, rolling_backtest
from scenarios import correlated_returns, importance_sample


def test_scenarios_deterministic_correlated_and_antithetic():
    covariance = [[1, .6], [.6, 2]]
    a = correlated_returns([0, 0], covariance, 50000, 7)
    assert np.array_equal(a, correlated_returns([0, 0], covariance, 50000, 7))
    assert np.allclose(np.cov(a, rowvar=False), covariance, atol=.035)
    paired = correlated_returns([.1, -.2], covariance, 100, 7, antithetic=True)
    assert np.allclose(paired[:50]+paired[50:], [.2, -.4])
    assert correlated_returns([0], [[0]], 1, 7).tolist() == [[0]]


def test_shared_student_t_scale_and_covariance():
    returns = correlated_returns([0, 0], [[1, .5], [.5, 1]], 100000, 7, "student-t", df=5)
    assert np.allclose(np.cov(returns, rowvar=False), [[1, .5], [.5, 1]], atol=.06)
    assert np.mean(returns[:, 0] > 4) > norm.sf(4)*10
    with pytest.raises(ValueError):
        correlated_returns([0], [[1]], 10, 7, "student-t", df=2)
    with pytest.raises(ValueError):
        correlated_returns([0], [[1]], 0, 7)


def test_importance_weights_recover_original_tail_and_zero_risk():
    portfolio = Portfolio(["A"], [1], 1)
    returns, weights, diagnostics = importance_sample(portfolio, [0], [[1]], 100000, 7, 2)
    probability = np.sum(weights[portfolio.losses(returns) > norm.ppf(.99)])
    assert probability == pytest.approx(.01, abs=.0007)
    assert diagnostics["weight_mean"] == pytest.approx(1, abs=.08)
    assert 0 < diagnostics["effective_sample_size"] < len(weights)
    assert empirical_risk(portfolio.losses(returns), .99, weights)["var"] == pytest.approx(norm.ppf(.99), abs=.04)
    zero_returns, zero_weights, _ = importance_sample(portfolio, [0], [[0]], 10, 7)
    assert np.all(zero_returns == 0)
    assert np.allclose(zero_weights, .1)


def test_no_lookahead_in_rolling_forecasts():
    losses = np.arange(20, dtype=float)
    original = rolling_backtest(losses, 10, .9)
    changed = losses.copy()
    changed[10:] = 999
    altered = rolling_backtest(changed, 10, .9)
    assert original["forecasts"][0] == altered["forecasts"][0] == 8
    assert original["losses"][0] == 10
    assert original["coverage"]["observations"] == 10


@pytest.mark.parametrize("losses,forecasts,expected", [([0]*20, [0]*20, 0), ([1]*20, [0]*20, 20), ([0, 1, 2], [0, 0, 1], 2)])
def test_backtest_boundary_exception_counts(losses, forecasts, expected):
    result = coverage_statistics(losses, forecasts, .99)
    assert result["exceptions"] == expected
    assert np.isfinite(result["kupiec_lr"])
    assert 0 <= result["exact_binomial_pvalue"] <= 1
    assert result["exception_rate_ci95"][0] <= result["exception_rate"] <= result["exception_rate_ci95"][1]


def test_invalid_backtest_window_and_alignment():
    with pytest.raises(ValueError):
        rolling_backtest([1, 2], 2)
    with pytest.raises(ValueError):
        coverage_statistics([1, 2], [1], .99)
