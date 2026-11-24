import numpy as np
import pytest
from statsmodels.tsa.stattools import adfuller

from project.research import stats
from project.research.hedge import fit_hedge, validation_gate

GATES = {"val_adf_pvalue": 0.10, "max_half_life": 60, "max_asset_weight": 0.9}


def ou(n, half_life, sigma, rng):
    phi, x = np.exp(-np.log(2) / half_life), np.zeros(n)
    for t in range(1, n):
        x[t] = phi * x[t - 1] + rng.normal(0, sigma)
    return x


def test_adf_tstat_matches_statsmodels():
    rng = np.random.default_rng(0)
    for series in (np.cumsum(rng.normal(size=400)), ou(400, 8, 1.0, rng)):
        ref = adfuller(series, maxlag=1, autolag=None, regression="c", result_object=False)[0]
        assert stats.adf_tstat(series) == pytest.approx(ref, rel=1e-8)


def test_adf_handles_degenerate_input():
    assert stats.adf_tstat(np.ones(100)) == 0.0
    assert stats.adf_tstat(np.arange(4.0)) == 0.0
    assert stats.adf_pvalue(np.ones(100)) == (0.0, 1.0)


def test_half_life_recovers_the_generating_value():
    rng = np.random.default_rng(1)
    assert stats.half_life(ou(20000, 10.0, 1.0, rng)) == pytest.approx(10.0, rel=0.15)
    assert stats.half_life(np.arange(200.0)) == float("inf")
    assert stats.half_life(np.array([1.0, 2.0])) == float("inf")


def test_pair_stats_separate_cointegrated_from_independent():
    rng = np.random.default_rng(2)
    trend = np.cumsum(rng.normal(0, 0.01, 600))
    logp = np.column_stack([trend + ou(600, 8, 0.004, rng), 0.8 * trend + ou(600, 8, 0.004, rng), np.cumsum(rng.normal(0, 0.01, 600))])
    ps = stats.pair_stats(logp)
    assert ps.strength[0, 1] > 0.8 > ps.strength[0, 2] + 0.3
    assert ps.stability[0, 1] > ps.stability[0, 2]
    assert ps.resid_var[0, 1] < ps.resid_var[0, 2]
    assert ps.half_life[0, 1] < 20
    np.testing.assert_allclose(ps.strength, ps.strength.T)
    assert np.all(np.diag(ps.strength) == 0)


def test_market_betas_average_to_one():
    rng = np.random.default_rng(3)
    logp = np.cumsum(rng.normal(0, 0.01, (300, 5)) + rng.normal(0, 0.01, (300, 1)), axis=0)
    assert stats.market_betas(logp).mean() == pytest.approx(1.0)


@pytest.mark.parametrize("method", ["johansen", "ols"])
def test_hedge_recovers_known_cointegrating_vector(method):
    rng = np.random.default_rng(4)
    trend = np.cumsum(rng.normal(0, 0.01, 1500))
    logp = np.column_stack([trend + ou(1500, 6, 0.003, rng), 0.5 * trend + ou(1500, 6, 0.003, rng)])
    fit = fit_hedge(logp, method)
    assert np.abs(fit.weights).sum() == pytest.approx(1.0)
    assert fit.weights[0] / fit.weights[1] == pytest.approx(-0.5, rel=0.1)  # x0 - 2*x1 is stationary
    assert fit.half_life < 15 and fit.std > 0
    if method == "johansen":
        assert fit.trace_stat > fit.trace_crit95


def test_hedge_rejects_single_asset():
    with pytest.raises(ValueError):
        fit_hedge(np.zeros((100, 1)))


def test_validation_gate_passes_stationary_and_rejects_random_walk():
    rng = np.random.default_rng(5)
    trend = np.cumsum(rng.normal(0, 0.01, 1200))
    good = np.column_stack([trend + ou(1200, 6, 0.003, rng), trend + ou(1200, 6, 0.003, rng)])
    fit = fit_hedge(good[:800])
    ok, info = validation_gate(fit, good[800:], GATES)
    assert ok and info["reasons"] == []
    broken = good[800:].copy()
    broken[:, 0] += np.cumsum(rng.normal(0, 0.01, 400))  # relationship breaks down out of sample
    ok, info = validation_gate(fit, broken, GATES)
    assert not ok and "validation_adf" in info["reasons"]
