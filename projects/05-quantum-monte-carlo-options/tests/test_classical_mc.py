import numpy as np
import pytest

from qmc_options import analytic, classical_mc as mc
from qmc_options.models import Market, Option


@pytest.mark.parametrize("method", mc.METHODS)
@pytest.mark.parametrize("kind", ["european_call", "european_put"])
def test_european_prices_match_black_scholes(market, method, kind):
    option = Option(kind, 100.0, 1.0)
    est = mc.mc_price(market, option, 100_000, method, seed=42)
    exact = analytic.reference_price(market, option)
    assert abs(est.price - exact) < 4 * est.stderr
    assert est.ci_low < est.price < est.ci_high
    assert est.n_paths <= 100_000


@pytest.mark.parametrize("method", mc.METHODS)
def test_fixed_seed_is_deterministic(market, call, method):
    a = mc.mc_price(market, call, 10_000, method, seed=7)
    b = mc.mc_price(market, call, 10_000, method, seed=7)
    c = mc.mc_price(market, call, 10_000, method, seed=8)
    assert a.price == b.price and a.stderr == b.stderr
    assert a.price != c.price


def test_plain_estimator_is_the_mean_of_seeded_payoffs(market, call):
    # Pins the path generator: a change here means stored results are no longer comparable.
    z = np.random.default_rng(7).standard_normal((10_000, 1))
    expected = float(mc.discounted_payoffs(market, call, z).mean())
    assert mc.mc_price(market, call, 10_000, "plain", seed=7).price == pytest.approx(expected, rel=1e-12)


def test_variance_reduction_reduces_standard_error(market, call):
    plain = mc.mc_price(market, call, 50_000, "plain", seed=1)
    for method in ("antithetic", "control_variate", "qmc"):
        assert mc.mc_price(market, call, 50_000, method, seed=1).stderr < plain.stderr


def test_asian_control_variate_agrees_with_plain_and_is_much_tighter(market):
    option = Option("asian_call", 100.0, 1.0, steps=6)
    plain = mc.mc_price(market, option, 200_000, "plain", seed=3)
    cv = mc.mc_price(market, option, 200_000, "control_variate", seed=4)
    assert abs(plain.price - cv.price) < 4 * np.hypot(plain.stderr, cv.stderr)
    assert cv.stderr < plain.stderr / 5
    # Arithmetic mean >= geometric mean, so the price is bounded below by the geometric closed form.
    assert cv.price > analytic.geometric_asian_call(market, 100.0, 1.0, 6)


def test_barrier_option_is_cheaper_than_vanilla(market):
    option = Option("barrier_up_out_call", 100.0, 1.0, steps=5, barrier=125.0)
    for method in mc.METHODS:
        est = mc.mc_price(market, option, 50_000, method, seed=5)
        assert 0 < est.price < analytic.black_scholes_price(market, 100.0, 1.0)


def test_deep_out_of_the_money_prices_to_zero(market):
    est = mc.mc_price(market, Option("european_call", 1e6, 1.0), 1000, "control_variate", seed=0)
    assert est.price == 0.0 and est.stderr == 0.0


def test_bad_inputs(market, call):
    with pytest.raises(ValueError):
        mc.mc_price(market, call, 1000, "importance_sampling")
    with pytest.raises(ValueError):
        mc.mc_price(market, call, 2)
    with pytest.raises(ValueError):
        mc.price_from_paths(market, Option("asian_call", 100.0, 1.0, steps=3), np.ones((10, 2)))


def test_price_from_paths_equals_direct_average(market, call):
    paths = np.array([[90.0], [110.0], [130.0], [100.0]])
    est = mc.price_from_paths(market, call, paths)
    assert est.price == pytest.approx(np.exp(-0.05) * 10.0)
    assert est.n_paths == 4


def test_finite_difference_greeks_match_closed_form(market, call):
    exact = analytic.black_scholes_greeks(market, 100.0, 1.0)
    greeks = mc.mc_greeks(market, call, 200_000, seed=11)
    for name in ("delta", "gamma", "vega"):
        assert abs(greeks[name].price - exact[name]) < 4 * greeks[name].stderr + 0.01 * abs(exact[name])
