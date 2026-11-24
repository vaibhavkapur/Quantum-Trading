import numpy as np
import pytest

from qmc_options import analytic, classical_mc
from qmc_options.models import Market, Option


def test_black_scholes_textbook_value(market):
    # S=K=100, r=5%, sigma=20%, T=1: the standard reference value.
    assert analytic.black_scholes_price(market, 100.0, 1.0, call=True) == pytest.approx(10.450584, abs=1e-6)
    assert analytic.black_scholes_price(market, 100.0, 1.0, call=False) == pytest.approx(5.573526, abs=1e-6)


@pytest.mark.parametrize("strike,maturity", [(80.0, 0.25), (100.0, 1.0), (135.0, 2.0)])
def test_put_call_parity(market, strike, maturity):
    c = analytic.black_scholes_price(market, strike, maturity, call=True)
    p = analytic.black_scholes_price(market, strike, maturity, call=False)
    assert c - p == pytest.approx(market.spot - strike * np.exp(-market.rate * maturity), abs=1e-10)


@pytest.mark.parametrize("is_call", [True, False])
def test_greeks_match_numerical_derivatives(market, is_call):
    greeks = analytic.black_scholes_greeks(market, 105.0, 0.75, call=is_call)

    def price(**bump):
        maturity = bump.pop("maturity", 0.75)
        return analytic.black_scholes_price(market.bumped(**bump), 105.0, maturity, call=is_call)

    h = 1e-4
    assert greeks["delta"] == pytest.approx((price(spot=100 + h) - price(spot=100 - h)) / (2 * h), abs=1e-6)
    assert greeks["gamma"] == pytest.approx((price(spot=100 + h) - 2 * price() + price(spot=100 - h)) / h**2, abs=1e-4)
    assert greeks["vega"] == pytest.approx((price(vol=0.2 + h) - price(vol=0.2 - h)) / (2 * h), abs=1e-5)
    assert greeks["rho"] == pytest.approx((price(rate=0.05 + h) - price(rate=0.05 - h)) / (2 * h), abs=1e-5)
    assert greeks["theta"] == pytest.approx(-(price(maturity=0.75 + h) - price(maturity=0.75 - h)) / (2 * h), abs=1e-5)


def test_geometric_asian_with_one_date_is_black_scholes(market):
    assert analytic.geometric_asian_call(market, 100.0, 1.0, steps=1) == pytest.approx(
        analytic.black_scholes_price(market, 100.0, 1.0), abs=1e-10)


def test_geometric_asian_matches_simulation(market):
    steps = 4
    z = np.random.default_rng(0).standard_normal((400_000, steps))
    option = Option("asian_call", 100.0, 1.0, steps=steps)
    from qmc_options.models import paths_from_normals
    geo = np.exp(np.log(paths_from_normals(market, option, z)).mean(axis=1))
    samples = np.exp(-0.05) * np.maximum(geo - 100.0, 0.0)
    stderr = samples.std() / np.sqrt(samples.size)
    assert abs(samples.mean() - analytic.geometric_asian_call(market, 100.0, 1.0, steps)) < 4 * stderr


def test_barrier_far_away_is_vanilla(market):
    assert analytic.barrier_up_out_call(market, 100.0, 1.0, barrier=1e6) == pytest.approx(
        analytic.black_scholes_price(market, 100.0, 1.0), abs=1e-8)


def test_barrier_already_breached_is_worthless(market):
    assert analytic.barrier_up_out_call(market, 90.0, 1.0, barrier=100.0) == 0.0


def test_barrier_formula_agrees_with_fine_monitoring_simulation(market):
    # 250 monitoring dates with the continuity correction: the two should agree to about 1%.
    option = Option("barrier_up_out_call", 100.0, 1.0, steps=250, barrier=130.0)
    simulated = classical_mc.mc_price(market, option, 200_000, "antithetic", seed=3)
    corrected = analytic.barrier_up_out_call(market, 100.0, 1.0, 130.0, monitoring_steps=250)
    assert corrected == pytest.approx(simulated.price, rel=0.02)


def test_reference_price_only_for_european(market, call):
    assert analytic.reference_price(market, call) == pytest.approx(10.450584, abs=1e-6)
    assert analytic.reference_price(market, Option("asian_call", 100.0, 1.0, steps=3)) is None
