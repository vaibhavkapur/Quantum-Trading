import numpy as np
import pytest

from qmc_options.models import Market, Option, paths_from_normals


@pytest.mark.parametrize("kwargs", [
    {"kind": "digital", "strike": 100, "maturity": 1},
    {"kind": "european_call", "strike": -1, "maturity": 1},
    {"kind": "european_call", "strike": 100, "maturity": 0},
    {"kind": "asian_call", "strike": 100, "maturity": 1, "steps": 0},
    {"kind": "barrier_up_out_call", "strike": 100, "maturity": 1, "steps": 2},
    {"kind": "barrier_up_out_call", "strike": 100, "maturity": 1, "steps": 2, "barrier": 90},
])
def test_invalid_options_are_rejected(kwargs):
    with pytest.raises(ValueError):
        Option(**kwargs)


@pytest.mark.parametrize("kwargs", [{"spot": 0, "rate": 0.0, "vol": 0.2}, {"spot": 100, "rate": 0.0, "vol": 0}])
def test_invalid_market_is_rejected(kwargs):
    with pytest.raises(ValueError):
        Market(**kwargs)


def test_payoffs_on_hand_built_paths():
    paths = np.array([[110.0, 140.0, 120.0],     # breaches 130, ends in the money
                      [105.0, 110.0, 120.0],     # stays below 130
                      [95.0, 90.0, 85.0]])       # ends out of the money
    assert Option("european_call", 100, 1, steps=3).payoff(paths).tolist() == [20.0, 20.0, 0.0]
    assert Option("european_put", 100, 1, steps=3).payoff(paths).tolist() == [0.0, 0.0, 15.0]
    assert Option("asian_call", 100, 1, steps=3).payoff(paths) == pytest.approx([70 / 3, 35 / 3, 0.0])
    assert Option("barrier_up_out_call", 100, 1, steps=3, barrier=130).payoff(paths).tolist() == [0.0, 20.0, 0.0]


def test_zero_shock_path_grows_at_the_drift():
    market = Market(100.0, 0.05, 0.2)
    option = Option("asian_call", 100.0, 1.0, steps=4)
    paths = paths_from_normals(market, option, np.zeros((1, 4)))
    assert paths[0] == pytest.approx(100 * np.exp((0.05 - 0.02) * np.array([0.25, 0.5, 0.75, 1.0])))
