import numpy as np
import pytest

from hqpo.data.synthetic import random_instance
from hqpo.models import Constraints


def make(n, seed=0, **kw):
    tau = kw.pop("turnover_penalty", 0.0)
    return random_instance(n, Constraints(**kw), seed=seed, turnover_penalty=tau)


# small instances covering every constraint family
TINY = {
    "budget_only": dict(n=5, lots=4, max_weight=0.5),
    "cardinality": dict(n=5, lots=4, max_weight=0.5, cardinality=3),
    "sector_cap": dict(n=5, lots=4, max_weight=0.5, sector_cap=0.5),
    "turnover": dict(n=4, lots=4, max_weight=0.75, cardinality=3, turnover_penalty=0.1),
    "odd_bound": dict(n=3, lots=6, max_weight=0.84),          # ub = 5 -> coefficients [1, 2, 2]
    "mn_plain": dict(n=3, lots=2, max_weight=0.5, mode="market_neutral"),
    "mn_cardinality": dict(n=3, lots=2, max_weight=0.5, mode="market_neutral", cardinality=2),
    "mn_sector": dict(n=4, lots=2, max_weight=0.25, mode="market_neutral", sector_cap=0.25),
}


@pytest.fixture(params=sorted(TINY))
def tiny(request):
    spec = dict(TINY[request.param])
    return make(spec.pop("n"), seed=5, **spec)


@pytest.fixture
def rng():
    return np.random.default_rng(0)
