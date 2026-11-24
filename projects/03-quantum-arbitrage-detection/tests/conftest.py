import copy

import numpy as np
import pytest

from qarb.config import DEFAULTS, validate
from qarb.core.book import OrderBook


def make_config(tmp_path=None, **over):
    cfg = copy.deepcopy(DEFAULTS)
    cfg["name"] = "test"
    cfg["seeds"] = [5]
    cfg["data"].update(
        currencies=["USD", "BTC", "ETH", "SOL"],
        venues=[
            {"name": "alpha", "fee_bps": 5.0, "feed_delay_ms": [5, 30], "symbol_style": "dash"},
            {"name": "beta", "fee_bps": 7.5, "feed_delay_ms": [10, 50], "symbol_style": "concat_xbt"},
        ],
        n_ticks=20,
        planted={"n_profitable": 3, "lengths": [2, 3], "edge_bps": [15.0, 35.0], "duration_ticks": [3, 6]},
        decoys={"n_thin": 1, "thin_notional": 25.0, "n_stale": 1, "stale_ticks": 6, "stale_jump_bps": 70.0},
    )
    cfg["graph"]["max_cycle_len"] = 3
    cfg["qaoa"].update(maxiter=14, shots_per_iter=48, final_shots=128)
    cfg["sa"].update(reads=12, sweeps=30)
    cfg["random_search"] = {"samples": 300}
    if tmp_path is not None:
        cfg["data"]["dir"] = str(tmp_path / "data")
        cfg["runs_dir"] = str(tmp_path / "runs")
    for k, v in over.items():
        cfg[k].update(v) if isinstance(v, dict) else cfg.__setitem__(k, v)
    validate(cfg)
    return cfg


@pytest.fixture
def cfg(tmp_path):
    return make_config(tmp_path)


def book(venue, base, quote, bid, ask, size=10.0, fee=0.0, ts=0, levels=1, step=0.0):
    bids = tuple((bid * (1 - i * step), size) for i in range(levels))
    asks = tuple((ask * (1 + i * step), size) for i in range(levels))
    return OrderBook(venue, base, quote, bids, asks, ts, fee)


@pytest.fixture
def triangle_books():
    """USD -> BTC -> ETH -> USD earns exactly 2% before fees at the top of book.

    Buy 1 BTC for 100 USD, sell it for 20.4 ETH (bid), sell ETH at 5 USD -> 102 USD.
    """
    return [
        book("v1", "BTC", "USD", 99.0, 100.0, size=1.0),
        book("v1", "ETH", "BTC", 1 / 20.4 * 0.99, 1 / 20.4, size=1000.0),
        book("v2", "ETH", "USD", 5.0, 5.05, size=1000.0),
    ]


def random_weights(rng: np.random.Generator, n: int, density: float = 1.0, neg_edges: int = 2) -> np.ndarray:
    W = rng.uniform(0.0003, 0.003, size=(n, n))
    for _ in range(neg_edges):
        i, j = rng.choice(n, size=2, replace=False)
        W[i, j] = -rng.uniform(0.001, 0.006)
    if density < 1.0:
        W[rng.random((n, n)) > density] = np.inf
    np.fill_diagonal(W, np.inf)
    return W
