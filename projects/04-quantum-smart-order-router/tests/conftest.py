import copy
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qsor.config import DEFAULTS, VenueCfg  # noqa: E402
from qsor.core import ASK, BID  # noqa: E402
from qsor.exchange_sim.data import Dataset  # noqa: E402
from qsor.optimizer.instances import random_instance  # noqa: E402


def flat_dataset(T=400, V=2, L=3, mid=10000, size=5, dt_ms=10, flow=0):
    """Constant books: best bid mid-1, best ask mid+1, `size` lots per level."""
    px = np.zeros((2, T, V, L), dtype=np.int32)
    lv = np.arange(L)
    px[BID] = mid - 1 - lv
    px[ASK] = mid + 1 + lv
    arrays = {"mid": np.full(T, mid, dtype=np.int32), "px": px,
              "sz": np.full((2, T, V, L), size, dtype=np.int32),
              "status": np.zeros((T, V), dtype=np.uint8),
              "flow": np.full((T, V, 2), flow, dtype=np.int32)}
    return Dataset.from_arrays(arrays, dt_ms, {"train": (0, T // 2), "validation": (T // 2, T // 2), "test": (T // 2, T)})


@pytest.fixture
def ds():
    return flat_dataset()


def venue_dicts(n=2, **over):
    base = dict(taker_fee=0.002, maker_fee=-0.001, order_fee=0.0, latency_ms=1.0, latency_jitter_ms=0.0,
                md_latency_ms=0.0, flow_rate=10.0)
    base.update(over)
    return [dict(base, name=f"V{i}") for i in range(n)]


def make_cfg(n=2, **venue_over):
    cfg = copy.deepcopy(DEFAULTS)
    cfg["name"] = "t"
    cfg["venues"] = venue_dicts(n, **venue_over)
    cfg["orders"].update(rate_per_s=20.0, sizes_lots=[2, 4], horizon_ms=200, warmup_ms=100)
    cfg["optimizer"].update(refresh_ms=200, publish_delay_ms=10, buckets_lots=[2, 4])
    return cfg


def random_problem(rng, V=3, max_units=3, target=None, cross=True):
    return random_instance(rng, V, max_units, target=target, cross=cross)[0]

