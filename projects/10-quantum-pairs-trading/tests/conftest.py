from pathlib import Path

import numpy as np
import pytest

from pairs_trading.config import load_config
from pairs_trading.research import Pair
from pairs_trading.selection import Problem


@pytest.fixture
def config(tmp_path):
    cfg = load_config(Path(__file__).parents[1] / "configs/ci.yaml")
    cfg["_root"] = str(tmp_path)
    return cfg


@pytest.fixture
def tiny_problem():
    pairs = [Pair("A", "B", ("one",), 0, 1, 0, 1, 4, 0.01, 5),
             Pair("A", "C", ("two",), 0, 1, 0, 1, 4, 0.01, 4),
             Pair("B", "D", ("three",), 0, 1, 0, 1, 4, 0.01, 4)]
    return Problem(pairs, np.array([-5.0, -4.0, -4.0]), np.zeros((3, 3)),
                   [("asset:A", np.array([1, 1, 0]), 1), ("asset:B", np.array([1, 0, 1]), 1),
                    ("capital", np.array([1, 1, 1]), 2)])
