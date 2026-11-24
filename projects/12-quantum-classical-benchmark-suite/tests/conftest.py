from pathlib import Path

import pytest

from benchmarks.adapters.base import Budget, Context
from experiments.config import DEFAULT_ALGORITHM, DEFAULT_BUDGET, load_config


@pytest.fixture
def config():
    return load_config(Path(__file__).resolve().parents[1] / "configs" / "ci.yaml")[0]


@pytest.fixture
def context():
    return Context(Budget(**DEFAULT_BUDGET), 42, DEFAULT_ALGORITHM.copy())
