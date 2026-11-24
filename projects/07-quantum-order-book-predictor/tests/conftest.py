from dataclasses import replace
import numpy as np
import pytest
from orderbook.config import Config
from orderbook.data import Book, generate_book


@pytest.fixture
def config():
    return replace(Config(), n_snapshots=240, folds=1, train_samples=18, seeds=[11],
                   qubits=[2], encodings=["angle"], shots=[None], max_iterations=3, latency_samples=5)


@pytest.fixture
def book(config):
    return generate_book(config)


def simple_book(mids, spread=0.02, size=10):
    mids = np.asarray(mids, dtype=float)
    n = len(mids)
    return Book(1000 + 100 * np.arange(n, dtype=np.int64), (mids-spread/2)[:, None],
                (mids+spread/2)[:, None], np.full((n, 1), float(size)), np.full((n, 1), float(size))).validate()
