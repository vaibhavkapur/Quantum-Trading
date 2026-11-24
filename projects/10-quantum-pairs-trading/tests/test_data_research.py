import json

import numpy as np
import pandas as pd
import pytest

from pairs_trading.config import validate_config
from pairs_trading.data import load_data, prepare_data, synthetic_prices, validate_prices
from pairs_trading.research import discover_pairs, fit_hedge, half_life, walk_forward_splits


def test_data_reproducible_and_content_addressed(config):
    pointer = prepare_data(config)
    first = pointer.read_bytes()
    frame, sectors, metadata = load_data(config)
    prepare_data(config)
    assert pointer.read_bytes() == first
    again, _, _ = load_data(config)
    pd.testing.assert_frame_equal(frame, again)
    assert len(sectors) == 6
    assert metadata["prices_sha256"]
    assert frame.index.tz is not None


def test_data_tampering_and_config_drift_rejected(config):
    pointer = prepare_data(config)
    data = json.loads(pointer.read_text())
    from pathlib import Path
    path = Path(config["_root"]) / data["directory"] / "prices.csv"
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="hash mismatch"):
        load_data(config)
    with pytest.raises(ValueError, match="immutable"):
        prepare_data(config)


def test_config_change_requires_new_dataset(config):
    prepare_data(config)
    config["data"]["seed"] += 1
    with pytest.raises(ValueError, match="dataset/config mismatch"):
        load_data(config)


@pytest.mark.parametrize("kind", ["missing", "negative", "duplicate", "unsorted", "timezone"])
def test_invalid_prices_rejected(config, kind):
    frame, sectors = synthetic_prices(config["data"])
    if kind == "missing":
        frame.iloc[3, 0] = np.nan
    elif kind == "negative":
        frame.iloc[3, 0] = -1
    elif kind == "duplicate":
        frame.index = [frame.index[0]] * len(frame)
    elif kind == "unsorted":
        frame = frame.iloc[::-1]
    else:
        frame.index = frame.index.tz_localize(None)
    with pytest.raises(ValueError):
        validate_prices(frame, sectors)


@pytest.mark.parametrize("section,key,value", [("execution", "borrow_rate", -1), ("research", "train", 20),
    ("solver", "max_qubits", 30), ("solver", "seeds", [1, 1]), ("execution", "entry_grid", [5]),
    ("solver", "wall_seconds", float("nan")), ("selection", "capital_budget", 1.5)])
def test_config_rejects_invalid_settings(config, section, key, value):
    config[section][key] = value
    with pytest.raises(ValueError):
        validate_config(config)


def test_known_cointegrating_process_and_ar_half_life():
    rng = np.random.default_rng(87)
    x = 100 + np.cumsum(rng.normal(0, 1, 2000))
    residual = np.zeros(len(x))
    for t in range(1, len(x)):
        residual[t] = 0.8 * residual[t-1] + rng.normal(0, 0.2)
    y = 3 + 1.5 * x + residual
    hedge = fit_hedge(y, x)
    assert hedge.beta == pytest.approx(1.5, abs=0.01)
    assert half_life(residual) == pytest.approx(-np.log(2) / np.log(0.8), rel=0.2)
    assert np.isinf(half_life(np.ones(50)))
    with pytest.raises(ValueError, match="constant"):
        fit_hedge(np.ones(50), np.ones(50))


def test_screening_is_train_only(config):
    prices, sectors = synthetic_prices(config["data"])
    cut = config["research"]["train"]
    pairs, corr, diagnostics = discover_pairs(prices.iloc[:cut], sectors, config["research"], 3)
    changed = prices.copy()
    changed.iloc[cut:] *= np.linspace(1, 4, len(changed) - cut)[:, None]
    other, other_corr, other_diagnostics = discover_pairs(changed.iloc[:cut], sectors, config["research"], 3)
    assert pairs == other and diagnostics == other_diagnostics
    np.testing.assert_array_equal(corr, other_corr)
    assert len(pairs) > 0
    assert all(p.pvalue <= config["research"]["pvalue"] for p in pairs)


def test_walk_forward_boundaries(config):
    folds = walk_forward_splits(config["data"]["periods"], config["research"])
    assert len(folds) == 2
    assert folds[0]["train_end"] == folds[0]["validation_start"]
    assert folds[0]["validation_end"] == folds[0]["test_start"]
    assert folds[0]["test_end"] == folds[1]["test_start"]
    with pytest.raises(ValueError, match="no complete"):
        walk_forward_splits(20, config["research"])
