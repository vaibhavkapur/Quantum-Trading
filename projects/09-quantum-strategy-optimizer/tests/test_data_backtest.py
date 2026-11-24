import json

import numpy as np
import pandas as pd
import pytest

from strategy_optimizer.backtest import backtest, ensemble, walk_forward
from strategy_optimizer.config import file_hash, validate
from strategy_optimizer.data import load_data, prepare_data, synthetic_data
from strategy_optimizer.strategies import Strategy, cached_positions, library, positions


def test_data_is_deterministic_and_immutable(config):
    path = prepare_data(config["data"])
    before = path.read_bytes()
    assert prepare_data(config["data"]) == path
    assert path.read_bytes() == before
    prices, manifest = load_data(config["data"])
    np.testing.assert_allclose(prices.to_numpy(), synthetic_data(config["data"]).to_numpy(), atol=1e-11)
    assert manifest["sha256"] == file_hash(path)
    assert str(prices.index.tz) == "UTC"
    config["data"]["seed"] += 1
    with pytest.raises(ValueError, match="parameters"):
        prepare_data(config["data"])
    assert path.read_bytes() == before


def test_data_tampering_is_detected(config):
    path = prepare_data(config["data"])
    path.write_text(path.read_text() + "\n")
    with pytest.raises(ValueError, match="SHA-256"):
        load_data(config["data"])


def test_licensed_csv_import_normalizes_and_records_provenance(config, tmp_path):
    source = tmp_path / "vendor.csv"
    prices = synthetic_data(config["data"])
    prices.to_csv(source)
    with pytest.raises(ValueError, match="requires"):
        prepare_data(config["data"], source_csv=source)
    output = prepare_data(config["data"], source, "Test provider", "Test fixture; redistribution allowed")
    imported, manifest = load_data(config["data"])
    np.testing.assert_allclose(imported, prices, atol=1e-11)
    assert manifest["source_sha256"] == file_hash(source)
    assert manifest["kind"] == "imported"
    assert "redistribution allowed" in manifest["license"]
    assert prepare_data(config["data"]) == output
    source.write_text(source.read_text() + "\n")
    with pytest.raises(ValueError, match="immutable"):
        prepare_data(config["data"], source, "Test provider", "Test fixture; redistribution allowed")


def test_nat_dates_are_rejected_on_import(config, tmp_path):
    source = tmp_path / "missing-date.csv"
    prices = synthetic_data(config["data"]).reset_index()
    prices.loc[5, "date"] = pd.NaT
    prices.to_csv(source, index=False)
    with pytest.raises(ValueError, match="positive, finite, chronological"):
        prepare_data(config["data"], source, "test", "test")


@pytest.mark.parametrize("damage", ["negative", "duplicate", "unsorted", "nan"])
def test_invalid_prices_fail_even_with_matching_checksum(config, damage):
    path = prepare_data(config["data"])
    frame = pd.read_csv(path)
    if damage == "negative":
        frame.iloc[5, 1] = -10
    elif damage == "duplicate":
        frame.iloc[5, 0] = frame.iloc[4, 0]
    elif damage == "unsorted":
        frame = frame.iloc[::-1]
    else:
        frame.iloc[5, 1] = np.nan
    frame.to_csv(path, index=False)
    metadata = path.with_suffix(".manifest.json")
    manifest = json.loads(metadata.read_text())
    manifest["sha256"] = file_hash(path)
    metadata.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="positive, finite, chronological"):
        load_data(config["data"])


def test_signal_lag_and_future_invariance(config):
    prices = synthetic_data(config["data"])
    strategies = library(config["data"]["assets"], [3, 10])
    base = positions(prices, strategies)
    assert np.all(base[:5, 0] == 0)  # momentum(3) first becomes available at index 5.
    assert base[5, 0] == np.sign(prices.iloc[3, 0] / prices.iloc[0, 0] - 1)
    modified = prices.copy()
    modified.iloc[70:] *= np.linspace(1, 2, len(prices) - 70)[:, None]
    np.testing.assert_array_equal(positions(modified, strategies)[:72], base[:72])
    np.testing.assert_array_equal(positions(prices.iloc[:70], strategies), base[:70])


def test_known_trade_costs_and_fixed_sleeves():
    prices = pd.DataFrame({"A": [100.0, 110.0, 99.0, 108.9]})
    strategies = [Strategy("A", "momentum", 2), Strategy("A", "mean_reversion", 2)]
    held = np.array([[0, 0], [1, -1], [1, -1], [-1, 1]], dtype=float)
    result = backtest(prices, held, strategies, 1, 4, 10, 2)
    np.testing.assert_allclose(result.gross[:, 0], [0.05, -0.05, -0.05])
    np.testing.assert_allclose(result.turnover[:, 0], [0.5, 0, 1.5])
    np.testing.assert_allclose(result.costs[:, 0], [0.0005, 0, 0.0015])
    # Opposing sleeves cancel gross exposure, but each pays its own trading costs.
    portfolio = ensemble(result, [1, 1])
    np.testing.assert_allclose(portfolio["gross"], 0)
    np.testing.assert_allclose(portfolio["net"], [-0.001, 0, -0.003])
    np.testing.assert_allclose(ensemble(result, [0, 0])["net"], 0)


def test_windows_start_and_end_flat():
    prices = pd.DataFrame({"A": [100.0] * 8})
    strategies = [Strategy("A", "momentum", 2)]
    result = backtest(prices, np.ones((8, 1)), strategies, 3, 6, 20, 1)
    np.testing.assert_allclose(result.turnover[:, 0], [1, 0, 1])
    np.testing.assert_allclose(result.net[:, 0], [-0.002, 0, -0.002])


def test_signal_cache_is_verified(config, tmp_path):
    prices = synthetic_data(config["data"])
    strategies = library(prices.columns, [3])
    values, hit = cached_positions(prices, strategies, "fixture", tmp_path / "cache")
    assert not hit
    repeated, hit = cached_positions(prices, strategies, "fixture", tmp_path / "cache")
    assert hit
    np.testing.assert_array_equal(values, repeated)
    path = next((tmp_path / "cache").glob("*.npy"))
    path.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="checksum"):
        cached_positions(prices, strategies, "fixture", tmp_path / "cache")


def test_walk_forward_is_chronological_and_tests_do_not_overlap():
    folds = walk_forward(504, 252, 63, 63)
    assert len(folds) == 3
    for fold in folds:
        assert fold["train"][1] == fold["validation"][0]
        assert fold["validation"][1] == fold["test"][0]
    assert folds[0]["test"][1] == folds[1]["test"][0]
    with pytest.raises(ValueError, match="No complete"):
        walk_forward(20, 10, 10, 10)


@pytest.mark.parametrize("section,key,value", [
    ("data", "seed", -1), ("data", "assets", ["A", "A"]),
    ("strategies", "windows", [True]), ("strategies", "windows", [3, 3]),
    ("backtest", "evaluation_cost_bps", -2), ("backtest", "train_cost_bps", float("nan")),
    ("optimization", "risk_aversion", [-1]), ("optimization", "evaluations", 2),
    ("optimization", "max_selected", 10), ("optimization", "max_qubits", 40),
    ("optimization", "seeds", []), ("data", "periods", 10),
])
def test_invalid_config(config, section, key, value):
    config[section][key] = value
    with pytest.raises(ValueError):
        validate(config)

