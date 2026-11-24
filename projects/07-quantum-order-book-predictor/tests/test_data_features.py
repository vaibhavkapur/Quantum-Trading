from dataclasses import replace
import json
import numpy as np
import pytest
from orderbook.config import load_config
from orderbook.data import prepare_data, load_book, generate_book
from orderbook.features import features, labels, FEATURE_NAMES
from orderbook.splits import walk_forward, stratified_train_sample
from orderbook.preprocessing import Reducer
from conftest import simple_book


def test_data_reproducible_immutable_and_verified(tmp_path, config):
    cfg = replace(config, data_path=str(tmp_path / "book.csv"))
    path = prepare_data(cfg)
    original = path.read_bytes()
    assert prepare_data(cfg).read_bytes() == original
    actual, manifest = load_book(path, cfg.levels)
    np.testing.assert_allclose(actual.mid, generate_book(cfg).mid)
    assert manifest["license"].startswith("CC0-1.0")
    with pytest.raises(ValueError, match="immutable"):
        prepare_data(replace(cfg, data_seed=7))
    path.write_bytes(original + b"\n")
    with pytest.raises(ValueError, match="SHA-256"):
        load_book(path, cfg.levels)


@pytest.mark.parametrize("mutation", ["duplicate", "nan", "crossed", "unsorted", "negative_size", "float_timestamp"])
def test_bad_data_rejected(book, mutation):
    if mutation == "duplicate":
        book.timestamp_ms[1] = book.timestamp_ms[0]
    elif mutation == "nan":
        book.bid_price[1, 0] = np.nan
    elif mutation == "crossed":
        book.bid_price[1, 0] = book.ask_price[1, 0]
    elif mutation == "unsorted":
        book.bid_price[:, 1] = book.bid_price[:, 0]
    elif mutation == "negative_size":
        book.ask_size[2, 0] = -1
    else:
        book.timestamp_ms = book.timestamp_ms.astype(float)
    with pytest.raises(ValueError):
        book.validate()


def test_known_labels_and_unavailable_tail():
    y, end = labels(simple_book([100, 101, 101, 100]), 1, 0)
    assert y.tolist() == [2, 1, 0, -1]
    assert end.tolist() == [1, 2, 3, 4]
    assert np.all(labels(simple_book([100, 101]), 3, 0)[0] == -1)


def test_known_feature_values_and_ofi():
    book = simple_book([100, 100])
    book.bid_size[:] = [[10], [15]]
    book.ask_size[:] = [[10], [5]]
    result = features(book)
    np.testing.assert_allclose(result[:, 0], [0, 0.5])
    assert result[1, FEATURE_NAMES.index("ofi_touch")] == pytest.approx(0.5)
    assert result[1, FEATURE_NAMES.index("microprice_offset_bps")] == pytest.approx(0.5)
    assert result[1, FEATURE_NAMES.index("spread_bps")] == pytest.approx(2)


def test_no_future_features_and_exact_rolling_context(book):
    original = features(book)
    for i in [5, 15, 99]:
        np.testing.assert_allclose(features(book.take(slice(i-5, i+1)))[-1], original[i])
    book.bid_price[100:] *= 2
    book.ask_price[100:] *= 2
    book.bid_size[100:] *= 3
    np.testing.assert_array_equal(features(book)[:100], original[:100])


def test_splits_purge_embargo_walkforward_and_train_only_sample(book, config):
    cfg = replace(config, folds=2)
    splits = walk_forward(len(book), cfg)
    y, ends = labels(book, cfg.horizon, cfg.label_threshold_bps)
    for split in splits:
        parts = [split.train, split.calibration, split.validation, split.test]
        for left, right in zip(parts[:-1], parts[1:]):
            assert ends[left].max() < right.min() - cfg.embargo
        assert ends[split.test].max() < len(book)
        sample = stratified_train_sample(split.train, y, 18, 11)
        assert set(sample) <= set(split.train)
        assert len(sample) == len(set(sample)) == 18
        assert set(y[sample]) == set(y[split.train])
        np.testing.assert_array_equal(sample, stratified_train_sample(split.train, y, 18, 11))
    assert not set(splits[0].test) & set(splits[1].test)
    assert len(splits[1].train) > len(splits[0].train)


def test_preprocessing_fits_training_only(book):
    x = features(book)
    reducer = Reducer(2).fit(x[:50])
    mean = reducer.scaler.mean_.copy()
    np.testing.assert_allclose(mean, x[:50].mean(0))
    reducer.transform(x[50:] * 100)
    np.testing.assert_array_equal(reducer.scaler.mean_, mean)
    assert np.max(np.abs(reducer.transform(x))) <= np.pi


@pytest.mark.parametrize("override", [{"horizon": 1}, {"shots": [0]}, {"qubits": [7]}, {"fee_bps": -1},
                                    {"embargo": -1}, {"seeds": [1, 1]}, {"fit_budget_seconds": float("nan")},
                                    {"calibration_fraction": 0.9}, {"encodings": ["wrong"]}])
def test_invalid_config(config, override):
    with pytest.raises(ValueError):
        replace(config, **override).validate()


def test_unknown_config_key(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("unknown: 12\n")
    with pytest.raises(ValueError, match="unknown"):
        load_config(path)
