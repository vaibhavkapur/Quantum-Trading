from dataclasses import replace
import numpy as np
import pytest
from orderbook.evaluation import backtest, select_threshold
from orderbook.features import features
from orderbook.preprocessing import Reducer
from orderbook.models import make_model
from orderbook.calibration import TemperatureCalibrator
from orderbook.serving import Predictor, StreamingPredictor, snapshot, profile_stream
from conftest import simple_book


def test_flat_market_pays_spread_fees_and_slippage(config):
    book = simple_book([100] * 20, spread=0.02)
    p = np.tile([0.01, 0.01, 0.98], (10, 1))
    cfg = replace(config, fee_bps=1, slippage_bps=0.5)
    result, trades = backtest(book, np.arange(10), p, 0.5, cfg)
    assert result["trades"] > 0
    assert result["mean_trade_bps"] == pytest.approx(-5, abs=1e-6)
    assert trades[0]["entry_index"] == 1 and trades[0]["exit_index"] == 3
    assert all(a["exit_index"] < b["entry_index"] for a, b in zip(trades[:-1], trades[1:]))
    threshold, trials = select_threshold(book, np.arange(10), p, cfg)
    assert threshold == 1.0
    assert backtest(book, np.arange(10), p, threshold, cfg)[0]["trades"] == 0


@pytest.mark.parametrize("direction", [-1, 1])
def test_delayed_fills_have_known_pnl(config, direction):
    book = simple_book([100, 100, 101, 102, 103, 104], spread=0.02)
    cfg = replace(config, fee_bps=0, slippage_bps=0)
    p = np.array([[0.99, 0.005, 0.005] if direction == -1 else [0.005, 0.005, 0.99]])
    result, trades = backtest(book, [0], p, 0.5, cfg)
    assert result["net_bps"] == pytest.approx(direction * 200 - 2)
    # The jump before the delayed entry cannot be counted as profit.
    book = simple_book([99, 100, 100, 100, 100])
    result, _ = backtest(book, [0], p, 0.5, cfg)
    assert result["net_bps"] == pytest.approx(-2)


def test_stream_equals_batch_and_invalid_input_abstains(book, tmp_path):
    x = features(book)
    reducer = Reducer(2).fit(x[5:50])
    model = make_model("logistic", seed=11, iterations=100, qubits=2).fit(reducer.transform(x[5:50]), np.arange(45) % 3)
    predictor = Predictor(reducer, model, TemperatureCalibrator(), 0.5, 3)
    path = tmp_path / "model.pkl"
    predictor.save(path)
    loaded = Predictor.load(path)
    stream = StreamingPredictor(loaded)
    for i in range(15):
        result = stream.update(snapshot(book, i))
        if i < 5:
            assert result["status"] == "warming_up"
        else:
            np.testing.assert_allclose(result["probabilities"], predictor.predict_features(x[i:i+1])[0])
    assert stream.update_safe(snapshot(book, 14))["direction"] == 0
    assert stream.update_safe({})["status"] == "error"
    np.testing.assert_allclose(stream.update(snapshot(book, 15))["probabilities"], predictor.predict_features(x[15:16])[0])
    latency, timings = profile_stream(predictor, book, np.arange(20, 30), 8)
    assert len(timings) == latency["samples"] == 8
    assert latency["p99_ms"] >= latency["p50_ms"] > 0


def test_future_depth_and_invalid_probability_fail(config):
    book = simple_book([100] * 10)
    book.ask_size[1, 0] = 0.1
    with pytest.raises(ValueError, match="depth"):
        backtest(book, [0], [[0, 0, 1]], 0.5, config)
    with pytest.raises(ValueError, match="normalized"):
        backtest(book, [0], [[0, 0, 2]], 0.5, config)
    with pytest.raises(ValueError, match="horizon"):
        backtest(book, [9], [[0, 0, 1]], 0.5, config)
