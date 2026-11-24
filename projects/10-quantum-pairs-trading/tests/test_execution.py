import numpy as np
import pandas as pd
import pytest

from pairs_trading.data import synthetic_prices
from pairs_trading.execution import backtest
from pairs_trading.research import Hedge, Pair, discover_pairs


def controlled_prices():
    dates = pd.bdate_range("2024-01-01", periods=48, tz="UTC")
    # Entry signal at 40, execution at 41. Convergence observed at 42, exit at 43.
    residual = np.zeros(48)
    residual[40:44] = [2, 3, 0.1, 0]
    return pd.DataFrame({"A": 100 + residual, "B": np.full(48, 100.0)}, index=dates)


def controlled_backtest(config, monkeypatch, **overrides):
    import pairs_trading.execution as execution
    monkeypatch.setattr(execution, "fit_hedge", lambda *args: Hedge(0, 1, 0, 1))
    settings = {**config["execution"], **overrides}
    pair = Pair("A", "B", ("one",), 0, 1, 0, 1, 4, 0.01, 2)
    prices = controlled_prices()
    return backtest(prices, [pair], 40, 48, settings, 2, 1.5), prices


def test_previous_close_fill_share_pnl_and_cost_reconciliation(config, monkeypatch):
    result, prices = controlled_backtest(config, monkeypatch)
    assert len(result.trades) == 1
    trade = result.trades[0]
    assert trade["entry_time"] == prices.index[41].isoformat()
    assert trade["signal_time"] == prices.index[40].isoformat()
    assert trade["exit_time"] == prices.index[43].isoformat()
    assert trade["exit_reason"] == "mean_reversion"
    q = 50000 / 203
    assert trade["left_quantity"] == pytest.approx(-q)
    assert trade["gross_pnl"] == pytest.approx(q * 3)
    assert trade["net_pnl"] == pytest.approx(trade["gross_pnl"] - trade["transaction_cost"] - trade["financing_cost"])
    assert result.metrics["net_pnl"] == pytest.approx(sum(t["net_pnl"] for t in result.trades))
    assert result.daily["mark_pnl"].sum() == pytest.approx(trade["gross_pnl"])
    assert result.daily["transaction_cost"].sum() == pytest.approx(trade["transaction_cost"])
    assert result.daily["financing_cost"].sum() == pytest.approx(trade["financing_cost"])
    assert result.daily.iloc[-1]["open_pairs"] == 0


def test_carry_accrues_over_actual_weekend_days(config, monkeypatch):
    result, prices = controlled_backtest(config, monkeypatch, financing_rate=0, borrow_rate=0.365)
    trade = result.trades[0]
    q = abs(trade["left_quantity"])
    expected = sum(q * prices.iloc[t-1]["A"] * 0.365 * (prices.index[t] - prices.index[t-1]).days / 365 for t in (42, 43))
    assert trade["financing_cost"] == pytest.approx(expected)


def test_holding_limit_and_no_same_close_reentry(config, monkeypatch):
    result, prices = controlled_backtest(config, monkeypatch, max_holding=1)
    assert result.trades[0]["holding_bars"] == 1
    assert result.trades[0]["exit_reason"] == "holding_limit"
    assert result.trades[0]["exit_time"] == prices.index[42].isoformat()


def test_stop_uses_prior_close(config, monkeypatch):
    result, prices = controlled_backtest(config, monkeypatch, stop_z=2.5)
    assert result.trades[0]["exit_reason"] == "stop_out"
    assert result.trades[0]["exit_time"] == prices.index[42].isoformat()


def test_fold_end_liquidation_is_costed(config, monkeypatch):
    import pairs_trading.execution as execution
    monkeypatch.setattr(execution, "fit_hedge", lambda *args: Hedge(0, 1, 0, 1))
    prices = controlled_prices()
    prices.iloc[42:, 0] = 102
    pair = Pair("A", "B", ("one",), 0, 1, 0, 1, 4, 0.01, 2)
    result = backtest(prices, [pair], 40, 45, config["execution"], 2, 1.5)
    assert result.trades[0]["exit_reason"] == "fold_end"
    assert result.daily.iloc[-1]["transaction_cost"] > 0
    assert result.daily.iloc[-1]["gross_exposure"] == 0


def test_backtest_prefix_invariant_and_refits_causal(config):
    prices, sectors = synthetic_prices(config["data"])
    pairs, _, _ = discover_pairs(prices.iloc[:126], sectors, config["research"], 3)
    selected = pairs[:1]
    original = backtest(prices, selected, 168, 252, config["execution"], 2, 1.25)
    changed = prices.copy()
    changed.iloc[215:] *= np.linspace(1, 2, 37)[:, None]
    altered = backtest(changed, selected, 168, 252, config["execution"], 2, 1.25)
    pd.testing.assert_frame_equal(original.daily.iloc[:47], altered.daily.iloc[:47])
    assert original.refits
    assert all(r["fit_end"] <= r["decision_time"] < r["fill_time"] for r in original.refits)
    assert all(t["model_train_end"] <= t["signal_time"] < t["entry_time"] for t in original.trades)
    assert original.metrics["net_pnl"] == pytest.approx(sum(t["net_pnl"] for t in original.trades), abs=1e-8)


def test_cash_backtest_is_finite(config):
    prices, _ = synthetic_prices(config["data"])
    result = backtest(prices, [], 168, 210, config["execution"], 2, 1.25)
    assert result.metrics["sharpe"] == 0 and result.metrics["trades"] == 0
    assert result.daily["equity"].nunique() == 1
    assert result.metrics["max_drawdown"] == 0


def test_higher_costs_reduce_same_path_pnl(config, monkeypatch):
    low, _ = controlled_backtest(config, monkeypatch, commission_bps=0, slippage_bps=0, borrow_rate=0, financing_rate=0)
    high, _ = controlled_backtest(config, monkeypatch)
    assert low.metrics["net_pnl"] > high.metrics["net_pnl"]
    assert low.trades[0]["entry_time"] == high.trades[0]["entry_time"]


def test_models_and_share_quantities_frozen_while_open(config, monkeypatch):
    import pairs_trading.execution as execution
    calls = []

    def fitting(*args):
        calls.append(len(args[0]))
        return Hedge(0, 1, 0, 1)

    monkeypatch.setattr(execution, "fit_hedge", fitting)
    prices = controlled_prices()
    pair = Pair("A", "B", ("one",), 0, 1, 0, 1, 4, 0.01, 2)
    settings = {**config["execution"], "refit_every": 1}
    result = backtest(prices, [pair], 40, 48, settings, 2, 1.5)
    assert 42 not in calls and 43 not in calls
    assert result.trades[0]["beta"] == 1
    assert result.trades[0]["gross_pnl"] == pytest.approx(3 * 50000 / 203)
