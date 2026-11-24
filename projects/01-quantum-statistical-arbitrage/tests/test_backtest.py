import numpy as np
import pandas as pd
import pytest

from project.backtest import metrics
from project.backtest.engine import BookExecution, ZScoreStrategy, run_backtest

SIGNAL = {"entry_z": 2.0, "exit_z": 0.5, "stop_z": 4.0, "max_holding_halflives": 4.0, "cooldown_bars": 2}
RISK = {"capital": 1_000_000, "gross_leverage": 1.0, "max_asset_notional_frac": 0.7, "kill_switch_drawdown": 0.08}
EXEC = {"latency_bars": 1, "fee_bps": 0.5, "book_levels": 3, "level_step": 0.5, "depth_frac_adv": 0.01,
        "beyond_book_bps": 25, "delist_haircut": 0.1}
FREE = {**EXEC, "fee_bps": 0.0, "beyond_book_bps": 0.0}


def frames(a, b):
    idx = pd.bdate_range("2022-01-03", periods=len(a), tz="UTC")
    close = pd.DataFrame({"A": a, "B": b}, index=idx)
    return close.copy(), close  # open == close keeps hand calculations simple


def strategy(std=0.005, hl=5.0, cfg=SIGNAL):
    return ZScoreStrategy(np.array([0.5, -0.5]), 0.0, std, hl, cfg)


def run(a, b, strat=None, spread=(4.0, 4.0), borrow=(0.0, 0.0), risk=RISK, execution=EXEC):
    o, c = frames(np.asarray(a, float), np.asarray(b, float))
    return run_backtest(o, c, strat or strategy(), np.array(spread), np.array(borrow), np.array([1e7, 1e7]), risk, execution)


def test_state_machine_entry_exit_stop_and_time_stop():
    s = ZScoreStrategy(np.array([1.0]), 0.0, 1.0, 1.0, SIGNAL)  # max holding = 4 bars
    z = lambda v: s.on_bar(np.array([v]))  # noqa: E731
    assert z(1.0) == 0 and z(2.5) == -1 and z(1.0) == -1 and z(0.4) == 0
    assert z(-2.5) == 1 and z(-4.2) == 0                      # stop loss
    assert z(-2.5) == 0 and z(-2.5) == 0 and z(-2.5) == 1     # two cooldown bars, then re-entry
    assert [z(-2.5) for _ in range(4)] == [1, 1, 1, 0]        # time stop
    assert s.exit_reasons == {"mean_reversion": 1, "stop_loss": 1, "time_stop": 1}
    assert z(4.5) == 0                                        # never enters beyond the stop band


def test_book_fill_walks_levels_and_charges_costs():
    book = BookExecution(np.array([10.0]), np.array([1000.0]), EXEC)  # half-spread 5 bps, 10 shares per level
    price, fee, spread_cost, impact = book.fill(0, 5, 100.0)
    assert price == pytest.approx(100.05) and impact == pytest.approx(0.0)
    assert fee == pytest.approx(500 * 0.5e-4) and spread_cost == pytest.approx(0.25)
    price, *_ = book.fill(0, -20, 100.0)  # 10 @ 5 bps + 10 @ 7.5 bps
    assert price == pytest.approx(100.0 * (1 - 6.25e-4))
    price, _, _, impact = book.fill(0, 40, 100.0)  # 10@5, 10@7.5, 10@10, 10 beyond @ 35
    assert price == pytest.approx(100.0 * (1 + 14.375e-4)) and impact > 0


def test_signal_never_fills_on_its_own_bar():
    a = [100, 100, 103, 103, 100, 100, 100]
    res = run(a, [100] * 7)
    entry = res.fills[0]
    assert entry.bar == 3 and entry.reason == "signal"       # z crosses on bar 2 -> fills at bar 3 open
    assert entry.ref_price == 103.0 and entry.qty < 0        # short the rich leg
    assert res.equity.iloc[2] == RISK["capital"]             # nothing happened yet on the signal bar


def test_zero_latency_is_refused():
    with pytest.raises(ValueError, match="latency_bars"):
        run([100] * 5, [100] * 5, execution={**EXEC, "latency_bars": 0})


def test_flat_market_produces_no_trades_and_flat_equity():
    res = run([100] * 30, [100] * 30)
    assert res.fills == [] and (res.equity == RISK["capital"]).all() and res.stats["pnl"] == 0.0


def test_round_trip_pnl_matches_hand_calculation_without_costs():
    # Short A at 103 with 350k notional (position cap 0.7 * capital applies to the 500k legs), cover at 100.
    a = [100, 100, 103, 103, 100, 100, 100, 100]
    res = run(a, [100] * 8, spread=(0.0, 0.0), execution=FREE, risk={**RISK, "max_asset_notional_frac": 0.35})
    assert res.stats["position_scale"] == pytest.approx(0.7)
    shares = 0.5 * 1_000_000 * 0.7 / 103
    assert res.stats["pnl"] == pytest.approx(shares * 3.0, rel=1e-9)
    assert res.stats["exit_reasons"] == {"mean_reversion": 1}


def test_costs_strictly_reduce_pnl():
    a = [100, 100, 103, 103, 100, 100, 100, 100]
    free = run(a, [100] * 8, spread=(0.0, 0.0), execution=FREE)
    paid = run(a, [100] * 8, spread=(8.0, 8.0), borrow=(300.0, 300.0))
    assert paid.stats["pnl"] < free.stats["pnl"]
    assert paid.stats["fees"] > 0 and paid.stats["spread_cost"] > 0 and paid.stats["borrow_cost"] > 0


def test_mean_reverting_spread_is_profitable_before_costs():
    rng = np.random.default_rng(0)
    n, phi, x = 1500, np.exp(-np.log(2) / 5), np.zeros(1500)
    for t in range(1, n):
        x[t] = phi * x[t - 1] + rng.normal(0, 0.01)
    res = run(100 * np.exp(x), [100.0] * n, strat=strategy(std=np.std(x) / 2), spread=(0.0, 0.0), execution=FREE)
    assert res.stats["n_entries"] > 10 and res.stats["pnl"] > 0


def test_kill_switch_flattens_and_halts():
    a = [100, 100, 103] + [103 * 1.03**k for k in range(1, 12)]  # spread keeps running against the short
    wide = {**SIGNAL, "stop_z": 1e9}
    res = run(a, [100] * len(a), strat=strategy(cfg=wide), risk={**RISK, "kill_switch_drawdown": 0.02})
    assert res.stats["kill_switch"] and res.stats["n_entries"] == 1
    assert any(f.reason == "kill_switch" for f in res.fills)
    last_trade = max(f.bar for f in res.fills)
    assert (res.equity.iloc[last_trade:].diff().dropna() == 0).all()  # flat and halted afterwards


def test_end_of_window_liquidation_pays_costs():
    a = [100, 100, 103, 103, 103, 103]
    res = run(a, [100] * 6, strat=strategy(hl=50.0))
    assert res.fills[-1].reason == "end_of_window" and res.fills[-1].bar == 5
    assert res.stats["pnl"] < 0  # entered and left at the same price, paid costs twice


def test_delisting_is_liquidated_with_haircut():
    a = [100, 100, 103, 103, 103, np.nan, np.nan, np.nan]
    res = run(a, [100] * 8, strat=strategy(hl=50.0), spread=(0.0, 0.0), execution=FREE)
    assert res.stats["delisting"]
    d = [f for f in res.fills if f.reason == "delisting"][0]
    assert d.symbol == "A" and d.price == pytest.approx(103 * 1.1)  # covering a short costs 10% more
    assert any(f.reason == "delisting_flatten" and f.symbol == "B" for f in res.fills)
    assert res.equity.notna().all()


def test_metrics_on_known_series():
    r = np.array([0.01, -0.01, 0.02, -0.03, 0.01])
    assert metrics.max_drawdown(r) == pytest.approx(0.03)
    assert metrics.annualized_return(r) == pytest.approx(0.0, abs=1e-12)
    assert metrics.sharpe(np.zeros(10)) == 0.0
    assert metrics.sharpe(np.array([0.01, 0.03])) == pytest.approx(0.02 / np.std([0.01, 0.03], ddof=1) * np.sqrt(252))
    assert metrics.annualized_turnover(2_000_000, 1_000_000, 126) == pytest.approx(4.0)
    lo, hi = metrics.bootstrap_ci(np.array([1.0, 2.0, 3.0, 4.0]))
    assert 1.0 <= lo < 2.5 < hi <= 4.0
    assert all(np.isnan(v) for v in metrics.bootstrap_ci(np.array([1.0])))
