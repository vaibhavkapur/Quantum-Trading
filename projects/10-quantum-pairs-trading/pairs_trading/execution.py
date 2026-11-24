from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .research import Hedge, Pair, fit_hedge


@dataclass
class Backtest:
    daily: pd.DataFrame
    trades: list[dict]
    metrics: dict
    refits: list[dict]


def performance(daily: pd.DataFrame, trades: list[dict], initial_capital: float) -> dict:
    if daily.empty:
        raise ValueError("cannot evaluate empty backtest")
    returns = daily["return"].to_numpy()
    volatility = float(np.std(returns, ddof=1)) if len(returns) > 1 else 0.0
    curve = np.concatenate(([initial_capital], daily["equity"].to_numpy()))
    drawdown = curve / np.maximum.accumulate(curve) - 1
    winners = [t["net_pnl"] > 0 for t in trades]
    return {"total_return": float(curve[-1] / initial_capital - 1),
            "sharpe": float(np.sqrt(252) * np.mean(returns) / volatility) if volatility > 1e-12 else 0.0,
            "max_drawdown": float(drawdown.min()), "win_rate": float(np.mean(winners)) if winners else 0.0,
            "mean_reversion_success": float(np.mean([t["exit_reason"] == "mean_reversion" for t in trades])) if trades else 0.0,
            "trades": len(trades), "turnover": float(daily["turnover_dollars"].sum() / initial_capital),
            "transaction_costs": float(daily["transaction_cost"].sum()), "financing_costs": float(daily["financing_cost"].sum()),
            "max_asset_overlap": int(daily["asset_overlap"].max()), "max_gross_exposure": float(daily["gross_exposure"].max()),
            "net_pnl": float(curve[-1] - initial_capital), "observations": len(daily)}


def backtest(prices: pd.DataFrame, pairs: list[Pair], start: int, end: int, settings: dict,
             capital_budget: int, entry_z: float, initial_capital: float | None = None) -> Backtest:
    """Close-to-close accounting with signals at t-1 and fills at t.

    Refits only use data through the signal close. Open trades freeze their model
    and share quantities; the marked share P&L does not depend on model changes.
    """
    if not 40 <= start < end <= len(prices) or capital_budget <= 0:
        raise ValueError("invalid execution window or capital budget")
    if not settings["exit_z"] < entry_z < settings["stop_z"]:
        raise ValueError("entry threshold outside exit/stop bounds")
    if sum(p.capital_units for p in pairs) > capital_budget:
        raise ValueError("selected pairs exceed reserved capital budget")
    if len({p.name for p in pairs}) != len(pairs):
        raise ValueError("duplicate selected pair")
    equity = float(initial_capital if initial_capital is not None else settings["initial_capital"])
    if not np.isfinite(equity) or equity <= 0:
        raise ValueError("initial equity must be finite and positive")
    starting_capital = equity
    positions, models, last_refits = {}, {}, {}
    rows, trades, refits = [], [], []
    values = prices.to_numpy(float)
    columns = {name: i for i, name in enumerate(prices.columns)}
    cost_rate = (settings["commission_bps"] + settings["slippage_bps"]) / 1e4
    for t in range(start, end):
        previous_equity = equity
        mark_pnl, financing, costs, turnover = 0.0, 0.0, 0.0, 0.0
        previous, current = values[t - 1], values[t]
        year_fraction = (prices.index[t] - prices.index[t - 1]).total_seconds() / (365 * 86400)
        # Price moves and carrying costs belong to shares held before today's fill.
        for position in positions.values():
            idx, qty = position["indices"], position["quantities"]
            pnl = float(qty @ (current[idx] - previous[idx]))
            market_values = qty * previous[idx]
            carry = float((-np.minimum(market_values, 0).sum() * settings["borrow_rate"] + np.maximum(market_values, 0).sum() * settings["financing_rate"]) * year_fraction)
            mark_pnl += pnl
            financing += carry
            position["gross_pnl"] += pnl
            position["financing"] += carry
        equity += mark_pnl - financing

        for pair in pairs:
            idx = [columns[pair.left], columns[pair.right]]
            position = positions.get(pair.name)
            if position is not None:
                z = position["model"].zscore(*previous[idx])
                # Stop-loss uses P&L known at t-1, before today's realized mark.
                previous_pnl = position["gross_pnl"] - float(position["quantities"] @ (current[idx] - previous[idx]))
                reason = None
                if abs(z) >= settings["stop_z"] or previous_pnl <= -settings["stop_loss_pct"] * position["entry_gross"]:
                    reason = "stop_out"
                elif t - position["entry_index"] >= settings["max_holding"]:
                    reason = "holding_limit"
                elif abs(z) <= settings["exit_z"] or z * position["entry_z"] < 0:
                    reason = "mean_reversion"
                if t == end - 1:
                    reason = reason or "fold_end"
                if equity <= 0:
                    reason = "insolvency"
                if reason:
                    notional = float(np.abs(position["quantities"] * current[idx]).sum())
                    close_cost = notional * cost_rate
                    costs += close_cost
                    turnover += notional
                    equity -= close_cost
                    trades.append({"pair": pair.name, "entry_time": position["entry_time"], "exit_time": prices.index[t].isoformat(),
                                   "signal_time": position["signal_time"], "exit_signal_time": prices.index[t - 1].isoformat(),
                                   "model_train_end": position["model_train_end"], "beta": position["model"].beta,
                                   "left_quantity": float(position["quantities"][0]), "right_quantity": float(position["quantities"][1]),
                                   "entry_z": position["entry_z"], "exit_z": float(z), "holding_bars": t - position["entry_index"],
                                   "gross_pnl": position["gross_pnl"], "financing_cost": position["financing"],
                                   "transaction_cost": position["entry_cost"] + close_cost,
                                   "net_pnl": position["gross_pnl"] - position["financing"] - position["entry_cost"] - close_cost,
                                   "exit_reason": reason})
                    del positions[pair.name]
                continue

            if t == end - 1 or equity <= 0:
                continue
            if pair.name not in models or t - last_refits[pair.name] >= settings["refit_every"]:
                begin = max(0, t - settings["lookback"])
                try:
                    model = fit_hedge(values[begin:t, idx[0]], values[begin:t, idx[1]])
                except ValueError:
                    models.pop(pair.name, None)
                    continue
                models[pair.name] = model
                last_refits[pair.name] = t
                refits.append({"pair": pair.name, "decision_time": prices.index[t - 1].isoformat(),
                               "fit_start": prices.index[begin].isoformat(), "fit_end": prices.index[t - 1].isoformat(),
                               "fill_time": prices.index[t].isoformat(), "beta": model.beta})
            model = models[pair.name]
            z = model.zscore(*previous[idx])
            if not entry_z <= abs(z) < settings["stop_z"]:
                continue
            gross_held = sum(float(np.abs(p["quantities"] * current[p["indices"]]).sum()) for p in positions.values())
            available = max(0.0, equity * settings["max_gross"] - gross_held)
            # Reserve capital after entry fees so the post-fill gross gate also holds.
            allocation = min(starting_capital * pair.capital_units / capital_budget, available / (1 + cost_rate * settings["max_gross"]))
            if allocation < 1e-8:
                continue
            direction = -np.sign(z)
            quantity = direction * allocation / (current[idx[0]] + abs(model.beta) * current[idx[1]])
            quantities = np.array([quantity, -model.beta * quantity])
            entry_cost = allocation * cost_rate
            costs += entry_cost
            turnover += allocation
            equity -= entry_cost
            positions[pair.name] = {"indices": idx, "quantities": quantities, "model": model,
                                    "entry_index": t, "entry_time": prices.index[t].isoformat(), "signal_time": prices.index[t - 1].isoformat(),
                                    "model_train_end": prices.index[last_refits[pair.name] - 1].isoformat(), "entry_z": float(z),
                                    "entry_gross": allocation, "entry_cost": entry_cost, "gross_pnl": 0.0, "financing": 0.0}
        counts = {}
        for position in positions.values():
            for index in position["indices"]:
                counts[index] = counts.get(index, 0) + 1
        gross = sum(float(np.abs(p["quantities"] * current[p["indices"]]).sum()) for p in positions.values())
        rows.append({"timestamp": prices.index[t].isoformat(), "equity": equity, "return": (equity - previous_equity) / previous_equity if previous_equity > 0 else 0.0,
                     "mark_pnl": mark_pnl, "transaction_cost": costs, "financing_cost": financing, "turnover_dollars": turnover,
                     "gross_exposure": gross / equity if equity > 0 else 0.0, "asset_overlap": max(counts.values(), default=0), "open_pairs": len(positions)})
    daily = pd.DataFrame(rows)
    return Backtest(daily, trades, performance(daily, trades, starting_capital), refits)
