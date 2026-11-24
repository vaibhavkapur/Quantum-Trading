"""Rolling, point-in-time backtest of one solver.

Timeline for a signal date ``s`` (a row of the price panel):

* inputs use prices with ``date <= s`` only;
* the trade executes at the close of ``s + 1`` (one-session lag), paying linear costs of
  ``cost_bps`` per unit of L1 turnover;
* the new weights earn returns from ``s + 2`` onwards and drift between rebalances.

Signal dates start after ``splits.train_end``.  Each row is tagged ``validation`` or ``test``
(by ``splits.validation_end``); hyper-parameters in the config are fixed before the test
window is looked at.  If a solve yields no feasible portfolio the previous weights are held
and the event is recorded.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import budget_from, constraints_from, make_solver
from ..data.estimation import estimate, simple_returns
from ..models.problem import PortfolioProblem
from .metrics import performance


def build_problem(cfg: dict, prices: pd.DataFrame, sectors: dict, asof, w_prev=None,
                  risk_aversion: float | None = None) -> PortfolioProblem:
    sigma, mu, info = estimate(prices, asof, cfg["estimation"])
    pf = cfg["portfolio"]
    return PortfolioProblem(
        sigma, mu, constraints_from(pf),
        risk_aversion=float(pf["risk_aversion"] if risk_aversion is None else risk_aversion),
        turnover_penalty=float(pf.get("turnover_penalty", 0.0)),
        w_prev=w_prev, sectors=[sectors.get(t, "UNKNOWN") for t in prices.columns],
        assets=list(prices.columns), name=f"{cfg['name']}@{pd.Timestamp(asof).date()}", meta=info)


def signal_dates(cfg: dict, prices: pd.DataFrame) -> list[pd.Timestamp]:
    train_end = cfg["splits"].get("train_end")
    lookback = int(cfg["estimation"]["lookback_days"])
    start = lookback
    if train_end is not None:
        start = max(start, int(prices.index.searchsorted(pd.Timestamp(train_end), side="right")))
    step = int(cfg["backtest"]["rebalance_every_days"])
    return [prices.index[i] for i in range(start, len(prices) - 2, step)]


def split_of(cfg: dict, date) -> str:
    val_end = cfg["splits"].get("validation_end")
    if val_end is None or pd.Timestamp(date) > pd.Timestamp(val_end):
        return "test"
    return "validation"


def run_backtest(cfg: dict, prices: pd.DataFrame, sectors: dict, solver_name: str,
                 seed: int | None) -> dict:
    solver = make_solver(cfg, solver_name)
    budget = budget_from(cfg)
    rets = simple_returns(prices)
    n = prices.shape[1]
    cost_rate = float(cfg["backtest"]["cost_bps"]) / 1e4
    signals = signal_dates(cfg, prices)
    if not signals:
        raise ValueError("no rebalance dates: check splits.train_end and estimation.lookback_days")
    pos = {d: i for i, d in enumerate(prices.index)}
    trade_day = {prices.index[pos[s] + 1]: s for s in signals}

    w = np.zeros(n)                       # current (drifting) weights; cash before first trade
    pending: dict[pd.Timestamp, np.ndarray | None] = {}
    signal_set = set(signals)
    first_trade = prices.index[pos[signals[0]] + 1]
    rebal_rows, weight_rows, daily_rows = [], [], []

    for d in prices.index[pos[signals[0]]:]:
        # 1. earn today's return on yesterday's weights, then let the weights drift
        if d >= first_trade:
            r = rets.loc[d].to_numpy() if d > first_trade else np.zeros(n)
            port = float(w @ r)
            daily_rows.append({"date": d, "gross_ret": port, "cost": 0.0,
                               "bench": float(r.mean()), "split": split_of(cfg, d)})
            w = w * (1 + r) / (1 + port)
        # 2. execute at today's close the signal computed at yesterday's close
        if d in pending:
            target = pending.pop(d)
            if target is not None:
                turnover = float(np.abs(target - w).sum())
                daily_rows[-1]["cost"] = cost_rate * turnover
                rebal_rows[-1].update(realized_turnover_l1=turnover, cost=cost_rate * turnover)
                w = target.copy()
        # 3. compute a signal from data up to today's close
        if d in signal_set:
            problem = build_problem(cfg, prices, sectors, d, w_prev=w.copy())
            k = len(rebal_rows)
            res = solver.solve(problem, budget, seed=None if seed is None else seed + 1000 * k)
            trade = prices.index[pos[d] + 1]
            row = {"signal_date": d, "trade_date": trade, "split": split_of(cfg, trade),
                   **res.record(), "realized_turnover_l1": 0.0, "cost": 0.0,
                   "held_previous": not res.feasible}
            if res.feasible:
                row.update(problem.describe(res.z))
                weight_rows.append({"signal_date": d, **dict(zip(prices.columns, res.weights))})
            rebal_rows.append(row)
            pending[trade] = res.weights if res.feasible else None

    daily = pd.DataFrame(daily_rows).set_index("date")
    daily["ret"] = daily["gross_ret"] - daily["cost"]
    rebal = pd.DataFrame(rebal_rows)
    weights = pd.DataFrame(weight_rows)
    summary = {"solver": solver_name, "seed": seed, "all": _summarize(daily, rebal)}
    for split in ("validation", "test"):
        dd, rr = daily[daily["split"] == split], rebal[rebal["split"] == split]
        if len(dd):
            summary[split] = _summarize(dd, rr)
    return {"daily": daily, "rebalances": rebal, "weights": weights, "summary": summary}


def _summarize(daily: pd.DataFrame, rebal: pd.DataFrame) -> dict:
    out = performance(daily)
    if len(rebal):
        ok = rebal["feasible"].astype(bool)
        out.update({
            "n_rebalances": int(len(rebal)),
            "n_infeasible_pre_repair": int((rebal["raw_violation"] > 0).sum()),
            "n_repaired": int(rebal["repaired"].astype(bool).sum()),
            "n_no_portfolio": int((~ok).sum()),
            "n_failed_or_skipped": int(rebal["status"].isin(["failed", "skipped"]).sum()),
            "mean_turnover_l1": float(rebal["realized_turnover_l1"].mean()),
            "total_cost": float(rebal["cost"].sum()),
            "mean_objective": float(rebal.loc[ok, "objective"].mean()) if ok.any() else float("nan"),
            "mean_wall_time_s": float(rebal["wall_time_s"].mean()),
            "total_wall_time_s": float(rebal["wall_time_s"].sum()),
            "mean_evals": float(rebal["evals"].mean()),
        })
    return out
