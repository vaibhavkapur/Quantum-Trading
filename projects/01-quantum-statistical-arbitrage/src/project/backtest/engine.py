"""Event-driven backtester for a single spread.

Events are processed in time order from a priority queue:

    BAR_OPEN(t)  -> orders whose latency has elapsed are filled against a
                    synthetic limit order book built around the open price
    BAR_CLOSE(t) -> mark to market, borrow accrual, delisting handling,
                    kill-switch check, then the strategy sees the close and may
                    emit orders that become executable ``latency_bars`` later

The strategy therefore never trades on the bar that produced its signal.
All parameters of the strategy are fixed before the first test bar.
"""
from __future__ import annotations

import heapq
import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

BAR_OPEN, BAR_CLOSE = 0, 1


@dataclass
class Fill:
    bar: int
    symbol: str
    qty: float
    price: float
    ref_price: float
    fee: float
    spread_cost: float
    impact_cost: float
    reason: str


class ZScoreStrategy:
    """Z-score entry/exit/stop state machine with a time stop and a cooldown."""

    def __init__(self, weights: np.ndarray, mean: float, std: float, half_life: float, cfg: dict) -> None:
        self.w, self.mean, self.std = np.asarray(weights, float), float(mean), float(std)
        self.entry, self.exit, self.stop = cfg["entry_z"], cfg["exit_z"], cfg["stop_z"]
        hl = half_life if np.isfinite(half_life) else 20.0
        self.max_holding = max(int(np.ceil(cfg["max_holding_halflives"] * hl)), 1)
        self.cooldown_bars = int(cfg["cooldown_bars"])
        self.direction = 0      # +1 long the spread, -1 short, 0 flat
        self.bars_held = 0
        self.cooldown = 0
        self.exit_reasons: dict[str, int] = {}
        self.entries = 0

    def zscore(self, log_prices: np.ndarray) -> float:
        return float((log_prices @ self.w - self.mean) / self.std)

    def _exit(self, reason: str, cooldown: bool) -> None:
        self.exit_reasons[reason] = self.exit_reasons.get(reason, 0) + 1
        self.direction, self.bars_held = 0, 0
        if cooldown:
            self.cooldown = self.cooldown_bars

    def on_bar(self, log_prices: np.ndarray) -> int:
        z = self.zscore(log_prices)
        if self.direction == 0:
            if self.cooldown > 0:
                self.cooldown -= 1
            elif self.entry < z < self.stop:
                self.direction, self.entries = -1, self.entries + 1
            elif -self.stop < z < -self.entry:
                self.direction, self.entries = 1, self.entries + 1
        else:
            self.bars_held += 1
            if abs(z) >= self.stop:
                self._exit("stop_loss", cooldown=True)
            elif (self.direction == 1 and z >= -self.exit) or (self.direction == -1 and z <= self.exit):
                self._exit("mean_reversion", cooldown=False)
            elif self.bars_held >= self.max_holding:
                self._exit("time_stop", cooldown=True)
        return self.direction


class BookExecution:
    """LOB-aware fill model for bar data.

    A book is synthesised around the reference price: level ``l`` sits
    ``half_spread * (1 + l * level_step)`` away and shows ``depth_frac_adv * ADV``
    shares. Marketable orders walk the levels; size beyond the displayed book
    pays ``beyond_book_bps`` on top of the last level.
    """

    def __init__(self, spread_bps: np.ndarray, adv_shares: np.ndarray, cfg: dict) -> None:
        self.half_bps = np.asarray(spread_bps, float) / 2.0
        self.level_size = np.maximum(np.asarray(adv_shares, float) * cfg["depth_frac_adv"], 1e-9)
        self.levels, self.level_step = int(cfg["book_levels"]), float(cfg["level_step"])
        self.beyond_bps, self.fee_bps = float(cfg["beyond_book_bps"]), float(cfg["fee_bps"])

    def fill(self, i: int, qty: float, ref_price: float) -> tuple[float, float, float, float]:
        """Return (avg_price, fee, spread_cost, impact_cost) for a signed share quantity."""
        remaining, cost_bps_shares = abs(qty), 0.0
        for level in range(self.levels):
            take = min(remaining, self.level_size[i])
            cost_bps_shares += take * self.half_bps[i] * (1 + level * self.level_step)
            remaining -= take
            if remaining <= 0:
                break
        if remaining > 0:
            worst = self.half_bps[i] * (1 + (self.levels - 1) * self.level_step) + self.beyond_bps
            cost_bps_shares += remaining * worst
        avg_bps = cost_bps_shares / abs(qty)
        price = ref_price * (1 + np.sign(qty) * avg_bps / 1e4)
        notional = abs(qty) * ref_price
        spread_cost = notional * self.half_bps[i] / 1e4
        impact_cost = notional * avg_bps / 1e4 - spread_cost
        return float(price), float(notional * self.fee_bps / 1e4), float(spread_cost), float(impact_cost)


@dataclass
class BacktestResult:
    equity: pd.Series
    returns: pd.Series          # daily P&L / initial capital
    fills: list[Fill]
    stats: dict = field(default_factory=dict)


def run_backtest(
    open_: pd.DataFrame,
    close: pd.DataFrame,
    strategy: ZScoreStrategy,
    spread_bps: np.ndarray,
    borrow_bps_annual: np.ndarray,
    adv_shares: np.ndarray,
    risk: dict,
    execution: dict,
) -> BacktestResult:
    symbols = list(close.columns)
    n_bars, n = close.shape
    capital = float(risk["capital"])
    latency = int(execution["latency_bars"])
    if latency < 1:
        raise ValueError("latency_bars must be >= 1 so a signal can never fill on its own bar")
    book = BookExecution(spread_bps, adv_shares, execution)
    close_v, open_v = close.to_numpy(float), open_.to_numpy(float)
    alive = ~np.isnan(close_v)
    last_px = pd.DataFrame(close_v).ffill().to_numpy()

    # Position limits: scale the whole basket so hedge ratios are preserved.
    gross = capital * float(risk["gross_leverage"])
    max_leg = float(np.max(np.abs(strategy.w))) * gross
    limit = float(risk["max_asset_notional_frac"]) * capital
    scale = min(1.0, limit / max_leg) if max_leg > 0 else 0.0

    cash, pos = capital, np.zeros(n)
    target = np.zeros(n)
    fills: list[Fill] = []
    equity = np.full(n_bars, np.nan)
    pending: list[tuple[int, np.ndarray, str]] = []  # (due_bar, target_shares, reason)
    halted, kill_switch, delisted = False, False, False
    peak = capital
    borrow_paid, traded_notional, bars_in_market = 0.0, 0.0, 0
    decision_ns: list[int] = []

    def execute(bar: int, new_target: np.ndarray, ref: np.ndarray, reason: str) -> None:
        nonlocal cash, pos, traded_notional
        for i in range(n):
            qty = new_target[i] - pos[i]
            if abs(qty) < 1e-9 or not np.isfinite(ref[i]):
                continue
            price, fee, spread_cost, impact = book.fill(i, qty, ref[i])
            cash -= qty * price + fee
            pos[i] += qty
            traded_notional += abs(qty) * ref[i]
            fills.append(Fill(bar, symbols[i], float(qty), price, float(ref[i]), fee, spread_cost, impact, reason))

    queue: list[tuple[int, int]] = []
    for t in range(n_bars):
        heapq.heappush(queue, (t, BAR_OPEN))
        heapq.heappush(queue, (t, BAR_CLOSE))

    while queue:
        t, kind = heapq.heappop(queue)
        if kind == BAR_OPEN:
            due = [p for p in pending if p[0] <= t]
            pending = [p for p in pending if p[0] > t]
            for _, tgt, reason in due:
                execute(t, tgt, open_v[t], reason)
            continue

        # ---- BAR_CLOSE
        px = last_px[t]
        dead = ~alive[t]
        if dead.any() and not delisted:
            # Delisting: close dead legs at the last price with an adverse haircut, flatten the rest.
            delisted = halted = True
            for i in np.flatnonzero(dead & (np.abs(pos) > 1e-9)):
                haircut = px[i] * (1 - np.sign(pos[i]) * float(execution["delist_haircut"]))
                cash += pos[i] * haircut
                fills.append(Fill(t, symbols[i], float(-pos[i]), float(haircut), float(px[i]), 0.0, 0.0, 0.0, "delisting"))
                pos[i] = 0.0
            target = np.zeros(n)
            pending = [(t + latency, target.copy(), "delisting_flatten")]
        short_value = float(np.sum(np.where(pos < 0, -pos * px, 0.0) * borrow_bps_annual)) / 1e4 / 252.0
        cash -= short_value
        borrow_paid += short_value
        eq = cash + float(np.nansum(pos * px))
        equity[t] = eq
        peak = max(peak, eq)
        bars_in_market += int(np.any(np.abs(pos) > 1e-9))

        if not halted and (peak - eq) >= float(risk["kill_switch_drawdown"]) * capital:
            halted = kill_switch = True
            target = np.zeros(n)
            pending = [(t + latency, target.copy(), "kill_switch")]

        if not halted and t < n_bars - 1:
            t0 = time.perf_counter_ns()
            prev_dir = strategy.direction
            direction = strategy.on_bar(np.log(px))
            if direction != prev_dir:
                new_target = direction * strategy.w * gross * scale / px if direction != 0 else np.zeros(n)
                target = new_target
                pending.append((t + latency, new_target.copy(), "signal"))
            decision_ns.append(time.perf_counter_ns() - t0)

        if t == n_bars - 1:
            # End of the test window: liquidate at the close through the same cost model.
            if np.any(np.abs(pos) > 1e-9):
                execute(t, np.zeros(n), px, "end_of_window")
                equity[t] = cash + float(np.nansum(pos * px))
            pending = []

    eq = pd.Series(equity, index=close.index, name="equity")
    rets = eq.diff().fillna(eq.iloc[0] - capital) / capital
    stats = {
        "pnl": float(eq.iloc[-1] - capital),
        "n_fills": len(fills),
        "n_entries": int(strategy.entries),
        "exit_reasons": dict(strategy.exit_reasons),
        "traded_notional": float(traded_notional),
        "fees": float(sum(f.fee for f in fills)),
        "spread_cost": float(sum(f.spread_cost for f in fills)),
        "impact_cost": float(sum(f.impact_cost for f in fills)),
        "borrow_cost": float(borrow_paid),
        "kill_switch": bool(kill_switch),
        "delisting": bool(delisted),
        "time_in_market": bars_in_market / max(n_bars, 1),
        "position_scale": float(scale),
        "decision_latency_us_p50": float(np.percentile(decision_ns, 50) / 1e3) if decision_ns else float("nan"),
        "decision_latency_us_p99": float(np.percentile(decision_ns, 99) / 1e3) if decision_ns else float("nan"),
    }
    return BacktestResult(eq, rets, fills, stats)
