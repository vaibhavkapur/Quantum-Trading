"""Deterministic synthetic market data.

The generator is a latent-factor model in log prices with *known* structure so
that selection quality can be judged against ground truth:

* group A  - one common stochastic trend + stationary (OU) idiosyncratic noise:
             every pair is cointegrated.
* group B  - two common trends: pairs are NOT cointegrated, triples are. This is
             a deliberate blind spot for a pairwise surrogate objective.
* group C  - independent random walks (decoys).
* a regime break: one group-A asset decouples part-way through the sample.
* point-in-time membership: one late listing and one delisting, so the tradable
  universe changes through time and a naive "survivors only" panel would be biased.

Nothing here is real market data and no result on it says anything about live
profitability.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# symbol, sector, group, beta_T1, beta_T2, ou_half_life (None = random walk), idio_vol
_SPEC = [
    ("TEC1", "TECH", "A", 1.00, 0.0, 9.0, 0.0060),
    ("TEC2", "TECH", "A", 0.80, 0.0, 12.0, 0.0055),
    ("FIN1", "FIN", "A", 1.20, 0.0, 15.0, 0.0065),
    ("FIN2", "FIN", "A", 0.90, 0.0, 10.0, 0.0060),  # decouples at the regime break
    ("TEC3", "TECH", "A", 1.10, 0.0, 8.0, 0.0060),  # lists late
    ("ENE1", "ENERGY", "B", 0.90, 1.00, 10.0, 0.0060),
    ("ENE2", "ENERGY", "B", 0.60, 0.70, 14.0, 0.0055),
    ("UTL1", "UTIL", "B", 0.50, 1.20, 11.0, 0.0050),
    ("UTL2", "UTIL", "B", 0.70, -0.60, 13.0, 0.0055),
    ("ENE3", "ENERGY", "B", 0.80, 0.90, 9.0, 0.0060),  # delists
    ("TEC4", "TECH", "C", 1.00, 0.0, None, 0.0120),
    ("FIN3", "FIN", "C", 0.90, 0.0, None, 0.0110),
    ("ENE4", "ENERGY", "C", 0.70, 0.4, None, 0.0130),
    ("UTL3", "UTIL", "C", 0.40, 0.3, None, 0.0100),
]
REGIME_BREAK_SYMBOL = "FIN2"
REGIME_BREAK_FRAC = 0.62
LATE_LISTING = ("TEC3", 0.30)
DELISTING = ("ENE3", 0.80)
SESSION_CLOSE_UTC = "21:00"


def generate_daily(n_days: int, start: str, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (ohlcv_long, assets). Timestamps are UTC session closes."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start=start, periods=n_days, tz="UTC") + pd.Timedelta(SESSION_CLOSE_UTC + ":00")
    t1 = np.cumsum(rng.normal(0.0003, 0.010, n_days))
    t2 = np.cumsum(rng.normal(0.0, 0.007, n_days))
    break_idx = int(n_days * REGIME_BREAK_FRAC)

    frames, meta = [], []
    for sym, sector, group, b1, b2, hl, vol in _SPEC:
        eps = rng.normal(0.0, vol, n_days)
        if hl is None:
            idio = np.cumsum(eps)
        else:
            phi = float(np.exp(-np.log(2) / hl))
            idio = np.empty(n_days)
            acc = 0.0
            for t in range(n_days):
                # After the regime break the decoupling asset stops mean-reverting.
                decay = 1.0 if (sym == REGIME_BREAK_SYMBOL and t >= break_idx) else phi
                acc = decay * acc + eps[t]
                idio[t] = acc
        p0 = float(rng.uniform(20, 200))
        close = p0 * np.exp(b1 * t1 + b2 * t2 + idio)
        prev = np.concatenate([[close[0]], close[:-1]])
        open_ = prev * np.exp(rng.normal(0, 0.002, n_days))
        high = np.maximum(open_, close) * np.exp(np.abs(rng.normal(0, 0.004, n_days)))
        low = np.minimum(open_, close) * np.exp(-np.abs(rng.normal(0, 0.004, n_days)))
        adv = float(rng.uniform(1.0e6, 6.0e6))
        volume = np.round(adv * np.exp(rng.normal(0, 0.35, n_days)))
        df = pd.DataFrame(
            {"ts": dates, "symbol": sym, "open": open_, "high": high, "low": low, "close": close, "volume": volume}
        )
        listed, delisted = dates[0], pd.NaT
        if sym == LATE_LISTING[0]:
            listed = dates[int(n_days * LATE_LISTING[1])]
            df = df[df["ts"] >= listed]
        if sym == DELISTING[0]:
            delisted = dates[int(n_days * DELISTING[1])]
            df = df[df["ts"] <= delisted]
        frames.append(df)
        meta.append(
            {
                "symbol": sym,
                "sector": sector,
                "true_group": group,
                "spread_bps": round(float(rng.uniform(2.0, 9.0)), 2),
                "borrow_bps_annual": round(float(rng.uniform(25, 150)), 1),
                "listed": listed,
                "delisted": delisted,
            }
        )
    ohlcv = pd.concat(frames, ignore_index=True).sort_values(["ts", "symbol"]).reset_index(drop=True)
    return ohlcv, pd.DataFrame(meta)


def generate_l2(n_events: int, seed: int, levels: int = 5) -> tuple[pd.DataFrame, dict]:
    """Synthetic L2 quote stream for a 3-leg cointegrated basket.

    Each row replaces one price level: ``ts_ns,sym,side,level,price,size``.
    Mid prices share one random-walk trend; the spread with weights
    ``true_weights`` is an OU process. Timestamps are irregular UTC nanoseconds.
    """
    rng = np.random.default_rng(seed)
    n_sym = 3
    n_ticks = max(n_events // (2 * levels), 100)
    dt = rng.exponential(2.5e6, n_ticks).astype(np.int64) + 1  # ~2.5 ms between ticks
    ts = 1_700_000_000_000_000_000 + np.cumsum(dt)
    trend = np.cumsum(rng.normal(0, 2e-5, n_ticks))
    betas = np.array([1.0, 0.8, 1.25])
    phi = np.exp(-np.log(2) / np.array([900.0, 1300.0, 1100.0]))
    idio = np.zeros((n_ticks, n_sym))
    eps = rng.normal(0, 6e-5, (n_ticks, n_sym))
    for t in range(1, n_ticks):
        idio[t] = phi * idio[t - 1] + eps[t]
    base = np.array([30000.0, 2000.0, 100.0])
    tick = np.array([0.5, 0.05, 0.01])
    mid = base * np.exp(betas * trend[:, None] + idio)

    which = rng.integers(0, n_sym, n_ticks)
    half_ticks = rng.integers(1, 3, n_ticks)
    rows_ts, rows_sym, rows_side, rows_lvl, rows_px, rows_sz = [], [], [], [], [], []
    for lvl in range(levels):
        # Displayed depth is roughly $40k-$120k per level whatever the price scale.
        size = np.round(rng.gamma(2.0, 1.5, (n_ticks, 2)) * (1 + lvl) * (40_000.0 / base[which])[:, None], 4) + 0.001
        m, tk = mid[np.arange(n_ticks), which], tick[which]
        bid = np.floor(m / tk) * tk - (half_ticks - 1 + lvl) * tk
        ask = bid + (2 * half_ticks - 1) * tk + 2 * lvl * tk
        for side, px, col in ((0, bid, 0), (1, ask, 1)):
            rows_ts.append(ts)
            rows_sym.append(which)
            rows_side.append(np.full(n_ticks, side))
            rows_lvl.append(np.full(n_ticks, lvl))
            rows_px.append(px)
            rows_sz.append(size[:, col])
    df = pd.DataFrame(
        {
            "ts_ns": np.concatenate(rows_ts),
            "sym": np.concatenate(rows_sym),
            "side": np.concatenate(rows_side),
            "level": np.concatenate(rows_lvl),
            "price": np.round(np.concatenate(rows_px), 2),
            "size": np.concatenate(rows_sz),
        }
    )
    df = df.sort_values(["ts_ns", "sym", "side", "level"], kind="stable").reset_index(drop=True)
    info = {"symbols": ["SYN-A", "SYN-B", "SYN-C"], "levels": levels, "n_ticks": int(n_ticks), "betas": betas.tolist()}
    return df, info
