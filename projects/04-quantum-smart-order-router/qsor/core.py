"""Shared constants, order/report messages and FIX-style encoding."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

BUY, SELL = 1, -1
BID, ASK = 0, 1  # book-side indices used by the dataset arrays

MS = 1_000_000  # nanoseconds per millisecond


def opp_book(side: int) -> int:
    """Book side an aggressive order of `side` trades against."""
    return ASK if side == BUY else BID


def own_book(side: int) -> int:
    """Book side a passive order of `side` rests on."""
    return BID if side == BUY else ASK


@dataclass
class Order:
    id: int
    parent_id: int
    venue: int
    side: int
    qty: int  # lots
    px: int  # limit price in ticks
    tif: str  # "IOC" or "DAY"
    sent_ns: int = 0
    latency_ns: int = 0
    leaves: int = 0
    est_queue: int = 0
    acked: bool = False


@dataclass
class ExecReport:
    order_id: int
    venue: int
    kind: str  # ACK FILL REJECT CANCELED REPLACED CANCEL_REJECT REPLACE_REJECT
    t_ns: int
    qty: int = 0
    px: int = 0
    leaves: int = 0
    reason: str = ""
    liquidity: str = ""  # "T" taker / "M" maker
    queue_ahead: Optional[int] = None


_MASK = (1 << 64) - 1


def mix64(*keys: int) -> int:
    """Deterministic splitmix64-style hash of integer keys."""
    x = 0x9E3779B97F4A7C15
    for k in keys:
        x = (x ^ (k & _MASK)) & _MASK
        x = (x + 0x9E3779B97F4A7C15) & _MASK
        x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & _MASK
        x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) & _MASK
        x ^= x >> 31
    return x


def hash_exponential(mean: float, *keys: int) -> float:
    """Exponential variate that depends only on `keys` (not on call order)."""
    if mean <= 0:
        return 0.0
    u = (mix64(*keys) >> 11) / float(1 << 53)
    return -mean * math.log(1.0 - u)


# --- FIX-style encoding (subset of FIX 4.4 tags; "|" stands in for SOH) -----

_ORD_STATUS = {"ACK": "0", "FILL": "1", "CANCELED": "4", "REPLACED": "5", "REJECT": "8"}


def _fix(fields) -> str:
    return "|".join(f"{k}={v}" for k, v in fields) + "|"


def fix_new_order(o: Order, venue_name: str, tick: float, lot: int) -> str:
    return _fix([(35, "D"), (11, o.id), (100, venue_name), (54, 1 if o.side == BUY else 2),
                 (38, o.qty * lot), (40, 2), (44, f"{o.px * tick:.2f}"),
                 (59, 3 if o.tif == "IOC" else 0), (60, o.sent_ns)])


def fix_cancel(o: Order, venue_name: str, t_ns: int) -> str:
    return _fix([(35, "F"), (41, o.id), (100, venue_name), (60, t_ns)])


def fix_replace(o: Order, venue_name: str, new_px: int, tick: float, lot: int, t_ns: int) -> str:
    return _fix([(35, "G"), (41, o.id), (100, venue_name), (38, o.leaves * lot),
                 (44, f"{new_px * tick:.2f}"), (60, t_ns)])


def fix_exec_report(r: ExecReport, venue_name: str, tick: float, lot: int) -> str:
    if r.kind in ("CANCEL_REJECT", "REPLACE_REJECT"):
        return _fix([(35, 9), (41, r.order_id), (100, venue_name), (58, r.reason), (60, r.t_ns)])
    status = _ORD_STATUS[r.kind]
    if r.kind == "FILL" and r.leaves == 0:
        status = "2"
    fields = [(35, 8), (11, r.order_id), (100, venue_name), (150, status), (39, status),
              (151, r.leaves * lot), (60, r.t_ns)]
    if r.kind == "FILL":
        fields += [(32, r.qty * lot), (31, f"{r.px * tick:.2f}"), (851, 2 if r.liquidity == "T" else 1)]
    if r.reason:
        fields.append((58, r.reason))
    return _fix(fields)
