"""Immutable, versioned policy snapshots and the store the hot path reads."""
from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Optional, Tuple

from ..core import BUY


@dataclass(frozen=True)
class PolicyEntry:
    weights: Tuple[int, ...]   # allocation units per venue for this bucket
    total: int
    rank: Tuple[int, ...]      # venues by descending weight (overflow order)


@dataclass(frozen=True)
class PolicySnapshot:
    version: int
    created_ns: int            # when the optimizer read the market
    published_ns: int
    source: str                # solver name, or "<solver>+fallback"
    buckets: Tuple[int, ...]
    bucket_of: Tuple[int, ...]  # qty (clamped) -> index into buckets
    entries: Tuple[Tuple[PolicyEntry, ...], Tuple[PolicyEntry, ...]]  # [buy, sell][bucket]

    def lookup(self, side: int, qty: int) -> PolicyEntry:
        b = self.bucket_of[qty] if qty < len(self.bucket_of) else len(self.buckets) - 1
        return self.entries[0 if side == BUY else 1][b]


def make_entry(units) -> PolicyEntry:
    units = tuple(int(u) for u in units)
    rank = tuple(sorted(range(len(units)), key=lambda v: (-units[v], v)))
    return PolicyEntry(units, sum(units), rank)


def make_snapshot(version, created_ns, published_ns, source, buckets, buy_units, sell_units) -> PolicySnapshot:
    buckets = tuple(buckets)
    bucket_of = tuple(min(range(len(buckets)), key=lambda i: (abs(buckets[i] - q), i))
                      for q in range(buckets[-1] + 1))
    return PolicySnapshot(version, created_ns, published_ns, source, buckets, bucket_of,
                          (tuple(make_entry(u) for u in buy_units), tuple(make_entry(u) for u in sell_units)))


class PolicyStore:
    """Single-writer store. Readers take one reference to an immutable snapshot,

    so a decision never observes a half-updated policy. (In CPython the
    reference swap is atomic; a native port would use an atomic pointer.)
    """

    def __init__(self):
        self.current: Optional[PolicySnapshot] = None
        self._lock = threading.Lock()  # serializes writers only
        self.published = 0

    def publish(self, snap: PolicySnapshot) -> bool:
        with self._lock:
            cur = self.current
            if cur is not None and snap.version <= cur.version:
                return False  # never roll back to an older policy
            self.current = snap
            self.published += 1
            return True
