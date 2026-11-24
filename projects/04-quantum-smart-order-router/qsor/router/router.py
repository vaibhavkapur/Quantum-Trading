"""Deterministic route decision (the hot path).

`SnapshotRouter.decide` only reads a precomputed policy snapshot and live
per-venue caps; it never calls a solver. If the snapshot is missing or older
than `max_age_ns` it falls back to best-price-first on the live view.
"""
from __future__ import annotations

from ..optimizer import classical
from ..optimizer.problem import build_problem


def best_price_alloc(side, qty, caps, best_px):
    order = sorted((v for v in range(len(caps)) if caps[v] > 0 and best_px[v] is not None),
                   key=lambda v: (best_px[v] * side, v))
    alloc = [0] * len(caps)
    for v in order:
        if qty <= 0:
            break
        take = caps[v] if caps[v] < qty else qty
        alloc[v] = take
        qty -= take
    return alloc


class SnapshotRouter:
    def __init__(self, store, max_age_ns: int):
        self.store = store
        self.max_age_ns = max_age_ns
        self.decisions = 0
        self.stale = 0
        self.age_ns_sum = 0
        self.last_version = -1

    def decide(self, side, qty, caps, best_px, now_ns, ctx=None):
        self.decisions += 1
        snap = self.store.current  # single read of an immutable snapshot
        if snap is None or now_ns - snap.published_ns > self.max_age_ns:
            self.stale += 1
            return best_price_alloc(side, qty, caps, best_px)
        self.age_ns_sum += now_ns - snap.created_ns
        self.last_version = snap.version
        e = snap.lookup(side, qty)
        tot = e.total
        if tot == 0:
            return best_price_alloc(side, qty, caps, best_px)
        w = e.weights
        alloc = [0] * len(caps)
        left = qty
        for v in e.rank:
            if w[v] == 0:
                break
            a = qty * w[v] // tot
            if a > caps[v]:
                a = caps[v]
            alloc[v] = a
            left -= a
        if left > 0:  # rounding remainder and capped/down venues spill over in rank order
            for v in e.rank:
                room = caps[v] - alloc[v]
                if room > 0:
                    a = room if room < left else left
                    alloc[v] += a
                    left -= a
                    if left == 0:
                        break
        return alloc


class LiveRouter:
    """Reference router that solves greedily on the hot path with the live view."""

    def __init__(self, model):
        self.model = model
        self.decisions = 0
        self.stale = 0
        self.age_ns_sum = 0
        self.last_version = -1

    def decide(self, side, qty, caps, best_px, now_ns, ctx=None):
        self.decisions += 1
        ladders, ref = ctx
        p = build_problem(side, qty, ladders, caps, ref, self.model, None)
        return list(classical.greedy(p))
