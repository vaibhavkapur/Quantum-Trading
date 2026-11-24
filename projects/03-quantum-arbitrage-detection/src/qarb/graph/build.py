"""Normalized currency conversion graph with fee- and depth-aware edges.

Each fresh order book contributes two directed edges:

    quote -> base   (buy base, walks the asks)
    base  -> quote  (sell base, walks the bids)

An edge is a ladder of levels `(rate, cap_in)`: `rate` is units of dst received
per unit of src before fees, `cap_in` the src amount available at that rate.
The effective executable rate for an input amount q is

    R(q) = convert(q) / q,        edge cost  w(q) = -log R(q).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from statistics import median
from typing import Iterable

import numpy as np

from ..core.book import OrderBook

EdgeKey = tuple[str, str, str, str]  # (venue, base, quote, side)


@dataclass(frozen=True)
class Edge:
    src: str
    dst: str
    venue: str
    base: str
    quote: str
    side: str  # "buy" (quote->base) or "sell" (base->quote)
    levels: tuple[tuple[float, float], ...]  # (rate, cap_in), best rate first
    fee: float
    ts: int
    mid_rate: float  # dst per src at mid, no fees

    @property
    def key(self) -> EdgeKey:
        return (self.venue, self.base, self.quote, self.side)

    def trimmed(self, consumed: float = 0.0) -> list[tuple[float, float]]:
        """Levels left after `consumed` units of src have already been taken."""
        if consumed <= 0:
            return list(self.levels)
        out = []
        for rate, cap in self.levels:
            if consumed >= cap:
                consumed -= cap
                continue
            out.append((rate, cap - consumed))
            consumed = 0.0
        return out

    def top_rate(self, consumed: float = 0.0) -> float:
        lv = self.trimmed(consumed)
        return lv[0][0] * (1 - self.fee) if lv else 0.0

    def convert(self, amount_in: float, min_rate: float = 0.0, consumed: float = 0.0) -> tuple[float, float, float]:
        """Walk the ladder with `amount_in` of src.

        Returns (net dst received, src actually filled, worst raw rate used).
        Levels with raw rate below `min_rate` are not taken (IOC limit), so the
        fill may be partial.
        """
        remaining = amount_in
        out = 0.0
        worst = 0.0
        for rate, cap in self.trimmed(consumed):
            if remaining <= 0 or rate < min_rate:
                break
            take = min(remaining, cap)
            out += take * rate
            remaining -= take
            worst = rate
        return out * (1 - self.fee), amount_in - remaining, worst

    def weight(self, amount_in: float, consumed: float = 0.0) -> float:
        """-log of the effective rate for a full `amount_in`; inf if no liquidity.

        Unfilled input counts as received nothing, so thin edges are penalized.
        """
        if amount_in <= 0:
            r = self.top_rate(consumed)
        else:
            out, _, _ = self.convert(amount_in, consumed=consumed)
            r = out / amount_in
        return -math.log(r) if r > 0 else math.inf


def edges_from_book(book: OrderBook) -> list[Edge]:
    mid = book.mid()
    edges = []
    if book.asks:
        levels = tuple((1.0 / p, p * s) for p, s in book.asks)
        edges.append(
            Edge(book.quote, book.base, book.venue, book.base, book.quote, "buy", levels, book.fee, book.ts,
                 1.0 / mid if mid else levels[0][0])
        )
    if book.bids:
        levels = tuple((p, s) for p, s in book.bids)
        edges.append(
            Edge(book.base, book.quote, book.venue, book.base, book.quote, "sell", levels, book.fee, book.ts,
                 mid if mid else levels[0][0])
        )
    return edges


def reference_values(books: Iterable[OrderBook], numeraire: str) -> dict[str, float]:
    """Value of one unit of each currency in the numeraire, from cross-venue median mids.

    Used only to convert notionals (probe size, capital caps, P&L marks), never
    as an executable price.
    """
    mids: dict[tuple[str, str], list[float]] = {}
    for b in books:
        m = b.mid()
        if m:
            mids.setdefault((b.base, b.quote), []).append(m)
    pair_mid = {k: median(v) for k, v in mids.items()}
    value = {numeraire: 1.0}
    frontier = [numeraire]
    while frontier:
        nxt = []
        for (base, quote), m in sorted(pair_mid.items()):
            if quote in frontier and base not in value:
                value[base] = m * value[quote]
                nxt.append(base)
            elif base in frontier and quote not in value:
                value[quote] = value[base] / m
                nxt.append(quote)
        frontier = nxt
    return value


@dataclass
class MarketGraph:
    currencies: list[str]
    edges: dict[tuple[str, str], list[Edge]]
    ref: dict[str, float]
    numeraire: str
    now: int
    n_books: int = 0
    dropped_stale: int = 0
    dropped_venue: int = 0
    index: dict[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.index = {c: i for i, c in enumerate(self.currencies)}

    def probe_amount(self, cur: str, notional: float) -> float:
        return notional / self.ref[cur]

    def weight_matrix(self, probe_notional: float) -> np.ndarray:
        """W[i, j] = min over venues of -log R_ij(q) at the probe notional; inf if no edge."""
        n = len(self.currencies)
        W = np.full((n, n), np.inf)
        for (src, dst), es in self.edges.items():
            if src not in self.ref:
                continue
            q = self.probe_amount(src, probe_notional)
            W[self.index[src], self.index[dst]] = min(e.weight(q) for e in es)
        return W

    def mid_weight_matrix(self) -> np.ndarray:
        """Weights from mid prices with no fees, spread or depth (the naive model)."""
        n = len(self.currencies)
        W = np.full((n, n), np.inf)
        for (src, dst), es in self.edges.items():
            W[self.index[src], self.index[dst]] = min(-math.log(e.mid_rate) for e in es)
        return W

    def to_networkx(self, probe_notional: float):
        import networkx as nx

        g = nx.MultiDiGraph()
        g.add_nodes_from(self.currencies)
        for (src, dst), es in self.edges.items():
            q = self.probe_amount(src, probe_notional) if src in self.ref else 0.0
            for e in es:
                g.add_edge(src, dst, key=e.venue, weight=e.weight(q), venue=e.venue, side=e.side)
        return g


def build_graph(
    books: Iterable[OrderBook],
    now: int,
    numeraire: str = "USD",
    stale_timeout_ms: int | None = None,
    allowed_venues: list[str] | None = None,
) -> MarketGraph:
    """Build the conversion graph from the latest book per venue market.

    Books older than `stale_timeout_ms` relative to `now` are dropped: a stale
    quote is not an executable price.
    """
    fresh: list[OrderBook] = []
    n_books = dropped_stale = dropped_venue = 0
    for b in books:
        n_books += 1
        if allowed_venues is not None and b.venue not in allowed_venues:
            dropped_venue += 1
            continue
        if stale_timeout_ms is not None and now - b.ts > stale_timeout_ms:
            dropped_stale += 1
            continue
        fresh.append(b)
    fresh.sort(key=lambda b: b.key)
    ref = reference_values(fresh, numeraire)
    edges: dict[tuple[str, str], list[Edge]] = {}
    currencies: set[str] = set()
    for b in fresh:
        if b.base not in ref or b.quote not in ref:
            continue  # not connected to the numeraire: cannot be valued
        for e in edges_from_book(b):
            edges.setdefault((e.src, e.dst), []).append(e)
            currencies.update((e.src, e.dst))
    return MarketGraph(sorted(currencies), edges, ref, numeraire, now, n_books, dropped_stale, dropped_venue)
