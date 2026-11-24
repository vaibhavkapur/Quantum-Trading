"""Order books and quote events."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator

from .symbols import parse_market

Level = tuple[float, float]  # (price in quote per base, size in base)


@dataclass(frozen=True)
class OrderBook:
    venue: str
    base: str
    quote: str
    bids: tuple[Level, ...]  # best (highest) first
    asks: tuple[Level, ...]  # best (lowest) first
    ts: int  # exchange timestamp, ms
    fee: float = 0.0  # taker fee rate charged on the received asset

    @property
    def key(self) -> tuple[str, str, str]:
        return (self.venue, self.base, self.quote)

    def mid(self) -> float | None:
        if not self.bids or not self.asks:
            return None
        return 0.5 * (self.bids[0][0] + self.asks[0][0])

    def validate(self) -> None:
        for name, side, sign in (("bids", self.bids, -1), ("asks", self.asks, 1)):
            prev = None
            for price, size in side:
                if not (price > 0 and size > 0):
                    raise ValueError(f"{self.key}: non-positive level in {name}")
                if prev is not None and sign * (price - prev) < 0:
                    raise ValueError(f"{self.key}: {name} not sorted best-first")
                prev = price
        if self.bids and self.asks and self.bids[0][0] >= self.asks[0][0]:
            raise ValueError(f"{self.key}: crossed book")
        if not 0 <= self.fee < 1:
            raise ValueError(f"{self.key}: fee out of range")


@dataclass(frozen=True)
class QuoteEvent:
    """A full-depth book snapshot for one venue market.

    `exchange_ts` is when the venue produced it, `recv_ts` when we received it.
    Replay orders by (recv_ts, seq), so arrival can be out of exchange order.
    """

    seq: int
    exchange_ts: int
    recv_ts: int
    venue: str
    symbol: str  # venue-specific raw symbol
    bids: tuple[Level, ...]
    asks: tuple[Level, ...]
    fee_bps: float

    def to_book(self, known: list[str] | None = None) -> OrderBook:
        base, quote = parse_market(self.symbol, known)
        return OrderBook(self.venue, base, quote, self.bids, self.asks, self.exchange_ts, self.fee_bps * 1e-4)

    def to_json(self) -> str:
        return json.dumps(
            {
                "seq": self.seq,
                "exchange_ts": self.exchange_ts,
                "recv_ts": self.recv_ts,
                "venue": self.venue,
                "symbol": self.symbol,
                "fee_bps": self.fee_bps,
                "bids": [list(l) for l in self.bids],
                "asks": [list(l) for l in self.asks],
            },
            separators=(",", ":"),
        )

    @staticmethod
    def from_json(line: str) -> "QuoteEvent":
        d = json.loads(line)
        return QuoteEvent(
            seq=int(d["seq"]),
            exchange_ts=int(d["exchange_ts"]),
            recv_ts=int(d["recv_ts"]),
            venue=str(d["venue"]),
            symbol=str(d["symbol"]),
            bids=tuple((float(p), float(s)) for p, s in d["bids"]),
            asks=tuple((float(p), float(s)) for p, s in d["asks"]),
            fee_bps=float(d["fee_bps"]),
        )


def write_events(path: Path, events: Iterable[QuoteEvent]) -> None:
    with open(path, "w") as f:
        for ev in events:
            f.write(ev.to_json() + "\n")


def read_events(path: Path) -> Iterator[QuoteEvent]:
    with open(path) as f:
        for line in f:
            if line.strip():
                yield QuoteEvent.from_json(line)
