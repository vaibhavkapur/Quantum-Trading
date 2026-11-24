import math

import pytest

from qarb.core.book import OrderBook, QuoteEvent
from qarb.core.symbols import canonical_currency, format_market, orient, parse_market
from qarb.graph.build import build_graph, edges_from_book, reference_values

from conftest import book


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("BTC-USD", ("BTC", "USD")),
        ("btc_usdt", ("BTC", "USDT")),
        ("ETH/BTC", ("ETH", "BTC")),
        ("XBTUSDT", ("BTC", "USDT")),
        ("XBTUSD", ("BTC", "USD")),
        ("USDTUSD", ("USDT", "USD")),
        ("SOLETH", ("SOL", "ETH")),
        ("ETHXBT", ("ETH", "BTC")),
    ],
)
def test_parse_market(raw, expected):
    assert parse_market(raw, known=["SOL"]) == expected


def test_symbol_roundtrip_all_styles():
    for style in ("slash", "dash", "underscore_lower", "concat_xbt"):
        for base, quote in (("BTC", "USD"), ("SOL", "BTC"), ("USDT", "USD"), ("ETH", "USDT")):
            assert parse_market(format_market(base, quote, style), known=["SOL"]) == (base, quote)


def test_symbol_errors():
    with pytest.raises(ValueError):
        parse_market("BTC-USD-PERP")
    with pytest.raises(ValueError):
        parse_market("???")
    with pytest.raises(ValueError):
        canonical_currency("")
    with pytest.raises(ValueError):
        parse_market("XBT/BTC")  # same currency after aliasing


def test_orient_is_order_independent():
    assert orient("USD", "BTC") == orient("BTC", "USD") == ("BTC", "USD")
    assert orient("ETH", "BTC") == ("ETH", "BTC")
    assert orient("USDT", "USD") == ("USDT", "USD")


def test_book_validation():
    book("v", "BTC", "USD", 99, 100).validate()
    with pytest.raises(ValueError):
        book("v", "BTC", "USD", 101, 100).validate()  # crossed
    with pytest.raises(ValueError):
        OrderBook("v", "BTC", "USD", ((99, 1), (99.5, 1)), ((100, 1),), 0).validate()  # unsorted bids
    with pytest.raises(ValueError):
        OrderBook("v", "BTC", "USD", ((99, 0),), ((100, 1),), 0).validate()  # empty level


def test_event_json_roundtrip():
    ev = QuoteEvent(3, 100, 130, "beta", "XBTUSD", ((99.0, 1.5),), ((100.0, 2.0),), 7.5)
    again = QuoteEvent.from_json(ev.to_json())
    assert again == ev
    b = again.to_book()
    assert (b.base, b.quote, b.fee, b.ts) == ("BTC", "USD", 7.5e-4, 100)


def test_edge_conversion_walks_depth_and_charges_fee():
    b = OrderBook("v", "BTC", "USD", ((99.0, 1.0), (98.0, 2.0)), ((100.0, 1.0), (102.0, 1.0)), 0, fee=0.01)
    buy, sell = edges_from_book(b)
    assert (buy.src, buy.dst, sell.src, sell.dst) == ("USD", "BTC", "BTC", "USD")
    # 151 USD buys 1 BTC at 100 and 0.5 BTC at 102, less 1% fee
    out, filled, worst = buy.convert(151.0)
    assert out == pytest.approx(1.5 * 0.99) and filled == pytest.approx(151.0) and worst == pytest.approx(1 / 102)
    # selling 2 BTC: 1 at 99, 1 at 98
    out, filled, _ = sell.convert(2.0)
    assert out == pytest.approx((99 + 98) * 0.99) and filled == 2.0
    # more than total depth: partial fill
    out, filled, _ = sell.convert(10.0)
    assert filled == pytest.approx(3.0) and out == pytest.approx((99 + 2 * 98) * 0.99)
    # IOC limit refuses the second level
    out, filled, _ = sell.convert(2.0, min_rate=98.5)
    assert filled == pytest.approx(1.0)
    # already-consumed liquidity is skipped
    out, filled, _ = sell.convert(1.0, consumed=1.0)
    assert out == pytest.approx(98 * 0.99)
    assert sell.weight(1.0) == pytest.approx(-math.log(99 * 0.99))
    assert sell.weight(1e9) > sell.weight(1.0)  # unfilled input is penalized


def test_reference_values_and_graph(triangle_books):
    ref = reference_values(triangle_books, "USD")
    assert ref["USD"] == 1.0 and ref["BTC"] == pytest.approx(99.5) and ref["ETH"] == pytest.approx(5.025)
    g = build_graph(triangle_books, now=0, numeraire="USD")
    assert g.currencies == ["BTC", "ETH", "USD"]
    assert set(g.edges) == {("USD", "BTC"), ("BTC", "USD"), ("BTC", "ETH"), ("ETH", "BTC"), ("ETH", "USD"), ("USD", "ETH")}
    W = g.weight_matrix(10.0)
    i = g.index
    cyc = W[i["USD"], i["BTC"]] + W[i["BTC"], i["ETH"]] + W[i["ETH"], i["USD"]]
    assert cyc == pytest.approx(-math.log(1.02))
    nxg = g.to_networkx(10.0)
    assert nxg.number_of_edges() == 6


def test_stale_and_venue_filters(triangle_books):
    stale = book("v3", "SOL", "USD", 9.9, 10.0, ts=0)
    fresh = [OrderBook(b.venue, b.base, b.quote, b.bids, b.asks, 1000, b.fee) for b in triangle_books]
    g = build_graph(fresh + [stale], now=1200, stale_timeout_ms=500)
    assert g.dropped_stale == 1 and "SOL" not in g.currencies
    g = build_graph(fresh + [stale], now=1200, stale_timeout_ms=None)
    assert "SOL" in g.currencies
    g = build_graph(fresh, now=1200, allowed_venues=["v1"])
    assert g.dropped_venue == 1 and ("ETH", "USD") not in g.edges


def test_currency_unreachable_from_numeraire_is_dropped():
    books = [book("v", "BTC", "USD", 99, 100), book("v", "SOL", "ADA", 300, 301)]
    g = build_graph(books, now=0)
    assert g.currencies == ["BTC", "USD"]
