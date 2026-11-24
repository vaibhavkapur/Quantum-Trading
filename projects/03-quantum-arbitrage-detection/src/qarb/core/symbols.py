"""Canonical currency and market symbols.

Venues spell the same instrument differently ("BTC-USD", "XBTUSDT", "btc_usd").
Everything downstream of the adapters uses canonical upper-case currency codes
and markets oriented as (base, quote).
"""

from __future__ import annotations

ALIASES = {
    "XBT": "BTC",
    "XDG": "DOGE",
    "BCC": "BCH",
    "WETH": "ETH",
}

# Higher priority = more "quote-like". In a pair, the quote currency is the one
# appearing earlier in this list; ties among unlisted currencies break alphabetically.
QUOTE_PRIORITY = ["USD", "USDT", "USDC", "EUR", "GBP", "JPY", "BTC", "ETH"]

_SEPARATORS = ("/", "-", "_", ":")


def canonical_currency(code: str) -> str:
    c = code.strip().upper()
    if not c or not c.isalnum():
        raise ValueError(f"invalid currency code: {code!r}")
    return ALIASES.get(c, c)


def _quote_rank(cur: str) -> tuple[int, str]:
    try:
        return (QUOTE_PRIORITY.index(cur), cur)
    except ValueError:
        return (len(QUOTE_PRIORITY), cur)


def orient(a: str, b: str) -> tuple[str, str]:
    """Return (base, quote) for an unordered currency pair."""
    a, b = canonical_currency(a), canonical_currency(b)
    if a == b:
        raise ValueError(f"degenerate pair {a}/{b}")
    return (b, a) if _quote_rank(a) < _quote_rank(b) else (a, b)


def parse_market(raw: str, known: list[str] | None = None) -> tuple[str, str]:
    """Parse a venue-specific market symbol into canonical (base, quote).

    Separated forms ("BTC-USD", "btc_usdt", "ETH/BTC") are split on the separator.
    Concatenated forms ("XBTUSDT") are split by matching the longest known
    quote-currency suffix; `known` extends the default quote list.
    """
    s = raw.strip().upper()
    for sep in _SEPARATORS:
        if sep in s:
            parts = s.split(sep)
            if len(parts) != 2:
                raise ValueError(f"cannot parse market symbol {raw!r}")
            base, quote = canonical_currency(parts[0]), canonical_currency(parts[1])
            break
    else:
        candidates = set(QUOTE_PRIORITY) | set(ALIASES) | {c.upper() for c in (known or [])}
        for suffix in sorted(candidates, key=lambda c: (-len(c), c)):
            if s.endswith(suffix) and len(s) > len(suffix):
                base, quote = canonical_currency(s[: -len(suffix)]), canonical_currency(suffix)
                break
        else:
            raise ValueError(f"cannot parse market symbol {raw!r}")
    if base == quote:
        raise ValueError(f"degenerate market {raw!r}")
    return base, quote


def format_market(base: str, quote: str, style: str = "slash") -> str:
    """Render a market the way a venue with the given style would spell it."""
    if style == "slash":
        return f"{base}/{quote}"
    if style == "dash":
        return f"{base}-{quote}"
    if style == "underscore_lower":
        return f"{base}_{quote}".lower()
    if style == "concat_xbt":
        inv = {"BTC": "XBT"}
        return f"{inv.get(base, base)}{inv.get(quote, quote)}"
    raise ValueError(f"unknown symbol style {style!r}")
