"""Integer-money accounting and the single objective shared by every solver."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from decimal import Decimal, ROUND_CEILING, ROUND_FLOOR
import hashlib
import json
import math
from typing import Any

OBJECTIVE_VERSION = "reference-nav-cash-tracking-v1"


class InfeasibleError(ValueError):
    pass


class ResourceLimitError(ValueError):
    pass


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def integer(name: str, value: Any, minimum: int = 0) -> None:
    if type(value) is not int or not minimum <= value <= 10**12:
        raise ValueError(f"{name} must be an integer in [{minimum}, 10^12]")


def number(name: str, value: Any, minimum: float = 0, maximum: float = math.inf) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not minimum <= value <= maximum:
        raise ValueError(f"{name} must be finite and in [{minimum}, {maximum}]")


def fraction_cents(total: int, fraction: float, *, up: bool = False) -> int:
    return int((Decimal(total) * Decimal(str(fraction))).to_integral_value(rounding=ROUND_CEILING if up else ROUND_FLOOR))


def charge(notional: int, bps: int) -> int:
    return (notional * bps + 9999) // 10000


@dataclass(frozen=True)
class TaxLot:
    shares: int
    basis_cents: int

    def __post_init__(self):
        integer("tax lot shares", self.shares, 1)
        integer("tax lot basis_cents", self.basis_cents)


@dataclass(frozen=True)
class Asset:
    symbol: str
    price_cents: int
    current_shares: int
    lot_size: int
    target_weight: float
    max_trade_lots: int = 2
    max_weight: float = 1.0
    group: str = "default"
    min_trade_notional_cents: int = 0
    max_trade_notional_cents: int | None = None
    fee_bps: int = 0
    fixed_fee_cents: int = 0
    tax_rate_bps: int = 0
    tax_lots: tuple[TaxLot, ...] = ()

    def __post_init__(self):
        if not isinstance(self.symbol, str) or not self.symbol.strip() or len(self.symbol) > 40:
            raise ValueError("symbol must be a nonempty string of at most 40 characters")
        if not isinstance(self.group, str) or not self.group:
            raise ValueError("group must be a nonempty string")
        for name in ("current_shares", "max_trade_lots", "min_trade_notional_cents", "fee_bps", "fixed_fee_cents", "tax_rate_bps"):
            integer(name, getattr(self, name))
        integer("price_cents", self.price_cents, 1)
        integer("lot_size", self.lot_size, 1)
        if self.max_trade_notional_cents is not None:
            integer("max_trade_notional_cents", self.max_trade_notional_cents)
        if self.max_trade_lots > 128:
            raise ResourceLimitError("max_trade_lots is capped at 128 per asset")
        if self.tax_rate_bps > 10000 or self.fee_bps > 10000:
            raise ValueError("fee_bps and tax_rate_bps cannot exceed 10000")
        number("target_weight", self.target_weight, 0, 1)
        number("max_weight", self.max_weight, 0, 1)
        if self.tax_lots and sum(lot.shares for lot in self.tax_lots) != self.current_shares:
            raise ValueError("FIFO tax lots must exactly cover current_shares")
        if self.tax_rate_bps and self.current_shares and not self.tax_lots:
            raise ValueError("tax_lots are required when taxing existing shares")


@dataclass(frozen=True)
class Portfolio:
    assets: tuple[Asset, ...]
    cash_cents: int
    cash_reserve_cents: int = 0
    cash_reserve_weight: float = 0.0
    group_caps: dict[str, float] = field(default_factory=dict)
    turnover_weight: float = 0.01
    cost_weight: float = 1.0
    schema_version: int = 1
    timestamp: str = "2026-01-01T00:00:00Z"

    def __post_init__(self):
        if not self.assets or len(self.assets) > 32:
            raise ValueError("provide between 1 and 32 assets")
        if len({a.symbol for a in self.assets}) != len(self.assets):
            raise ValueError("asset symbols must be unique")
        integer("cash_cents", self.cash_cents)
        integer("cash_reserve_cents", self.cash_reserve_cents)
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError("unsupported portfolio schema_version")
        from datetime import datetime
        try:
            dt = datetime.fromisoformat(self.timestamp.replace("Z", "+00:00"))
            if dt.utcoffset() is None or dt.utcoffset().total_seconds() != 0:
                raise ValueError()
        except (ValueError, TypeError, AttributeError):
            raise ValueError("timestamp must be an ISO-8601 UTC timestamp") from None
        number("cash_reserve_weight", self.cash_reserve_weight, 0, 1)
        number("turnover_weight", self.turnover_weight, 0, 1000000)
        number("cost_weight", self.cost_weight, 0, 1000000)
        if sum(Decimal(str(a.target_weight)) for a in self.assets) > Decimal(1):
            raise ValueError("target weights cannot sum above one")
        for group, cap in self.group_caps.items():
            if group not in {a.group for a in self.assets}:
                raise ValueError(f"unknown exposure group: {group}")
            number(f"group cap {group}", cap, 0, 1)
        if not 0 < self.nav_cents <= 10**12:
            raise ValueError("initial NAV must be positive and <= 10^12 cents")
        if self.reserve_cents > self.nav_cents:
            raise InfeasibleError("cash reserve exceeds initial NAV")

    @property
    def nav_cents(self) -> int:
        return self.cash_cents + sum(a.price_cents * a.current_shares for a in self.assets)

    @property
    def reserve_cents(self) -> int:
        return max(self.cash_reserve_cents, fraction_cents(self.nav_cents, self.cash_reserve_weight, up=True))

    @property
    def target_cash_weight(self) -> float:
        return float(1 - sum(Decimal(str(a.target_weight)) for a in self.assets))

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> Portfolio:
        if not isinstance(data, dict):
            raise ValueError("portfolio must be an object")
        try:
            raw = dict(data)
            raw["assets"] = tuple(Asset(**{**a, "tax_lots": tuple(TaxLot(**lot) for lot in a.get("tax_lots", []))}) for a in raw["assets"])
            return cls(**raw)
        except (TypeError, KeyError, AttributeError) as exc:
            raise ValueError(f"invalid portfolio: {exc}") from exc


@dataclass(frozen=True)
class Choice:
    delta_lots: int
    shares: int
    value_cents: int
    notional_cents: int
    fee_cents: int
    tax_cents: int
    cash_delta_cents: int


def make_choice(asset: Asset, delta: int) -> Choice:
    traded_shares = delta * asset.lot_size
    notional = abs(traded_shares) * asset.price_cents
    fee = asset.fixed_fee_cents + charge(notional, asset.fee_bps) if delta else 0
    remaining, gain = max(0, -traded_shares), 0
    for lot in asset.tax_lots:
        used = min(remaining, lot.shares)
        gain += used * max(0, asset.price_cents - lot.basis_cents)
        remaining -= used
    tax = charge(gain, asset.tax_rate_bps)
    shares = asset.current_shares + traded_shares
    return Choice(delta, shares, shares * asset.price_cents, notional, fee, tax, -traded_shares * asset.price_cents - fee - tax)


@dataclass
class Problem:
    portfolio: Portfolio
    choices: tuple[tuple[Choice, ...], ...] = field(init=False)

    def __post_init__(self):
        domains = []
        for asset in self.portfolio.assets:
            options = []
            for delta in range(-min(asset.max_trade_lots, asset.current_shares // asset.lot_size), asset.max_trade_lots + 1):
                c = make_choice(asset, delta)
                if c.value_cents > fraction_cents(self.portfolio.nav_cents, asset.max_weight):
                    continue
                if delta and c.notional_cents < asset.min_trade_notional_cents:
                    continue
                if asset.max_trade_notional_cents is not None and c.notional_cents > asset.max_trade_notional_cents:
                    continue
                options.append(c)
            if not options:
                raise InfeasibleError(f"no locally feasible order for {asset.symbol}")
            domains.append(tuple(options))
        self.choices = tuple(domains)

    @property
    def combinations(self) -> int:
        return math.prod(map(len, self.choices))

    def evaluate(self, indices: tuple[int, ...] | list[int]) -> dict:
        if len(indices) != len(self.choices) or any(type(j) is not int or not 0 <= j < len(opts) for j, opts in zip(indices, self.choices)):
            raise ValueError("invalid choice indices")
        p = self.portfolio
        selected = [opts[j] for opts, j in zip(self.choices, indices)]
        cash = p.cash_cents + sum(c.cash_delta_cents for c in selected)
        fees, taxes = sum(c.fee_cents for c in selected), sum(c.tax_cents for c in selected)
        nav_after = p.nav_cents - fees - taxes
        tracking = sum((c.value_cents / p.nav_cents - a.target_weight)**2 for a, c in zip(p.assets, selected))
        tracking += (cash / p.nav_cents - p.target_cash_weight)**2
        turnover = sum(c.notional_cents for c in selected) / p.nav_cents
        cost = (fees + taxes) / p.nav_cents
        objective = tracking + p.turnover_weight * turnover + p.cost_weight * cost
        groups = {g: sum(c.value_cents for a, c in zip(p.assets, selected) if a.group == g) for g in sorted({a.group for a in p.assets})}
        violations = []
        if cash < p.reserve_cents:
            violations.append("cash_reserve")
        for group, cap in p.group_caps.items():
            if groups[group] > fraction_cents(p.nav_cents, cap):
                violations.append(f"group_cap:{group}")
        rows = []
        for a, c in zip(p.assets, selected):
            rows.append({"symbol": a.symbol, "group": a.group, "before_shares": a.current_shares,
                         "after_shares": c.shares, "delta_lots": c.delta_lots, "lot_size": a.lot_size,
                         "side": "buy" if c.delta_lots > 0 else "sell" if c.delta_lots < 0 else "hold",
                         "trade_shares": abs(c.delta_lots) * a.lot_size, "price_cents": a.price_cents,
                         "notional_cents": c.notional_cents, "fee_cents": c.fee_cents, "tax_cents": c.tax_cents,
                         "before_weight": a.current_shares * a.price_cents / p.nav_cents,
                         "reference_nav_weight": c.value_cents / p.nav_cents,
                         "post_nav_weight": c.value_cents / nav_after if nav_after > 0 else None,
                         "target_weight": a.target_weight})
        orders = []
        running_cash = p.cash_cents
        # Instantaneous fixed-price settlement: cash-raising sales execute first.
        # This prevents a net-feasible basket from temporarily borrowing cash.
        fills = sorted(((r, c) for r, c in zip(rows, selected) if c.delta_lots),
                       key=lambda pair: (-pair[1].cash_delta_cents, pair[0]["symbol"]))
        for sequence, (row, choice) in enumerate(fills, 1):
            before_cash = running_cash
            running_cash += choice.cash_delta_cents
            orders.append({**row, "sequence": sequence, "cash_before_cents": before_cash,
                           "cash_after_cents": running_cash, "settlement": "simulated_instantaneous"})
        return {"objective_version": OBJECTIVE_VERSION, "objective": objective, "tracking_error": tracking,
                "turnover": turnover, "cost_fraction": cost, "fees_cents": fees, "taxes_cents": taxes,
                "initial_nav_cents": p.nav_cents, "post_nav_cents": nav_after, "cash_before_cents": p.cash_cents,
                "cash_after_cents": cash, "cash_reserve_cents": p.reserve_cents,
                "cash_usage_cents": p.cash_cents - cash, "group_exposure_cents": groups,
                "conservation_residual_cents": p.nav_cents - cash - sum(c.value_cents for c in selected) - fees - taxes,
                "feasible": not violations, "violations": violations, "holdings": rows,
                "simulated_orders": orders}

