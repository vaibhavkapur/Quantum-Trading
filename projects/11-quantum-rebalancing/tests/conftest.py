import pytest
from rebalancing.model import Portfolio


@pytest.fixture
def known():
    # NAV=100000 cents; (-2,+4) lots reaches (40%,40%,20% cash) exactly.
    return Portfolio.from_dict({"cash_cents": 40000, "turnover_weight": 0, "cost_weight": 0,
        "assets": [
            {"symbol": "A", "price_cents": 10000, "current_shares": 6, "lot_size": 1, "target_weight": 0.4, "max_trade_lots": 4},
            {"symbol": "B", "price_cents": 10000, "current_shares": 0, "lot_size": 1, "target_weight": 0.4, "max_trade_lots": 4}]})


@pytest.fixture
def tiny():
    return Portfolio.from_dict({"cash_cents": 10000, "cash_reserve_cents": 1000,
        "assets": [
            {"symbol": "A", "price_cents": 4000, "current_shares": 2, "lot_size": 1, "target_weight": 0.3, "max_trade_lots": 1, "fee_bps": 7, "fixed_fee_cents": 5},
            {"symbol": "B", "price_cents": 3000, "current_shares": 2, "lot_size": 1, "target_weight": 0.5, "max_trade_lots": 1, "fee_bps": 11}]})

