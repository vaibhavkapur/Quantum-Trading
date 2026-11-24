from dataclasses import replace
import math
import pytest
from rebalancing.model import Asset, InfeasibleError, Portfolio, Problem, TaxLot, charge, fraction_cents, make_choice


def test_known_optimum_accounting(known):
    problem = Problem(known)
    indices = tuple(next(i for i,c in enumerate(opts) if c.delta_lots == d) for opts,d in zip(problem.choices, (-2,4)))
    report = problem.evaluate(indices)
    assert report['objective'] == pytest.approx(0)
    assert report['cash_after_cents'] == 20000
    assert report['conservation_residual_cents'] == 0
    assert report['feasible']


def test_fees_fifo_tax_and_rounding():
    a = Asset('A', 1001, 7, 2, 0.5, fee_bps=7, fixed_fee_cents=3, tax_rate_bps=1500,
              tax_lots=(TaxLot(3, 900), TaxLot(4, 1100)))
    sell = make_choice(a, -2)
    assert sell.shares == 3  # odd-lot remainder survives
    assert sell.fee_cents == 6
    assert sell.tax_cents == 46  # ceil(3 * 101 * 15%); no loss offset
    assert sell.cash_delta_cents == 4004 - 6 - 46
    assert make_choice(a, 0).fee_cents == make_choice(a, 0).tax_cents == 0
    assert make_choice(a, 1).tax_cents == 0
    assert charge(1, 1) == 1
    assert fraction_cents(100, 0.29) == 29


def test_local_constraints_and_odd_lots():
    p = Portfolio((Asset('A', 100, 7, 2, 0.5, max_trade_lots=3, min_trade_notional_cents=300,
                         max_trade_notional_cents=400, max_weight=0.7),), cash_cents=300)
    assert [c.delta_lots for c in Problem(p).choices[0]] == [-2, 0]


def test_cash_and_group_cap_are_hard(tiny):
    p = replace(tiny, cash_reserve_cents=9000, group_caps={'default': 0.5})
    r = Problem(p).evaluate((2, 2))
    assert not r['feasible']
    assert set(r['violations']) == {'cash_reserve', 'group_cap:default'}
    assert r['conservation_residual_cents'] == 0


@pytest.mark.parametrize('field,value', [('price_cents',0),('price_cents',True),('lot_size',0),('current_shares',-1),
    ('target_weight',math.nan),('max_weight',1.1),('fee_bps',-1),('max_trade_lots',1000),('fixed_fee_cents',1.5)])
def test_invalid_asset(field,value):
    data = dict(symbol='A',price_cents=100,current_shares=1,lot_size=1,target_weight=0.5)
    data[field] = value
    with pytest.raises(ValueError):
        Asset(**data)


@pytest.mark.parametrize('change', [{'cash_cents':-1}, {'timestamp':'2026-01-01'}, {'timestamp':'2026-01-01T00:00:00+05:30'},
    {'schema_version':2}, {'schema_version':True}, {'cost_weight':float('inf')}, {'group_caps':{'typo':0.5}}])
def test_invalid_portfolio(tiny,change):
    with pytest.raises(ValueError):
        replace(tiny, **change)


def test_invalid_tax_and_targets(tiny):
    with pytest.raises(ValueError,match='exactly cover'):
        replace(tiny.assets[0], tax_lots=(TaxLot(1,100),))
    with pytest.raises(ValueError,match='tax_lots are required'):
        replace(tiny.assets[0], tax_rate_bps=10)
    with pytest.raises(ValueError,match='sum above'):
        replace(tiny, assets=tuple(replace(a,target_weight=0.6) for a in tiny.assets))
    with pytest.raises(ValueError,match='unique'):
        replace(tiny, assets=(tiny.assets[0],tiny.assets[0]))
    with pytest.raises(ValueError):
        Portfolio.from_dict({'assets':[], 'cash_cents':100, 'unknown':3})


def test_empty_domain_and_impossible_reserve(tiny):
    with pytest.raises(InfeasibleError):
        Problem(replace(tiny, assets=(replace(tiny.assets[0], max_weight=0, max_trade_lots=0),)))
    with pytest.raises(InfeasibleError):
        replace(tiny,cash_reserve_cents=tiny.nav_cents+1)


def test_roundtrip_and_index_validation(tiny):
    assert Portfolio.from_dict(tiny.to_dict()) == tiny
    for indices in [(0,), (3,0), (True,0), (-1,1)]:
        with pytest.raises(ValueError):
            Problem(tiny).evaluate(indices)


def test_sells_fund_buys_before_execution(known):
    # Reverse the input order, so naïve input-order execution would buy first.
    p = replace(known, assets=tuple(reversed(known.assets)), cash_cents=0)
    problem = Problem(p)
    indices = tuple(next(j for j,c in enumerate(opts) if c.delta_lots == d) for opts,d in zip(problem.choices,(2,-2)))
    report = problem.evaluate(indices)
    assert report['feasible']
    assert [o['side'] for o in report['simulated_orders']] == ['sell','buy']
    assert all(o['cash_after_cents'] >= 0 for o in report['simulated_orders'])
    assert report['simulated_orders'][-1]['cash_after_cents'] == report['cash_after_cents']

