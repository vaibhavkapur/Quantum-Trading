import pytest
from market_making.native import Book
from market_making.policy import Policy


def test_price_time_priority_partial_fills_and_cancel(config):
    with Book(config['execution']) as b:
        b.add(1,1,100,3)
        b.add(2,1,100,2,own=True)
        b.add(3,1,101,1)
        assert b.queue_ahead(2)==3
        assert b.add(4,-1,1,3,market=True)==3
        assert b.stats.inventory==0  # Better price, then external FIFO quantity.
        assert b.queue_ahead(2)==1
        b.add(5,-1,1,2,market=True)
        assert b.stats.inventory==1
        assert b.queue_ahead(2)==0
        assert b.fills_since(0)[0]['price']==100
        assert b.cancel(2)
        assert not b.cancel(2)
        assert b.stats.open_buy==0


def test_ask_priority_and_limit_price(config):
    with Book(config['execution']) as b:
        b.add(1,-1,102,1,own=True)
        b.add(2,-1,101,1,own=True)
        b.add(3,1,101,4)
        assert b.stats.inventory==-1
        assert b.stats.open_sell==1
        assert b.fills_since(0)[0]['order_id']==2


def test_fees_accounting_and_taker_liquidation(config):
    e=dict(config['execution'],tick_value=1.0,maker_fee_bps=-1.,taker_fee_bps=2.)
    with Book(e) as b:
        b.add(1,1,100,2,own=True)
        b.add(2,-1,1,2,market=True)
        assert b.stats.cash==pytest.approx(-199.98)
        assert b.stats.maker_fees==pytest.approx(-.02)
        b.add(3,1,99,10)
        b.flatten(1)
        assert b.stats.inventory==0
        assert b.stats.taker_fees==pytest.approx(.0396)
        assert b.stats.cash==pytest.approx(-2.0196)


def test_liquidation_exhausts_visible_depth_without_inventing_fills(config):
    with Book(config['execution']) as b:
        b.add(1,1,100,3,own=True)
        b.add(2,-1,1,3,market=True)
        b.add(3,1,99,1)
        b.flatten(1)
        assert b.stats.inventory==2
        assert b.stats.open_buy==b.stats.open_sell==0


def test_l2_queue_reconciliation(config):
    with Book(config['execution']) as b:
        frame={'bids':[[100,5]],'asks':[[102,5]]}
        b.reconcile(frame)
        b.add(1,1,100,2,own=True)
        assert b.queue_ahead(1)==5
        b.reconcile({'bids':[[100,8]],'asks':[[102,5]]})
        assert b.queue_ahead(1)==5  # Increases join the tail.
        b.reconcile({'bids':[[100,4]],'asks':[[102,5]]})
        assert b.queue_ahead(1)==4  # Newest external volume cancels first.
        b.reconcile({'bids':[[99,8]],'asks':[[102,5]]})
        assert b.queue_ahead(1)==0


def test_quotes_preserve_queue_and_soft_band(config):
    with Book(config['execution']) as b:
        b.reconcile({'bids':[[99,3]],'asks':[[101,3]]})
        p=Policy(1,0,4,1)
        b.quote(0,100,p,0)
        first=b.stats.cancels
        b.quote(1,100,p,0)
        assert b.stats.cancels==first
        b.add(1,-1,1,5,market=True)
        assert b.stats.inventory==2
        b.quote(2,100,p,0)
        b.add(2,-1,1,2,market=True)
        assert b.stats.inventory==4
        b.quote(3,100,p,0)
        assert b.stats.open_buy==0
        assert b.stats.open_sell==2


def test_hard_limit_counts_all_outstanding_orders(config):
    with Book(config['execution']) as b:
        b.add(1,1,100,7,own=True)
        with pytest.raises(ValueError,match='worst-case'):
            b.add(2,1,99,6,own=True)
        b.add(3,-1,1,7,market=True)
        with pytest.raises(ValueError,match='worst-case'):
            b.add(4,1,100,6,own=True)
        assert b.stats.inventory==7


def test_emergency_cancel_is_latched(config):
    with Book(config['execution']) as b:
        p=Policy(1,.2,8,1)
        b.quote(0,100,p,0)
        b.cancel_all(emergency=True)
        b.quote(1,100,p,0)
        assert b.stats.halted
        assert b.stats.open_buy==b.stats.open_sell==0
        with pytest.raises(ValueError,match='latched'):
            b.add(1,1,100,1,own=True)


@pytest.mark.parametrize('trigger',['loss','drawdown','volatility'])
def test_risk_triggers_cancel_quotes(config,trigger):
    e=dict(config['execution'],max_loss=.1,max_drawdown=.1)
    with Book(e) as b:
        p=Policy(1,.2,8,1)
        b.add(1,1,10000,1,own=True)
        b.add(2,-1,1,1,market=True)
        if trigger=='drawdown':
            b.quote(0,10100,p,0)
        mid=9900 if trigger=='loss' else 10000
        b.quote(1,mid,p,100 if trigger=='volatility' else 0)
        assert b.stats.halted and b.stats.open_buy==b.stats.open_sell==0


def test_post_only_stale_policy_and_self_trade(config):
    with Book(config['execution']) as b:
        b.add(1,1,99,1,own=True)
        with pytest.raises(ValueError,match='self-trade'):
            b.add(2,-1,98,1,own=True)
        assert b.stats.inventory==0
        b.cancel_all()
        b.add(3,-1,100,10)
        b.quote(0,102,Policy(1,0,8,2),0)
        assert b.stats.inventory==0  # Bid clamped below best ask.
        with pytest.raises(ValueError,match='stale'):
            b.quote(1,102,Policy(1,0,8,1),0)


@pytest.mark.parametrize('side,price,quantity',[(0,100,1),(1,0,1),(1,100,0),(1,-1,2),(1,100,1000001)])
def test_reject_invalid_orders(config,side,price,quantity):
    with Book(config['execution']) as b:
        with pytest.raises(ValueError):
            b.add(1,side,price,quantity)


def test_duplicate_id_and_closed_handle(config):
    b=Book(config['execution'])
    b.add(1,1,100,1)
    with pytest.raises(ValueError,match='duplicate'):
        b.add(1,1,99,1)
    b.close()
    b.close()
    with pytest.raises(RuntimeError,match='closed'):
        b.cancel(1)
