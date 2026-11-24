from conftest import VenueCfg, flat_dataset
from qsor.core import ASK, BID, BUY, MS, SELL, Order
from qsor.exchange_sim.venue import VenueSim


def venue(ds, **kw):
    return VenueSim(0, VenueCfg(name="V0", **kw), ds, halflife_ms=100)


def kinds(reps):
    return [r.kind for r in reps]


def test_ioc_walks_book_then_cancels_remainder(ds):
    reps = venue(ds).new_order(Order(1, 0, 0, BUY, 8, 10002, "IOC"), 5 * MS)
    assert kinds(reps) == ["ACK", "FILL", "FILL"]  # fully filled: nothing to cancel
    assert [(r.px, r.qty, r.leaves) for r in reps[1:]] == [(10001, 5, 3), (10002, 3, 0)]
    # the limit makes only two levels (10 lots) eligible; the other 2 lots are cancelled
    reps = venue(ds).new_order(Order(2, 0, 0, BUY, 12, 10002, "IOC"), 5 * MS)
    assert kinds(reps) == ["ACK", "FILL", "FILL", "CANCELED"]
    assert sum(r.qty for r in reps if r.kind == "FILL") == 10


def test_sell_ioc_respects_limit(ds):
    reps = venue(ds).new_order(Order(1, 0, 0, SELL, 7, 9999, "IOC"), 0)
    assert [(r.px, r.qty) for r in reps if r.kind == "FILL"] == [(9999, 5)]


def test_own_fills_deplete_and_replenish(ds):
    v = venue(ds)
    v.new_order(Order(1, 0, 0, BUY, 4, 10001, "IOC"), 0)
    assert v.book(ASK, 0)[1][0] == 1
    assert v.book(ASK, 100 * MS)[1][0] == 3      # one half-life: 2 of 4 lots are back
    assert v.book(ASK, 2000 * MS)[1][0] == 5
    assert v.book(BID, 0)[1][0] == 5             # other side untouched


def test_fade_depletes_top_of_book_from_its_start_time(ds):
    v = venue(ds)
    v.deplete(ASK, 10 * MS, 7.0)
    assert v.book(ASK, 9 * MS)[1][:2] == [5, 5]
    assert v.book(ASK, 10 * MS)[1][:2] == [0, 3]


def test_reject_when_down_and_when_throttled():
    ds = flat_dataset()
    ds.status[10:20, 0] = 1
    v = venue(ds, throttle_max_orders=2, throttle_window_ms=50)
    assert v.new_order(Order(1, 0, 0, BUY, 1, 10001, "IOC"), 150 * MS)[0].reason == "VENUE_DOWN"
    t = 300 * MS
    assert kinds(v.new_order(Order(2, 0, 0, BUY, 1, 10001, "IOC"), t))[0] == "ACK"
    assert kinds(v.new_order(Order(3, 0, 0, BUY, 1, 10001, "IOC"), t + MS))[0] == "ACK"
    r = v.new_order(Order(4, 0, 0, BUY, 1, 10001, "IOC"), t + 2 * MS)
    assert kinds(r) == ["REJECT"] and r[0].reason == "THROTTLE"
    assert kinds(v.new_order(Order(5, 0, 0, BUY, 1, 10001, "IOC"), t + 60 * MS))[0] == "ACK"


def test_resting_order_queue_and_passive_fill():
    ds = flat_dataset(flow=2)
    v = venue(ds)
    o = Order(1, 0, 0, BUY, 3, 9999, "DAY")
    reps = v.new_order(o, 0)
    assert kinds(reps) == ["ACK"] and reps[0].queue_ahead == 5
    assert v.on_tick(1, 10 * MS) == {} and v.queue[1] == 3   # flow 2 eats queue 5 -> 3
    assert v.on_tick(2, 20 * MS) == {} and v.queue[1] == 1
    out = v.on_tick(3, 30 * MS)                               # 1 queue + 1 fill
    assert [(r.kind, r.qty, r.px, r.liquidity) for r in out[1]] == [("FILL", 1, 9999, "M")]
    out = v.on_tick(4, 40 * MS)
    assert out[1][0].qty == 2 and out[1][0].leaves == 0 and not v.resting


def test_resting_order_fills_when_market_crosses_it():
    ds = flat_dataset()
    ds.px[ASK][5:, 0, :] -= 2   # ask drops to 9999
    v = venue(ds)
    v.new_order(Order(1, 0, 0, BUY, 3, 9999, "DAY"), 0)
    assert v.on_tick(4, 40 * MS) == {}
    assert v.on_tick(5, 50 * MS)[1][0].qty == 3


def test_cancel_replace_and_cancel_on_disconnect():
    ds = flat_dataset(flow=0)
    v = venue(ds)
    v.new_order(Order(1, 0, 0, BUY, 3, 9998, "DAY"), 0)
    rep = v.replace(1, 9999, MS)[0]
    assert rep.kind == "REPLACED" and rep.queue_ahead == 5 and v.resting[1].px == 9999
    assert kinds(v.cancel(1, 2 * MS)) == ["CANCELED"]
    assert kinds(v.cancel(1, 3 * MS)) == ["CANCEL_REJECT"]
    assert kinds(v.replace(1, 9999, 3 * MS)) == ["REPLACE_REJECT"]
    v.new_order(Order(2, 0, 0, BUY, 3, 9999, "DAY"), 4 * MS)
    ds.status[7, 0] = 1
    out = v.on_tick(7, 70 * MS)
    assert out[2][0].kind == "CANCELED" and out[2][0].reason == "CANCEL_ON_DISCONNECT" and not v.resting
