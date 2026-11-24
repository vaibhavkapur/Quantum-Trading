"""Simulated venue gateway: L2 book replay, matching, throttles, outages.

The displayed book is the replayed dataset minus liquidity removed by the
simulation itself:

* liquidity taken by our own fills (price-specific), and
* quote fade triggered by our fills on *other* venues (top-of-book).

Both replenish exponentially. Resting orders track a queue position that is
worked down by the dataset's exogenous aggressor flow.
"""
from __future__ import annotations

from collections import deque

from ..config import VenueCfg
from ..core import BUY, MS, ExecReport, Order, opp_book, own_book


class VenueSim:
    def __init__(self, idx: int, cfg: VenueCfg, ds, halflife_ms: float):
        self.idx, self.cfg, self.ds = idx, cfg, ds
        self.hl_ns = max(1.0, halflife_ms * MS)
        self.depl = ([], [])  # per book side: [t_ns, lots, px or None]
        self.resting = {}     # order id -> Order (insertion ordered)
        self.queue = {}       # order id -> lots ahead
        self._recent = deque()
        self._win_ns = int(cfg.throttle_window_ms * MS)

    # -- book ---------------------------------------------------------------
    def book(self, bside: int, t_ns: int):
        """Effective displayed (prices, sizes), best level first, at time t."""
        k = self.ds.step(t_ns)
        px = self.ds.px[bside][k, self.idx].tolist()
        sz = self.ds.sz[bside][k, self.idx].tolist()
        ev = self.depl[bside]
        if ev:
            top, by_px = 0.0, {}
            for te, lots, p in ev:
                if te > t_ns:
                    continue
                rem = lots * 0.5 ** ((t_ns - te) / self.hl_ns)
                if p is None:
                    top += rem
                else:
                    by_px[p] = by_px.get(p, 0.0) + rem
            for p, rem in by_px.items():
                if p in px:
                    i = px.index(p)
                    sz[i] = max(0, sz[i] - int(round(rem)))
            r = int(round(top))
            for i in range(len(sz)):
                if r <= 0:
                    break
                take = min(sz[i], r)
                sz[i] -= take
                r -= take
        return px, sz

    def deplete(self, bside: int, t_ns: int, lots: float, px=None) -> None:
        ev = self.depl[bside]
        horizon = t_ns - 12 * self.hl_ns
        if ev and ev[0][0] < horizon:
            ev[:] = [e for e in ev if e[0] >= horizon]
        ev.append((t_ns, lots, px))

    def is_down(self, t_ns: int) -> bool:
        return bool(self.ds.status[self.ds.step(t_ns), self.idx])

    # -- order entry --------------------------------------------------------
    def new_order(self, o: Order, t: int):
        if self.is_down(t):
            return [ExecReport(o.id, self.idx, "REJECT", t, leaves=0, reason="VENUE_DOWN")]
        rec = self._recent
        while rec and rec[0] <= t - self._win_ns:
            rec.popleft()
        if len(rec) >= self.cfg.throttle_max_orders:
            return [ExecReport(o.id, self.idx, "REJECT", t, leaves=0, reason="THROTTLE")]
        rec.append(t)

        bs = opp_book(o.side)
        px, sz = self.book(bs, t)
        leaves, fills = o.qty, []
        for p, s in zip(px, sz):
            if leaves == 0 or (p - o.px) * o.side > 0:
                break
            if s <= 0:
                continue
            q = min(s, leaves)
            leaves -= q
            fills.append(ExecReport(o.id, self.idx, "FILL", t, qty=q, px=p, leaves=leaves, liquidity="T"))
            self.deplete(bs, t, q, p)

        ack = ExecReport(o.id, self.idx, "ACK", t, leaves=o.qty)
        reports = [ack] + fills
        if leaves > 0:
            if o.tif == "IOC":
                reports.append(ExecReport(o.id, self.idx, "CANCELED", t, leaves=0, reason="IOC"))
            else:
                o.leaves = leaves
                self.resting[o.id] = o
                self.queue[o.id] = ack.queue_ahead = self._size_at(own_book(o.side), o.px, t)
        return reports

    def _size_at(self, bside: int, price: int, t: int) -> int:
        px, sz = self.book(bside, t)
        return sz[px.index(price)] if price in px else 0

    def cancel(self, order_id: int, t: int):
        o = self.resting.pop(order_id, None)
        if o is None:
            return [ExecReport(order_id, self.idx, "CANCEL_REJECT", t, reason="UNKNOWN_ORDER")]
        self.queue.pop(order_id, None)
        return [ExecReport(order_id, self.idx, "CANCELED", t, leaves=0, reason="USER")]

    def replace(self, order_id: int, new_px: int, t: int):
        o = self.resting.get(order_id)
        if o is None:
            return [ExecReport(order_id, self.idx, "REPLACE_REJECT", t, reason="UNKNOWN_ORDER")]
        o.px = new_px  # loses queue priority
        q = self.queue[order_id] = self._size_at(own_book(o.side), new_px, t)
        return [ExecReport(order_id, self.idx, "REPLACED", t, px=new_px, leaves=o.leaves, queue_ahead=q)]

    # -- passive fills ------------------------------------------------------
    def on_tick(self, k: int, t: int):
        """Advance resting orders through dataset step k. Returns {order_id: [reports]}."""
        out = {}
        if not self.resting:
            return out
        if self.ds.status[k, self.idx]:
            for oid in list(self.resting):  # cancel-on-disconnect
                del self.resting[oid]
                self.queue.pop(oid, None)
                out[oid] = [ExecReport(oid, self.idx, "CANCELED", t, leaves=0, reason="CANCEL_ON_DISCONNECT")]
            return out
        flow = self.ds.flow[k, self.idx].tolist()
        for oid in list(self.resting):
            o = self.resting[oid]
            own, opp = own_book(o.side), opp_book(o.side)
            pxs = self.ds.px[own][k, self.idx].tolist()
            szs = self.ds.sz[own][k, self.idx].tolist()
            fill = 0
            if (int(self.ds.px[opp][k, self.idx, 0]) - o.px) * o.side <= 0:
                fill = o.leaves  # market moved through our price
            elif (pxs[0] - o.px) * o.side <= 0:  # we are at (or better than) the touch
                disp = szs[pxs.index(o.px)] if o.px in pxs else 0
                q = min(self.queue[oid], disp)  # cancels ahead of us shorten the queue
                take = min(q, flow[own])
                flow[own] -= take
                self.queue[oid] = q - take
                if q - take == 0:
                    fill = min(o.leaves, flow[own])
                    flow[own] -= fill
            if fill > 0:
                o.leaves -= fill
                out[oid] = [ExecReport(oid, self.idx, "FILL", t, qty=fill, px=o.px, leaves=o.leaves, liquidity="M")]
                if o.leaves == 0:
                    del self.resting[oid]
                    self.queue.pop(oid, None)
        return out
