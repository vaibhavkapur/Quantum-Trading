"""Event-driven execution simulator on a single simulated clock.

Flow per parent order: IOC sweep(s) allocated by the router -> residual rests
passively at the near touch (repriced with cancel/replace when the touch moves
away) -> at the deadline the passive order is cancelled and one final wider
IOC sweep runs. Rejects put the venue in a cooldown so later sweeps fail over.

Wall-clock time is used only to *measure* the router decision and the solver;
with ``delay_model: fixed`` it never influences simulated outcomes.
"""
from __future__ import annotations

import heapq
import time

import numpy as np

from ..config import venues as venue_cfgs
from ..core import (BUY, MS, SELL, ExecReport, Order, fix_cancel, fix_exec_report, fix_new_order,
                    fix_replace, hash_exponential, opp_book, own_book)
from ..exchange_sim.venue import VenueSim
from ..optimizer.service import market_ladders

PARENT, ORDER, CANCEL, REPLACE, REPORTS, TICK, TIMER, REFRESH, PUBLISH = range(9)


class Parent:
    __slots__ = ("id", "side", "qty", "t0", "deadline", "remaining", "sweeps", "open", "passive",
                 "cancel_sent", "replace_pending", "final_done", "done", "fills", "fees", "arr_mid",
                 "far0", "children", "rejects", "t_done", "end_mid", "timer_on")

    def __init__(self, pid, side, qty, t0, deadline):
        self.id, self.side, self.qty, self.t0, self.deadline = pid, side, qty, t0, deadline
        self.remaining, self.sweeps, self.open, self.passive = qty, 0, 0, None
        self.cancel_sent = self.replace_pending = self.final_done = self.done = self.timer_on = False
        self.fills, self.fees, self.arr_mid, self.far0 = [], 0.0, 0.0, None
        self.children = self.rejects = 0
        self.t_done, self.end_mid = None, 0.0


def make_parents(cfg, ds, window, seed):
    """Seeded parent-order flow (identical for every solver given the seed)."""
    o = cfg["orders"]
    rng = np.random.default_rng([seed, 0xA11CE])
    start = window[0] * ds.dt_ns + int(o["warmup_ms"] * MS)
    end = window[1] * ds.dt_ns - int((o["horizon_ms"] + 200) * MS)
    out, t = [], float(start)
    while True:
        t += rng.exponential(1e9 / o["rate_per_s"])
        if t >= end:
            return out
        side = BUY if rng.random() < 0.5 else SELL
        qty = int(rng.choice(o["sizes_lots"]))
        out.append(Parent(len(out), side, qty, int(t), int(t) + int(o["horizon_ms"] * MS)))


class Engine:
    def __init__(self, cfg, ds, window, router, optimizer=None, store=None, seed=0, parents=None,
                 fix_log_limit=0):
        self.cfg, self.ds, self.window, self.seed = cfg, ds, window, seed
        self.vcfg = venue_cfgs(cfg)
        self.V = len(self.vcfg)
        exe = cfg["execution"]
        self.exe = exe
        self.venues = [VenueSim(i, v, ds, exe["replenish_halflife_ms"]) for i, v in enumerate(self.vcfg)]
        self.router, self.optimizer, self.store = router, optimizer, store
        self.parents = parents if parents is not None else make_parents(cfg, ds, window, seed)
        self.orders = {}
        self.heap, self._seq, self._oid = [], 0, 0
        self.cooldown = [0] * self.V
        self.md_ns = [int(v.md_latency_ms * MS) for v in self.vcfg]
        self.hb_ns = int(exe["heartbeat_ms"] * MS)
        self.max_child = [v.max_child_lots for v in self.vcfg]
        self.tick_on = False
        self.busy_until = 0
        # measurements
        self.decision_ns, self.o2a_ns, self.queue_err = [], [], []
        self.skipped_refresh = 0
        self.counts = {"children": 0, "rejects": 0, "cancels": 0, "replaces": 0, "passive_orders": 0,
                       "reject_reasons": {}}
        self.fix_log, self._fix_limit = [], fix_log_limit
        inst = cfg["instrument"]
        self.tick_size, self.lot = inst["tick_size"], inst["lot_size"]

    # -- plumbing -----------------------------------------------------------
    def _push(self, t, kind, data):
        self._seq += 1
        heapq.heappush(self.heap, (t, self._seq, kind, data))

    def _fix(self, line):
        if len(self.fix_log) < self._fix_limit:
            self.fix_log.append(line)

    def _avail(self, v, t):
        return self.cooldown[v] <= t and not self.ds.status[self.ds.step(t - self.hb_ns), v]

    def _view(self, v, bside, t):
        return self.venues[v].book(bside, t - self.md_ns[v])

    def _books(self, side, t):
        bs = opp_book(side)
        return [self._view(v, bs, t) if self._avail(v, t) else None for v in range(self.V)]

    def _latency(self, oid, v):
        c = self.vcfg[v]
        return int((c.latency_ms + hash_exponential(c.latency_jitter_ms, self.seed, oid, v)) * MS)

    def _deliver(self, t, oid, reps):
        if reps:
            self._push(t + self.orders[oid].latency_ns, REPORTS, (oid, reps))

    # -- run ----------------------------------------------------------------
    def run(self):
        dt = self.ds.dt_ns
        t_start, t_end = self.window[0] * dt, self.window[1] * dt
        for p in self.parents:
            self._push(p.t0, PARENT, p)
        if self.optimizer is not None:
            step = int(self.cfg["optimizer"]["refresh_ms"] * MS)
            for t in range(t_start, t_end, step):
                self._push(t, REFRESH, None)
        handlers = {PARENT: self._on_parent, ORDER: self._on_order, CANCEL: self._on_cancel,
                    REPLACE: self._on_replace, REPORTS: self._on_reports, TICK: self._on_tick,
                    TIMER: self._on_timer, REFRESH: self._on_refresh, PUBLISH: self._on_publish}
        heap = self.heap
        while heap:
            t, _, kind, data = heapq.heappop(heap)
            handlers[kind](t, data)
        return self

    # -- optimizer events ---------------------------------------------------
    def _on_refresh(self, t, _):
        if t < self.busy_until:
            self.skipped_refresh += 1  # previous solve still running: keep the old policy
            return
        books = {BUY: self._books(BUY, t), SELL: self._books(SELL, t)}
        snap, wall = self.optimizer.compute(books, t)
        o = self.cfg["optimizer"]
        delay = wall * 1e9 if o["delay_model"] == "measured" else o["publish_delay_ms"] * MS
        self.busy_until = t + int(delay)
        self._push(self.busy_until, PUBLISH, snap)

    def _on_publish(self, t, snap):
        from dataclasses import replace
        self.store.publish(replace(snap, published_ns=t))

    # -- parent lifecycle ---------------------------------------------------
    def _on_parent(self, t, p):
        p.arr_mid = self.ds.mid_ticks(t)
        self._sweep(p, t)

    def _sweep(self, p, t, final=False):
        exe, side = self.exe, p.side
        books = self._books(side, t)
        ladders, caps, best, far = market_ladders(
            side, books, exe["final_limit_ticks"] if final else exe["limit_ticks"],
            None if final else exe["participation"], self.max_child)
        if p.far0 is None:
            p.far0 = far
        p.sweeps += 1
        if final:
            p.final_done = True
        if far is None:
            return self._after_sweep(p, t)
        qty = p.remaining
        t0 = time.perf_counter_ns()
        alloc = self.router.decide(side, qty, caps, best, t, (ladders, far))
        self.decision_ns.append(time.perf_counter_ns() - t0)
        limit = far + side * (exe["final_limit_ticks"] if final else exe["limit_ticks"])
        sent = 0
        for v in range(self.V):
            q = min(alloc[v], caps[v], qty - sent)
            if q > 0:
                self._send(p, v, q, limit, "IOC", t)
                sent += q
        if sent == 0:
            self._after_sweep(p, t)

    def _send(self, p, v, qty, px, tif, t, est_queue=0):
        self._oid += 1
        o = Order(self._oid, p.id, v, p.side, qty, px, tif, sent_ns=t, leaves=qty, est_queue=est_queue)
        o.latency_ns = self._latency(o.id, v)
        self.orders[o.id] = o
        p.children += 1
        self.counts["children"] += 1
        if tif == "IOC":
            p.open += 1
        else:
            p.passive = o
            self.counts["passive_orders"] += 1
        self._fix(fix_new_order(o, self.vcfg[v].name, self.tick_size, self.lot))
        self._push(t + o.latency_ns, ORDER, o)
        return o

    def _after_sweep(self, p, t):
        if p.done or p.open > 0:
            return
        if p.remaining == 0 or p.final_done:
            return self._complete(p, t)
        if t >= p.deadline:
            return self._sweep(p, t, final=True)
        if p.sweeps < self.exe["max_sweeps"]:
            return self._sweep(p, t)
        self._post_passive(p, t)
        if not p.timer_on:
            p.timer_on = True
            self._push(min(t + int(self.exe["reprice_ms"] * MS), p.deadline), TIMER, p)

    def _post_passive(self, p, t):
        bs = own_book(p.side)
        best_v, best_key = None, None
        for v in range(self.V):
            if not self._avail(v, t):
                continue
            px, sz = self._view(v, bs, t)
            # most aggressive near-touch price first, then shortest expected queue time
            key = (-px[0] * p.side, (sz[0] + 1) / max(self.vcfg[v].flow_rate, 1e-9), v)
            if best_key is None or key < best_key:
                best_v, best_key, post_px, est = v, key, px[0], sz[0]
        if best_v is not None:
            self._send(p, best_v, p.remaining, post_px, "DAY", t, est_queue=est)

    def _on_timer(self, t, p):
        if p.done:
            return
        o = p.passive
        if t >= p.deadline:
            if o is not None:
                if not p.cancel_sent:
                    p.cancel_sent = True
                    self.counts["cancels"] += 1
                    self._fix(fix_cancel(o, self.vcfg[o.venue].name, t))
                    self._push(t + o.latency_ns, CANCEL, o)
            elif p.open == 0:
                self._after_sweep(p, t)
            return
        if o is None:
            if p.open == 0:
                self._post_passive(p, t)
        elif o.acked and not p.replace_pending:
            px, _ = self._view(o.venue, own_book(p.side), t)
            if (px[0] - o.px) * p.side > 0:  # touch moved away: chase with cancel/replace
                _, sz = self._view(o.venue, own_book(p.side), t)
                o.est_queue = sz[0]
                p.replace_pending = True
                self.counts["replaces"] += 1
                self._fix(fix_replace(o, self.vcfg[o.venue].name, px[0], self.tick_size, self.lot, t))
                self._push(t + o.latency_ns, REPLACE, (o, px[0]))
        self._push(min(t + int(self.exe["reprice_ms"] * MS), p.deadline), TIMER, p)

    def _complete(self, p, t):
        p.done, p.t_done = True, t
        p.end_mid = self.ds.mid_ticks(t)

    # -- venue-side events --------------------------------------------------
    def _on_order(self, t, o):
        venue = self.venues[o.venue]
        reps = venue.new_order(o, t)
        taken = sum(r.qty for r in reps if r.kind == "FILL")
        if taken:
            bs = opp_book(o.side)
            for b, other in enumerate(self.venues):  # quoters elsewhere react to our print
                c = self.vcfg[b]
                if b != o.venue and c.fade_coeff > 0:
                    other.deplete(bs, t + int(c.fade_delay_ms * MS), c.fade_coeff * taken)
        if venue.resting and not self.tick_on:
            self.tick_on = True
            self._push((t // self.ds.dt_ns + 1) * self.ds.dt_ns, TICK, None)
        self._deliver(t, o.id, reps)

    def _on_cancel(self, t, o):
        self._deliver(t, o.id, self.venues[o.venue].cancel(o.id, t))

    def _on_replace(self, t, data):
        o, px = data
        self._deliver(t, o.id, self.venues[o.venue].replace(o.id, px, t))

    def _on_tick(self, t, _):
        k = self.ds.step(t)
        live = False
        for venue in self.venues:
            if venue.resting:
                for oid, reps in venue.on_tick(k, t).items():
                    self._deliver(t, oid, reps)
                live = live or bool(venue.resting)
        if live and k < self.ds.T - 1:
            self._push(t + self.ds.dt_ns, TICK, None)
        else:
            self.tick_on = False

    # -- router-side report handling ---------------------------------------
    def _on_reports(self, t, data):
        oid, reps = data
        o = self.orders[oid]
        p = self.parents[o.parent_id]
        c = self.vcfg[o.venue]
        ioc = o.tif == "IOC"
        for r in reps:
            self._fix(fix_exec_report(r, c.name, self.tick_size, self.lot))
            k = r.kind
            if k == "ACK":
                o.acked = True
                self.o2a_ns.append(t - o.sent_ns)
                p.fees += c.order_fee
                if r.queue_ahead is not None:
                    self.queue_err.append(abs(o.est_queue - r.queue_ahead))
            elif k == "FILL":
                p.fills.append((r.t_ns, o.venue, r.px, r.qty, r.liquidity))
                p.remaining -= r.qty
                o.leaves = r.leaves
                p.fees += (c.taker_fee if r.liquidity == "T" else c.maker_fee) * r.qty * self.lot
                if not ioc and r.leaves == 0 and p.passive is o:
                    p.passive = None
            elif k == "REJECT":
                p.rejects += 1
                self.counts["rejects"] += 1
                rr = self.counts["reject_reasons"]
                rr[r.reason] = rr.get(r.reason, 0) + 1
                self.cooldown[o.venue] = t + int(self.exe["cooldown_ms"] * MS)
                if not ioc and p.passive is o:
                    p.passive = None
            elif k == "CANCELED":
                if not ioc and p.passive is o:
                    p.passive = None
            elif k == "REPLACED":
                o.px = r.px
                p.replace_pending = False
                self.queue_err.append(abs(o.est_queue - r.queue_ahead))
            elif k == "REPLACE_REJECT":
                p.replace_pending = False
            elif k == "CANCEL_REJECT":
                if p.passive is o:
                    p.passive = None
        if p.done:
            return
        if ioc:
            p.open -= 1
            self._after_sweep(p, t)
        elif p.remaining == 0 and p.open == 0:
            self._complete(p, t)
        elif p.passive is None and p.cancel_sent and p.open == 0:
            self._after_sweep(p, t)  # passive leg is gone at the deadline: final sweep
