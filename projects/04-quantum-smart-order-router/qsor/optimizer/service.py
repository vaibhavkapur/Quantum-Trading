"""Policy optimizer: solves every (side, bucket) problem and builds a snapshot.

`PolicyOptimizer.compute` is synchronous and is what the simulator calls at
each refresh. `AsyncPolicyService` runs the same computation on a background
thread and publishes into a `PolicyStore` for a live router.

Failure handling: any solver error, unsupported instance size or wall-time
overrun for a bucket falls back to the classical greedy solution for that
bucket, and the snapshot is tagged ``<solver>+fallback``.
"""
from __future__ import annotations

import threading
import time

import numpy as np

from ..core import BUY, SELL
from ..router.policy import make_snapshot
from . import classical, registry
from .problem import build_problem


def market_ladders(side, books, limit_ticks, participation, max_child):
    """Common view -> (ladders, caps, best_px, far) used by optimizer and router.

    books[v] is (prices, sizes) best-first for the side being taken, or None
    if the venue is unavailable.
    """
    V = len(books)
    best, far = [None] * V, None
    for v in range(V):
        if books[v] is None:
            continue
        for p, s in zip(*books[v]):
            if s > 0:
                best[v] = p
                if far is None or (p - far) * side < 0:
                    far = p
                break
    ladders, caps = [()] * V, [0] * V
    if far is None:
        return ladders, caps, best, None
    limit = far + side * limit_ticks
    for v in range(V):
        if best[v] is None:
            continue
        lad = tuple((p, s) for p, s in zip(*books[v]) if s > 0 and (p - limit) * side <= 0)
        if not lad:
            continue
        depth = sum(s for _, s in lad)
        cap = depth if participation is None else min(max_child[v], -int(-participation * depth // 1))
        ladders[v], caps[v] = lad, cap
    return ladders, caps, best, far


class PolicyOptimizer:
    def __init__(self, solver: str, model, cfg: dict, max_child, seed: int = 0):
        self.solver, self.model = solver, model
        registry.warmup(solver)
        self.opt, self.exe = cfg["optimizer"], cfg["execution"]
        self.solver_cfg = cfg["solvers"]
        self.max_child = max_child
        self.buckets = tuple(sorted(self.opt["buckets_lots"]))
        self.rng = np.random.default_rng([seed, 0xC0FFEE])
        self.version = 0
        self.warm = {}
        self.stats = {"refreshes": 0, "problems": 0, "fallbacks": 0, "evals": 0, "wall_s": [],
                      "t_build": 0.0, "t_solve": 0.0, "t_encode": 0.0, "t_optimize": 0.0,
                      "t_sample": 0.0, "t_decode": 0.0, "shots": 0, "fallback_reasons": {}}

    def compute(self, books_by_side, now_ns: int):
        """books_by_side[side] -> per-venue books. Returns (snapshot, wall seconds)."""
        st = self.stats
        t_start = time.perf_counter()
        deadline = t_start + self.opt["timeout_ms"] / 1000.0
        units, fell_back = {BUY: [], SELL: []}, False
        for side in (BUY, SELL):
            ladders, caps, _, far = market_ladders(
                side, books_by_side[side], self.exe["limit_ticks"], self.exe["participation"], self.max_child)
            for q in self.buckets:
                if far is None:
                    units[side].append([0] * len(caps))
                    continue
                t0 = time.perf_counter()
                prob = build_problem(side, q, ladders, caps, far, self.model, self.opt["max_units"])
                t1 = time.perf_counter()
                st["problems"] += 1
                try:
                    x, info = registry.solve(self.solver, prob, self.rng, self.solver_cfg,
                                             warm=self.warm.get((side, q)), deadline=deadline)
                    if not prob.is_feasible(x):
                        raise ValueError("solver returned an infeasible allocation")
                    if "theta" in info:
                        self.warm[(side, q)] = info["theta"]
                    st["evals"] += info.get("evals", 0)
                    st["shots"] += info.get("shots", 0)
                    for k in ("t_encode", "t_optimize", "t_sample", "t_decode"):
                        st[k] += info.get(k, 0.0)
                except Exception as e:  # timeout, too many qubits, solver failure
                    x = classical.greedy(prob)
                    fell_back = True
                    st["fallbacks"] += 1
                    key = type(e).__name__
                    st["fallback_reasons"][key] = st["fallback_reasons"].get(key, 0) + 1
                st["t_build"] += t1 - t0
                st["t_solve"] += time.perf_counter() - t1
                units[side].append(x)
        wall = time.perf_counter() - t_start
        st["refreshes"] += 1
        st["wall_s"].append(wall)
        self.version += 1
        source = self.solver + ("+fallback" if fell_back else "")
        snap = make_snapshot(self.version, now_ns, now_ns, source, self.buckets, units[BUY], units[SELL])
        return snap, wall


class AsyncPolicyService(threading.Thread):
    """Background optimizer: market view in, immutable policy snapshots out."""

    def __init__(self, optimizer: PolicyOptimizer, store, market_fn, clock_ns, interval_s: float):
        super().__init__(daemon=True)
        self.optimizer, self.store = optimizer, store
        self.market_fn, self.clock_ns, self.interval_s = market_fn, clock_ns, interval_s
        self._stop_evt = threading.Event()
        self.errors = 0

    def run(self):
        while not self._stop_evt.is_set():
            t0 = time.perf_counter()
            try:
                snap, _ = self.optimizer.compute(self.market_fn(), self.clock_ns())
                self.store.publish(_stamp(snap, self.clock_ns()))
            except Exception:
                self.errors += 1  # keep serving the last good policy
            self._stop_evt.wait(max(0.0, self.interval_s - (time.perf_counter() - t0)))

    def stop(self):
        self._stop_evt.set()
        self.join(timeout=10)


def _stamp(snap, published_ns):
    from dataclasses import replace
    return replace(snap, published_ns=published_ns)
