"""Deterministic replay of an asynchronous quote stream.

Two views of the same event log:

* observed state at time t: the latest book per venue market among events
  *received* by t (what a solver can see, including delayed/stale quotes);
* true state at time t: the latest book per venue market *produced* by t
  (what an order would actually hit).

At every decision time each solver searches the observed graph; resulting
plans are executed against the true state `latency_ms` later. Ground truth is
recomputed at every decision time by exhaustive search on the true state.
"""

from __future__ import annotations

import bisect
import time
import zlib
from dataclasses import dataclass, field

import numpy as np

from ..core.book import OrderBook, QuoteEvent
from ..graph.build import EdgeKey, MarketGraph, build_graph
from ..graph.cycles import enumerate_cycles
from ..graph.sizing import Inventory, Plan, best_plan
from ..solvers import SearchContext, solve
from .execution import execute_plan

NameCycle = tuple[str, ...]


def canonical_names(cycle: NameCycle) -> NameCycle:
    k = cycle.index(min(cycle))
    return cycle[k:] + cycle[:k]


class EventStore:
    def __init__(self, events: list[QuoteEvent], known: list[str]):
        self.events = sorted(events, key=lambda e: (e.recv_ts, e.seq))
        self._ptr = 0
        self._observed: dict[tuple, OrderBook] = {}
        self._by_seq: dict[int, OrderBook] = {}
        per_market: dict[tuple, list[OrderBook]] = {}
        for ev in self.events:
            book = ev.to_book(known)
            book.validate()
            self._by_seq[ev.seq] = book
            per_market.setdefault(book.key, []).append(book)
        self._true: dict[tuple, tuple[list[int], list[OrderBook]]] = {}
        for key, books in sorted(per_market.items()):
            books.sort(key=lambda b: b.ts)
            self._true[key] = ([b.ts for b in books], books)

    @property
    def end_ts(self) -> int:
        return max(e.exchange_ts for e in self.events) if self.events else 0

    def observed(self, t: int) -> list[OrderBook]:
        """Latest received book per market at local time t (t must not decrease)."""
        while self._ptr < len(self.events) and self.events[self._ptr].recv_ts <= t:
            book = self._by_seq[self.events[self._ptr].seq]
            cur = self._observed.get(book.key)
            if cur is None or book.ts >= cur.ts:  # late out-of-order updates never overwrite newer ones
                self._observed[book.key] = book
            self._ptr += 1
        return list(self._observed.values())

    def true(self, t: int) -> list[OrderBook]:
        out = []
        for ts, books in self._true.values():
            i = bisect.bisect_right(ts, t)
            if i:
                out.append(books[i - 1])
        return out


@dataclass
class SolverState:
    name: str
    rng: np.random.Generator
    inventory: Inventory
    resources: dict = field(default_factory=dict)
    in_episode: set = field(default_factory=set)  # cycles already executed in their current episode
    ticks: list[dict] = field(default_factory=list)
    opportunities: list[dict] = field(default_factory=list)
    detections: list[tuple[int, NameCycle, tuple[str, ...]]] = field(default_factory=list)
    captured: set = field(default_factory=set)  # (cycle, t) with positive realised P&L


def _truth(graph: MarketGraph, cfg: dict, inventory: Inventory) -> dict[NameCycle, float]:
    """Every cycle executable for at least min_profit on `graph`, by exhaustive search.

    On the true books this is ground truth; on the observed books it is the
    best any search could have reported from the data it was given.
    """
    g, s = cfg["graph"], cfg["sizing"]
    W = graph.weight_matrix(0.0)  # only used for connectivity
    cycles, _, _ = enumerate_cycles(W, g["min_cycle_len"], g["max_cycle_len"])
    out: dict[NameCycle, float] = {}
    for cyc, _ in cycles:
        names = tuple(graph.currencies[i] for i in cyc)
        plan = best_plan(graph, names, s["max_notional"], None, g["max_venues_per_cycle"], inventory,
                         funding=cfg["funding_currencies"])
        if plan is not None and plan.profit_num >= s["min_profit"]:
            out[canonical_names(names)] = plan.profit_num
    return out


def run_replay(cfg: dict, seed: int, solvers: list[str], events: list[QuoteEvent], manifest: dict) -> dict:
    """Replay one dataset for the given solvers. Returns per-solver raw records plus shared truth."""
    g, s, x = cfg["graph"], cfg["sizing"], cfg["execution"]
    store = EventStore(events, manifest["currencies"])
    decision_times = list(range(x["decision_offset_ms"], store.end_ts + 1, x["decision_interval_ms"]))
    states: dict[str, SolverState] = {}
    truth_by_t: dict[int, dict[NameCycle, float]] = {}
    initial_inventory: Inventory | None = None

    for t in decision_times:
        t0 = time.perf_counter()
        graph = build_graph(store.observed(t), t, cfg["numeraire"], g["stale_timeout_ms"], g["allowed_venues"])
        W = graph.weight_matrix(g["probe_notional"])
        W_mid = graph.mid_weight_matrix()
        graph_ms = (time.perf_counter() - t0) * 1e3

        true_now = build_graph(store.true(t), t, cfg["numeraire"], None, g["allowed_venues"])
        true_exec = build_graph(store.true(t + x["latency_ms"]), t + x["latency_ms"], cfg["numeraire"], None,
                                g["allowed_venues"])
        if initial_inventory is None:
            if len(true_now.ref) < 2:
                continue
            initial_inventory = {
                (v, c): s["inventory_notional"] / true_now.ref[c] for v in manifest["venues"] for c in true_now.ref
            }
            for name in solvers:
                states[name] = SolverState(
                    name, np.random.default_rng([seed, zlib.crc32(name.encode())]), dict(initial_inventory)
                )
        truth = _truth(true_now, cfg, initial_inventory)
        truth_by_t[t] = truth
        observable = set(_truth(graph, cfg, initial_inventory))

        for name in solvers:
            st = states[name]
            ctx = SearchContext(W, W_mid, g["min_cycle_len"], g["max_cycle_len"], st.rng, cfg, st.resources)
            res = solve(name, ctx)

            t1 = time.perf_counter()
            plans: list[Plan] = []
            for cyc in res.cycles:
                names = tuple(graph.currencies[i] for i in cyc)
                # Detection is judged against the same starting inventory as ground truth, so
                # recall does not depend on a solver's own trading history.
                plan = best_plan(graph, names, s["max_notional"], s["venue_top_k"], g["max_venues_per_cycle"],
                                 initial_inventory, funding=cfg["funding_currencies"])
                if plan is not None and plan.profit_num >= s["min_profit"]:
                    plans.append(plan)
            plans.sort(key=lambda p: (-p.profit_num, p.cycle))
            detected = {canonical_names(p.cycle): p for p in plans}
            sizing_ms = (time.perf_counter() - t1) * 1e3

            t2 = time.perf_counter()
            consumed_obs: dict[EdgeKey, float] = {}
            consumed_true: dict[EdgeKey, float] = {}
            st.in_episode &= set(detected)
            expected = realised = 0.0
            n_exec = 0
            for key, plan in detected.items():
                st.detections.append((t, key, plan.venues))
                row = {
                    "t": t, "cycle": "->".join(plan.cycle), "venues": "|".join(plan.venues),
                    "in_truth": key in truth, "notional": plan.notional_num, "expected_profit": plan.profit_num,
                    "executed": False, "realised_pnl": 0.0, "fill_ratio": 0.0, "residual_exposure": 0.0,
                }
                if key not in st.in_episode:
                    # Re-size against current inventory and liquidity already claimed this tick.
                    live = best_plan(graph, plan.cycle, s["max_notional"], s["venue_top_k"],
                                     g["max_venues_per_cycle"], st.inventory, consumed_obs,
                                     cfg["funding_currencies"])
                    if live is not None and live.profit_num >= s["min_profit"]:
                        fill = execute_plan(live, true_exec, st.inventory, consumed_true, x["slippage_bps"])
                        for e, a in zip(live.edges, live.amounts_in):
                            consumed_obs[e.key] = consumed_obs.get(e.key, 0.0) + a
                        st.in_episode.add(key)
                        n_exec += 1
                        expected += live.profit_num
                        realised += fill["realised_pnl"]
                        if fill["realised_pnl"] > 0:
                            st.captured.add((key, t))
                        row.update(executed=True, expected_profit=live.profit_num, notional=live.notional_num,
                                   venues="|".join(live.venues), realised_pnl=fill["realised_pnl"],
                                   fill_ratio=fill["fill_ratio"], residual_exposure=fill["residual_exposure"])
                st.opportunities.append(row)
            exec_ms = (time.perf_counter() - t2) * 1e3

            keys = set(detected)
            unrepaired = None
            if res.unrepaired is not None:
                unrepaired = {canonical_names(tuple(graph.currencies[i] for i in c)) for c in res.unrepaired}
            st.ticks.append({
                "t": t, "n_currencies": len(graph.currencies), "n_edges": sum(len(e) for e in graph.edges.values()),
                "dropped_stale": graph.dropped_stale, "graph_ms": graph_ms, "search_ms": res.stats["search_ms"],
                "sizing_ms": sizing_ms, "exec_sim_ms": exec_ms,
                "decision_ms": graph_ms + res.stats["search_ms"] + sizing_ms,
                "candidates": len(res.cycles), "detected": len(keys), "truth": len(truth),
                "tp": len(keys & set(truth)), "fp": len(keys - set(truth)), "fn": len(set(truth) - keys),
                "observable": len(observable), "observable_found": len(keys & observable),
                "observable_found_unrepaired": len(keys & observable if unrepaired is None
                                                   else keys & observable & unrepaired),
                "executed": n_exec, "expected_profit": expected, "realised_pnl": realised,
                **{f"s_{k}": v for k, v in res.stats.items() if k != "search_ms"},
            })
    return {"states": states, "truth_by_t": truth_by_t, "decision_times": decision_times}
