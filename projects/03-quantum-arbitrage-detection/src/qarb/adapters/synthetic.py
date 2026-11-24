"""Synthetic multi-venue order-book stream with planted cycles.

Background books are arbitrage-free by construction: every venue quotes around
a common fair value with mid noise strictly smaller than the half-spread, so
every conversion edge costs more than the fair-value potential difference and
no cycle can be profitable. On top of that the generator plants:

* profitable cycles: the books on the cycle's legs are dislocated so the
  top-of-book round trip earns `edge_bps` after spread and fees, with normal depth;
* thin decoys: the same dislocation but only on a tiny top level, so the
  cycle is negative at the top of book and not worth trading at size;
* stale decoys: a fair-value jump while one venue's feed is delayed, so its
  last *received* quotes look like an arbitrage that no longer exists.

Planted dislocations can create additional profitable cycles through the same
legs; ground truth is therefore always recomputed by exhaustive search on the
true books, never read from the plant list.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
from pathlib import Path

import numpy as np

from ..core.book import QuoteEvent, write_events
from ..core.symbols import format_market, orient

DEFAULT_FAIR = {
    "USD": 1.0, "USDT": 1.0004, "USDC": 0.9998, "EUR": 1.085, "GBP": 1.27,
    "BTC": 64250.0, "ETH": 3420.0, "SOL": 148.0, "XRP": 0.52, "ADA": 0.45,
    "DOGE": 0.12, "LTC": 84.0, "LINK": 14.5, "AVAX": 36.0, "DOT": 7.1, "BCH": 480.0,
}


def _price_round(p: float) -> float:
    return float(f"{p:.10g}")


def generate_dataset(cfg: dict, seed: int) -> tuple[list[QuoteEvent], dict]:
    """Generate events and the ground-truth manifest for one seed. Deterministic."""
    d = cfg["data"]
    rng = np.random.default_rng([seed, 0xA5B])
    currencies: list[str] = list(d["currencies"])
    venues: list[dict] = list(d["venues"])
    n_ticks, tick_ms, n_levels = int(d["n_ticks"]), int(d["tick_ms"]), int(d["levels"])
    hs = d["half_spread_bps"] * 1e-4
    step = d["level_step_bps"] * 1e-4
    noise_amp = d["noise_frac"] * hs
    vol = d["vol_bps_per_tick"] * 1e-4
    numeraire = cfg["numeraire"]

    fair0 = {}
    for c in currencies:
        fair0[c] = DEFAULT_FAIR.get(c) or float(np.exp(rng.uniform(math.log(0.1), math.log(500.0))))

    # markets listed per venue
    pairs = [orient(a, b) for a, b in itertools.combinations(currencies, 2)]
    listed: dict[str, list[tuple[str, str]]] = {}
    for v in venues:
        cov = float(v.get("coverage", d.get("market_coverage", 1.0)))
        listed[v["name"]] = [p for p in pairs if cov >= 1.0 or numeraire in p or rng.random() < cov]
    fee_bps = {
        (v["name"], p): float(v["fee_bps"]) * (float(d.get("stable_fee_mult", 1.0)) if _is_stable_pair(p) else 1.0)
        for v in venues for p in listed[v["name"]]
    }

    def leg_market(src: str, dst: str) -> tuple[tuple[str, str], str]:
        base, quote = orient(src, dst)
        return (base, quote), ("buy" if src == quote else "sell")

    # ---- plants -----------------------------------------------------------
    plants: list[dict] = []
    busy: dict[tuple[str, tuple[str, str]], list[tuple[int, int]]] = {}

    def free(venue: str, market: tuple[str, str], t0: int, t1: int) -> bool:
        return all(t1 < a or t0 > b for a, b in busy.get((venue, market), []))

    def plant(kind: str, count: int) -> None:
        p = d["planted"]
        made = attempts = 0
        while made < count and attempts < 500:
            attempts += 1
            L = int(rng.choice(p["lengths"]))
            if L > len(currencies):
                continue
            cyc = [currencies[i] for i in rng.permutation(len(currencies))[:L]]
            dur = int(rng.integers(p["duration_ticks"][0], p["duration_ticks"][1] + 1))
            t0 = int(rng.integers(2, max(3, n_ticks - dur - 1)))
            t1 = t0 + dur - 1
            legs = []
            for i in range(L):
                src, dst = cyc[i], cyc[(i + 1) % L]
                market, side = leg_market(src, dst)
                options = [v["name"] for v in venues if market in listed[v["name"]] and free(v["name"], market, t0, t1)]
                options = [v for v in options if (v, market) not in {(l["venue"], l["market"]) for l in legs}]
                if not options:
                    break
                legs.append({"venue": str(rng.choice(options)), "market": market, "side": side})
            if len(legs) != L:
                continue
            if L == 2 and legs[0]["venue"] == legs[1]["venue"]:
                continue  # a round trip on one book cannot be dislocated
            edge = float(rng.uniform(*p["edge_bps"])) * 1e-4
            for leg in legs:
                busy.setdefault((leg["venue"], leg["market"]), []).append((t0, t1))
                leg["tilt"] = hs + fee_bps[(leg["venue"], leg["market"])] * 1e-4 + edge / L
            plants.append({
                "id": f"{kind}-{len(plants)}", "kind": kind, "currencies": cyc,
                "venues": [l["venue"] for l in legs], "start_tick": t0, "end_tick": t1,
                "start_ms": t0 * tick_ms, "end_ms": (t1 + 1) * tick_ms,
                "edge_bps": edge * 1e4, "legs": legs,
            })
            made += 1

    plant("profitable", int(d["planted"]["n_profitable"]))
    plant("thin", int(d["decoys"]["n_thin"]))

    # stale decoys: fair-value jump + delayed feed on one venue
    outages: list[dict] = []
    risky = [c for c in currencies if c != numeraire and not _is_stable(c)]
    for k in range(int(d["decoys"]["n_stale"])):
        if not risky or len(venues) < 2:
            break
        dur = int(d["decoys"]["stale_ticks"])
        t0 = int(rng.integers(2, max(3, n_ticks - dur - 1)))
        outages.append({
            "id": f"stale-{k}", "kind": "stale", "venue": str(rng.choice([v["name"] for v in venues])),
            "currency": str(rng.choice(risky)), "start_tick": t0, "end_tick": t0 + dur - 1,
            "start_ms": t0 * tick_ms, "end_ms": (t0 + dur) * tick_ms,
            "jump_bps": float(d["decoys"]["stale_jump_bps"]) * (1 if rng.random() < 0.5 else -1),
        })

    # ---- event stream -----------------------------------------------------
    events: list[tuple[int, int, str, str, tuple, tuple, float]] = []
    log_fair = {c: math.log(fair0[c]) for c in currencies}
    styles = {v["name"]: v.get("symbol_style", "dash") for v in venues}
    for t in range(n_ticks):
        for c in currencies:
            if c != numeraire:
                log_fair[c] += vol * (0.1 if _is_stable(c) else 1.0) * float(rng.standard_normal())
        for o in outages:
            if o["start_tick"] == t:
                log_fair[o["currency"]] += o["jump_bps"] * 1e-4
        for v in venues:
            vname = v["name"]
            lo, hi = v["feed_delay_ms"]
            for market in listed[vname]:
                base, quote = market
                log_mid = log_fair[base] - log_fair[quote] + float(rng.uniform(-noise_amp, noise_amp))
                sizes = d["depth_notional"] / math.exp(log_fair[base]) * rng.uniform(0.5, 1.5, size=(2, n_levels))
                thin_side = None
                for pl in plants:
                    if not pl["start_tick"] <= t <= pl["end_tick"]:
                        continue
                    for leg in pl["legs"]:
                        if leg["venue"] == vname and leg["market"] == market:
                            sign = -1.0 if leg["side"] == "buy" else 1.0
                            if pl["kind"] == "profitable":
                                log_mid = log_fair[base] - log_fair[quote] + sign * leg["tilt"]
                            else:
                                thin_side = (leg["side"], leg["tilt"])
                mid = math.exp(log_mid)
                bids = [(mid * (1 - hs - i * step), float(sizes[0, i])) for i in range(n_levels)]
                asks = [(mid * (1 + hs + i * step), float(sizes[1, i])) for i in range(n_levels)]
                if thin_side is not None:
                    side, tilt = thin_side
                    fairp = math.exp(log_fair[base] - log_fair[quote])
                    tiny = d["decoys"]["thin_notional"] / math.exp(log_fair[base])
                    if side == "buy":  # one small aggressive ask; bids pulled below it
                        px = fairp * math.exp(-tilt) * (1 + hs)
                        asks = [(px, tiny)] + asks[:-1]
                        shift = px * (1 - 2 * hs) / bids[0][0]
                        bids = [(p * min(1.0, shift), s) for p, s in bids]
                    else:
                        px = fairp * math.exp(tilt) * (1 - hs)
                        bids = [(px, tiny)] + bids[:-1]
                        shift = px * (1 + 2 * hs) / asks[0][0]
                        asks = [(p * max(1.0, shift), s) for p, s in asks]
                exchange_ts = t * tick_ms + int(rng.integers(0, max(1, tick_ms // 4)))
                delay = int(rng.integers(lo, hi + 1))
                for o in outages:
                    if o["venue"] == vname and o["start_tick"] <= t <= o["end_tick"]:
                        delay += o["end_ms"] - t * tick_ms
                events.append((
                    exchange_ts + delay, exchange_ts, vname, format_market(base, quote, styles[vname]),
                    tuple((_price_round(p), _price_round(s)) for p, s in bids),
                    tuple((_price_round(p), _price_round(s)) for p, s in asks),
                    fee_bps[(vname, market)],
                ))
    events.sort(key=lambda e: (e[0], e[1], e[2], e[3]))
    out = [QuoteEvent(i, e[1], e[0], e[2], e[3], e[4], e[5], e[6]) for i, e in enumerate(events)]

    for pl in plants:
        for leg in pl["legs"]:
            leg["market"] = "/".join(leg["market"])
    manifest = {
        "seed": seed,
        "generator": "qarb.adapters.synthetic",
        "provenance": "synthetic; no real market data",
        "numeraire": numeraire,
        "currencies": currencies,
        "venues": [v["name"] for v in venues],
        "n_ticks": n_ticks,
        "tick_ms": tick_ms,
        "n_events": len(out),
        "timestamps": "integer milliseconds since stream start; exchange_ts = venue clock, recv_ts = local clock",
        "planted": plants,
        "outages": outages,
    }
    return out, manifest


def _is_stable(c: str) -> bool:
    return c in ("USD", "USDT", "USDC")


def _is_stable_pair(p: tuple[str, str]) -> bool:
    return _is_stable(p[0]) and _is_stable(p[1])


def dataset_paths(cfg: dict, seed: int) -> tuple[Path, Path]:
    root = Path(cfg["data"]["dir"])
    return root / f"seed{seed}.events.jsonl", root / f"seed{seed}.manifest.json"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_dataset(cfg: dict, seed: int) -> dict:
    events, manifest = generate_dataset(cfg, seed)
    ev_path, man_path = dataset_paths(cfg, seed)
    ev_path.parent.mkdir(parents=True, exist_ok=True)
    write_events(ev_path, events)
    manifest["events_file"] = ev_path.name
    manifest["events_sha256"] = sha256_file(ev_path)
    man_path.write_text(json.dumps(manifest, indent=2))
    return manifest
