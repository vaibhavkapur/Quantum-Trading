"""Transaction-cost analysis and latency summaries for one simulation run."""
from __future__ import annotations

import numpy as np

from ..core import MS


def pct(samples, q):
    return float(np.percentile(samples, q)) if len(samples) else float("nan")


def parent_rows(engine):
    tick, lot = engine.tick_size, engine.lot
    rows = []
    for p in engine.parents:
        filled = sum(f[3] for f in p.fills)
        slip = sum(p.side * (f[2] - p.arr_mid) * f[3] for f in p.fills) * tick * lot
        opp = p.side * (p.end_mid - p.arr_mid) * (p.qty - filled) * tick * lot
        notional = p.arr_mid * tick * p.qty * lot
        pi = (sum(p.side * (p.far0 - f[2]) * f[3] for f in p.fills) * tick * lot) if p.far0 is not None else 0.0
        rows.append({
            "parent": p.id, "side": p.side, "qty_lots": p.qty, "filled_lots": filled,
            "arrival_ns": p.t0, "done_ns": p.t_done, "arrival_mid": p.arr_mid * tick,
            "slippage": slip, "fees": p.fees, "opportunity": opp, "notional": notional,
            "price_improvement": pi, "children": p.children, "rejects": p.rejects,
            "passive_lots": sum(f[3] for f in p.fills if f[4] == "M"),
            "is_bps": (slip + p.fees + opp) / notional * 1e4,
            "completed": bool(p.done),
        })
    return rows


def summarize(engine, rows):
    notional = sum(r["notional"] for r in rows) or float("nan")
    qty = sum(r["qty_lots"] for r in rows) or float("nan")
    filled = sum(r["filled_lots"] for r in rows)
    filled_notional = sum(r["notional"] * r["filled_lots"] / r["qty_lots"] for r in rows) or float("nan")
    dec = np.asarray(engine.decision_ns, dtype=float) / 1e3  # microseconds
    o2a = np.asarray(engine.o2a_ns, dtype=float) / MS
    r = engine.router
    m = {
        "parents": len(rows),
        "incomplete_parents": sum(1 for x in rows if not x["completed"]),
        "fill_rate": filled / qty,
        "is_bps": sum(x["slippage"] + x["fees"] + x["opportunity"] for x in rows) / notional * 1e4,
        "slippage_bps": sum(x["slippage"] for x in rows) / notional * 1e4,
        "fees_bps": sum(x["fees"] for x in rows) / notional * 1e4,
        "opportunity_bps": sum(x["opportunity"] for x in rows) / notional * 1e4,
        "price_improvement_bps": sum(x["price_improvement"] for x in rows) / filled_notional * 1e4,
        "passive_fill_share": sum(x["passive_lots"] for x in rows) / filled if filled else float("nan"),
        "decision_us_p50": pct(dec, 50), "decision_us_p99": pct(dec, 99),
        "decision_us_p999": pct(dec, 99.9), "decision_us_mean": float(dec.mean()) if dec.size else float("nan"),
        "o2a_ms_p50": pct(o2a, 50), "o2a_ms_p99": pct(o2a, 99), "o2a_ms_p999": pct(o2a, 99.9),
        "queue_error_lots_mean": float(np.mean(engine.queue_err)) if engine.queue_err else float("nan"),
        "queue_error_samples": len(engine.queue_err),
        "decisions": r.decisions,
        "stale_policy_frequency": r.stale / r.decisions if r.decisions else float("nan"),
        "policy_age_ms_mean": (r.age_ns_sum / max(1, r.decisions - r.stale)) / MS,
        "children": engine.counts["children"], "rejects": engine.counts["rejects"],
        "reject_reasons": engine.counts["reject_reasons"],
        "cancels": engine.counts["cancels"], "replaces": engine.counts["replaces"],
        "passive_orders": engine.counts["passive_orders"],
        "skipped_refreshes": engine.skipped_refresh,
    }
    shares = [0] * engine.V
    for p in engine.parents:
        for f in p.fills:
            shares[f[1]] += f[3]
    m["venue_fill_share"] = {engine.vcfg[v].name: (shares[v] / filled if filled else 0.0) for v in range(engine.V)}
    opt = engine.optimizer
    if opt is not None:
        st = opt.stats
        wall = np.asarray(st["wall_s"]) * 1e3
        m.update({
            "solver_refreshes": st["refreshes"], "solver_problems": st["problems"],
            "solver_fallbacks": st["fallbacks"], "solver_fallback_reasons": st["fallback_reasons"],
            "solver_wall_ms_mean": float(wall.mean()) if wall.size else float("nan"),
            "solver_wall_ms_p99": pct(wall, 99), "solver_wall_ms_total": float(wall.sum()),
            "solver_overrun_frequency": float((wall > engine.cfg["optimizer"]["refresh_ms"]).mean()) if wall.size else 0.0,
            "solver_evals_per_problem": st["evals"] / max(1, st["problems"]),
            "solver_shots_per_problem": st["shots"] / max(1, st["problems"]),
            "solver_ms_build": st["t_build"] * 1e3, "solver_ms_solve": st["t_solve"] * 1e3,
            "solver_ms_encode": st["t_encode"] * 1e3, "solver_ms_optimize": st["t_optimize"] * 1e3,
            "solver_ms_sample": st["t_sample"] * 1e3, "solver_ms_decode_repair": st["t_decode"] * 1e3,
            "policies_published": engine.store.published,
        })
    return m


PROM_BUCKETS_US = (1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 5000)


def prometheus_text(engine, metrics, labels):
    """Prometheus text exposition of the run (scrape-file / pushgateway friendly)."""
    lab = ",".join(f'{k}="{v}"' for k, v in labels.items())
    dec = np.asarray(engine.decision_ns, dtype=float) / 1e3
    out = ["# HELP sor_decision_latency_microseconds Router decision latency (wall clock).",
           "# TYPE sor_decision_latency_microseconds histogram"]
    for b in PROM_BUCKETS_US:
        out.append(f'sor_decision_latency_microseconds_bucket{{{lab},le="{b}"}} {int((dec <= b).sum())}')
    out.append(f'sor_decision_latency_microseconds_bucket{{{lab},le="+Inf"}} {dec.size}')
    out.append(f"sor_decision_latency_microseconds_sum{{{lab}}} {dec.sum():.3f}")
    out.append(f"sor_decision_latency_microseconds_count{{{lab}}} {dec.size}")
    for name, key, help_ in [
        ("sor_implementation_shortfall_bps", "is_bps", "Notional-weighted implementation shortfall."),
        ("sor_fill_rate", "fill_rate", "Filled lots / ordered lots."),
        ("sor_stale_policy_frequency", "stale_policy_frequency", "Decisions served by the stale-policy fallback."),
        ("sor_child_orders_total", "children", "Child orders sent."),
        ("sor_rejects_total", "rejects", "Child orders rejected."),
        ("sor_solver_fallbacks_total", "solver_fallbacks", "Optimizer problems solved by the classical fallback."),
    ]:
        if key in metrics:
            kind = "counter" if name.endswith("_total") else "gauge"
            out += [f"# HELP {name} {help_}", f"# TYPE {name} {kind}", f"{name}{{{lab}}} {metrics[key]}"]
    return "\n".join(out) + "\n"
