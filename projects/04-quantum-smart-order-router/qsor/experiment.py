"""Experiment orchestration: calibration, simulation runs and run manifests."""
from __future__ import annotations

import csv
import datetime as dt
import json
import platform
import sys
import time
import traceback
from pathlib import Path

import numpy as np

from . import __version__
from .config import base_dir, config_hash, public_config, venues
from .core import BUY, MS, SELL, Order, opp_book
from .exchange_sim.data import load_for
from .exchange_sim.venue import VenueSim
from .optimizer import registry
from .optimizer.problem import CostModel
from .optimizer.service import PolicyOptimizer, market_ladders
from .router.policy import PolicyStore
from .router.router import LiveRouter, SnapshotRouter
from .sim.engine import Engine
from .sim.tca import parent_rows, prometheus_text, summarize

REFERENCE_SOLVERS = ("exact", "cpsat")


def cost_model(cfg, fill_prob, nonfill_penalty) -> CostModel:
    inst, vs = cfg["instrument"], venues(cfg)
    tick, lot = inst["tick_size"], inst["lot_size"]
    fade = tuple(
        tuple(b.est_fade_coeff if (a is not b and b.latency_ms - a.latency_ms > b.fade_delay_ms) else 0.0
              for b in vs) for a in vs)
    return CostModel(
        fee=tuple(v.taker_fee / tick for v in vs), fixed=tuple(v.order_fee / (tick * lot) for v in vs),
        impact=tuple(v.impact_coeff for v in vs), fill_prob=tuple(fill_prob), fade=fade,
        nonfill_penalty=float(nonfill_penalty))


def probe_fill_probs(cfg, ds, window):
    """Single-venue IOC probes on the training window -> fill ratio per venue."""
    exe, opt, vs = cfg["execution"], cfg["optimizer"], venues(cfg)
    rng = np.random.default_rng([opt["calibration_seed"], 1])
    sent, got = [0] * len(vs), [0] * len(vs)
    lo, hi = window[0] * ds.dt_ns + 100 * MS, window[1] * ds.dt_ns - 100 * MS
    for i in range(int(opt["calibration_probes"])):
        t = int(rng.integers(lo, hi))
        side = BUY if rng.random() < 0.5 else SELL
        for v, c in enumerate(vs):
            venue = VenueSim(v, c, ds, exe["replenish_halflife_ms"])
            if venue.is_down(t):
                continue
            books = [None] * len(vs)
            books[v] = venue.book(opp_book(side), t - int(c.md_latency_ms * MS))
            _, caps, _, far = market_ladders(side, books, exe["limit_ticks"], exe["participation"],
                                             [x.max_child_lots for x in vs])
            if far is None or caps[v] == 0:
                continue
            o = Order(i, 0, v, side, caps[v], far + side * exe["limit_ticks"], "IOC")
            reps = venue.new_order(o, t + int(c.latency_ms * MS))
            sent[v] += caps[v]
            got[v] += sum(r.qty for r in reps if r.kind == "FILL")
    return [round(got[v] / sent[v], 4) if sent[v] else 1.0 for v in range(len(vs))]


def calibrate(cfg, ds, force=False) -> dict:
    """Fit router-side estimates on train, choose the non-fill penalty on validation.

    The result does not depend on which solver is later evaluated, so every
    solver is compared under the same objective.
    """
    key = config_hash(cfg, ["instrument", "data", "venues", "orders", "execution", "optimizer"])
    path = base_dir(cfg) / cfg["runs_root"] / "_calibration" / f"{cfg['name']}-{key}.json"
    if path.exists() and not force:
        return json.loads(path.read_text())
    fill_prob = probe_fill_probs(cfg, ds, ds.splits["train"])
    grid, scores = cfg["optimizer"]["nonfill_penalty_grid"], {}
    for lam in grid:
        _, _, m = simulate(cfg, ds, "validation", "greedy", cfg["optimizer"]["calibration_seed"],
                           cost_model(cfg, fill_prob, lam))
        scores[str(lam)] = m["is_bps"]
    best = min(grid, key=lambda g: (scores[str(g)], g))
    out = {"fill_prob": fill_prob, "nonfill_penalty": best, "validation_is_bps": scores,
           "fitted_on": "train", "selected_on": "validation", "selection_solver": "greedy"}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2))
    return out


def simulate(cfg, ds, window_name, solver, seed, model, fix_log_limit=0):
    window = ds.splits[window_name]
    max_child = [v.max_child_lots for v in venues(cfg)]
    if solver in registry.LIVE_SOLVERS:
        router, optimizer, store = LiveRouter(model), None, None
    else:
        store = PolicyStore()
        router = SnapshotRouter(store, int(cfg["optimizer"]["max_policy_age_ms"] * MS))
        optimizer = PolicyOptimizer(solver, model, cfg, max_child, seed)
    eng = Engine(cfg, ds, window, router, optimizer, store, seed, fix_log_limit=fix_log_limit).run()
    rows = parent_rows(eng)
    return eng, rows, summarize(eng, rows)


def _versions():
    out = {"python": sys.version.split()[0], "qsor": __version__, "platform": platform.platform(),
           "machine": platform.machine()}
    for mod in ("numpy", "scipy", "ortools", "qiskit", "yaml"):
        try:
            out[mod] = __import__(mod).__version__
        except Exception:
            out[mod] = None
    return out


def experiment_id(cfg) -> str:
    return f"{cfg['name']}-{config_hash(cfg)}"


def run_experiment(cfg, solver_arg: str, window="test", log=print) -> Path:
    ds = load_for(cfg)
    calib = calibrate(cfg, ds)
    model = cost_model(cfg, calib["fill_prob"], calib["nonfill_penalty"])
    root = base_dir(cfg) / cfg["runs_root"] / experiment_id(cfg)
    log(f"experiment {root.name}: calibration fill_prob={calib['fill_prob']} "
        f"nonfill_penalty={calib['nonfill_penalty']}")
    for solver in registry.expand(solver_arg):
        sdir = root / solver
        sdir.mkdir(parents=True, exist_ok=True)
        seeds = {}
        t_all = time.perf_counter()
        for i, seed in enumerate(cfg["seeds"]):
            t0 = time.perf_counter()
            try:
                eng, rows, metrics = simulate(cfg, ds, window, solver, seed, model,
                                              fix_log_limit=2000 if i == 0 else 0)
                out = sdir / f"seed_{seed}"
                out.mkdir(exist_ok=True)
                (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
                with open(out / "parents.csv", "w", newline="") as f:
                    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["parent"])
                    w.writeheader()
                    w.writerows(rows)
                np.savetxt(out / "decision_latency_ns.csv", np.asarray(eng.decision_ns, dtype=np.int64), fmt="%d")
                (out / "metrics.prom").write_text(
                    prometheus_text(eng, metrics, {"solver": solver, "seed": seed, "experiment": root.name}))
                if eng.fix_log:
                    (out / "messages.fix").write_text("\n".join(eng.fix_log) + "\n")
                seeds[str(seed)] = {"status": "ok", "wall_s": round(time.perf_counter() - t0, 3)}
                log(f"  {solver:14s} seed={seed:<4d} IS={metrics['is_bps']:.3f}bps fill={metrics['fill_rate']:.3f} "
                    f"decision p50/p99={metrics['decision_us_p50']:.1f}/{metrics['decision_us_p99']:.1f}us "
                    f"({seeds[str(seed)]['wall_s']}s)")
            except Exception as e:
                seeds[str(seed)] = {"status": "failed", "error": repr(e), "traceback": traceback.format_exc()}
                log(f"  {solver:14s} seed={seed:<4d} FAILED: {e!r}")
        manifest = {
            "experiment": root.name, "solver": solver, "solver_kind": registry.KIND[solver],
            "hardware": "none (CPU simulation only; no QPU results)",
            "window": window, "window_steps": list(ds.splits[window]), "seeds": seeds,
            "dataset": {"path": str(ds.path), "sha256": ds.manifest["sha256"]},
            "calibration": calib, "config": public_config(cfg), "versions": _versions(),
            "wall_s": round(time.perf_counter() - t_all, 3),
            "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        }
        (sdir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return root
