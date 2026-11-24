"""Hot-path decision latency, alone and while the optimizer thread is solving.

The simulator runs the optimizer inline (so simulated outcomes are
reproducible); this script measures what the simulator cannot: interference
between a concurrently running optimizer and the route decision in one
CPython process.

    python benchmarks/hotpath_benchmark.py [--decisions 200000]
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qsor.config import load_config  # noqa: E402
from qsor.core import BUY, SELL  # noqa: E402
from qsor.experiment import cost_model  # noqa: E402
from qsor.optimizer.service import AsyncPolicyService, PolicyOptimizer  # noqa: E402
from qsor.router.policy import PolicyStore  # noqa: E402
from qsor.router.router import LiveRouter, SnapshotRouter  # noqa: E402


def books():
    mk = lambda px, sizes: [(list(px), list(s)) for s in sizes]
    sizes = [(4, 5, 5), (3, 4, 4), (3, 3, 4)]
    return {BUY: mk((10001, 10002, 10003), sizes), SELL: mk((9999, 9998, 9997), sizes)}


def measure(router, n, ctx):
    rng = np.random.default_rng(0)
    qty = rng.integers(1, 9, n).tolist()
    caps = rng.integers(0, 6, (n, 3)).tolist()
    best = [10001, 10001, 10002]
    lat = np.empty(n, dtype=np.int64)
    clock = time.perf_counter_ns
    for i in range(n):
        t0 = clock()
        router.decide(BUY, qty[i], caps[i], best, t0, ctx)
        lat[i] = clock() - t0
    return lat / 1e3


def row(name, us, extra=""):
    return (f"| {name} | {np.percentile(us, 50):.2f} | {np.percentile(us, 99):.2f} | "
            f"{np.percentile(us, 99.9):.2f} | {us.max():.0f} | {extra} |")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--decisions", type=int, default=200_000)
    ap.add_argument("--config", default=str(Path(__file__).resolve().parents[1] / "configs" / "demo.yaml"))
    ap.add_argument("--out", default=str(Path(__file__).parent / "results"))
    args = ap.parse_args()
    cfg = load_config(args.config)
    model = cost_model(cfg, [1.0] * len(cfg["venues"]), 2.0)
    max_child = [v["max_child_lots"] if "max_child_lots" in v else 20 for v in cfg["venues"]]
    n = args.decisions
    lines = ["# Hot-path decision latency", "",
             f"{n} decisions per row, wall clock (`perf_counter_ns`), single CPython process. "
             "Microseconds.", "", "| Scenario | p50 | p99 | p99.9 | max | Policies published |", "|---|---|---|---|---|---|"]

    for solver in ("exact", "qaoa"):
        store = PolicyStore()
        opt = PolicyOptimizer(solver, model, cfg, max_child, seed=1)
        snap, _ = opt.compute(books(), 0)
        store.publish(snap)
        router = SnapshotRouter(store, max_age_ns=10**18)
        if solver == "exact":
            measure(router, 20_000, None)  # warm-up
            lines.append(row("snapshot router, optimizer idle", measure(router, n, None), "1"))
        svc = AsyncPolicyService(opt, store, books, time.perf_counter_ns, 0.0)
        svc.start()
        time.sleep(0.2)
        us = measure(router, n, None)
        svc.stop()
        assert svc.errors == 0
        lines.append(row(f"snapshot router, `{solver}` optimizer thread solving continuously", us,
                         str(store.published)))

    ladders = [((10001, 4), (10002, 5)), ((10001, 3), (10002, 4)), ((10002, 3),)]
    lines.append(row("live greedy solve on the hot path (no snapshot)", measure(LiveRouter(model), n, (ladders, 10001)), "–"))
    lines += ["", "Caps are randomized per decision; the snapshot router never calls a solver. "
              "With the optimizer on a thread of the same interpreter, the tail reflects GIL hand-offs, "
              "which is why a production layout would put the optimizer in a separate process (or the "
              "router in native code) and share snapshots through shared memory."]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "hotpath_benchmark.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
