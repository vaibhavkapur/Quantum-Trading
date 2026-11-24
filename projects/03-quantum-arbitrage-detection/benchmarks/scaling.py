"""Scaling of search cost with the number of currencies.

    python benchmarks/scaling.py --output reports/scaling.csv

Random arbitrage-free weight matrices with planted negative cycles, three
seeds per size. Reports wall time, objective evaluations and recall against
exhaustive enumeration for every solver, plus the QUBO register size. QAOA is
only run while its largest sub-problem fits `--max-qubits`; beyond that the
row records that it was skipped rather than extrapolating.
"""

from __future__ import annotations

import argparse
import copy
import csv
import time
from pathlib import Path

import numpy as np

from qarb.config import DEFAULTS
from qarb.graph.cycles import enumerate_cycles
from qarb.solvers import SearchContext, solve

SOLVERS = ["enumeration", "bellman_ford", "bounded_dp", "random_search", "qubo_sa", "qaoa"]


def instance(n: int, rng: np.random.Generator, n_planted: int = 3, max_len: int = 4) -> np.ndarray:
    """Potential-difference weights plus positive costs (no arbitrage), then planted negative cycles."""
    phi = rng.normal(size=n)
    W = phi[None, :] - phi[:, None] + rng.uniform(5e-4, 2e-3, size=(n, n))
    for _ in range(n_planted):
        L = int(rng.integers(2, min(max_len, n) + 1))
        cyc = rng.permutation(n)[:L]
        for i in range(L):
            u, v = cyc[i], cyc[(i + 1) % L]
            W[u, v] = phi[v] - phi[u] - rng.uniform(2e-4, 1.5e-3)
    np.fill_diagonal(W, np.inf)
    return W


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output", default="reports/scaling.csv")
    ap.add_argument("--sizes", default="4,5,6,8,12,16,24")
    ap.add_argument("--seeds", default="1,2,3")
    ap.add_argument("--max-len", type=int, default=4)
    ap.add_argument("--max-qubits", type=int, default=16)
    args = ap.parse_args()

    cfg = copy.deepcopy(DEFAULTS)
    cfg["graph"]["max_cycle_len"] = args.max_len
    cfg["qaoa"].update(max_qubits=args.max_qubits, fallback=None, circuit_metrics=False)
    cfg["budget"]["max_objective_evals"] = 10**9
    rows = []
    for n in [int(x) for x in args.sizes.split(",")]:
        for seed in [int(x) for x in args.seeds.split(",")]:
            W = instance(n, np.random.default_rng([seed, n]), max_len=args.max_len)
            truth = {c for c, w in enumerate_cycles(W, 2, args.max_len)[0] if w < -1e-12}
            for name in SOLVERS:
                row = {"n_currencies": n, "seed": seed, "solver": name, "max_len": args.max_len,
                       "negative_cycles": len(truth),
                       "qubo_vars_anchored": (n - 1) * (min(args.max_len, n) - 1),
                       "qubo_vars_unanchored": n * min(args.max_len, n)}
                if name == "random_search":
                    cfg["random_search"] = {"samples": 5000}
                ctx = SearchContext(W, W, 2, args.max_len, np.random.default_rng([seed, n, 7]), cfg)
                t0 = time.perf_counter()
                try:
                    res = solve(name, ctx)
                    found = set(res.cycles)
                    row.update(status="ok", wall_ms=(time.perf_counter() - t0) * 1e3,
                               objective_evals=res.stats["objective_evals"],
                               recall=len(found & truth) / len(truth) if truth else None,
                               spurious=len(found - truth))
                except Exception as exc:
                    row.update(status=f"skipped: {exc}", wall_ms=None, objective_evals=None, recall=None, spurious=None)
                rows.append(row)
                rec = "n/a" if row["recall"] is None else f"{row['recall']:.2f}"
                ms = "n/a" if row["wall_ms"] is None else f"{row['wall_ms']:.2f}"
                print(f"N={n:<3} seed={seed} {name:<14} recall={rec:<5} wall_ms={ms:<10} {row['status']}")
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
