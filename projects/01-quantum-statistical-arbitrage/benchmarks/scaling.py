"""Scaling benchmark: solver quality and wall time versus universe size.

    python benchmarks/scaling.py --runs runs/ [--sizes 8 10 12 14 16] [--quick]

Random planted instances (one strongly "cointegrated" cluster hidden among
decoys) are solved by every solver with the same budget; the exact optimum
comes from brute force. QAOA uses the fast numpy backend, which is verified
against the Qiskit circuit in tests/test_qaoa.py.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from project.research.qubo import BasketProblem  # noqa: E402
from project.solvers.base import Budget  # noqa: E402
from project.solvers.registry import STOCHASTIC, run_solver  # noqa: E402

SOLVERS = ["greedy", "cpsat", "random_search", "simulated_annealing", "qaoa"]


def planted_instance(n: int, seed: int, k: int = 4, n_sectors: int = 4) -> BasketProblem:
    rng = np.random.default_rng(seed)
    C = rng.normal(0.0, 0.35, (n, n))
    C = (C + C.T) / 2
    cluster = rng.choice(n, size=k + 1, replace=False)
    for i in cluster:
        for j in cluster:
            if i != j:
                C[i, j] -= 0.6
    np.fill_diagonal(C, 0.0)
    return BasketProblem(
        symbols=[f"A{i}" for i in range(n)],
        linear=rng.uniform(0.0, 0.1, n),
        quad=C,
        sectors=np.arange(n) % n_sectors,
        basket_size=k,
        max_per_sector=2,
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", type=Path, default=Path("runs"))
    ap.add_argument("--sizes", type=int, nargs="+", default=[8, 10, 12, 14, 16, 18])
    ap.add_argument("--instances", type=int, default=3)
    ap.add_argument("--seeds", type=int, nargs="+", default=[11, 23, 37])
    ap.add_argument("--max-evals", type=int, default=64)
    ap.add_argument("--quick", action="store_true", help="sizes 8/10/12, 2 instances, 2 seeds")
    args = ap.parse_args(argv)
    if args.quick:
        args.sizes, args.instances, args.seeds = [8, 10, 12], 2, args.seeds[:2]

    budget = Budget(args.max_evals, 5.0)
    qaoa_kwargs = {"backend": "numpy", "p": 2, "shots": 256, "maxiter": 40, "cvar_alpha": 0.5}
    rows, t0 = [], time.perf_counter()
    for n in args.sizes:
        for inst in range(args.instances):
            problem = planted_instance(n, seed=1000 * n + inst)
            vals = []
            for combo in problem.feasible_baskets():
                idx = list(combo)
                vals.append(problem.linear[idx].sum() + np.triu(problem.quad[np.ix_(idx, idx)], 1).sum())
            vals = np.sort(np.array(vals))
            opt, denom = vals[0], np.median(vals) - vals[0]
            for name in SOLVERS:
                for seed in args.seeds if name in STOCHASTIC else [-1]:
                    res = run_solver(name, problem, budget, None if seed < 0 else seed, **(qaoa_kwargs if name == "qaoa" else {}))
                    rows.append(
                        {
                            "n": n, "instance": inst, "solver": name, "seed": seed,
                            "n_feasible": len(vals),
                            "norm_gap": float((res.objective - opt) / denom),
                            "raw_norm_gap": float((res.raw_best_objective - opt) / denom),
                            "is_optimal": bool(res.objective <= opt + 1e-9),
                            "raw_feasible_rate": res.raw_feasible_rate,
                            "t_total": res.wall_time, "n_evals": res.n_evals, "n_shots": res.n_shots,
                            "failed": res.status.startswith("fallback"), "status": res.status,
                        }
                    )
        done = pd.DataFrame([r for r in rows if r["n"] == n])
        print(f"N={n}: " + "  ".join(f"{s} gap={g:.3f}" for s, g in done.groupby("solver")["norm_gap"].mean().items()))

    out = args.runs / "scaling"
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / "scaling.csv", index=False)
    manifest = {
        "run_id": "scaling",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "solver_family": "scaling",
        "sizes": args.sizes, "instances": args.instances, "seeds": args.seeds,
        "budget": {"max_evals": args.max_evals, "time_limit_s": 5.0},
        "qaoa": qaoa_kwargs,
        "instance_generator": "planted cluster of K+1 assets, K=4, 4 sectors, max 2 per sector",
        "elapsed_s": round(time.perf_counter() - t0, 2),
        "files": ["scaling.csv"],
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"wrote {len(rows)} rows to {out / 'scaling.csv'} in {manifest['elapsed_s']}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
