"""Solver quality, cost and scaling on static instances with known optima.

Every instance is solved by exhaustive enumeration first; all other solvers
are scored against that optimum. For QAOA the script also reports the
probability that a single shot is an optimal, feasible bitstring *before*
repair, next to the same probability for a uniformly random bitstring.

    python benchmarks/static_benchmark.py [--instances 20] [--quick]
"""
import argparse
import csv
import itertools
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qsor.optimizer import classical, quantum, registry  # noqa: E402
from qsor.optimizer.instances import random_instance  # noqa: E402
from qsor.optimizer.qubo import build_qubo  # noqa: E402

SIZES = [(2, 2), (2, 4), (3, 3), (3, 4), (4, 3), (4, 4)]   # (venues, units per venue)
SOLVERS = ["best_price", "pro_rata", "greedy", "cpsat", "sa", "random_repair", "qaoa"]
CFG = {"qaoa": {"layers": 2, "maxiter": 40, "shots": 256, "max_qubits": 16, "warm_start": False},
       "sa": {"sweeps": 60, "restarts": 2}, "random_repair": {"shots": 256}}


def feasible_states(p, qubo):
    out = {}
    for x in itertools.product(*[range(n + 1) for n in p.n_units]):
        if sum(x) == p.target_units:
            out[sum(1 << i for i, (v, k) in enumerate(qubo.var_map) if x[v] == k)] = p.cost(x)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instances", type=int, default=20)
    ap.add_argument("--quick", action="store_true", help="sizes up to 12 qubits, 5 instances")
    ap.add_argument("--out", default=str(Path(__file__).parent / "results"))
    args = ap.parse_args()
    sizes = SIZES[:4] if args.quick else SIZES
    n_inst = 5 if args.quick else args.instances
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for V, m in sizes:
        for inst in range(n_inst):
            rng = np.random.default_rng([V, m, inst])        # predeclared instance seeds
            p, fine = random_instance(rng, V, m, unit_lots=1 + inst % 2)
            t0 = time.perf_counter()
            x_star = classical.exact(p)
            t_exact = time.perf_counter() - t0
            c_star = p.cost(x_star)
            qubo = build_qubo(p)
            feas = feasible_states(p, qubo)
            c_worst = max(feas.values())
            fine_cost = fine.cost(classical.exact(fine))
            base = {"venues": V, "units": m, "instance": inst, "qubits": qubo.n,
                    "feasible_allocations": len(feas), "optimum": c_star,
                    "granularity_gap_pct": 100 * (c_star - fine_cost) / abs(fine_cost) if fine_cost else 0.0}
            rows.append(dict(base, solver="exact", cost=c_star, optimal=1, gap_pct=0.0,
                             wall_ms=t_exact * 1e3, evals=0))
            for s in SOLVERS:
                t0 = time.perf_counter()
                try:
                    x, info = registry.solve(s, p, np.random.default_rng([V, m, inst, 99]), CFG)
                    err = ""
                except Exception as e:      # failed runs are reported, not dropped
                    x, info, err = None, {}, repr(e)
                wall = time.perf_counter() - t0
                row = dict(base, solver=s, wall_ms=wall * 1e3, evals=info.get("evals", 0), error=err)
                if x is not None:
                    c = p.cost(x)
                    row.update(cost=c, optimal=int(c <= c_star + 1e-9),
                               gap_pct=100 * (c - c_star) / (c_worst - c_star) if c_worst > c_star else 0.0)
                if s == "qaoa" and x is not None:
                    en = qubo.energies()
                    mean, std = quantum.normalize(en)
                    L = CFG["qaoa"]["layers"]
                    psi = quantum.qaoa_state((en - mean) / std, info["theta"][:L], info["theta"][L:])
                    probs = np.abs(psi) ** 2
                    opt_states = [z for z, c in feas.items() if c <= c_star + 1e-9]
                    row.update(p_opt_per_shot=float(probs[opt_states].sum()),
                               p_opt_uniform=len(opt_states) / probs.size,
                               p_feasible_per_shot=float(probs[list(feas)].sum()),
                               p_feasible_uniform=len(feas) / probs.size,
                               t_optimize_ms=info["t_optimize"] * 1e3, t_encode_ms=info["t_encode"] * 1e3,
                               t_sample_ms=info["t_sample"] * 1e3, t_decode_ms=info["t_decode"] * 1e3)
                rows.append(row)
        print(f"done: {V} venues x {m} units")

    keys = []
    for r in rows:
        keys += [k for k in r if k not in keys]
    with open(out / "static_benchmark.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)

    def agg(sel, key):
        v = [r[key] for r in sel if r.get(key) not in (None, "")]
        return float(np.mean(v)) if v else float("nan")

    md = ["# Static solver benchmark", "",
          f"{n_inst} random instances per size, seeds `[venues, units, instance]`. Optimum from exhaustive "
          "enumeration. Gap is the cost excess as a percentage of the feasible cost range (worst - best). "
          "QAOA: 2 layers, 40 COBYLA iterations, 256 shots, noiseless statevector simulation on CPU. "
          "`random_repair` draws 256 uniform bitstrings and uses the same decode + repair.", "",
          "| Venues x units | Qubits | Feasible allocs | Solver | Optimal found | Mean gap % | Wall ms | Evals | Failed |",
          "|---|---|---|---|---|---|---|---|---|"]
    for V, m in sizes:
        for s in ["exact"] + SOLVERS:
            sel = [r for r in rows if (r["venues"], r["units"], r["solver"]) == (V, m, s)]
            ok = [r for r in sel if "cost" in r]
            md.append(f"| {V} x {m} | {agg(sel, 'qubits'):.1f} | {agg(sel, 'feasible_allocations'):.0f} | {s} | "
                      f"{sum(r['optimal'] for r in ok)}/{len(sel)} | {agg(ok, 'gap_pct'):.2f} | "
                      f"{agg(sel, 'wall_ms'):.3f} | {agg(sel, 'evals'):.0f} | {len(sel) - len(ok)} |")
    md += ["", "## QAOA sampling quality before repair", "",
           "Probability that one measured bitstring is feasible / optimal, against a uniformly random bitstring.", "",
           "| Venues x units | Qubits | P(feasible) QAOA | P(feasible) uniform | P(optimal) QAOA | P(optimal) uniform | Optimize ms | Sample ms | Decode+repair ms |",
           "|---|---|---|---|---|---|---|---|---|"]
    for V, m in sizes:
        sel = [r for r in rows if (r["venues"], r["units"], r["solver"]) == (V, m, "qaoa") and "p_opt_per_shot" in r]
        md.append(f"| {V} x {m} | {agg(sel, 'qubits'):.1f} | {agg(sel, 'p_feasible_per_shot'):.4f} | "
                  f"{agg(sel, 'p_feasible_uniform'):.4f} | {agg(sel, 'p_opt_per_shot'):.4f} | "
                  f"{agg(sel, 'p_opt_uniform'):.5f} | {agg(sel, 't_optimize_ms'):.1f} | "
                  f"{agg(sel, 't_sample_ms'):.2f} | {agg(sel, 't_decode_ms'):.2f} |")
    md += ["", "## Cost of coarse allocation units", "",
           "Excess cost of the best allocation at the QUBO's granularity over the best one-lot allocation.", "",
           "| Venues x units | Mean granularity gap % |", "|---|---|"]
    for V, m in sizes:
        sel = [r for r in rows if (r["venues"], r["units"], r["solver"]) == (V, m, "exact")]
        md.append(f"| {V} x {m} | {agg(sel, 'granularity_gap_pct'):.2f} |")
    (out / "static_benchmark.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
