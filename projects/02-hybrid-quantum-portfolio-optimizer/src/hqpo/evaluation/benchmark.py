"""Benchmark harness: identical instances, identical budget, every solver, several seeds.

Produces four tidy tables (written as CSV by ``python -m hqpo.benchmark``):

* ``results``       one row per (instance, solver, seed) with objective gap to the reference
* ``penalty_sweep`` QUBO ground-state validity (and QAOA quality) versus penalty scale
* ``frontier``      risk/return points across risk aversions
* ``qubit_growth``  qubit requirement versus asset count
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import budget_from, constraints_from, make_solver
from ..data.synthetic import random_instance
from ..models.qubo import build_qubo, qubit_count
from ..solvers import SOLVERS
from ..solvers.base import SolverUnavailable
from ..solvers.classical import brute_force_optimum, solve_relaxation


def _instance(spec: dict, seed: int, risk_aversion: float):
    return random_instance(int(spec["n"]), constraints_from(spec), seed,
                           n_sectors=int(spec.get("n_sectors", 3)), risk_aversion=risk_aversion,
                           turnover_penalty=float(spec.get("turnover_penalty", 0.0)))


def _label(spec: dict) -> str:
    bits = [f"n={spec['n']}", f"L={spec['lots']}", f"wmax={spec['max_weight']}"]
    if spec.get("cardinality"):
        bits.append(f"K<={spec['cardinality']}")
    if spec.get("sector_cap"):
        bits.append(f"sec<={spec['sector_cap']}")
    if spec.get("mode", "long_only") != "long_only":
        bits.append("MN")
    return " ".join(bits)


def reference_optimum(problem, max_states: int = 3_000_000):
    """Exact optimum by enumeration when affordable, else ``(None, nan)``."""
    try:
        z, f, _ = brute_force_optimum(problem, max_states)
        return z, f
    except SolverUnavailable:
        return None, float("nan")


def random_feasible_mean(problem, k: int = 40, seed: int = 0) -> float:
    """Mean objective of random feasible portfolios: the "no optimisation" yardstick."""
    from ..models.repair import repair

    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(k):
        z0 = np.zeros(problem.m, dtype=int)
        for c in problem.budget_constraints:
            for j in rng.choice(np.flatnonzero(c.a != 0), size=problem.lots):
                z0[j] += 1
        z, ok = repair(problem, z0)
        if ok:
            vals.append(problem.objective(z))
    return float(np.mean(vals)) if vals else float("nan")


def run_solver_grid(cfg: dict, log=print) -> pd.DataFrame:
    bm = cfg["benchmark"]
    budget = budget_from(cfg)
    rows = []
    for spec in bm["instances"]:
        for iseed in bm["instance_seeds"]:
            problem = _instance(spec, iseed, float(bm.get("risk_aversion", 4.0)))
            _, f_exact = reference_optimum(problem)
            _, bound, _ = solve_relaxation(problem)
            f_random = random_feasible_mean(problem)
            qc = qubit_count(problem)
            batch = []
            for name in bm["solvers"]:
                seeds = cfg["seeds"] if SOLVERS[name].stochastic else [None]
                for seed in seeds:
                    res = make_solver(cfg, name).solve(problem, budget, seed=seed)
                    rec = res.record()
                    rec.update(instance=_label(spec), n=problem.n, instance_seed=iseed, seed=seed,
                               required_qubits=qc["total"], relaxation_bound=bound,
                               exact_optimum=f_exact)
                    batch.append(rec)
            # reference = exact optimum, else a proven-optimal MIQP, else best objective seen
            feas = [r["objective"] for r in batch if r["feasible"]]
            proven = [r["objective"] for r in batch if r["feasible"] and r.get("optimal_proven")]
            if np.isfinite(f_exact):
                ref, kind = f_exact, "brute-force"
            elif proven:
                ref, kind = min(proven), "proven-optimal MIP"
            elif feas:
                ref, kind = min(feas), "best-known"
            else:
                ref, kind = float("nan"), "none"
            for r in batch:
                r["reference"], r["reference_kind"] = ref, kind
                r["gap"] = r["objective"] - ref if r["feasible"] else float("nan")
                # normalised gap: 0 % = reference optimum, 100 % = average random feasible portfolio
                denom = f_random - ref
                r["random_feasible_mean"] = f_random
                r["gap_pct"] = (100.0 * r["gap"] / denom
                                if r["feasible"] and denom > 1e-12 else float("nan"))
            rows += batch
            ok = sum(r["feasible"] for r in batch)
            log(f"  {_label(spec):44s} seed={iseed} qubits={qc['total']:4d} ref={kind:18s} "
                f"feasible {ok}/{len(batch)}")
    return pd.DataFrame(rows)


def run_penalty_sweep(cfg: dict, log=print) -> pd.DataFrame:
    """For tiny instances: is the QUBO ground state feasible *and* the true optimum?"""
    sweep = cfg["benchmark"].get("penalty_sweep")
    if not sweep:
        return pd.DataFrame()
    budget = budget_from(cfg)
    rows = []
    for spec in sweep["instances"]:
        for iseed in cfg["benchmark"]["instance_seeds"]:
            problem = _instance(spec, iseed, float(cfg["benchmark"].get("risk_aversion", 4.0)))
            _, f_exact = reference_optimum(problem)
            for scale in sweep["scales"]:
                qubo = build_qubo(problem, method=cfg["penalty"]["method"], scale=scale)
                E = qubo.energies_all()
                z = qubo.decode(qubo.index_to_bits(int(np.argmin(E))))
                feasible = problem.is_feasible(z)
                row = {"instance": _label(spec), "instance_seed": iseed, "scale": scale,
                       "penalty": qubo.penalty, "n_qubits": qubo.n_qubits,
                       "ground_state_feasible": feasible,
                       "ground_state_optimal": bool(feasible and abs(problem.objective(z) - f_exact) < 1e-9)}
                if sweep.get("qaoa", True):
                    solver = make_solver({**cfg, "penalty": {**cfg["penalty"], "scale": scale}}, "qaoa")
                    solver.params["circuit_metrics"] = False
                    res = solver.solve(problem, budget, seed=cfg["seeds"][0])
                    row.update(qaoa_gap=res.objective - f_exact if res.feasible else float("nan"),
                               qaoa_raw_violation=res.raw_violation,
                               qaoa_p_ground_state=res.meta.get("p_ground_state"),
                               qaoa_sample_feasible_fraction=res.meta.get("sample_feasible_fraction"))
                rows.append(row)
        log(f"  penalty sweep done: {_label(spec)}")
    return pd.DataFrame(rows)


def run_frontier(cfg: dict, problem_factory, log=print) -> pd.DataFrame:
    """Risk/return points per solver across risk aversions (turnover term off)."""
    fr = cfg["benchmark"].get("frontier")
    if not fr:
        return pd.DataFrame()
    budget = budget_from(cfg)
    rows = []
    for lam in fr["risk_aversions"]:
        problem = problem_factory(lam)
        z_cont, bound, _ = solve_relaxation(problem)
        if z_cont is not None:
            w = problem.weights(z_cont)
            rows.append({"solver": "continuous relaxation", "risk_aversion": lam, "seed": None,
                         "risk": float(np.sqrt(w @ problem.sigma @ w)),
                         "expected_return": float(problem.mu @ w), "objective": bound,
                         "feasible": True})
        for name in fr["solvers"]:
            seeds = cfg["seeds"] if SOLVERS[name].stochastic else [None]
            for seed in seeds:
                res = make_solver(cfg, name).solve(problem, budget, seed=seed)
                row = {"solver": name, "risk_aversion": lam, "seed": seed,
                       "objective": res.objective, "feasible": res.feasible, "status": res.status}
                if res.feasible:
                    d = problem.describe(res.z)
                    row.update(risk=d["risk"], expected_return=d["expected_return"])
                rows.append(row)
        log(f"  frontier lambda={lam} done")
    return pd.DataFrame(rows)


def qubit_growth(cfg: dict) -> pd.DataFrame:
    """Qubit requirement as the universe grows, for three constraint sets (lots = n, w_max = 4 lots)."""
    sizes = cfg["benchmark"].get("qubit_growth", {}).get("sizes", [])
    rows = []
    for n in sizes:
        base = {"lots": n, "max_weight": min(1.0, 4.0 / n)}
        variants = {
            "budget + position limit": base,
            "+ cardinality (n/2)": {**base, "cardinality": max(n // 2, int(np.ceil(n / 4)))},
            "+ cardinality + sector caps": {**base, "cardinality": max(n // 2, int(np.ceil(n / 4))),
                                            "sector_cap": 0.4},
            # same 4-lot position cap per leg (leg exposure 0.5 -> lot = 0.5 / n)
            "market neutral + cardinality": {"lots": n, "max_weight": min(0.5, 2.0 / n),
                                             "mode": "market_neutral", "cardinality": n // 2 + n // 4},
        }
        for label, spec in variants.items():
            try:
                problem = random_instance(n, constraints_from(spec), seed=0, n_sectors=4)
            except ValueError:
                continue
            qc = qubit_count(problem)
            rows.append({"n": n, "constraints": label, **qc,
                         "couplers_dense": qc["total"] * (qc["total"] - 1) // 2})
    return pd.DataFrame(rows)
