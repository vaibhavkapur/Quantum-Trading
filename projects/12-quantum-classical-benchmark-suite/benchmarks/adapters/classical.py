from __future__ import annotations

import math
from itertools import combinations

import numpy as np

from benchmarks.instances.schema import encode_qubo, feasible, objective
from .base import Context


class ClassicalAdapter:
    def __init__(self, name: str):
        self.name = name

    def solve(self, problem: dict, context: Context) -> dict:
        return getattr(self, self.name)(problem, context)

    def exact(self, p: dict, ctx: Context) -> dict:
        if p["size"] > 16:
            raise ValueError("exact enumeration is capped at 16 variables")
        best_value = math.inf
        with ctx.stage("execution"):
            for selected in combinations(range(p["size"]), p["cardinality"]):
                ctx.check()
                x = [int(i in selected) for i in range(p["size"])]
                if feasible(p, x):
                    ctx.charge(evaluations=1)
                    value = objective(p, x)
                    if value < best_value:
                        best_value = value
                        ctx.partial = {"candidate": x}
        return ctx.partial | {"termination": "optimal" if ctx.partial else "infeasible", "proven_optimal": bool(ctx.partial)}

    def milp(self, p: dict, ctx: Context) -> dict:
        with ctx.stage("setup"):
            from scipy.optimize import Bounds, LinearConstraint, milp
            from scipy.sparse import lil_matrix

        with ctx.stage("encoding"):
            n = p["size"]
            q = np.asarray(p["quadratic"])
            pairs = [(i, j) for i in range(n) for j in range(i + 1, n) if q[i, j] != 0]
            c = np.r_[p["linear"], [q[i, j] for i, j in pairs]]
        with ctx.stage("compilation"):
            # y_ij = x_i x_j; three McCormick constraints are exact for binary x.
            rows = 1 + len(p["conflicts"]) + 3 * len(pairs)
            a = lil_matrix((rows, len(c)), dtype=float)
            lower, upper = np.full(rows, -np.inf), np.zeros(rows)
            a[0, :n] = 1
            lower[0] = upper[0] = p["cardinality"]
            row = 1
            for i, j in p["conflicts"]:
                a[row, i] = a[row, j] = 1
                upper[row] = 1
                row += 1
            for index, (i, j) in enumerate(pairs, n):
                a[row, index], a[row, i] = 1, -1
                a[row + 1, index], a[row + 1, j] = 1, -1
                a[row + 2, i], a[row + 2, j], a[row + 2, index] = 1, 1, -1
                upper[row + 2] = 1
                row += 3
            constraints = LinearConstraint(a.tocsc(), lower, upper)
        ctx.resources.update(objective_evaluations=None, evaluation_unit="internal_MILP_evaluations_unavailable",
                             evaluation_budget_enforced=False, solver_engine="SciPy/HiGHS MILP")
        with ctx.stage("execution"):
            ctx.check()
            result = milp(c, integrality=np.r_[np.ones(n), np.zeros(len(pairs))],
                          bounds=Bounds(np.zeros(len(c)), np.ones(len(c))), constraints=constraints,
                          options={"time_limit": ctx.remaining_seconds(), "node_limit": ctx.budget.mip_nodes,
                                   "mip_rel_gap": 0.0})
        with ctx.stage("decoding"):
            candidate = None
            if result.x is not None and np.max(np.abs(result.x[:n] - np.rint(result.x[:n]))) < 1e-6:
                candidate = np.rint(result.x[:n]).astype(int).tolist()
            ctx.resources["mip_nodes"] = int(getattr(result, "mip_node_count", 0) or 0)
            gap = getattr(result, "mip_gap", None)
            return {"candidate": candidate, "termination": result.message, "proven_optimal": result.status == 0,
                    "solver_status": "ok" if result.status == 0 else "budget_exhausted" if result.status == 1 else "failed",
                    "solver_gap": float(gap) if gap is not None and np.isfinite(gap) else None}

    def greedy(self, p: dict, ctx: Context) -> dict:
        x = np.zeros(p["size"], dtype=int)
        conflicts = p["conflicts"]
        incumbent_value = math.inf
        with ctx.stage("execution"):
            for _ in range(p["cardinality"]):
                best, best_value = None, math.inf
                for i in np.flatnonzero(x == 0):
                    candidate = x.copy()
                    candidate[i] = 1
                    if any(candidate[a] and candidate[b] for a, b in conflicts):
                        continue
                    ctx.charge(evaluations=1)
                    value = objective(p, candidate)
                    if value < best_value:
                        best, best_value = candidate, value
                    if feasible(p, candidate):
                        if value < incumbent_value:
                            incumbent_value = value
                            ctx.partial = {"candidate": candidate.tolist()}
                if best is None:
                    break
                x = best
        return {"candidate": x.tolist(), "termination": "greedy construction"}

    def annealing(self, p: dict, ctx: Context) -> dict:
        with ctx.stage("encoding"):
            c, q, offset, penalty = encode_qubo(p)
            x = np.zeros(p["size"], dtype=int)
            x[ctx.rng.choice(p["size"], p["cardinality"], replace=False)] = 1
        best_value, best_energy = math.inf, math.inf

        def score(candidate):
            nonlocal best_value, best_energy
            ctx.charge(evaluations=1)
            energy = float(c @ candidate + candidate @ q @ candidate + offset)
            if energy < best_energy and best_value == math.inf:
                ctx.partial = {"candidate": candidate.tolist()}
                best_energy = energy
            if feasible(p, candidate):
                value = objective(p, candidate)
                if value < best_value:
                    best_value = value
                    ctx.partial = {"candidate": candidate.tolist()}
            return energy

        with ctx.stage("execution"):
            energy = score(x)
            for step in range(1, ctx.budget.max_evaluations):
                candidate = x.copy()
                # Cardinality-preserving swaps; conflict penalties remain in the energy.
                selected, unselected = np.flatnonzero(x), np.flatnonzero(1 - x)
                if not len(selected) or not len(unselected):
                    break
                candidate[ctx.rng.choice(selected)] = 0
                candidate[ctx.rng.choice(unselected)] = 1
                candidate_energy = score(candidate)
                temperature = max(1e-6, penalty * 0.5 * (1 - step / ctx.budget.max_evaluations) ** 3)
                if candidate_energy < energy or ctx.rng.random() < np.exp(min(0, (energy - candidate_energy) / temperature)):
                    x, energy = candidate, candidate_energy
        return ctx.partial | {"termination": "annealing schedule completed", "penalty": penalty}
