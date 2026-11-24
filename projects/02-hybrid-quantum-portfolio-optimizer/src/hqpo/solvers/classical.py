"""Classical baselines: brute force, convex relaxation, exact MIQP, CP-SAT, greedy, SA."""
from __future__ import annotations

import math
import time

import numpy as np

from ..models.problem import PortfolioProblem
from ..models.repair import local_search, repair
from .base import Budget, Solver, SolverUnavailable, Timer


# --------------------------------------------------------------------------- brute force
def enumerate_lots(problem: PortfolioProblem, max_states: int = 20_000_000):
    """Yield chunks of all lot vectors in the box that satisfy the budget equalities."""
    p = problem

    def compositions(nvars: int, total: int, ub: int):
        # all vectors in [0, ub]^nvars summing to total
        def rec(k, remaining):
            if k == nvars - 1:
                if remaining <= ub:
                    yield (remaining,)
                return
            lo = max(0, remaining - ub * (nvars - k - 1))
            for v in range(lo, min(ub, remaining) + 1):
                for tail in rec(k + 1, remaining - v):
                    yield (v,) + tail
        return rec(0, total)

    def count(nvars: int, total: int, ub: int) -> int:
        # bounded compositions via inclusion-exclusion
        return sum((-1) ** k * math.comb(nvars, k) * math.comb(total - k * (ub + 1) + nvars - 1, nvars - 1)
                   for k in range(nvars + 1) if total - k * (ub + 1) >= 0)

    ub = int(p.ub[0])
    per_leg = count(p.n, p.lots, ub)
    total = per_leg ** (2 if p.market_neutral else 1)
    if total > max_states:
        raise SolverUnavailable(f"brute force needs {total:.3g} states (> {max_states:.3g})")
    leg = np.array(list(compositions(p.n, p.lots, ub)), dtype=np.int16)
    if not p.market_neutral:
        for s in range(0, len(leg), 200_000):
            yield leg[s: s + 200_000]
        return
    chunk = max(1, 200_000 // max(len(leg), 1))
    for s in range(0, len(leg), chunk):
        a = leg[s: s + chunk]
        yield np.hstack([np.repeat(a, len(leg), axis=0), np.tile(leg, (len(a), 1))])


def brute_force_optimum(problem: PortfolioProblem, max_states: int = 20_000_000):
    """Exact constrained optimum by enumeration. Returns ``(z, objective, n_states)``."""
    best_z, best_f, seen = None, np.inf, 0
    for Z in enumerate_lots(problem, max_states):
        seen += len(Z)
        feas = problem.total_violation_batch(Z) == 0
        if not feas.any():
            continue
        Zf = Z[feas]
        f = problem.objective_batch(Zf)
        k = int(np.argmin(f))
        if f[k] < best_f:
            best_f, best_z = float(f[k]), Zf[k].astype(int)
    return best_z, best_f, seen


class BruteForce(Solver):
    name = "brute-force"

    def _solve(self, problem, budget, rng, timer):
        with timer.section("optimize"):
            z, f, seen = brute_force_optimum(problem, self.params.get("max_states", 20_000_000))
        return z, seen, {"optimal_proven": z is not None, "message": "no feasible point"}


# --------------------------------------------------------------------------- convex relaxation
def _cvx_model(problem: PortfolioProblem, integer: bool):
    import cvxpy as cp

    p = problem
    z = cp.Variable(p.m, integer=integer)
    # z'Az = || R z ||^2 with A = R'R (PSD by construction)
    evals, evecs = np.linalg.eigh(p.A)
    R = (evecs * np.sqrt(np.clip(evals, 0, None))).T
    obj = cp.sum_squares(R @ z) + p.b @ z + p.c0
    cons = [z >= 0, z <= p.ub]
    for c in p.budget_constraints:
        cons.append(c.a @ z == c.lo)
    for c in p.sector_constraints:
        cons += [c.a @ z <= c.hi, c.a @ z >= c.lo]
    return cp, z, obj, cons


def solve_relaxation(problem: PortfolioProblem):
    """Continuous convex relaxation (no lots, cardinality or exclusivity): a valid lower bound."""
    cp, z, obj, cons = _cvx_model(problem, integer=False)
    prob = cp.Problem(cp.Minimize(obj), cons)
    prob.solve(solver=cp.CLARABEL)
    if z.value is None:
        return None, float("nan"), prob.status
    return np.asarray(z.value, dtype=float), float(prob.value), prob.status


def round_to_lots(problem: PortfolioProblem, z_cont: np.ndarray) -> np.ndarray:
    """Largest-remainder rounding per budget leg (keeps each leg's lot total)."""
    z = np.floor(z_cont + 1e-9).astype(int)
    for c in problem.budget_constraints:
        members = np.flatnonzero(c.a != 0)
        short = c.lo - int(z[members].sum())
        frac = z_cont[members] - z[members]
        for k in np.argsort(-frac, kind="stable")[: max(short, 0)]:
            z[members[k]] += 1
    return np.clip(z, 0, problem.ub)


class CvxpyRelaxRound(Solver):
    """CVXPY mean-variance (min-variance when ``mu = 0``) relaxation, then rounding to lots."""

    name = "cvxpy-relax"

    def _solve(self, problem, budget, rng, timer):
        with timer.section("optimize"):
            z_cont, bound, status = solve_relaxation(problem)
        if z_cont is None:
            return None, 1, {"message": f"relaxation status {status}"}
        with timer.section("decode"):
            z = round_to_lots(problem, z_cont)
        return z, 1, {"lower_bound": bound, "backend": "cvxpy/CLARABEL"}


# --------------------------------------------------------------------------- exact MIQP
def _miqp_scip(problem: PortfolioProblem, time_limit: float):
    try:
        import pyscipopt  # noqa: F401
    except ImportError as exc:
        raise SolverUnavailable("SCIP (pyscipopt) is not installed") from exc
    cp, z, obj, cons = _cvx_model(problem, integer=True)
    p = problem
    if p.constraints.cardinality is not None and p.cardinality_binding:
        y = cp.Variable(p.n, boolean=True)
        for i, group in enumerate(p.groups):
            for j in group:
                cons.append(z[j] <= int(p.ub[j]) * y[i])
        cons.append(cp.sum(y) <= p.constraints.cardinality)
    if p.market_neutral:
        u = cp.Variable(p.n, boolean=True)
        for i, (j1, j2) in enumerate(p.exclusive_pairs):
            cons += [z[j1] <= int(p.ub[j1]) * u[i], z[j2] <= int(p.ub[j2]) * (1 - u[i])]
    prob = cp.Problem(cp.Minimize(obj), cons)
    prob.solve(solver=cp.SCIP, scip_params={"limits/time": float(max(time_limit, 0.1))})
    if z.value is None:
        return None, {"message": f"SCIP status {prob.status}", "backend": "cvxpy/SCIP"}
    return np.rint(z.value).astype(int), {
        "backend": "cvxpy/SCIP", "optimal_proven": prob.status == "optimal",
        "solver_status": prob.status}


def _cpsat(problem: PortfolioProblem, time_limit: float, seed: int = 0, scale: float = 1e7,
           workers: int = 1):
    from ortools.sat.python import cp_model

    p = problem
    model = cp_model.CpModel()
    z = [model.NewIntVar(0, int(p.ub[j]), f"z{j}") for j in range(p.m)]
    for c in p.budget_constraints:
        model.Add(sum(int(c.a[j]) * z[j] for j in range(p.m)) == c.lo)
    for c in p.sector_constraints:
        expr = sum(int(c.a[j]) * z[j] for j in range(p.m))
        model.Add(expr <= c.hi)
        model.Add(expr >= c.lo)
    if p.cardinality_binding:
        y = [model.NewBoolVar(f"y{i}") for i in range(p.n)]
        for i, group in enumerate(p.groups):
            for j in group:
                model.Add(z[j] <= int(p.ub[j]) * y[i])
        model.Add(sum(y) <= p.constraints.cardinality)
    if p.market_neutral:
        for i, (j1, j2) in enumerate(p.exclusive_pairs):
            u = model.NewBoolVar(f"u{i}")
            model.Add(z[j1] <= int(p.ub[j1]) * u)
            model.Add(z[j2] <= int(p.ub[j2]) * (1 - u))
    # integer-scaled quadratic objective with explicit product variables
    norm = scale / max(np.abs(p.A).max(), np.abs(p.b).max(), 1e-300)
    terms = []
    for j in range(p.m):
        cj = int(round(p.b[j] * norm))
        if cj:
            terms.append(cj * z[j])
        for k in range(j, p.m):
            coef = p.A[j, k] if j == k else 2.0 * p.A[j, k]
            ck = int(round(coef * norm))
            if ck == 0:
                continue
            prod = model.NewIntVar(0, int(p.ub[j] * p.ub[k]), f"p{j}_{k}")
            model.AddMultiplicationEquality(prod, [z[j], z[k]])
            terms.append(ck * prod)
    model.Minimize(sum(terms))
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = float(max(time_limit, 0.1))
    solver.parameters.num_workers = workers
    solver.parameters.random_seed = int(seed) % (2 ** 31 - 1)
    status = solver.Solve(model)
    name = solver.StatusName(status)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return None, {"message": f"CP-SAT status {name}", "backend": "ortools/CP-SAT"}
    zz = np.array([solver.Value(v) for v in z], dtype=int)
    return zz, {"backend": "ortools/CP-SAT", "optimal_proven": status == cp_model.OPTIMAL,
                "solver_status": name, "objective_scale": norm}


class CpSat(Solver):
    """OR-Tools CP-SAT on the integer-lot model (objective scaled to integers)."""

    name = "cpsat"

    def _solve(self, problem, budget, rng, timer):
        with timer.section("optimize"):
            z, meta = _cpsat(problem, budget.wall_time_s, seed=int(rng.integers(2 ** 31 - 1)),
                             workers=self.params.get("workers", 1))
        return z, 1, meta


class Miqp(Solver):
    """Exact mixed-integer QP: CVXPY + SCIP, falling back to CP-SAT if SCIP is unavailable/fails."""

    name = "miqp"

    def _solve(self, problem, budget, rng, timer):
        backend = self.params.get("backend", "auto")
        fallback_reason = None
        if backend in ("auto", "scip"):
            try:
                with timer.section("optimize"):
                    z, meta = _miqp_scip(problem, budget.wall_time_s)
                if z is not None or backend == "scip":
                    return z, 1, meta
                fallback_reason = meta.get("message", "SCIP returned no solution")
            except Exception as exc:
                if backend == "scip":
                    raise
                fallback_reason = f"{type(exc).__name__}: {exc}"
        remaining = max(budget.wall_time_s - timer.elapsed, 0.5)
        with timer.section("optimize"):
            z, meta = _cpsat(problem, remaining, seed=int(rng.integers(2 ** 31 - 1)))
        if fallback_reason:
            meta["fallback"] = "cpsat"
            meta["fallback_reason"] = fallback_reason
        return z, 1, meta


# --------------------------------------------------------------------------- heuristics
class Greedy(Solver):
    """Lot-by-lot greedy construction (best marginal objective) plus lot-shift local search."""

    name = "greedy"

    def _solve(self, problem, budget, rng, timer):
        deadline = timer.t0 + budget.wall_time_s
        with timer.section("optimize"):
            z, ok = repair(problem, np.zeros(problem.m, dtype=int))
            evals = problem.m * problem.lots
            if ok and self.params.get("local_search", True):
                z, ls = local_search(problem, z, deadline=deadline)
                evals += ls
        return z, evals, {}


class SimulatedAnnealing(Solver):
    """Simulated annealing directly on feasible lot vectors (lot-shift moves, hard constraints)."""

    name = "sa"
    stochastic = True

    def _solve(self, problem, budget, rng, timer):
        p = problem
        deadline = timer.t0 + budget.wall_time_s * 0.98
        with timer.section("optimize"):
            z0 = np.zeros(p.m, dtype=int)
            for c in p.budget_constraints:               # random start, then repaired
                members = np.flatnonzero(c.a != 0)
                for j in rng.choice(members, size=p.lots):
                    z0[j] += 1
            z, ok = repair(p, z0)
            if not ok:
                z, ok = repair(p, np.zeros(p.m, dtype=int))
            if not ok:
                return None, 0, {"message": "could not construct a feasible start"}
            g = p.A @ z
            f = p.objective(z)
            best_z, best_f = z.copy(), f
            legs = [np.flatnonzero(c.a != 0) for c in p.budget_constraints]
            # temperature scale from typical move sizes
            t_hot = self.params.get("t_hot", 2.0) * float(np.abs(np.diag(p.A)).mean() + np.abs(p.b).std() + 1e-12)
            t_cold = t_hot * self.params.get("t_ratio", 1e-3)
            max_evals = budget.max_evals
            evals, accepted = 0, 0
            check_every = 256
            frac = 0.0
            while evals < max_evals:
                if evals % check_every == 0:
                    now = time.perf_counter()
                    if now > deadline:
                        break
                    frac = max(evals / max_evals, (now - timer.t0) / budget.wall_time_s)
                temp = t_hot * (t_cold / t_hot) ** min(frac, 1.0)
                evals += 1
                move = rng.random()
                if move < 0.15 and p.n > 1:
                    # swap the complete positions of two names (escapes cardinality plateaus)
                    i, j = rng.choice(p.n, size=2, replace=False)
                    z_new = z.copy()
                    gi, gj = p.groups[i], p.groups[j]
                    z_new[gi], z_new[gj] = z[gj], z[gi]
                    if np.array_equal(z_new, z):
                        continue
                    f_new = p.objective(z_new)
                    d = f_new - f
                    if d > 0 and rng.random() >= math.exp(-d / temp):
                        continue
                    if p.ineq_violation(z_new) != 0:
                        continue
                    z, g = z_new, p.A @ z_new
                else:
                    members = legs[rng.integers(len(legs))]
                    i, j = rng.choice(members, size=2, replace=False)
                    if z[i] == 0:
                        continue
                    q = int(z[i]) if move < 0.40 else 1      # whole position or a single lot
                    if z[j] + q > p.ub[j]:
                        continue
                    d = (q * (2.0 * (g[j] - g[i]) + p.b[j] - p.b[i])
                         + q * q * (p.A[j, j] + p.A[i, i] - 2.0 * p.A[i, j]))
                    if d > 0 and rng.random() >= math.exp(-d / temp):
                        continue
                    z[i] -= q
                    z[j] += q
                    if p.ineq_violation(z) != 0:
                        z[i] += q
                        z[j] -= q
                        continue
                    g = g + q * (p.A[:, j] - p.A[:, i])
                f += d
                accepted += 1
                if f < best_f - 1e-15:
                    best_f, best_z = f, z.copy()
            best_z, ls = local_search(p, best_z, max_passes=3)
        return best_z, evals + ls, {"accepted_moves": accepted}
