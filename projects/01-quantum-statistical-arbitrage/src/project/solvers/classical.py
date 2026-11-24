"""Classical baselines: exact enumeration, greedy, CP-SAT, random search, simulated annealing."""
from __future__ import annotations

import numpy as np

from ..research.qubo import BasketProblem
from .base import Budget, SolverResult, Timer, best_of_candidates


def brute_force(problem: BasketProblem, budget: Budget, seed: int | None = None) -> SolverResult:
    """Reference optimum (ignores the eval budget - it defines the ground truth)."""
    timer = Timer()
    with timer.stage("optimize"):
        z, val, count = problem.brute_force()
    return SolverResult("brute_force", z, val, "optimal", n_evals=count, raw_feasible_rate=1.0, raw_best_objective=val, timings=timer.t)


def greedy(problem: BasketProblem, budget: Budget, seed: int | None = None) -> SolverResult:
    """Forward selection: best feasible pair, then the best marginal addition."""
    timer = Timer()
    evals = 0
    with timer.stage("optimize"):
        n, cap = problem.n, problem.max_per_sector
        z = np.zeros(n, dtype=int)
        if problem.basket_size == 1:
            z[int(np.argmin(problem.linear))] = 1
            evals = n
        else:
            best = None
            for i in range(n):
                for j in range(i + 1, n):
                    if cap < 2 and problem.sectors[i] == problem.sectors[j]:
                        continue
                    evals += 1
                    val = problem.linear[i] + problem.linear[j] + problem.quad[i, j]
                    if best is None or val < best[0]:
                        best = (val, i, j)
            z[[best[1], best[2]]] = 1
        dead_end = False
        while z.sum() < problem.basket_size:
            counts = np.bincount(problem.sectors[z == 1], minlength=problem.sectors.max() + 1)
            cands = [i for i in np.flatnonzero(z == 0) if counts[problem.sectors[i]] < cap]
            if not cands:
                dead_end = True
                break
            evals += len(cands)
            z[min(cands, key=lambda i: problem.linear[i] + problem.quad[i] @ z)] = 1
        if dead_end:
            z = problem.repair(z)
    return SolverResult(
        "greedy", z, problem.base_objective(z), "ok", n_evals=evals, raw_feasible_rate=1.0, repaired=dead_end,
        raw_best_objective=float("nan") if dead_end else problem.base_objective(z), timings=timer.t,
    )


def cpsat(problem: BasketProblem, budget: Budget, seed: int | None = None) -> SolverResult:
    """OR-Tools CP-SAT on the native constrained model (hard constraints, no penalties)."""
    from ortools.sat.python import cp_model

    timer = Timer()
    scale = 1_000_000
    with timer.stage("encode"):
        model = cp_model.CpModel()
        n = problem.n
        z = [model.NewBoolVar(f"z{i}") for i in range(n)]
        terms = [int(round(problem.linear[i] * scale)) * z[i] for i in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                c = int(round(problem.quad[i, j] * scale))
                if c == 0:
                    continue
                y = model.NewBoolVar(f"y{i}_{j}")
                model.AddBoolOr([z[i].Not(), z[j].Not(), y])
                model.AddImplication(y, z[i])
                model.AddImplication(y, z[j])
                terms.append(c * y)
        model.Add(sum(z) == problem.basket_size)
        for s in np.unique(problem.sectors):
            members = [z[i] for i in np.flatnonzero(problem.sectors == s)]
            if len(members) > problem.max_per_sector:
                model.Add(sum(members) <= problem.max_per_sector)
        model.Minimize(sum(terms))
    with timer.stage("optimize"):
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = float(budget.time_limit_s)
        solver.parameters.num_search_workers = 1
        solver.parameters.random_seed = int(seed or 0)
        status = solver.Solve(model)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        raise RuntimeError(f"CP-SAT returned {solver.StatusName(status)} within {budget.time_limit_s}s")
    sol = np.array([int(solver.Value(v)) for v in z])
    return SolverResult(
        "cpsat",
        sol,
        problem.base_objective(sol),
        "optimal" if status == cp_model.OPTIMAL else "feasible",
        n_evals=int(solver.NumBranches()),
        raw_feasible_rate=1.0,
        raw_best_objective=problem.base_objective(sol),
        timings=timer.t,
        meta={"branches": int(solver.NumBranches()), "conflicts": int(solver.NumConflicts())},
    )


def random_search(problem: BasketProblem, budget: Budget, seed: int | None = None) -> SolverResult:
    """Budget-matched control: uniform random bitstrings through the same decode/repair path as QAOA."""
    timer = Timer()
    rng = np.random.default_rng(seed)
    with timer.stage("sample"):
        cands = rng.integers(0, 2, size=(budget.max_evals, problem.n), dtype=np.int8)
    z, val, rate, repaired, raw_best = best_of_candidates(problem, cands, timer)
    return SolverResult(
        "random_search", z, val, "ok", n_evals=budget.max_evals, raw_feasible_rate=rate, repaired=repaired,
        raw_best_objective=raw_best, timings=timer.t,
    )


def simulated_annealing(problem: BasketProblem, budget: Budget, seed: int | None = None) -> SolverResult:
    """Single-bit-flip annealing on the same penalised QUBO the quantum solver sees."""
    timer = Timer()
    rng = np.random.default_rng(seed)
    with timer.stage("encode"):
        Q, _ = problem.to_qubo()
        S = Q + Q.T - np.diag(np.diag(Q))  # symmetric couplings, diagonal = linear terms
        diag = np.diag(Q)
    with timer.stage("optimize"):
        n, steps = problem.n, max(budget.max_evals - 1, 1)
        z = rng.integers(0, 2, n)
        visited = [z.copy()]
        scale = float(np.mean(np.abs(Q[Q != 0]))) if np.any(Q) else 1.0
        t_hot, t_cold = 2.0 * scale, 0.02 * scale
        for step in range(steps):
            temp = t_hot * (t_cold / t_hot) ** (step / max(steps - 1, 1))
            i = int(rng.integers(n))
            # Energy change of flipping bit i.
            field = diag[i] + S[i] @ z - S[i, i] * z[i]
            delta = field * (1 - 2 * z[i])
            if delta <= 0 or rng.random() < np.exp(-delta / temp):
                z[i] ^= 1
            visited.append(z.copy())
    cands = np.array(visited, dtype=np.int8)
    zb, val, rate, repaired, raw_best = best_of_candidates(problem, cands, timer)
    return SolverResult(
        "simulated_annealing", zb, val, "ok", n_evals=len(cands), raw_feasible_rate=rate, repaired=repaired,
        raw_best_objective=raw_best, timings=timer.t,
    )
