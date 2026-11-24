"""Classical candidates, exact MILP lifting, and a common feasibility repair."""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product
from time import perf_counter
import math
import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp, minimize
from scipy.sparse import coo_matrix, csc_matrix

from .encoding import Encoding
from .model import InfeasibleError, Problem, ResourceLimitError, integer, number


class SolverFailure(RuntimeError):
    pass


@dataclass(frozen=True)
class Budget:
    seed: int = 7
    max_evaluations: int = 80
    time_limit_s: float = 10.0
    repair_time_limit_s: float = 10.0
    shots: int = 512
    qaoa_depth: int = 1
    max_qubits: int = 16

    def __post_init__(self):
        integer("seed", self.seed)
        integer("max_evaluations", self.max_evaluations, 1)
        integer("shots", self.shots, 1)
        integer("qaoa_depth", self.qaoa_depth, 1)
        integer("max_qubits", self.max_qubits, 1)
        number("time_limit_s", self.time_limit_s, 0.001, 300)
        number("repair_time_limit_s", self.repair_time_limit_s, 0.001, 300)
        if self.seed > 2**32 - 1 or self.max_qubits > 20 or self.qaoa_depth > 5 or self.shots > 100000 or self.max_evaluations > 100000:
            raise ResourceLimitError("budget exceeds CPU simulator limits")


@dataclass
class Candidate:
    indices: tuple[int, ...]
    status: str
    evaluations: int | None = None
    metadata: dict = field(default_factory=dict)


def _milp_result(result, encoding: Encoding, elapsed: float) -> Candidate:
    if result.x is None:
        if result.status == 2:
            raise InfeasibleError("no portfolio satisfies the lot, budget and exposure constraints")
        raise SolverFailure(f"MILP did not produce an incumbent: {result.message}")
    x = result.x[:encoding.size]
    indices = encoding.decode(x)
    if np.max(np.abs(x - encoding.bits(indices))) > 1e-5 or not encoding.problem.evaluate(indices)["feasible"]:
        raise SolverFailure("MILP incumbent failed the integer-cent risk gate")
    bound = getattr(result, "mip_dual_bound", None)
    gap = getattr(result, "mip_gap", None)
    return Candidate(indices, "optimal" if result.status == 0 else "feasible_limit", metadata={
        "engine": "scipy.optimize.milp/HiGHS", "solver_message": str(result.message),
        "mip_gap": float(gap) if gap is not None and math.isfinite(gap) else None,
        "objective_lower_bound": float(bound + encoding.offset) if bound is not None and math.isfinite(bound) else None,
        "nodes": int(getattr(result, "mip_node_count", 0) or 0), "milp_elapsed_s": elapsed})


def solve_milp(encoding: Encoding, budget: Budget, *, linear_objective: np.ndarray | None = None) -> Candidate:
    start = perf_counter()
    n = encoding.size
    if n > 2048:
        raise ResourceLimitError("MILP supports at most 2048 order-choice variables")
    pairs = []
    if linear_objective is None:
        asset_of = {j: i for i, block in enumerate(encoding.blocks) for j in block}
        pairs = [(i, j, encoding.quadratic[i, j]) for i in range(n) for j in range(i + 1, n)
                 if asset_of[i] != asset_of[j] and abs(encoding.quadratic[i, j]) > 1e-15]
    if n + len(pairs) > 60000:
        raise ResourceLimitError("MILP lifting exceeds 60000 variables; reduce assets or trade domains")
    objective = np.r_[encoding.linear if linear_objective is None else linear_objective, [q for _, _, q in pairs]]
    rows, cols = np.nonzero(encoding.constraints)
    values = list(encoding.constraints[rows, cols])
    rows, cols = list(rows), list(cols)
    lower, upper = list(encoding.lower), list(encoding.upper)
    for k, (i, j, _) in enumerate(pairs, n):
        for coefficients, lb, ub in [({k: 1, i: -1}, -np.inf, 0), ({k: 1, j: -1}, -np.inf, 0),
                                      ({k: 1, i: -1, j: -1}, -1, np.inf)]:
            row = len(lower)
            for col, value in coefficients.items():
                rows.append(row)
                cols.append(col)
                values.append(value)
            lower.append(lb)
            upper.append(ub)
    matrix = csc_matrix(coo_matrix((values, (rows, cols)), shape=(len(lower), len(objective))))
    integrality = np.r_[np.ones(n), np.zeros(len(pairs))]
    result = milp(objective, integrality=integrality, bounds=Bounds(0, 1),
                  constraints=LinearConstraint(matrix, lower, upper),
                  options={"time_limit": budget.time_limit_s, "mip_rel_gap": 0.0})
    candidate = _milp_result(result, encoding, perf_counter() - start)
    candidate.metadata.update({"binary_variables": n, "product_variables": len(pairs)})
    if linear_objective is not None:
        candidate.metadata.pop("objective_lower_bound", None)
    return candidate


def brute_force(problem: Problem, *, max_combinations: int = 1000000) -> Candidate:
    if problem.combinations > max_combinations:
        raise ResourceLimitError("brute-force combination limit exceeded")
    best, value, count = None, math.inf, 0
    for indices in product(*(range(len(c)) for c in problem.choices)):
        result = problem.evaluate(indices)
        count += 1
        if result["feasible"] and result["objective"] < value - 1e-14:
            best, value = indices, result["objective"]
    if best is None:
        raise InfeasibleError("enumeration found no feasible portfolio")
    return Candidate(best, "optimal", count, {"engine": "exhaustive enumeration", "objective_lower_bound": value})


def greedy(encoding: Encoding, budget: Budget) -> Candidate:
    p = encoding.problem.portfolio
    indices = tuple(min(range(len(options)), key=lambda j: (
        (options[j].value_cents / p.nav_cents - asset.target_weight)**2
        + (p.turnover_weight * options[j].notional_cents + p.cost_weight * (options[j].fee_cents + options[j].tax_cents)) / p.nav_cents,
        abs(options[j].delta_lots), j)) for asset, options in zip(p.assets, encoding.problem.choices))
    return Candidate(indices, "heuristic", sum(map(len, encoding.blocks)), {"engine": "independent greedy target rounding"})


def convex_round(encoding: Encoding, budget: Budget) -> Candidate:
    start = perf_counter()
    # Fractional one-hot choices form a convex hull relaxation. Rounded results
    # need the same integer-cent feasibility gate as quantum samples.
    x0 = np.concatenate([np.full(len(b), 1 / len(b)) for b in encoding.blocks])
    scale = np.maximum(1, np.max(np.abs(encoding.constraints), axis=1))
    a = encoding.constraints / scale[:, None]
    lower, upper = encoding.lower / scale, encoding.upper / scale
    eq = lower == upper
    constraints = [LinearConstraint(a[eq], lower[eq], upper[eq]), LinearConstraint(a[~eq], lower[~eq], upper[~eq])]
    evaluations = 0
    best_x = x0.copy()

    class Stop(Exception):
        pass

    def objective(x):
        nonlocal evaluations, best_x
        if evaluations >= budget.max_evaluations or perf_counter() - start >= budget.time_limit_s:
            raise Stop()
        evaluations += 1
        best_x = x.copy()
        residual = encoding.matrix @ x + encoding.residual
        return float(residual @ residual + encoding.cost @ x)

    def jac(x):
        return 2 * encoding.matrix.T @ (encoding.matrix @ x + encoding.residual) + encoding.cost

    try:
        result = minimize(objective, x0, jac=jac, method="SLSQP", bounds=Bounds(0, 1), constraints=constraints,
                          options={"ftol": 1e-12, "maxiter": budget.max_evaluations})
        best_x = result.x
        status, message = ("relaxed_converged" if result.success else "relaxed_limit"), str(result.message)
    except Stop:
        status, message = "relaxed_limit", "evaluation or wall-time budget reached"
    return Candidate(encoding.decode(best_x), status, evaluations, {"engine": "SLSQP convex hull relaxation + rounding", "solver_message": message})


def anneal(encoding: Encoding, budget: Budget) -> Candidate:
    rng = np.random.default_rng(budget.seed)
    start = perf_counter()
    current = greedy(encoding, budget).indices

    def energy(indices):
        report = encoding.problem.evaluate(indices)
        # Infeasibility count is a search heuristic, never a substitute for repair.
        return report["objective"] + 1000 * len(report["violations"])

    value = energy(current)
    best, best_value, evaluations = current, value, 1
    while evaluations < budget.max_evaluations and perf_counter() - start < budget.time_limit_s:
        i = int(rng.integers(len(current)))
        candidate = list(current)
        candidate[i] = int(rng.integers(len(encoding.blocks[i])))
        candidate = tuple(candidate)
        score = energy(candidate)
        temperature = 0.2 * (0.001 / 0.2)**(evaluations / budget.max_evaluations)
        if score < value or rng.random() < math.exp(min(0, (value - score) / temperature)):
            current, value = candidate, score
        if score < best_value:
            best, best_value = candidate, score
        evaluations += 1
    return Candidate(best, "heuristic", evaluations, {"engine": "seeded simulated annealing"})


def repair(encoding: Encoding, candidate: Candidate, budget: Budget) -> tuple[tuple[int, ...], dict]:
    before = encoding.problem.evaluate(candidate.indices)
    if before["feasible"]:
        return candidate.indices, {"applied": False, "distance_lots": 0, "raw_violations": [], "method": "integer-cent risk gate"}
    desired = [options[index].delta_lots for options, index in zip(encoding.problem.choices, candidate.indices)]
    distance = np.array([abs(c.delta_lots - delta) for delta, options in zip(desired, encoding.problem.choices) for c in options], dtype=float)
    from dataclasses import replace
    repaired = solve_milp(encoding, replace(budget, time_limit_s=budget.repair_time_limit_s), linear_objective=distance)
    actual_distance = sum(abs(options[j].delta_lots - delta) for options, j, delta in zip(encoding.problem.choices, repaired.indices, desired))
    return repaired.indices, {"applied": True, "distance_lots": actual_distance, "raw_violations": before["violations"],
                              "method": "MILP nearest feasible lot vector (no tracking-objective optimization)",
                              "status": repaired.status, "solver": repaired.metadata}

