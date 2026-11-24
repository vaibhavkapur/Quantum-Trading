from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from time import perf_counter

import numpy as np

from .research import Pair


@dataclass
class Problem:
    pairs: list[Pair]
    linear: np.ndarray
    quadratic: np.ndarray  # strictly upper triangular; counted once
    constraints: list[tuple[str, np.ndarray, int]]

    def __post_init__(self):
        n = len(self.pairs)
        if self.linear.shape != (n,) or self.quadratic.shape != (n, n):
            raise ValueError("objective dimension mismatch")
        if not np.isfinite(self.linear).all() or not np.isfinite(self.quadratic).all():
            raise ValueError("objective coefficients must be finite")
        if np.any(np.tril(self.quadratic) != 0):
            raise ValueError("quadratic coefficients must be strictly upper triangular")
        for _, a, bound in self.constraints:
            if a.shape != (n,) or not np.isfinite(a).all() or np.any(a < 0) or np.any(a != np.floor(a)) or type(bound) is not int or bound < 0:
                raise ValueError("constraints require nonnegative integral coefficients and bounds")

    @property
    def n(self) -> int:
        return len(self.pairs)

    def objective(self, x: np.ndarray) -> float:
        x = np.asarray(x)
        return float(self.linear @ x + x @ self.quadratic @ x)

    def feasible(self, x: np.ndarray) -> bool:
        x = np.asarray(x)
        return x.shape == (self.n,) and bool(np.isin(x, [0, 1]).all()) and all(a @ x <= b for _, a, b in self.constraints)


def make_problem(pairs: list[Pair], correlation: np.ndarray, settings: dict) -> Problem:
    n = len(pairs)
    if correlation.shape != (n, n) or not np.isfinite(correlation).all() or np.any(np.abs(correlation) > 1 + 1e-9):
        raise ValueError("invalid residual correlation matrix")
    if any(type(p.capital_units) is not int or p.capital_units <= 0 or p.left == p.right for p in pairs):
        raise ValueError("each pair needs distinct assets and positive integer capital units")
    if len({p.name for p in pairs}) != n:
        raise ValueError("duplicate pairs")
    rows = [("pair_count", np.ones(n, dtype=int), settings["max_pairs"]),
            ("capital", np.array([p.capital_units for p in pairs], dtype=int), settings["capital_budget"])]
    for asset in sorted({a for p in pairs for a in (p.left, p.right)}):
        rows.append((f"asset:{asset}", np.array([int(asset in (p.left, p.right)) for p in pairs]), settings["asset_limit"]))
    for sector in sorted({s for p in pairs for s in p.sectors}):
        rows.append((f"sector:{sector}", np.array([int(sector in p.sectors) for p in pairs]), settings["sector_limit"]))
    # Deduplicate equal rows, keeping the strongest bound. Both MILP and QUBO use these rows.
    deduplicated = {}
    for name, a, bound in rows:
        key = tuple(a)
        if a.sum() > bound and (key not in deduplicated or bound < deduplicated[key][2]):
            deduplicated[key] = (name, a, bound)
    return Problem(pairs, -np.array([p.score for p in pairs]), np.triu(np.abs(correlation) * settings["correlation_penalty"], 1), list(deduplicated.values()))


@dataclass
class Qubo:
    matrix: np.ndarray  # upper triangular including diagonal
    offset: float
    decisions: int
    penalty: float
    slack: list[dict]

    @property
    def n(self) -> int:
        return len(self.matrix)

    def energy(self, bits: np.ndarray) -> float:
        return float(bits @ self.matrix @ bits + self.offset)

    def energies(self) -> np.ndarray:
        bits = ((np.arange(1 << self.n, dtype=np.uint64)[:, None] >> np.arange(self.n, dtype=np.uint64)) & 1).astype(float)
        return np.einsum("bi,ij,bj->b", bits, self.matrix, bits) + self.offset


def bounded_weights(bound: int) -> list[int]:
    weights, covered = [], 0
    while covered < bound:
        weight = min(covered + 1, bound - covered)
        weights.append(weight)
        covered += weight
    return weights


def encode_qubo(problem: Problem, max_qubits: int = 20) -> Qubo:
    # Greater than the full possible objective range. An invalid assignment cannot
    # beat the feasible empty selection, even if every quality term is favorable.
    penalty = float(np.abs(problem.linear).sum() + np.abs(problem.quadratic).sum() + 1)
    terms, slack, n = [], [], problem.n
    for name, a, bound in problem.constraints:
        if bound == 1 and np.all((a == 0) | (a == 1)):
            terms.append((a, bound, None))
        else:
            weights = bounded_weights(bound)
            indices = list(range(n, n + len(weights)))
            slack.append({"constraint": name, "indices": indices, "weights": weights, "bound": bound})
            terms.append((a, bound, (indices, weights)))
            n += len(weights)
    if n > max_qubits:
        raise ValueError(f"QUBO requires {n} qubits, exceeding configured cap {max_qubits}")
    matrix = np.zeros((n, n))
    matrix[:problem.n, :problem.n] = problem.quadratic
    matrix[np.arange(problem.n), np.arange(problem.n)] = problem.linear
    offset = 0.0
    for a, bound, encoding in terms:
        if encoding is None:
            for i, j in combinations(np.flatnonzero(a), 2):
                matrix[i, j] += penalty
        else:
            indices, weights = encoding
            coefficients = np.zeros(n)
            coefficients[:problem.n] = a
            coefficients[indices] = weights
            matrix[np.arange(n), np.arange(n)] += penalty * (coefficients**2 - 2 * bound * coefficients)
            matrix += np.triu(2 * penalty * np.outer(coefficients, coefficients), 1)
            offset += penalty * bound**2
    return Qubo(matrix, offset, problem.n, penalty, slack)


def repair(problem: Problem, x: np.ndarray, accounting: dict | None = None) -> np.ndarray:
    x = np.asarray(x, dtype=int).copy()
    if x.shape != (problem.n,) or not np.isin(x, [0, 1]).all():
        raise ValueError("repair expects binary decision vector")
    while not problem.feasible(x):
        violated = [a for _, a, b in problem.constraints if a @ x > b]
        eligible = [i for i in np.flatnonzero(x) if any(a[i] for a in violated)]
        choices = []
        for i in eligible:
            candidate = x.copy()
            candidate[i] = 0
            choices.append((problem.objective(candidate), int(i), candidate))
            if accounting is not None:
                accounting["repair_objective_evaluations"] += 1
        x = min(choices, key=lambda item: (item[0], item[1]))[2]
    return x


@dataclass
class SelectionResult:
    solver: str
    x: np.ndarray
    objective: float
    status: str = "ok"
    evaluations: int = 0
    timings: dict = field(default_factory=dict)
    details: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"solver": self.solver, "bits": self.x.tolist(), "objective": self.objective, "status": self.status,
                "evaluations": self.evaluations, "timings": self.timings, "details": self.details}


def brute_force(problem: Problem, max_variables: int = 20) -> SelectionResult:
    if problem.n > max_variables:
        raise ValueError("brute-force oracle exceeds safety cap")
    start = perf_counter()
    best, value = np.zeros(problem.n, dtype=int), 0.0
    for mask in range(1 << problem.n):
        x = np.array([(mask >> i) & 1 for i in range(problem.n)])
        if problem.feasible(x) and problem.objective(x) < value - 1e-12:
            best, value = x, problem.objective(x)
    return SelectionResult("brute-force", best, value, evaluations=1 << problem.n, timings={"total_seconds": perf_counter() - start})


def ranked(problem: Problem, diversified: bool = False, settings: dict | None = None) -> SelectionResult:
    start, evaluations = perf_counter(), 0
    x = np.zeros(problem.n, dtype=int)
    current_value, exhausted = 0.0, False
    remaining = list(np.argsort(problem.linear, kind="stable"))
    while remaining:
        choices = []
        for i in remaining if diversified else remaining[:1]:
            if settings and (evaluations >= settings["max_evaluations"] or perf_counter() - start >= settings["wall_seconds"]):
                exhausted = True
                break
            candidate = x.copy()
            candidate[i] = 1
            value = problem.objective(candidate)
            evaluations += 1
            if problem.feasible(candidate) and value < current_value - 1e-12:
                choices.append((value, int(i), candidate))
        if choices:
            current_value, i, x = min(choices, key=lambda item: (item[0], item[1]))
            remaining.remove(i)
        elif diversified:
            break
        else:
            remaining.pop(0)
        if exhausted:
            break
    return SelectionResult("greedy" if diversified else "top-k", x, current_value, evaluations=evaluations, timings={"total_seconds": perf_counter() - start}, details={"budget_exhausted": exhausted})


def milp(problem: Problem, settings: dict) -> SelectionResult:
    from ortools.linear_solver import pywraplp

    start = perf_counter()
    solver = pywraplp.Solver.CreateSolver("SCIP")
    if solver is None:
        raise RuntimeError("OR-Tools SCIP backend unavailable")
    solver.SetNumThreads(1)
    solver.SetTimeLimit(max(1, int(settings["wall_seconds"] * 1000)))
    x = [solver.BoolVar(f"pair_{i}") for i in range(problem.n)]
    for _, a, bound in problem.constraints:
        solver.Add(sum(float(a[i]) * x[i] for i in range(problem.n)) <= bound)
    objective = solver.Objective()
    for i in range(problem.n):
        objective.SetCoefficient(x[i], float(problem.linear[i]))
    for i, j in zip(*np.nonzero(problem.quadratic)):
        both = solver.BoolVar(f"both_{i}_{j}")
        solver.Add(both <= x[i])
        solver.Add(both <= x[j])
        solver.Add(both >= x[i] + x[j] - 1)
        objective.SetCoefficient(both, float(problem.quadratic[i, j]))
    objective.SetMinimization()
    built = perf_counter()
    status = solver.Solve()
    solved = perf_counter()
    if status not in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE):
        raise RuntimeError(f"MILP returned no incumbent (status={status})")
    result = np.array([round(v.solution_value()) for v in x], dtype=int)
    if not problem.feasible(result):
        raise RuntimeError("MILP returned an infeasible selection")
    return SelectionResult("milp", result, problem.objective(result), status="optimal" if status == pywraplp.Solver.OPTIMAL else "time_limit_feasible",
                           timings={"model_build_seconds": built - start, "optimizer_seconds": solved - built, "total_seconds": perf_counter() - start},
                           details={"backend": solver.SolverVersion(), "nodes": int(solver.nodes()), "best_bound": objective.BestBound(), "objective_evaluations": "not exposed by SCIP; shared wall-time cap applies"})


def decode_samples(problem: Problem, samples: np.ndarray) -> tuple[np.ndarray, dict]:
    """Feasibility repair never adds pairs and is accounted separately from search."""
    start = perf_counter()
    decisions = samples[:, :problem.n]
    unique = np.unique(decisions, axis=0)
    best, value, repaired = np.zeros(problem.n, dtype=int), 0.0, 0
    accounting = {"repair_objective_evaluations": 0}
    for x in unique:
        repaired += int(not problem.feasible(x))
        feasible = repair(problem, x, accounting)
        objective = problem.objective(feasible)
        if objective < value - 1e-12:
            best, value = feasible, objective
    return best, {**accounting, "sample_feasible_fraction": float(np.mean([problem.feasible(x) for x in decisions])),
                  "unique_decisions": len(unique), "repaired_unique_decisions": repaired,
                  "decode_objective_evaluations": len(unique), "repair_decode_seconds": perf_counter() - start}


def anneal(problem: Problem, settings: dict, seed: int) -> SelectionResult:
    start = perf_counter()
    qubo = encode_qubo(problem, settings["max_qubits"])
    prepared = perf_counter()
    rng = np.random.default_rng(seed)
    x = rng.integers(0, 2, qubo.n)
    current = qubo.energy(x)
    samples, evaluations = [x.copy()], 1
    while evaluations < settings["max_evaluations"] and perf_counter() - start < settings["wall_seconds"]:
        trial = x.copy()
        trial[rng.integers(qubo.n)] ^= 1
        energy = qubo.energy(trial)
        temperature = max(0.01, qubo.penalty * (1 - evaluations / settings["max_evaluations"])**2)
        if energy < current or rng.random() < np.exp(min(0, (current - energy) / temperature)):
            x, current = trial, energy
        samples.append(trial.copy())
        evaluations += 1
    optimized = perf_counter()
    best, details = decode_samples(problem, np.array(samples))
    return SelectionResult("annealing", best, problem.objective(best), evaluations=evaluations,
                           timings={"encoding_seconds": prepared - start, "optimizer_seconds": optimized - prepared, "repair_decode_seconds": details.pop("repair_decode_seconds"), "total_seconds": perf_counter() - start},
                           details={**details, "qubits": qubo.n, "budget_exhausted": evaluations < settings["max_evaluations"]})


def solve(problem: Problem, name: str, settings: dict, seed: int) -> SelectionResult:
    start = perf_counter()
    if name not in ("top-k", "greedy", "milp", "annealing", "quantum-sim"):
        raise ValueError(f"unknown solver: {name}")
    if not problem.n:
        return SelectionResult(name, np.zeros(0, dtype=int), 0, status="no_candidates", timings={"total_seconds": 0.0})
    if name in ("top-k", "greedy"):
        return ranked(problem, diversified=name == "greedy", settings=settings)
    try:
        if name == "milp":
            return milp(problem, settings)
        if name == "annealing":
            return anneal(problem, settings, seed)
        if name == "quantum-sim":
            from .quantum import qaoa
            return qaoa(problem, settings, seed)
        raise ValueError(f"unknown solver: {name}")
    except (ImportError, RuntimeError, ValueError) as exc:
        if name not in ("milp", "annealing", "quantum-sim") or settings["fallback"] != "greedy":
            raise
        failed_attempt_seconds = perf_counter() - start
        result = ranked(problem, diversified=True, settings=settings)
        result.solver = name
        result.status = "fallback_greedy"
        result.details.update({"fallback_reason": str(exc), "actual_solver": "greedy", "failed_attempt_seconds": failed_attempt_seconds})
        result.timings["total_seconds"] = perf_counter() - start
        return result
