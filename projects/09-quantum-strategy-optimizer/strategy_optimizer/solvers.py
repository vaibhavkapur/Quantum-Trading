"""Budgeted optimizers, explicit failures, and auditable candidate evaluations."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from time import perf_counter

import numpy as np

from .qubo import feasible_portfolios

CLASSICAL = ("grid", "random", "tpe", "annealing")
SOLVERS = (*CLASSICAL, "qaoa")


@dataclass
class SolverResult:
    solver: str
    seed: int
    status: str = "failed"
    selected: object = None
    objective: object = None
    evaluations: int = 0
    feasible_evaluations: int = 0
    error: object = None
    elapsed_s: float = 0.0
    timing: dict = field(default_factory=dict)
    resources: dict = field(default_factory=dict)
    trace: list = field(default_factory=list)

    def to_dict(self):
        return asdict(self)


class BudgetExhausted(Exception):
    pass


class Ledger:
    def __init__(self, problem, budget):
        self.problem, self.budget = problem, budget
        self.trace = []
        self.best = None
        self.best_value = np.inf
        self.scoring_s = 0.0

    def evaluate(self, bits):
        if len(self.trace) >= self.budget:
            raise BudgetExhausted()
        start = perf_counter()
        bits = self.problem.bits(bits)
        energy = self.problem.energy(bits)
        feasible = self.problem.feasible(bits)
        z = bits[:self.problem.n_strategies].astype(int)
        objective = self.problem.objective(z)
        if feasible and objective < self.best_value:
            self.best, self.best_value = z.copy(), objective
        self.trace.append({"evaluation": len(self.trace) + 1, "bits": bits.astype(int).tolist(),
                           "energy": energy, "objective": objective, "feasible": feasible})
        self.scoring_s += perf_counter() - start
        return energy


def solve(problem, solver, budget, seed, quantum_options=None):
    if solver not in SOLVERS or type(budget) is not int or budget < 1:
        raise ValueError("Unknown solver or invalid evaluation budget")
    started = perf_counter()
    result = SolverResult(solver, seed)
    ledger = Ledger(problem, budget)
    rng = np.random.default_rng(seed)
    try:
        if solver in ("grid", "random"):
            candidates = list(feasible_portfolios(problem))
            indices = range(min(budget, len(candidates))) if solver == "grid" else rng.integers(len(candidates), size=budget)
            for index in indices:
                ledger.evaluate(problem.encode(candidates[index]))
        elif solver == "tpe":
            import optuna
            optuna.logging.set_verbosity(optuna.logging.WARNING)
            study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=seed))
            def objective(trial):
                z = [trial.suggest_int(f"strategy_{i}", 0, 1) for i in range(problem.n_strategies)]
                return ledger.evaluate(problem.encode(z))
            study.optimize(objective, n_trials=budget, n_jobs=1)
        elif solver == "annealing":
            current = np.zeros(problem.n_strategies, dtype=int)
            energy = ledger.evaluate(problem.encode(current))
            scale = max(float(np.abs(problem.base).sum()), 1e-6)
            for step in range(1, budget):
                proposal = current.copy()
                index = rng.integers(problem.n_strategies)
                proposal[index] = 1 - proposal[index]
                candidate_energy = ledger.evaluate(problem.encode(proposal))
                temperature = scale * 0.001 ** (step / max(budget - 1, 1))
                if candidate_energy <= energy or rng.random() < np.exp(min(0, (energy - candidate_energy) / temperature)):
                    current, energy = proposal, candidate_energy
        else:
            from .quantum import optimize_qaoa
            timing, resources = optimize_qaoa(problem, ledger, rng, **(quantum_options or {}))
            result.timing.update(timing)
            result.resources.update(resources)
        if ledger.best is None:
            raise RuntimeError("No feasible sample within the objective-evaluation budget; no repair or fallback")
        result.status = "ok"
        result.selected = ledger.best.tolist()
        result.objective = float(ledger.best_value)
    except Exception as error:
        # Persist failures, including missing optional runtime dependencies. Never
        # relabel a classical fallback as a quantum result or a partial solve as OK.
        result.error = f"{type(error).__name__}: {error}"
    result.elapsed_s = perf_counter() - started
    result.evaluations = len(ledger.trace)
    result.feasible_evaluations = sum(item["feasible"] for item in ledger.trace)
    result.trace = ledger.trace
    result.timing["objective_scoring_s"] = ledger.scoring_s
    accounted = sum(result.timing.values())
    result.timing["optimizer_and_setup_s"] = max(0.0, result.elapsed_s - accounted)
    for name in ("state_preparation_s", "compilation_s", "simulation_s", "sampling_decoding_s", "communication_s", "repair_s"):
        result.timing.setdefault(name, 0.0)
    return result

