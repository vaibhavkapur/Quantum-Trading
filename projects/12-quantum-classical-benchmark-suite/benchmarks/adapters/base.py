from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Protocol

import numpy as np

STAGES = ("setup", "data_prep", "encoding", "compilation", "execution", "decoding", "repair", "validation")


class BudgetExceeded(RuntimeError):
    pass


@dataclass(frozen=True)
class Budget:
    wall_seconds: float
    max_evaluations: int
    max_shots: int
    max_oracle_queries: int
    mip_nodes: int


class Context:
    def __init__(self, budget: Budget, seed: int, algorithm: dict):
        self.budget, self.seed, self.algorithm = budget, seed, algorithm
        self.rng = np.random.default_rng(seed)
        self.start = time.perf_counter()
        self.timings = dict.fromkeys(STAGES, 0.0)
        self._stack: list[list[float]] = []
        self.resources = {"objective_evaluations": 0, "shots": 0, "oracle_queries": 0,
                          "logical_qubits": 0, "circuit_depth": 0, "two_qubit_gates": 0,
                          "optimizer_evaluations": 0, "queue_seconds": 0.0,
                          "communication_seconds": 0.0, "qpu_cost_usd": 0.0,
                          "total_compute_cost_usd": None, "noise_model": "not applicable",
                          "hardware_kind": "classical_cpu", "evaluation_unit": "candidate_objective",
                          "evaluation_budget_enforced": True}
        self.partial: dict = {}

    @contextmanager
    def stage(self, name: str):
        frame = [time.perf_counter(), 0.0]
        self._stack.append(frame)
        try:
            yield
        finally:
            elapsed = time.perf_counter() - frame[0]
            self.timings[name] += elapsed - frame[1]
            self._stack.pop()
            if self._stack:
                self._stack[-1][1] += elapsed

    def remaining_seconds(self) -> float:
        return max(0.0, self.budget.wall_seconds - (time.perf_counter() - self.start))

    def check(self) -> None:
        if self.remaining_seconds() <= 0:
            raise BudgetExceeded("wall-time budget exhausted")

    def charge(self, evaluations: int = 0, shots: int = 0, queries: int = 0) -> None:
        self.check()
        charges = {"objective_evaluations": (evaluations, self.budget.max_evaluations),
                   "shots": (shots, self.budget.max_shots),
                   "oracle_queries": (queries, self.budget.max_oracle_queries)}
        for key, (value, limit) in charges.items():
            if self.resources[key] + value > limit:
                raise BudgetExceeded(f"{key} budget exhausted")
        for key, (value, _) in charges.items():
            self.resources[key] += value


class SolverAdapter(Protocol):
    """solve accepts only public problem data, never the reference optimum."""
    def solve(self, problem: dict, context: Context) -> dict: ...
