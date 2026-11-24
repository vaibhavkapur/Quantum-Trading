"""The objective convention is c·x + sum(i<j, Q[i,j] x[i] x[j])."""
from __future__ import annotations

import hashlib
import json
import math
from itertools import combinations
from typing import Any

import numpy as np

SCHEMA_VERSION = 1


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


def integer(value: Any, name: str, minimum: int = 0, maximum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    if maximum is not None and value > maximum:
        raise ValueError(f"{name} must be <= {maximum}")
    return value


def validate_problem(problem: dict) -> None:
    track = problem.get("track")
    if track in {"portfolio", "basket"}:
        n = integer(problem["size"], "size", 1, 20)
        integer(problem["cardinality"], "cardinality", 0, n)
        c, q = np.asarray(problem["linear"], dtype=float), np.asarray(problem["quadratic"], dtype=float)
        if c.shape != (n,) or q.shape != (n, n) or not np.isfinite(c).all() or not np.isfinite(q).all():
            raise ValueError("objective dimensions or finite coefficients are invalid")
        if np.any(np.tril(q) != 0):
            raise ValueError("quadratic coefficients must be strictly upper triangular")
        seen = set()
        for pair in problem["conflicts"]:
            if len(pair) != 2:
                raise ValueError("conflicts must be pairs")
            i, j = pair
            integer(i, "conflict index", 0, n - 1)
            integer(j, "conflict index", 0, n - 1)
            if i >= j or (i, j) in seen:
                raise ValueError("conflicts must be unique ordered pairs i < j")
            seen.add((i, j))
    elif track == "estimation":
        n = integer(problem["size"], "size", 2, 256)
        if n & (n - 1):
            raise ValueError("estimation size must be a power of two")
        p, f = np.asarray(problem["probabilities"], dtype=float), np.asarray(problem["payoffs"], dtype=float)
        if p.shape != (n,) or f.shape != (n,) or not np.isfinite(p).all() or not np.isfinite(f).all():
            raise ValueError("invalid probability/payoff dimensions")
        if np.any(p < 0) or not math.isclose(float(p.sum()), 1.0, abs_tol=1e-12):
            raise ValueError("probabilities must be nonnegative and sum to one")
        if np.any(f < 0) or np.any(f > 1):
            raise ValueError("normalized payoffs must lie in [0, 1]")
        if not math.isfinite(problem["scale"]) or problem["scale"] <= 0:
            raise ValueError("payoff scale must be positive and finite")
    else:
        raise ValueError(f"unknown track {track!r}")


def validate_instance(instance: dict) -> None:
    if instance.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported instance schema")
    if not isinstance(instance.get("instance_id"), str) or not instance["instance_id"]:
        raise ValueError("instance_id is required")
    validate_problem(instance["problem"])
    reference = instance["reference"]
    if not math.isfinite(reference["value"]):
        raise ValueError("reference must be finite")
    p = instance["problem"]
    if p["track"] == "estimation":
        actual = float(np.dot(p["probabilities"], p["payoffs"]) * p["scale"])
    else:
        if not feasible(p, reference["solution"]):
            raise ValueError("reference solution is infeasible")
        actual = objective(p, reference["solution"])
    if not math.isclose(actual, reference["value"], rel_tol=1e-10, abs_tol=1e-10):
        raise ValueError("reference value does not match the shared evaluator")


def feasible(problem: dict, candidate: Any) -> bool:
    if candidate is None:
        return False
    x = np.asarray(candidate)
    if x.shape != (problem["size"],) or not np.isin(x, [0, 1]).all():
        return False
    return int(x.sum()) == problem["cardinality"] and all(not (x[i] and x[j]) for i, j in problem["conflicts"])


def objective(problem: dict, candidate: Any) -> float:
    x = np.asarray(candidate, dtype=float)
    if x.shape != (problem["size"],) or not np.isin(x, [0, 1]).all():
        raise ValueError("candidate must be a binary vector of the specified size")
    return float(np.asarray(problem["linear"]) @ x + x @ np.asarray(problem["quadratic"]) @ x)


def exact_reference(problem: dict) -> tuple[list[int], float]:
    """Offline reference generation, never a fallback or solver input."""
    if problem["size"] > 16:
        raise ValueError("reference enumeration is capped at 16 binary variables")
    best, value = None, math.inf
    for selected in combinations(range(problem["size"]), problem["cardinality"]):
        x = [int(i in selected) for i in range(problem["size"])]
        if feasible(problem, x) and (v := objective(problem, x)) < value:
            best, value = x, v
    if best is None:
        raise ValueError("instance has no feasible solution")
    return best, value


def encode_qubo(problem: dict) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Exact penalty: any nonzero integer violation costs more than the objective range."""
    c, q = np.array(problem["linear"], dtype=float), np.array(problem["quadratic"], dtype=float)
    penalty = 2 * (np.abs(c).sum() + np.abs(q).sum()) + 1
    k = problem["cardinality"]
    c += penalty * (1 - 2 * k)
    q += np.triu(np.ones(q.shape), 1) * 2 * penalty
    for i, j in problem["conflicts"]:
        q[i, j] += penalty
    return c, q, float(penalty * k * k), float(penalty)


def qubo_to_ising(c: np.ndarray, q: np.ndarray, offset: float) -> tuple[np.ndarray, np.ndarray, float]:
    h = -c / 2 - (q.sum(axis=0) + q.sum(axis=1)) / 4
    return h, q / 4, float(offset + c.sum() / 2 + q.sum() / 4)


def repair(problem: dict, candidate: list[int]) -> list[int]:
    """Deterministic linear-score repair, no reference access or exact search."""
    n, k = problem["size"], problem["cardinality"]
    if len(candidate) != n:
        return candidate
    conflicts = {tuple(pair) for pair in problem["conflicts"]}
    order = sorted(range(n), key=lambda i: (not bool(candidate[i]), problem["linear"][i], i))
    selected: list[int] = []
    for i in order:
        if len(selected) == k:
            break
        if all(tuple(sorted((i, j))) not in conflicts for j in selected):
            selected.append(i)
    return [int(i in selected) for i in range(n)]
