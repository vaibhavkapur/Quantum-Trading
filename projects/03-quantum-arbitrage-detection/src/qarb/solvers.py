"""Uniform solver interface over the weight matrix.

Every solver gets the same fee- and depth-adjusted weight matrix W and the same
length bounds, and returns candidate currency cycles with negative weight.
Venue choice, sizing and the cost gate are applied afterwards by shared code,
so solvers differ only in *search*.

`objective_evals` is the common budget unit: the number of candidate solutions
scored against the objective.

    enumeration / random_search   cycles scored
    bellman_ford / bounded_dp     edge relaxations / max_cycle_len
    qubo_sa                       reads * sweeps (full-register sweeps)
    qaoa                          circuit executions (optimizer shots + final shots)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from .graph.cycles import (
    EPS,
    Cycle,
    bellman_ford_cycles,
    bounded_dp_cycles,
    cycle_weight,
    enumerate_cycles,
    random_cycles,
)
from .quantum.anneal import simulated_annealing
from .quantum.qaoa import run_qaoa
from .quantum.qubo import (
    CycleQubo,
    argmin_is_valid_optimum,
    build_cycle_qubo,
    energy_vector,
    index_to_bits,
)

CLASSICAL = ("enumeration", "bellman_ford", "bounded_dp", "random_search", "qubo_sa", "naive_mid")
QUANTUM = ("qaoa",)


@dataclass
class SearchContext:
    W: np.ndarray
    W_mid: np.ndarray
    min_len: int
    max_len: int
    rng: np.random.Generator
    cfg: dict
    resources: dict = field(default_factory=dict)  # run-level cache (circuit metrics)

    @property
    def max_evals(self) -> int:
        return int(self.cfg["budget"]["max_objective_evals"])

    @property
    def max_wall_ms(self) -> float:
        return float(self.cfg["budget"]["max_wall_ms"])


@dataclass
class SearchResult:
    cycles: list[Cycle]
    stats: dict
    unrepaired: list[Cycle] | None = None  # QUBO solvers: cycles read directly from feasible samples


def _negative(pairs) -> list[Cycle]:
    return sorted(c for c, w in pairs if w < -EPS)


def _enumeration(ctx: SearchContext) -> SearchResult:
    cycles, evals, truncated = enumerate_cycles(ctx.W, ctx.min_len, ctx.max_len, ctx.max_evals)
    return SearchResult(_negative(cycles), {"objective_evals": evals, "budget_exhausted": truncated})


def _bellman_ford(ctx: SearchContext, W: np.ndarray | None = None) -> SearchResult:
    cycles, st = bellman_ford_cycles(
        ctx.W if W is None else W, ctx.min_len, ctx.max_len, max_relaxations=ctx.max_evals * ctx.max_len
    )
    return SearchResult(
        _negative(cycles),
        {
            "objective_evals": st["relaxations"] / ctx.max_len,
            "relaxations": st["relaxations"],
            "bf_runs": st["bf_runs"],
            "rejected_length": st["rejected_length"],
            "budget_exhausted": st["truncated"],
        },
    )


def _bounded_dp(ctx: SearchContext) -> SearchResult:
    cycles, st = bounded_dp_cycles(ctx.W, ctx.min_len, ctx.max_len)
    return SearchResult(
        _negative(cycles), {"objective_evals": st["relaxations"] / ctx.max_len, "relaxations": st["relaxations"]}
    )


def _random_search(ctx: SearchContext) -> SearchResult:
    n = int(ctx.cfg.get("random_search", {}).get("samples", 0)) or ctx.max_evals
    cycles, evals = random_cycles(ctx.W, ctx.rng, min(n, ctx.max_evals), ctx.min_len, ctx.max_len)
    return SearchResult(_negative(cycles), {"objective_evals": evals})


def _naive_mid(ctx: SearchContext) -> SearchResult:
    """Bellman-Ford on mid prices with no costs: the textbook detector."""
    res = _bellman_ford(ctx, ctx.W_mid)
    res.stats["weights"] = "mid, no fees/spread/depth"
    return res


def _qubo_instances(ctx: SearchContext):
    """(L, anchor) sub-problems covering all cycle lengths."""
    n = ctx.W.shape[0]
    anchored = bool(ctx.cfg["qubo"]["anchored"])
    for L in range(ctx.min_len, min(ctx.max_len, n) + 1):
        for anchor in (range(n) if anchored else [None]):
            yield L, anchor


def _collect(qubo: CycleQubo, W: np.ndarray, states, shots, stats: dict, found: set, raw: set) -> None:
    """Decode sampled bit vectors, repairing infeasible ones.

    `raw` receives only cycles decoded from samples that satisfied every
    constraint, i.e. what the sampler found without classical repair.
    """
    for x, n_shots in zip(states, shots):
        t0 = time.perf_counter()
        cyc = qubo.decode(x, W)
        t1 = time.perf_counter()
        stats["decode_ms"] += (t1 - t0) * 1e3
        stats["samples_distinct"] += 1
        stats["shots_total"] += int(n_shots)
        if cyc is None:
            cyc = qubo.repair(x, W)
            stats["repair_ms"] += (time.perf_counter() - t1) * 1e3
            stats["repaired"] += 1
            if cyc is None:
                stats["unrepairable"] += 1
                continue
            if cycle_weight(W, cyc) < -EPS:
                found.add(cyc)
            continue
        stats["valid"] += 1
        stats["shots_valid"] += int(n_shots)
        if cycle_weight(W, cyc) < -EPS:
            found.add(cyc)
            raw.add(cyc)


def _qubo_stats() -> dict:
    return {
        "objective_evals": 0, "instances": 0, "max_qubits": 0, "build_ms": 0.0, "optimize_ms": 0.0,
        "sample_ms": 0.0, "decode_ms": 0.0, "repair_ms": 0.0, "samples_distinct": 0, "valid": 0,
        "repaired": 0, "unrepairable": 0, "shots_total": 0, "shots_valid": 0, "budget_exhausted": False,
    }


def _qubo_sa(ctx: SearchContext) -> SearchResult:
    sa = ctx.cfg["sa"]
    stats, found, raw = _qubo_stats(), set(), set()
    for L, anchor in _qubo_instances(ctx):
        if stats["objective_evals"] + sa["reads"] * sa["sweeps"] > ctx.max_evals:
            stats["budget_exhausted"] = True
            break
        t0 = time.perf_counter()
        qubo = build_cycle_qubo(ctx.W, L, anchor, ctx.cfg["qubo"]["penalty"])
        t1 = time.perf_counter()
        states, _ = simulated_annealing(qubo.Q, ctx.rng, sa["reads"], sa["sweeps"])
        t2 = time.perf_counter()
        stats["build_ms"] += (t1 - t0) * 1e3
        stats["optimize_ms"] += (t2 - t1) * 1e3
        stats["objective_evals"] += sa["reads"] * sa["sweeps"]
        stats["instances"] += 1
        stats["max_qubits"] = max(stats["max_qubits"], qubo.n_vars)
        uniq, counts = np.unique(states, axis=0, return_counts=True)
        _collect(qubo, ctx.W, uniq, counts, stats, found, raw)
    return SearchResult(sorted(found), stats, sorted(raw))


class QuantumSolverError(RuntimeError):
    pass


def _qaoa(ctx: SearchContext) -> SearchResult:
    q = ctx.cfg["qaoa"]
    stats, found, raw = _qubo_stats(), set(), set()
    stats.update({"optimizer_iters": 0, "circuit_executions": 0, "p_ground_sum": 0.0, "p_uniform_sum": 0.0,
                  "exact_instances": 0,
                  "engine": q["engine"], "shots_final": 0})
    instances = list(_qubo_instances(ctx))
    per_instance = ctx.max_evals // max(1, len(instances))
    per_iter = q["shots_per_iter"] if q["expectation"] == "sampled" else 1
    maxiter = int(q["maxiter"])
    affordable = (per_instance - q["final_shots"]) // max(1, per_iter * max(1, q["restarts"]))
    if affordable < maxiter:
        stats["budget_exhausted"] = True
        maxiter = int(affordable)
    if maxiter < 2 * q["layers"] + 2:
        raise QuantumSolverError("evaluation budget too small for the configured QAOA")
    t_start = time.perf_counter()
    for L, anchor in instances:
        if (time.perf_counter() - t_start) * 1e3 > ctx.max_wall_ms:
            stats["budget_exhausted"] = True
            break
        t0 = time.perf_counter()
        qubo = build_cycle_qubo(ctx.W, L, anchor, ctx.cfg["qubo"]["penalty"])
        if qubo.n_vars > q["max_qubits"]:
            raise QuantumSolverError(
                f"instance needs {qubo.n_vars} qubits (L={L}), above qaoa.max_qubits={q['max_qubits']}"
            )
        E = energy_vector(qubo)
        t1 = time.perf_counter()
        probs_fn = None
        if q["engine"] == "qiskit":
            from .quantum.qiskit_backend import probabilities_fn

            probs_fn = probabilities_fn(qubo, q["layers"])
        elif q["engine"] != "numpy":
            raise QuantumSolverError(f"unknown qaoa.engine {q['engine']!r}")
        res = run_qaoa(
            E, ctx.rng, q["layers"], maxiter, q["shots_per_iter"], q["final_shots"],
            q["cvar_alpha"], q["restarts"], q["expectation"], probs_fn,
        )
        stats["build_ms"] += (t1 - t0) * 1e3
        stats["optimize_ms"] += res.timings_ms["optimize"]
        stats["sample_ms"] += res.timings_ms["sample"]
        stats["optimizer_iters"] += res.optimizer_iters
        stats["circuit_executions"] += res.circuit_executions
        stats["shots_final"] += q["final_shots"]
        stats["objective_evals"] += res.circuit_executions
        stats["instances"] += 1
        stats["max_qubits"] = max(stats["max_qubits"], qubo.n_vars)
        stats["p_ground_sum"] += res.p_ground
        stats["p_uniform_sum"] += float(np.isclose(E, E.min()).mean())
        stats["exact_instances"] += int(argmin_is_valid_optimum(qubo, ctx.W, E))
        states = [index_to_bits(k, qubo.n_vars) for k in sorted(res.counts)]
        _collect(qubo, ctx.W, states, [res.counts[k] for k in sorted(res.counts)], stats, found, raw)
        key = (qubo.n_vars, q["layers"])
        if q.get("circuit_metrics") and key not in ctx.resources:
            try:
                from .quantum.qiskit_backend import circuit_metrics

                ctx.resources[key] = circuit_metrics(qubo, q["layers"])
            except Exception as exc:  # resource reporting must never break a run
                ctx.resources[key] = {"n_qubits": qubo.n_vars, "layers": q["layers"], "error": repr(exc)}
    return SearchResult(sorted(found), stats, sorted(raw))


_REGISTRY = {
    "enumeration": _enumeration,
    "bellman_ford": _bellman_ford,
    "bounded_dp": _bounded_dp,
    "random_search": _random_search,
    "naive_mid": _naive_mid,
    "qubo_sa": _qubo_sa,
    "qaoa": _qaoa,
}


def solver_names(group: str, cfg: dict) -> list[str]:
    """Resolve a CLI solver argument (group or single solver) to solver names."""
    if group in cfg["solvers"]:
        return list(cfg["solvers"][group])
    if group == "all":
        return [s for g in cfg["solvers"].values() for s in g]
    if group in _REGISTRY:
        return [group]
    raise ValueError(f"unknown solver or group {group!r}; known: {sorted(_REGISTRY)} / {sorted(cfg['solvers'])}")


def solve(name: str, ctx: SearchContext) -> SearchResult:
    """Run a solver. Quantum solvers fall back to a classical one on failure."""
    t0 = time.perf_counter()
    status, error = "ok", None
    try:
        res = _REGISTRY[name](ctx)
    except Exception as exc:
        if name not in QUANTUM:
            raise
        fallback = ctx.cfg["qaoa"].get("fallback")
        if not fallback:
            raise
        res = _REGISTRY[fallback](ctx)
        status, error = f"fallback:{fallback}", f"{type(exc).__name__}: {exc}"
    res.stats["search_ms"] = (time.perf_counter() - t0) * 1e3
    res.stats["status"] = status
    if error:
        res.stats["error"] = error
    return res
