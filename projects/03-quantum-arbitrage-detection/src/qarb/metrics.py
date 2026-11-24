"""Run-level metrics from raw replay records."""

from __future__ import annotations

import numpy as np

from .simulation.replay import SolverState, canonical_names


def _episodes(truth_by_t: dict[int, dict], times: list[int]) -> list[dict]:
    """Maximal runs of consecutive decision times in which a cycle is in ground truth."""
    episodes: list[dict] = []
    open_eps: dict[tuple, dict] = {}
    for t in times:
        truth = truth_by_t.get(t, {})
        for cyc in list(open_eps):
            if cyc not in truth:
                episodes.append(open_eps.pop(cyc))
        for cyc, profit in truth.items():
            ep = open_eps.setdefault(cyc, {"cycle": cyc, "times": [], "best_profit": 0.0})
            ep["times"].append(t)
            ep["best_profit"] = max(ep["best_profit"], profit)
    episodes.extend(open_eps.values())
    return episodes


def _pct(values, q) -> float:
    return float(np.percentile(values, q)) if len(values) else 0.0


def _mean(ticks: list[dict], key: str) -> float:
    vals = [t[key] for t in ticks if key in t]
    return float(np.mean(vals)) if vals else 0.0


def _sum(ticks: list[dict], key: str) -> float:
    return float(sum(t.get(key, 0) for t in ticks))


def summarize(state: SolverState, truth_by_t: dict, times: list[int], manifest: dict, cfg: dict) -> dict:
    ticks = state.ticks
    interval = cfg["execution"]["decision_interval_ms"]
    detected_at: dict[tuple, list[int]] = {}
    for t, cyc, _ in state.detections:
        detected_at.setdefault(cyc, []).append(t)

    def seen(plant: dict) -> bool:
        cyc = canonical_names(tuple(plant["currencies"]))
        return any(plant["start_ms"] <= t <= plant["end_ms"] + interval for t in detected_at.get(cyc, []))

    profitable = [p for p in manifest["planted"] if p["kind"] == "profitable"]
    thin = [p for p in manifest["planted"] if p["kind"] == "thin"]
    planted_detected = sum(seen(p) for p in profitable)
    stale_flagged = 0
    for o in manifest["outages"]:
        stale_flagged += any(
            o["start_ms"] <= t <= o["end_ms"] and o["currency"] in cyc and o["venue"] in venues
            and cyc not in truth_by_t.get(t, {})
            for t, cyc, venues in state.detections
        )

    episodes = _episodes(truth_by_t, times)
    captured = sum(any((ep["cycle"], t) in state.captured for t in ep["times"]) for ep in episodes)
    available = float(sum(ep["best_profit"] for ep in episodes))

    tp, fp, fn = _sum(ticks, "tp"), _sum(ticks, "fp"), _sum(ticks, "fn")
    execs = [o for o in state.opportunities if o["executed"]]
    realised = float(sum(o["realised_pnl"] for o in execs))
    search = [t["search_ms"] for t in ticks]
    decision = [t["decision_ms"] for t in ticks]
    statuses = [t.get("s_status", "ok") for t in ticks]
    distinct = _sum(ticks, "s_samples_distinct")
    shots = _sum(ticks, "s_shots_total")
    instances = _sum(ticks, "s_instances")

    m = {
        "solver": state.name,
        "n_decisions": len(ticks),
        # detection quality
        "planted_total": len(profitable),
        "planted_detected": planted_detected,
        "planted_recall": planted_detected / len(profitable) if profitable else None,
        "search_recall": _sum(ticks, "observable_found") / _sum(ticks, "observable")
        if _sum(ticks, "observable") else None,
        "search_recall_unrepaired": _sum(ticks, "observable_found_unrepaired") / _sum(ticks, "observable")
        if _sum(ticks, "observable") else None,
        "truth_recall": tp / (tp + fn) if tp + fn else None,
        "precision": tp / (tp + fp) if tp + fp else None,
        "true_positives": tp,
        "false_positives_after_costs": fp,
        "false_negatives": fn,
        "nominal_candidates": _sum(ticks, "candidates"),
        "rejected_by_cost_gate": _sum(ticks, "candidates") - _sum(ticks, "detected"),
        "thin_decoys_flagged": sum(seen(p) for p in thin),
        "stale_decoys_flagged": stale_flagged,
        # economics (simulated)
        "truth_episodes": len(episodes),
        "episodes_captured": captured,
        "missed_opportunity_rate": 1 - captured / len(episodes) if episodes else None,
        "available_profit_upper_bound": available,
        "expected_profit": float(sum(o["expected_profit"] for o in execs)),
        "realised_pnl": realised,
        "capture_ratio": realised / available if available > 0 else None,
        "n_executions": len(execs),
        "mean_fill_ratio": float(np.mean([o["fill_ratio"] for o in execs])) if execs else None,
        "residual_exposure": float(sum(o["residual_exposure"] for o in execs)),
        # latency and budget
        "search_ms_mean": float(np.mean(search)) if search else 0.0,
        "search_ms_p50": _pct(search, 50),
        "search_ms_p95": _pct(search, 95),
        "search_ms_max": float(max(search)) if search else 0.0,
        "decision_ms_p50": _pct(decision, 50),
        "decision_ms_p95": _pct(decision, 95),
        "graph_ms_mean": _mean(ticks, "graph_ms"),
        "sizing_ms_mean": _mean(ticks, "sizing_ms"),
        "objective_evals_mean": _mean(ticks, "s_objective_evals"),
        "budget_exhausted_decisions": int(sum(bool(t.get("s_budget_exhausted")) for t in ticks)),
        "fallback_decisions": int(sum(s != "ok" for s in statuses)),
    }
    if instances:  # QUBO-based solvers
        m.update({
            "max_qubits": int(max(t.get("s_max_qubits", 0) for t in ticks)),
            "qubo_instances_mean": instances / len(ticks),
            "valid_shot_fraction": _sum(ticks, "s_shots_valid") / shots if shots else None,
            "distinct_samples_repaired_fraction": _sum(ticks, "s_repaired") / distinct if distinct else None,
            "build_ms_mean": _mean(ticks, "s_build_ms"),
            "optimize_ms_mean": _mean(ticks, "s_optimize_ms"),
            "sample_ms_mean": _mean(ticks, "s_sample_ms"),
            "decode_ms_mean": _mean(ticks, "s_decode_ms"),
            "repair_ms_mean": _mean(ticks, "s_repair_ms"),
        })
    if any("s_circuit_executions" in t for t in ticks):
        qi = sum(t.get("s_instances", 0) for t in ticks if "s_circuit_executions" in t)
        m.update({
            "optimizer_iters_mean": _mean(ticks, "s_optimizer_iters"),
            "circuit_executions_mean": _mean(ticks, "s_circuit_executions"),
            "p_ground_mean": _sum(ticks, "s_p_ground_sum") / qi if qi else None,
            "p_ground_uniform_mean": _sum(ticks, "s_p_uniform_sum") / qi if qi else None,
            "formulation_exact_fraction": _sum(ticks, "s_exact_instances") / qi if qi else None,
        })
    return m
