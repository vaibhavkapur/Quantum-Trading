"""Chronological train/validation/test experiments and durable run records."""
from __future__ import annotations

import importlib.metadata
import platform
import sys
import uuid
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from . import __version__
from .backtest import backtest, ensemble, walk_forward
from .config import digest, file_hash, validate, write_json
from .data import load_data
from .metrics import block_bootstrap_sharpe, performance, sharpe, stability
from .qubo import build_qubo, exact_reference, milp_reference
from .solvers import SOLVERS, solve
from .strategies import cached_positions, library

PROTOCOL = "fixed-sleeves-next-close-nested-walk-forward-v1"


def source_hash():
    root = Path(__file__).parent.parent
    return digest({str(p.relative_to(root)): file_hash(p)
                   for folder in ("strategy_optimizer", "project") for p in sorted((root / folder).glob("*.py"))})


def select_on_validation(problem, train_result, validation_outcomes, annualization):
    if train_result.status != "ok":
        return None
    selected = train_result.selected
    if not problem.feasible(problem.encode(selected)):
        raise ValueError("Solver returned an infeasible selection")
    score = sharpe(ensemble(validation_outcomes, selected)["net"], annualization)
    return score


def run_experiment(config, solvers, launch_started=None):
    validate(config)
    if not solvers or len(set(solvers)) != len(solvers) or any(s not in SOLVERS for s in solvers):
        raise ValueError("Require a unique nonempty list of supported solvers")
    started = perf_counter() if launch_started is None else launch_started
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{timestamp}-{uuid.uuid4().hex[:8]}"
    run_dir = Path(config["output"]["runs"]) / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    manifest = {"schema_version": 1, "run_id": run_id, "status": "running",
                "config": config, "config_sha256": digest(config), "source_sha256": source_hash(),
                "protocol": PROTOCOL, "package_version": __version__, "solvers": solvers,
                "started_utc": datetime.now(timezone.utc).isoformat(),
                "python": sys.version, "platform": platform.platform(),
                "versions": {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()},
                "execution": "CPU only; noiseless sampled Qiskit statevector, no QPU",
                "budget_unit": "individual QUBO candidate evaluations, including repeated/infeasible samples",
                "reference_budget": "exhaustive/MILP verification is outside search budgets and timed separately",
                "fallback": "none", "timing": {}}
    write_json(run_dir / "manifest.json", manifest)
    try:
        stage = perf_counter()
        prices, data_manifest = load_data(config["data"])
        manifest["timing"]["data_loading_s"] = perf_counter() - stage
        manifest["dataset"] = data_manifest
        protocol_config = {key: value for key, value in config.items() if key not in ("data", "output")}
        manifest["comparison_key"] = digest({"config": protocol_config, "data": data_manifest["sha256"],
                                               "protocol": PROTOCOL, "source": manifest["source_sha256"]})
        strategies = library(config["data"]["assets"], config["strategies"]["windows"])
        manifest["strategy_library"] = [asdict(s) for s in strategies]
        stage = perf_counter()
        held, cache_hit = cached_positions(prices, strategies, data_manifest["sha256"],
                                           Path(config["data"]["path"]).parent / "cache")
        manifest["signal_cache_hit"] = cache_hit
        manifest["timing"]["feature_cache_s"] = perf_counter() - stage
        folds = walk_forward(len(prices), **config["walk_forward"])
        manifest["folds"] = [{**fold, "dates": {part: [prices.index[fold[part][0]].isoformat(), prices.index[fold[part][1]-1].isoformat()]
                                                   for part in ("train", "validation", "test")}}
                             for fold in folds]
        opt, bt = config["optimization"], config["backtest"]
        # Predeclare all five search methods even if this invocation runs only
        # classical or quantum methods, keeping correction factors comparable.
        trials = len(SOLVERS) * len(opt["seeds"]) * len(folds) * len(opt["risk_aversion"]) * opt["evaluations"]
        manifest["multiplicity_trials"] = trials
        records, daily_rows, instances = [], [], []
        shared_s = search_s = persistence_s = 0.0
        for fold in folds:
            stage = perf_counter()
            train = backtest(prices, held, strategies, *fold["train"], bt["train_cost_bps"], opt["max_selected"])
            validation = backtest(prices, held, strategies, *fold["validation"], bt["evaluation_cost_bps"], opt["max_selected"])
            problems, references = [], []
            for risk in opt["risk_aversion"]:
                problem = build_qubo(train.net, opt["max_selected"], risk, opt["cardinality_cost"], bt["annualization"])
                ref_started = perf_counter()
                reference = exact_reference(problem)
                reference["elapsed_s"] = perf_counter() - ref_started
                ref_started = perf_counter()
                try:
                    milp = milp_reference(problem)
                    if not np.isclose(milp["objective"], reference["objective"], atol=1e-7, rtol=1e-7):
                        raise ValueError("MILP and exhaustive objective disagree")
                    milp["status"] = "ok"
                except Exception as error:
                    milp = {"status": "failed", "error": f"{type(error).__name__}: {error}"}
                milp["elapsed_s"] = perf_counter() - ref_started
                instances.append({"fold": fold["fold"], "risk_aversion": risk, "matrix": problem.matrix,
                                  "offset": problem.offset, "base": problem.base, "penalty": problem.penalty,
                                  "qubits": problem.n_bits, "reference": reference, "milp": milp})
                problems.append(problem)
                references.append(reference)
            shared_s += perf_counter() - stage
            for solver in solvers:
                for seed in opt["seeds"]:
                    case_started = perf_counter()
                    candidates, best = [], None
                    for index, (risk, problem) in enumerate(zip(opt["risk_aversion"], problems)):
                        # Same fold/candidate seed across solvers; independent of loop order.
                        solver_seed = int(np.random.SeedSequence([seed, fold["fold"], index]).generate_state(1)[0])
                        result = solve(problem, solver, opt["evaluations"], solver_seed, {
                            "layers": opt["qaoa_layers"], "shots_per_evaluation": opt["shots_per_evaluation"],
                            "max_qubits": opt["max_qubits"],
                        })
                        score = select_on_validation(problem, result, validation, bt["annualization"])
                        candidate = {"risk_aversion": risk, "validation_sharpe": score, "result": result.to_dict(),
                                     "reference_objective": references[index]["objective"],
                                     "objective_gap": None if result.objective is None else
                                         float(result.objective - references[index]["objective"])}
                        candidates.append(candidate)
                        # Stable config-order tie break; no test information is consulted.
                        if score is not None and (best is None or score > candidates[best]["validation_sharpe"]):
                            best = index
                    record = {"fold": fold["fold"], "solver": solver, "seed": seed, "candidates": candidates,
                              "status": "failed" if best is None else "ok", "selected_candidate": best,
                              "candidate_failures": sum(c["result"]["status"] != "ok" for c in candidates)}
                    if best is None:
                        record["error"] = "No candidate with a feasible selection and defined validation Sharpe"
                    if best is not None:
                        chosen = candidates[best]
                        record["selected"] = chosen["result"]["selected"]
                        record["risk_aversion"] = chosen["risk_aversion"]
                        # Evaluate the held-out period only after freezing the selection.
                        test = backtest(prices, held, strategies, *fold["test"], bt["evaluation_cost_bps"], opt["max_selected"])
                        evaluated = ensemble(test, record["selected"])
                        record["test_metrics"] = performance(evaluated["net"], evaluated["turnover"], evaluated["costs"],
                                                             bt["annualization"], trials)
                        for offset, date in enumerate(prices.index[slice(*fold["test"]) ]):
                            daily_rows.append({"solver": solver, "seed": seed, "fold": fold["fold"],
                                               "date": date.isoformat(),
                                               **{key: float(value[offset]) for key, value in evaluated.items()}})
                    record["elapsed_s"] = perf_counter() - case_started
                    search_s += record["elapsed_s"]
                    records.append(record)
                    stage = perf_counter()
                    write_json(run_dir / "cases" / f"{solver}-seed{seed}-fold{fold['fold']}.json", record)
                    persistence_s += perf_counter() - stage
        stage = perf_counter()
        daily = pd.DataFrame(daily_rows, columns=["solver", "seed", "fold", "date", "gross", "net", "turnover", "costs"])
        daily.to_csv(run_dir / "daily_returns.csv", index=False)
        write_json(run_dir / "instances.json", instances)
        summaries = []
        for solver in solvers:
            for seed in opt["seeds"]:
                cases = [r for r in records if r["solver"] == solver and r["seed"] == seed]
                failures = sum(r["status"] != "ok" for r in cases)
                row = {"solver": solver, "seed": seed, "status": "failed" if failures else "ok",
                       "failed_folds": failures, "folds": len(cases),
                       "candidate_failures": sum(r["candidate_failures"] for r in cases),
                       "elapsed_s": sum(r["elapsed_s"] for r in cases),
                       "solver_s": sum(c["result"]["elapsed_s"] for r in cases for c in r["candidates"]),
                       "evaluations": sum(c["result"]["evaluations"] for r in cases for c in r["candidates"]),
                       "feasible_evaluations": sum(c["result"]["feasible_evaluations"] for r in cases for c in r["candidates"])}
                if not failures:
                    series = daily[(daily.solver == solver) & (daily.seed == seed)].sort_values("date")
                    row.update(performance(series.net.to_numpy(), series.turnover.to_numpy(), series.costs.to_numpy(), bt["annualization"], trials))
                    row["sharpe_ci95"] = block_bootstrap_sharpe(
                        series.net.to_numpy(), bt["annualization"], config["report"]["bootstrap_samples"], config["report"]["block_length"])
                    row["stability"] = stability([r["selected"] for r in cases], [s.name for s in strategies])
                    row["mean_selected_objective_gap"] = float(np.mean([r["candidates"][r["selected_candidate"]]["objective_gap"] for r in cases]))
                summaries.append(row)
        write_json(run_dir / "summary.json", summaries)
        persistence_s += perf_counter() - stage
        manifest["status"] = "complete_with_failures" if any(r["candidate_failures"] or r["status"] != "ok" for r in records) or any(i["milp"]["status"] != "ok" for i in instances) else "complete"
        manifest["artifacts"] = {str(p.relative_to(run_dir)): file_hash(p) for p in sorted(run_dir.rglob("*")) if p.is_file() and p.name != "manifest.json"}
        manifest["timing"].update({"training_validation_and_references_s": shared_s,
                                    "search_validation_test_s": search_s, "artifacts_and_summary_s": persistence_s,
                                    "total_s": perf_counter() - started})
    except Exception as error:
        manifest["status"] = "failed"
        manifest["error"] = f"{type(error).__name__}: {error}"
        manifest["timing"]["total_s"] = perf_counter() - started
        write_json(run_dir / "manifest.json", manifest)
        raise
    write_json(run_dir / "manifest.json", manifest)
    return run_dir, manifest

