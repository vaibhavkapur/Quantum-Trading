from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from uuid import uuid4

import numpy as np
import pandas as pd

from .config import public_config, resolve, write_json
from .data import load_data
from .execution import backtest, performance
from .research import discover_pairs, walk_forward_splits
from .selection import brute_force, make_problem, solve


SOLVERS = {"classical": ["top-k", "greedy", "milp", "annealing"], "quantum-sim": ["quantum-sim"],
           "all": ["top-k", "greedy", "milp", "annealing", "quantum-sim"]}


def source_hash(root: Path) -> str:
    sha = hashlib.sha256()
    for folder in ("pairs_trading", "project"):
        for path in sorted((root / folder).glob("*.py")):
            sha.update(str(path.relative_to(root)).encode())
            sha.update(path.read_bytes())
    return sha.hexdigest()


def run_experiment(cfg: dict, solver_group: str) -> Path:
    started = perf_counter()
    prices, sectors, dataset = load_data(cfg)
    loading_seconds = perf_counter() - started
    splits = walk_forward_splits(len(prices), cfg["research"])
    names = SOLVERS.get(solver_group, [solver_group])
    if any(name not in SOLVERS["all"] for name in names):
        raise ValueError(f"unsupported solver group {solver_group}")
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "-" + uuid4().hex[:8]
    directory = resolve(cfg, cfg["output"]["runs"]) / run_id
    directory.mkdir(parents=True, exist_ok=False)
    canonical = public_config(cfg)
    comparison_config = json.loads(json.dumps(canonical))
    comparison_config["output"].pop("runs", None)
    code_hash = source_hash(Path(__file__).resolve().parent.parent)
    versions = {}
    for package in ("numpy", "pandas", "scipy", "statsmodels", "ortools", "qiskit", "PyYAML"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not installed"
    comparison_key = hashlib.sha256(json.dumps({"config": comparison_config, "dataset": dataset["dataset_id"], "code": code_hash, "versions": versions}, sort_keys=True).encode()).hexdigest()
    manifest = {"schema_version": 1, "run_id": run_id, "status": "running", "experiment": cfg["experiment"], "comparison_key": comparison_key,
                "created_at_utc": datetime.now(timezone.utc).isoformat(), "config": canonical, "dataset": dataset,
                "source_sha256": code_hash, "versions": versions, "python": sys.version, "platform": platform.platform(), "processor": platform.machine(),
                "requested_solvers": names, "seeds": cfg["solver"]["seeds"], "splits": splits, "data_loading_seconds": loading_seconds,
                "unused_tail_rows": len(prices) - splits[-1]["test_end"],
                "protocol": {"selection": "train only", "entry_tuning": "validation only", "test_windows": "non-overlapping; trades closed at fold end",
                             "fill": "signal t-1 close, fill t close, frozen share hedge during trade",
                             "quantum": "noiseless CPU simulation, exact expectation optimizer, final finite-shot sampling",
                             "budget": "QAOA and annealing share optimizer-evaluation and cooperative wall caps; MILP uses wall cap. Decoding/repair evaluations additional and disclosed.",
                             "confidence_intervals": "bootstrap over predeclared solver seeds on the same market path; not market uncertainty",
                             "capital": "capital units reserve equal slots; gross exposure can drift between fills",
                             "qpu": "No hardware accessed; queue/communication zero; QPU cost not applicable."}}
    write_json(directory / "manifest.json", manifest)
    records, daily_rows, all_trades, sensitivities, summaries = [], [], [], [], []
    failures = []
    for name in names:
        for seed in cfg["solver"]["seeds"]:
            equity = cfg["execution"]["initial_capital"]
            seed_daily, seed_trades = [], []
            for split in splits:
                tick = perf_counter()
                identity = {"solver": name, "seed": seed, "fold": split["fold"]}
                try:
                    train = prices.iloc[split["train_start"]:split["train_end"]]
                    pairs, correlation, diagnostics = discover_pairs(train, sectors, cfg["research"], cfg["execution"]["commission_bps"] + cfg["execution"]["slippage_bps"])
                    discovered = perf_counter()
                    problem = make_problem(pairs, correlation, cfg["selection"])
                    built = perf_counter()
                    selected = solve(problem, name, cfg["solver"], seed)
                    solved = perf_counter()
                    if not problem.feasible(selected.x):
                        raise RuntimeError("selection failed the classical feasibility risk gate")
                    # The verifier never supplies its result to any solver or trade decision.
                    oracle = brute_force(problem) if problem.n <= 16 else None
                    verified = perf_counter()
                    active = [pair for pair, bit in zip(pairs, selected.x) if bit]
                    validation = []
                    for threshold in cfg["execution"]["entry_grid"]:
                        trial = backtest(prices, active, split["validation_start"], split["validation_end"], cfg["execution"], cfg["selection"]["capital_budget"], threshold)
                        validation.append({"entry_z": threshold, **trial.metrics})
                    winner = max(validation, key=lambda row: (row["sharpe"], -row["turnover"], row["entry_z"]))
                    validated = perf_counter()
                    test = backtest(prices, active, split["test_start"], split["test_end"], cfg["execution"], cfg["selection"]["capital_budget"], winner["entry_z"], equity)
                    tested = perf_counter()
                    scenarios = [("base", cfg["execution"], winner["entry_z"])]
                    for multiple in cfg["output"]["sensitivity_cost_multipliers"]:
                        execution = dict(cfg["execution"])
                        for key in ("commission_bps", "slippage_bps", "borrow_rate", "financing_rate"):
                            execution[key] *= multiple
                        scenarios.append((f"cost_x{multiple:g}", execution, winner["entry_z"]))
                    for offset in cfg["output"]["sensitivity_entry_offsets"]:
                        scenarios.append((f"entry_offset_{offset:+g}", cfg["execution"], winner["entry_z"] + offset))
                    for scenario, execution, threshold in scenarios:
                        result = test if scenario == "base" else backtest(prices, active, split["test_start"], split["test_end"], execution, cfg["selection"]["capital_budget"], threshold, equity)
                        sensitivities.append({**identity, "scenario": scenario, **result.metrics})
                    sensitivity_done = perf_counter()
                    equity = float(test.daily.iloc[-1]["equity"])
                    frame = test.daily.assign(**identity)
                    daily_rows.append(frame)
                    seed_daily.append(frame)
                    trades = [{**identity, **trade} for trade in test.trades]
                    all_trades.extend(trades)
                    seed_trades.extend(trades)
                    records.append({**identity, "status": "ok", "split": split, "train_end": train.index[-1].isoformat(),
                                    "test_start": prices.index[split["test_start"]].isoformat(), "test_end": prices.index[split["test_end"] - 1].isoformat(),
                                    "candidates": [pair.to_dict() for pair in pairs], "diagnostics": diagnostics,
                                    "correlation": correlation.tolist(), "constraints": [{"name": n, "coefficients": a.tolist(), "bound": b} for n, a, b in problem.constraints],
                                    "selection": selected.to_dict(), "selected_pairs": [pair.name for pair in active], "validation": validation,
                                    "chosen_entry_z": winner["entry_z"], "metrics": test.metrics, "refits": test.refits,
                                    "oracle_objective": oracle.objective if oracle else None,
                                    "optimality_gap": max(0, selected.objective - oracle.objective) if oracle else None,
                                    "timings": {"screening_seconds": discovered - tick, "problem_build_seconds": built - discovered,
                                                "selection_seconds": solved - built, "verification_seconds": verified - solved,
                                                "validation_seconds": validated - verified, "backtest_seconds": tested - validated,
                                                "sensitivity_seconds": sensitivity_done - tested, "total_seconds": perf_counter() - tick}})
                except Exception as exc:
                    failure = {**identity, "status": "failed", "error_type": type(exc).__name__, "error": str(exc), "elapsed_seconds": perf_counter() - tick}
                    failures.append(failure)
                    records.append(failure)
                    # Later folds cannot silently resume a broken capital path.
                    break
            if len(seed_daily) == len(splits):
                rows = [r for r in records if r["solver"] == name and r["seed"] == seed and r["status"] == "ok"]
                summaries.append({"solver": name, "seed": seed, **performance(pd.concat(seed_daily, ignore_index=True), seed_trades, cfg["execution"]["initial_capital"]),
                                  "folds": len(splits), "mean_objective_gap": float(np.mean([r["optimality_gap"] for r in rows])) if all(r["optimality_gap"] is not None for r in rows) else None,
                                  "selection_seconds": sum(r["timings"]["selection_seconds"] for r in rows),
                                  "pipeline_seconds": sum(r["timings"]["total_seconds"] for r in rows),
                                  "fallback_folds": sum(r["selection"]["status"] == "fallback_greedy" for r in rows)})
    computation_done = perf_counter()
    write_json(directory / "run.json", {"run_id": run_id, "comparison_key": comparison_key, "records": records, "summaries": summaries, "failures": failures})
    if daily_rows:
        pd.concat(daily_rows, ignore_index=True).to_csv(directory / "daily.csv", index=False)
    pd.DataFrame(all_trades, columns=list(all_trades[0]) if all_trades else ["solver", "seed", "fold", "pair", "net_pnl"]).to_csv(directory / "trades.csv", index=False)
    pd.DataFrame(sensitivities).to_csv(directory / "sensitivity.csv", index=False)
    pd.DataFrame(summaries).to_csv(directory / "summary.csv", index=False)
    manifest.update({"status": "completed_with_failures" if failures else "complete", "failure_count": len(failures),
                     "successful_fold_count": sum(r["status"] == "ok" for r in records), "serialization_seconds": perf_counter() - computation_done,
                     "end_to_end_seconds": perf_counter() - started})
    write_json(directory / "manifest.json", manifest)
    return directory
