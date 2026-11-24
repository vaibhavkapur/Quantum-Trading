from __future__ import annotations

import argparse
import csv
from dataclasses import replace
from datetime import datetime, timezone
import json
from pathlib import Path
from time import perf_counter

from rebalancing.data import load_config, load_portfolio, prepare
from rebalancing.engine import SOLVERS, plan, versions
from rebalancing.model import Problem, canonical, digest
from rebalancing.reporting import write_benchmark_report, write_run_report
from rebalancing.solvers import Budget
from rebalancing.storage import RunStore


def run_benchmarks(config_path: str, output: str | Path, *, quick: bool = False) -> list[dict]:
    config = load_config(config_path)
    declared = config.get("benchmark", {})
    seeds = declared.get("seeds", [7, 19, 41])
    counts = declared.get("asset_counts", [2, 3, 4, 6])
    solvers = declared.get("solvers", list(SOLVERS))
    if not seeds or not counts or not solvers or any(s not in SOLVERS for s in solvers) or "classical" not in solvers:
        raise ValueError("benchmark requires seeds, asset counts and a classical reference")
    if len(set(seeds)) != len(seeds) or len(set(counts)) != len(counts) or len(set(solvers)) != len(solvers):
        raise ValueError("benchmark seeds, cases and solvers must be unique")
    solvers = ["classical"] + [s for s in solvers if s != "classical"]
    budget = Budget(**config.get("budget", {}))
    if quick:
        counts = counts[:2]
        budget = replace(budget, max_evaluations=min(20, budget.max_evaluations), shots=min(128, budget.shots))
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    datasets = {}
    for count in counts:
        dataset = output / "datasets" / f"assets-{count}-v1.json"
        manifest = prepare({"dataset": str(dataset), "generator": {**config.get("generator", {}), "assets": count}})
        datasets[count] = {"path": str(dataset), "portfolio_hash": manifest["portfolio_hash"], "content_sha256": manifest["content_sha256"]}
    from dataclasses import asdict
    manifest = {"schema_version": 1, "created_at": datetime.now(timezone.utc).isoformat(), "seeds": seeds,
                "asset_counts": counts, "solvers": solvers, "budget": asdict(budget), "quick": quick,
                "datasets": {str(k): {**v, "path": str(Path(v['path']).relative_to(output))} for k, v in datasets.items()},
                "environment": versions(), "hardware_results": False, "fallback": None,
                "timing_scope": "warm-process end-to-end: load + encoding + solver + repair + decoding + audit persistence; no process startup or report rendering",
                "limits": "wall times are cooperative and may overrun by one solver call/evaluation; MILP uses its native time limit; repair has a separate budget"}
    root = Path(__file__).resolve().parents[1]
    manifest['source_sha256'] = digest({str(path.relative_to(root)): path.read_text()
                                       for package in ('rebalancing', 'project', 'api', 'benchmarks')
                                       for path in sorted((root / package).glob('*.py'))})
    # Make per-run import state consistent. Record startup separately rather than
    # charging the first quantum seed for initialization and all later seeds zero.
    initialization_start = perf_counter()
    try:
        import qiskit  # noqa: F401
        from qiskit.quantum_info import Statevector  # noqa: F401
        initialization_error = None
    except ImportError as exc:
        initialization_error = str(exc)
    manifest['quantum_backend_initialization_s'] = perf_counter() - initialization_start
    manifest['quantum_backend_initialization_error'] = initialization_error
    manifest["experiment_id"] = digest({k: v for k, v in manifest.items() if k not in ('created_at', 'quantum_backend_initialization_s')})
    (output / "manifest.json").write_text(canonical(manifest) + "\n")
    store = RunStore(output / "raw")
    rows = []
    for count in counts:
        for seed in seeds:
            reference = None
            for solver in solvers:
                start = perf_counter()
                row = {"assets": count, "seed": seed, "solver": solver, "input_hash": datasets[count]["portfolio_hash"],
                       "success": False, "status": "failed", "error": None, "run_id": None, "objective": None,
                       "absolute_gap": None, "tracking_error": None, "turnover": None, "fees_cents": None,
                       "taxes_cents": None, "cash_after_cents": None, "cash_usage_cents": None,
                       "feasible": False, "repair_applied": False, "qubits": None, "shots": None, "evaluations": None,
                       "data_loading_s": None, "encoding_s": None, "solver_s": None, "repair_s": None, "decoding_s": None,
                       "persistence_s": None, "end_to_end_s": None,
                       "state_preparation_s": None, "compilation_s": None, "optimizer_s": None,
                       "statevector_evolution_s": None, "measurement_s": None, "queue_s": None, "communication_s": None}
                try:
                    portfolio = load_portfolio(datasets[count]["path"])
                    row["data_loading_s"] = perf_counter() - start
                    if solver == 'quantum-sim':
                        row['qubits'] = sum(map(len, Problem(portfolio).choices))
                    result = plan(portfolio, solver, replace(budget, seed=seed))
                    row.update({key: result["report"][key] for key in ("objective", "tracking_error", "turnover", "fees_cents", "taxes_cents", "cash_after_cents", "cash_usage_cents", "feasible")})
                    if solver == "classical" and result['status'] == 'optimal':
                        reference = row['objective']
                    row['absolute_gap'] = max(0.0, row['objective'] - reference) if reference is not None else None
                    row.update({"status": result['status'], "success": True, "repair_applied": result['repair']['applied'],
                                "evaluations": result['evaluations'], "qubits": result['solver_metadata'].get('qubits', 0),
                                "shots": result['solver_metadata'].get('shots', 0)})
                    for key in ("encoding_s", "solver_s", "repair_s", "decoding_s"):
                        row[key] = result['timings'][key]
                    for key in ("state_preparation_s", "compilation_s", "optimizer_s", "statevector_evolution_s", "measurement_s", "queue_s", "communication_s"):
                        row[key] = result['solver_metadata'].get('timings', {}).get(key, 0.0)
                    result['experiment_id'] = manifest['experiment_id']
                    result['timings']['data_loading_s'] = row['data_loading_s']
                    t = perf_counter()
                    row['run_id'] = store.save(result)
                    row['persistence_s'] = perf_counter() - t
                except (ValueError, RuntimeError, ImportError, OSError) as exc:
                    row['success'], row['status'], row['error'] = False, 'failed', f"{type(exc).__name__}: {exc}"
                row['end_to_end_s'] = perf_counter() - start
                rows.append(row)
                print(f"assets={count} seed={seed} solver={solver} success={row['success']} seconds={row['end_to_end_s']:.4f}", flush=True)
    with (output / "results.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output / "results.json").write_text(json.dumps(rows, indent=2, allow_nan=False) + "\n")
    write_benchmark_report(rows, output / "comparison.html")
    if any(r['success'] for r in rows):
        write_run_report(output / "raw", output / "holdings.html", run_ids={r['run_id'] for r in rows if r['success']})
    (output / "audit-verification.json").write_text(canonical(store.verify()) + "\n")
    return rows


def main():
    parser = argparse.ArgumentParser(description="Run fixed-seed scaling comparisons, retaining failed runs")
    parser.add_argument("--config", default="configs/demo.yaml")
    parser.add_argument("--output", default="benchmarks/results/demo")
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    run_benchmarks(args.config, args.output, quick=args.quick)


if __name__ == "__main__":
    main()

