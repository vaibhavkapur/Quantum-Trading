"""Self-contained reports with strict cohort and artifact-integrity checks."""
from __future__ import annotations

import html
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from .config import file_hash, write_json


def _text(value):
    return html.escape(str(value))


def _number(value, digits=3):
    return "—" if value is None or not np.isfinite(value) else f"{value:.{digits}f}"


def _table(headers, rows):
    return ("<div class='scroll'><table><thead><tr>" + "".join(f"<th>{_text(x)}</th>" for x in headers) +
            "</tr></thead><tbody>" + "".join("<tr>" + "".join(f"<td>{_text(x)}</td>" for x in row) + "</tr>" for row in rows) +
            "</tbody></table></div>")


def equity_chart(daily_frames, first_seed):
    curves = []
    for frame in daily_frames:
        for solver, group in frame[frame.seed == first_seed].groupby("solver"):
            group = group.sort_values("date")
            curves.append((solver, np.r_[1.0, np.cumprod(1 + group.net.to_numpy())]))
    if not curves:
        return "<p>No complete equity series available.</p>"
    low = min(float(v.min()) for _, v in curves)
    high = max(float(v.max()) for _, v in curves)
    padding = max((high - low) * 0.1, 0.01)
    low, high = low - padding, high + padding
    colors = ["#2563eb", "#b45309", "#047857", "#9333ea", "#dc2626"]
    elements = ["<svg viewBox='0 0 920 320' role='img' aria-label='Held-out equity by solver, first predeclared seed'>"]
    for tick in np.linspace(low, high, 5):
        y = 265 - (tick - low) / (high - low) * 230
        elements.append(f"<line x1='60' x2='900' y1='{y:.1f}' y2='{y:.1f}' stroke='#dce3ee'/><text x='5' y='{y+4:.1f}' font-size='12'>{tick:.3f}</text>")
    for index, (solver, values) in enumerate(sorted(curves)):
        color = colors[index % len(colors)]
        points = " ".join(f"{60 + i / max(len(values)-1, 1)*840:.1f},{265-(v-low)/(high-low)*230:.1f}" for i, v in enumerate(values))
        elements.append(f"<polyline fill='none' stroke='{color}' stroke-width='2.5' points='{points}'/>")
        elements.append(f"<text x='{65+index*170}' y='302' fill='{color}' font-size='14'>{_text(solver)}</text>")
    return "".join(elements) + "</svg>"


def create_report(runs, output):
    runs, output = Path(runs), Path(output)
    manifests = sorted(runs.rglob("manifest.json"))
    if not manifests:
        raise ValueError(f"No run manifests under {runs}")
    cohort = None
    seen, rows, failures, timings, resources, curves, sources = set(), [], [], [], [], [], []
    first_config = None
    for manifest_path in manifests:
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("schema_version") != 1 or "run_id" not in manifest:
            raise ValueError(f"Unsupported run manifest: {manifest_path}")
        run_dir = manifest_path.parent
        sources.append((manifest["run_id"], os.path.relpath(manifest_path, output.parent), manifest["status"]))
        if manifest["status"] not in ("complete", "complete_with_failures"):
            failures.append([manifest["run_id"], "pipeline", "—", "—", manifest.get("error", "Run is incomplete")])
            continue
        key = manifest["comparison_key"]
        if cohort is not None and key != cohort:
            raise ValueError("Incompatible run cohorts: data, code, constraints, splits, or budgets differ. Report them separately.")
        cohort, first_config = key, manifest["config"]
        for relative, checksum in manifest["artifacts"].items():
            artifact = (run_dir / relative).resolve()
            if run_dir.resolve() not in artifact.parents or file_hash(artifact) != checksum:
                raise ValueError(f"Run artifact checksum/path mismatch: {relative}")
        for row in json.loads((run_dir / "summary.json").read_text()):
            identity = (row["solver"], row["seed"])
            if identity in seen:
                raise ValueError(f"Duplicate solver/seed {identity}; choose one run per comparison")
            seen.add(identity)
            rows.append({"run_id": manifest["run_id"], **row})
        timing = manifest["timing"]
        timings.append([manifest["run_id"], ", ".join(manifest["solvers"]),
                        _number(timing["data_loading_s"]), _number(timing["feature_cache_s"]),
                        str(manifest["signal_cache_hit"]), _number(timing["training_validation_and_references_s"]),
                        _number(timing["search_validation_test_s"]), _number(timing["artifacts_and_summary_s"]),
                        _number(timing["total_s"])])
        daily = pd.read_csv(run_dir / "daily_returns.csv")
        successful = {(r["solver"], r["seed"]) for r in rows if r["status"] == "ok"}
        mask = np.array([(r.solver, r.seed) in successful for r in daily.itertuples()], dtype=bool)
        curves.append(daily.loc[mask])
        for case_path in sorted((run_dir / "cases").glob("*.json")):
            case = json.loads(case_path.read_text())
            if case["status"] != "ok":
                failures.append([manifest["run_id"], case["solver"], case["seed"], case["fold"], case.get("error", "Fold failed")])
            for candidate in case["candidates"]:
                result = candidate["result"]
                if result["status"] != "ok":
                    failures.append([manifest["run_id"], case["solver"], case["seed"], case["fold"], result["error"]])
                resources.append({"run_id": manifest["run_id"], "solver": case["solver"], "seed": case["seed"],
                                  "fold": case["fold"], "risk_aversion": candidate["risk_aversion"],
                                  "status": result["status"], "evaluations": result["evaluations"],
                                  "objective_gap": candidate["objective_gap"], "elapsed_s": result["elapsed_s"],
                                  **result["timing"], **result["resources"]})
        for instance in json.loads((run_dir / "instances.json").read_text()):
            if instance["milp"]["status"] != "ok":
                failures.append([manifest["run_id"], "milp-reference", "—", instance["fold"], instance["milp"]["error"]])
    rows.sort(key=lambda row: (row["solver"], row["seed"]))
    table_rows = []
    flattened = []
    for row in rows:
        ci = row.get("sharpe_ci95")
        table_rows.append([row["solver"], row["seed"], row["status"], _number(row.get("sharpe")),
                           "—" if ci is None else f"[{ci[0]:.2f}, {ci[1]:.2f}]",
                           _number(row.get("total_return")), _number(row.get("max_drawdown")),
                           _number(row.get("annualized_turnover")), _number(row.get("bonferroni_pvalue")),
                           _number(row.get("mean_selected_objective_gap"), 6), _number(row["solver_s"]),
                           row["evaluations"], row["candidate_failures"],
                           _number(row.get("stability", {}).get("mean_adjacent_jaccard"))])
        flattened.append({**{k: v for k, v in row.items() if k not in ("sharpe_ci95", "stability")},
                          "sharpe_ci_lower": ci[0] if ci else None, "sharpe_ci_upper": ci[1] if ci else None,
                          "stability_jaccard": row.get("stability", {}).get("mean_adjacent_jaccard")})
    output.parent.mkdir(parents=True, exist_ok=True)
    csv_path = output.with_suffix(".csv")
    pd.DataFrame(flattened).to_csv(csv_path, index=False)
    resources_path = output.with_name(output.stem + "-resources.csv")
    pd.DataFrame(resources).to_csv(resources_path, index=False)
    write_json(output.with_suffix(".json"), {"comparison_key": cohort, "summaries": rows, "failures": failures})
    config = first_config or {}
    settings = "No completed experiments."
    if config:
        opt = config["optimization"]
        settings = (f"{opt['evaluations']} candidate evaluations per training objective; risk grid {opt['risk_aversion']}; "
                    f"predeclared seeds {opt['seeds']}; at most {opt['max_selected']} fixed capital sleeves. "
                    f"Training cost {config['backtest']['train_cost_bps']} bps; held-out cost {config['backtest']['evaluation_cost_bps']} bps per unit turnover.")
    document = """<!doctype html><html lang='en'><meta charset='utf-8'><meta name='viewport' content='width=device-width, initial-scale=1'>
<title>Quantum strategy optimizer — benchmark</title><style>
body{font:15px/1.6 system-ui,sans-serif;color:#17273b;background:#f4f7fb;margin:0}main{max-width:1400px;margin:auto;padding:40px 24px}
h1{font-size:34px;line-height:1.2}h2{margin-top:36px}p{max-width:1000px}.card{background:white;border:1px solid #dce3ee;border-radius:12px;padding:22px;margin:22px 0}
.scroll{overflow-x:auto}table{border-collapse:collapse;width:100%;font-size:13px}td,th{padding:10px;text-align:left;border-bottom:1px solid #e2e8f0;white-space:nowrap}th{background:#edf2f9}svg{width:100%;height:auto}a{color:#1d4ed8}.tag{color:#526780;font-size:12px;letter-spacing:1px}
</style><main><div class='tag'>REPRODUCIBLE RESEARCH · CPU SIMULATION</div><h1>Quantum strategy parameter optimizer</h1>
<p>Synthetic daily prices, chronological validation, and held-out walk-forward returns. These experiments establish neither live profitability nor quantum advantage.</p>"""
    document += f"<p>{_text(settings)}</p>"
    document += (f"<p><a href='{_text(csv_path.name)}'>Summary CSV</a> · <a href='{_text(resources_path.name)}'>Full timing and resource CSV</a> · "
                 f"<a href='{_text(output.with_suffix('.json').name)}'>Machine-readable report</a></p>")
    document += "<section class='card'><h2>Held-out performance by predeclared seed</h2><p>Returns and drawdown are fractions. Confidence intervals are circular moving-block bootstrap intervals on each held-out time series; they do not establish performance on other markets. Failed folds invalidate the aggregate series.</p>"
    document += _table(["Solver", "Seed", "Status", "Sharpe", "95% block CI", "Return", "Drawdown", "Turnover/year", "Adjusted p", "Objective gap", "Search s", "Evaluations", "Candidate failures", "Stability"], table_rows) + "</section>"
    if config:
        document += f"<section class='card'><h2>Held-out equity · seed {config['optimization']['seeds'][0]}</h2><p>The first predeclared seed is shown; all seeds appear above. Every fold starts and ends flat with costs charged.</p>"
        document += equity_chart(curves, config["optimization"]["seeds"][0]) + "</section>"
    document += "<section class='card'><h2>Entire pipeline timing</h2><p>Seconds per invocation. Total includes startup/import time when invoked through the CLI, data checks, cache access, verification, all searches, evaluation and artifact creation; the final manifest write is excluded. Shared preparation is performed once. Cold and warm caches are identified. Exact enumeration and MILP are separate references, outside optimizer evaluation budgets.</p>"
    document += _table(["Run", "Solvers", "Data s", "Features s", "Cache hit", "Training + references s", "Search + evaluation s", "Artifacts s", "Total s"], timings) + "</section>"
    document += "<section class='card'><h2>Failures and incomplete runs</h2>"
    document += (_table(["Run", "Solver", "Seed", "Fold", "Error"], failures) if failures else "<p>No failed candidates or reference solves in this comparison.</p>") + "</section>"
    document += """<section class='card'><h2>Interpretation and constraints</h2>
<p>Each selected strategy receives 1/K capital. Unused capital stays in cash. Sleeves are charged independently without cross-strategy trade netting. QUBO means and covariances use training returns only; validation chooses risk aversion; the test never chooses a configuration.</p>
<p>Grid visits feasible configurations; random samples them uniformly with replacement. TPE and annealing may evaluate penalized infeasible portfolios. QAOA uses finite measurements of a noiseless statevector; all measured candidates, repeats, and infeasible slack encodings consume budget. Equal evaluation caps are not equal computation or wall time. Grid stops early after exhausting its feasible domain. There is no classical repair or fallback in QAOA.</p>
<p>Objective gaps refer to the selected training objective, relative to exhaustive feasible search. MILP independently verifies each reference. Stability is adjacent-fold selection Jaccard similarity. Adjusted p-values use a one-sided HAC mean test and Bonferroni correction for the predeclared five-method search budget; normal approximations and finite samples limit inference. No deflated Sharpe claim is made.</p>
<p>The full resource CSV separates state preparation, compilation, simulation, sampling/decoding, scoring and optimizer overhead. Queue, communication, repair, and QPU cost are zero for these local simulations. No hardware benchmark or asymptotic speedup is implied.</p></section><h2>Source records</h2><ul>"""
    for run_id, relative, status in sources:
        document += f"<li><a href='{_text(relative)}'>{_text(run_id)}</a> — {_text(status)}</li>"
    document += "</ul></main></html>"
    output.write_text(document)
    return output

