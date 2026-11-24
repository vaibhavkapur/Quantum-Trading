"""Aggregate stored runs into a comparison report.

    python -m qarb.report --runs runs/ --output reports/comparison.html

Writes the HTML report plus `<output stem>.csv` (aggregated) and
`<output stem>_per_seed.csv` (raw per-seed metrics). Every stored run is
included; failed runs are counted, never dropped.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy import stats as sps

# (metric key, column label, format)
QUALITY = [
    ("planted_recall", "Planted recall", ".2f"),
    ("search_recall", "Search recall (vs observable optimum)", ".2f"),
    ("search_recall_unrepaired", "Search recall without repair", ".2f"),
    ("truth_recall", "Recall vs true books", ".2f"),
    ("nominal_candidates", "Nominal candidates", ".1f"),
    ("rejected_by_cost_gate", "Rejected by cost gate", ".1f"),
    ("false_positives_after_costs", "False positives after costs", ".1f"),
]
ECON = [
    ("missed_opportunity_rate", "Missed-opportunity rate", ".2f"),
    ("available_profit_upper_bound", "Available profit, standalone upper bound", ".1f"),
    ("expected_profit", "Expected profit at decision", ".1f"),
    ("realised_pnl", "Simulated net P&L", ".1f"),
    ("mean_fill_ratio", "Mean fill ratio", ".2f"),
    ("residual_exposure", "Residual exposure", ".1f"),
    ("n_executions", "Executions", ".1f"),
]
COST = [
    ("objective_evals_mean", "Objective evals / decision", ".0f"),
    ("search_ms_p50", "Search p50 (ms)", ".3f"),
    ("search_ms_p95", "Search p95 (ms)", ".3f"),
    ("search_ms_max", "Search max (ms)", ".3f"),
    ("decision_ms_p95", "Graph+search+sizing p95 (ms)", ".3f"),
    ("graph_ms_mean", "Graph build mean (ms)", ".3f"),
    ("sizing_ms_mean", "Sizing mean (ms)", ".3f"),
    ("budget_exhausted_decisions", "Budget-capped decisions", ".1f"),
    ("fallback_decisions", "Fallback decisions", ".1f"),
]
QUBO = [
    ("max_qubits", "Max register (qubits / binary vars)", ".0f"),
    ("qubo_instances_mean", "QUBO instances / decision", ".1f"),
    ("valid_shot_fraction", "Feasible shot fraction", ".3f"),
    ("distinct_samples_repaired_fraction", "Distinct samples needing repair", ".3f"),
    ("formulation_exact_fraction", "Instances where QUBO minimum is a valid cycle", ".3f"),
    ("p_ground_mean", "P(ground state), optimized", ".4f"),
    ("p_ground_uniform_mean", "P(ground state), uniform guess", ".4f"),
    ("optimizer_iters_mean", "Optimizer iterations / decision", ".0f"),
    ("circuit_executions_mean", "Circuit executions / decision", ".0f"),
    ("build_ms_mean", "State prep / QUBO build (ms)", ".2f"),
    ("optimize_ms_mean", "Optimize (ms)", ".2f"),
    ("sample_ms_mean", "Final sampling (ms)", ".2f"),
    ("decode_ms_mean", "Decode (ms)", ".2f"),
    ("repair_ms_mean", "Repair (ms)", ".2f"),
]


def load_runs(root: Path) -> list[dict]:
    runs = []
    for path in sorted(root.rglob("metrics.json")):
        m = json.loads(path.read_text())
        m["_dir"] = str(path.parent)
        man = path.parent / "manifest.json"
        m["_manifest"] = json.loads(man.read_text()) if man.exists() else {}
        runs.append(m)
    return runs


def mean_ci(values: list[float]) -> tuple[float, float, int]:
    """Mean and half-width of the 95% Student-t interval across seeds."""
    vals = [float(v) for v in values if v is not None and not (isinstance(v, float) and math.isnan(v))]
    n = len(vals)
    if n == 0:
        return math.nan, math.nan, 0
    mean = float(np.mean(vals))
    if n < 2:
        return mean, math.nan, n
    half = float(sps.t.ppf(0.975, n - 1) * np.std(vals, ddof=1) / math.sqrt(n))
    return mean, half, n


def aggregate(runs: list[dict]) -> list[dict]:
    groups: dict[tuple[str, str], list[dict]] = {}
    for r in runs:
        groups.setdefault((r.get("config_name", "?"), r["solver"]), []).append(r)
    rows = []
    for (config, solver), rs in sorted(groups.items()):
        ok = [r for r in rs if r.get("status") == "ok"]
        row = {"config": config, "solver": solver, "seeds_ok": len(ok), "seeds_failed": len(rs) - len(ok),
               "seeds": sorted(r["seed"] for r in rs)}
        for key, _, _ in QUALITY + ECON + COST + QUBO:
            mean, half, n = mean_ci([r.get(key) for r in ok])
            row[key], row[key + "_ci95"], row[key + "_n"] = mean, half, n
        rows.append(row)
    return rows


def _cell(row: dict, key: str, fmt: str) -> str:
    mean, half = row[key], row[key + "_ci95"]
    if row[key + "_n"] == 0 or math.isnan(mean):
        return '<td class="na">n/a</td>'
    ci = "" if math.isnan(half) else f' <span class="ci">± {half:{fmt}}</span>'
    return f"<td>{mean:{fmt}}{ci}</td>"


def _table(rows: list[dict], cols: list[tuple[str, str, str]], only_with: str | None = None) -> str:
    rows = [r for r in rows if only_with is None or r[only_with + "_n"] > 0]
    if not rows:
        return "<p>No runs of this kind.</p>"
    head = "".join(f"<th>{html.escape(label)}</th>" for _, label, _ in cols)
    body = ""
    for r in rows:
        failed = f' <span class="fail">({r["seeds_failed"]} failed)</span>' if r["seeds_failed"] else ""
        body += (f'<tr><th>{html.escape(r["solver"])}</th><td>{r["seeds_ok"]}{failed}</td>'
                 + "".join(_cell(r, k, f) for k, _, f in cols) + "</tr>\n")
    return (f'<div class="scroll"><table><thead><tr><th>Solver</th><th>Seeds</th>{head}</tr></thead>'
            f"<tbody>{body}</tbody></table></div>")


CSS = """
:root{color-scheme:light dark;--fg:#1c2024;--muted:#60646c;--line:#d9dce0;--bg:#fff;--head:#f4f5f7;--warn:#b42318}
@media(prefers-color-scheme:dark){:root{--fg:#e8eaed;--muted:#9aa0a8;--line:#3a3f46;--bg:#15171a;--head:#1e2125;--warn:#ff8a7a}}
body{font:15px/1.5 system-ui,-apple-system,Segoe UI,sans-serif;color:var(--fg);background:var(--bg);max-width:1180px;margin:2rem auto;padding:0 1.25rem}
h1{font-size:1.5rem;margin-bottom:.25rem}h2{font-size:1.1rem;margin-top:2.2rem}
p,li{max-width:78ch}.muted{color:var(--muted)}
.scroll{overflow-x:auto;border:1px solid var(--line);border-radius:6px}
table{border-collapse:collapse;width:100%;font-size:13px;font-variant-numeric:tabular-nums}
th,td{padding:.45rem .7rem;text-align:right;border-bottom:1px solid var(--line);white-space:nowrap}
thead th{background:var(--head);font-weight:600;white-space:normal;vertical-align:bottom;min-width:5.5rem}
tbody th,thead th:first-child{text-align:left}tbody tr:last-child>*{border-bottom:0}
.ci{color:var(--muted)}.na{color:var(--muted)}.fail{color:var(--warn)}
.note{border-left:3px solid var(--line);padding:.1rem 1rem;margin:1rem 0}
code{font-size:.92em}
"""


def render_html(runs: list[dict], rows: list[dict]) -> str:
    configs = sorted({r["config"] for r in rows})
    sections = ""
    for config in configs:
        crow = [r for r in rows if r["config"] == config]
        cruns = [r for r in runs if r.get("config_name") == config]
        failed = [r for r in cruns if r.get("status") != "ok"]
        man = next((r["_manifest"] for r in cruns if r["_manifest"]), {})
        cfg = man.get("config", {})
        versions = man.get("versions", {})
        resources = {}
        for r in cruns:
            for q in r["_manifest"].get("quantum_resources", []):
                resources[(q.get("n_qubits"), q.get("layers"))] = q
        res_html = ""
        if resources:
            res_rows = "".join(
                f"<tr><th>L={q.get('cycle_len', '?')}{', anchored' if q.get('anchored') else ''}</th>"
                f"<td>{q.get('n_qubits')}</td><td>{q.get('layers')}</td><td>{q.get('depth', 'n/a')}</td>"
                f"<td>{q.get('two_qubit_gates', 'n/a')}</td><td>{q.get('total_gates', 'n/a')}</td>"
                f"<td>{q.get('compile_ms', float('nan')):.1f}</td></tr>"
                for _, q in sorted(resources.items())
            )
            res_html = (
                "<h2>Compiled circuit resources (not executed on hardware)</h2>"
                "<p class='muted'>QAOA circuits transpiled with Qiskit to a CX/RZ/SX/X basis assuming all-to-all "
                "connectivity. Real devices add routing overhead and noise; none of that is modelled.</p>"
                '<div class="scroll"><table><thead><tr><th>Instance</th><th>Qubits</th><th>Layers</th><th>Depth</th>'
                f"<th>CX gates</th><th>Total gates</th><th>Compile (ms)</th></tr></thead><tbody>{res_rows}</tbody></table></div>"
            )
        budget = cfg.get("budget", {})
        sections += f"""
<h2>Experiment <code>{html.escape(config)}</code></h2>
<p class="muted">{len(cruns)} stored runs, {len(failed)} failed. Seeds: {sorted({r['seed'] for r in cruns})}.
Currencies: {html.escape(', '.join(cfg.get('data', {}).get('currencies', [])))};
venues: {len(cfg.get('data', {}).get('venues', []))};
cycle length {cfg.get('graph', {}).get('min_cycle_len')}–{cfg.get('graph', {}).get('max_cycle_len')}.
Per-decision budget for every solver: {budget.get('max_objective_evals')} objective evaluations, {budget.get('max_wall_ms')} ms.
Software: {html.escape(', '.join(f'{k} {v}' for k, v in versions.items() if k != 'platform'))}; {html.escape(str(versions.get('platform', '')))}.
Hardware: {html.escape(str(man.get('hardware', 'n/a')))}.</p>
<h2>Detection quality</h2>{_table(crow, QUALITY)}
<h2>Simulated economics (numeraire units, sum over the replay)</h2>{_table(crow, ECON)}
<h2>Search cost and latency</h2>{_table(crow, COST)}
<h2>QUBO-based solvers: formulation and sampling diagnostics</h2>{_table(crow, QUBO, only_with="max_qubits")}
{res_html}
"""
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Arbitrage cycle search: classical vs QUBO/QAOA</title><style>{CSS}</style></head><body>
<h1>Arbitrage cycle search: classical vs QUBO/QAOA</h1>
<p class="muted">Values are means across predeclared seeds ± the half-width of a 95% Student-t interval.
Every solver saw the same datasets, edge weights, length bounds, cost gate and budget.</p>
<div class="note"><p><strong>How to read this.</strong> All quantum results are noiseless statevector
simulations on a CPU. Their run times measure the simulator, not a quantum device, and no hardware was used.
P&amp;L comes from a synthetic replay with a simple cost and fill model; it does not establish live profitability.
Nothing here supports a claim of quantum advantage.</p></div>
{sections}
<h2>Metric definitions</h2>
<ul>
<li><strong>Planted recall</strong>: share of planted profitable cycles detected at least once while active.</li>
<li><strong>Search recall</strong>: detections divided by what exhaustive search finds on the <em>same observed</em> books. Isolates the search algorithm from data staleness.</li>
<li><strong>Search recall without repair</strong>: the same, counting only cycles read from samples that already satisfied every QUBO constraint (equals search recall for non-QUBO solvers).</li>
<li><strong>Recall vs true books</strong>: detections divided by cycles executable on the true (undelayed) books; includes misses caused by delayed or filtered quotes.</li>
<li><strong>Nominal candidates / rejected by cost gate</strong>: negative-weight cycles a solver emitted, and how many failed liquidity-aware sizing against the minimum profit.</li>
<li><strong>False positives after costs</strong>: detections that passed the cost gate on observed books but were not executable on the true books.</li>
<li><strong>Missed-opportunity rate</strong>: share of ground-truth opportunity episodes with no profitable simulated execution.</li>
<li><strong>Objective evals</strong>: candidate solutions scored. Cycles for enumeration and random search; edge relaxations divided by the maximum cycle length for Bellman–Ford and the bounded DP; reads × sweeps for simulated annealing; circuit executions (optimizer shots plus final shots) for QAOA.</li>
</ul>
</body></html>
"""


def write_csvs(output: Path, runs: list[dict], rows: list[dict]) -> tuple[Path, Path]:
    agg_path = output.with_suffix(".csv")
    seed_path = output.with_name(output.stem + "_per_seed.csv")
    with open(agg_path, "w", newline="") as f:
        fields = [k for k in rows[0] if k != "seeds"] if rows else []
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    with open(seed_path, "w", newline="") as f:
        fields: list[str] = []
        for r in runs:
            fields += [k for k in r if not k.startswith("_") and k not in fields]
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(runs)
    return agg_path, seed_path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args(argv)
    runs = load_runs(Path(args.runs))
    if not runs:
        print(f"no metrics.json found under {args.runs}", file=sys.stderr)
        return 1
    rows = aggregate(runs)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render_html(runs, rows))
    agg, per_seed = write_csvs(output, runs, rows)
    failed = sum(r.get("status") != "ok" for r in runs)
    print(f"{len(runs)} runs ({failed} failed) -> {output}, {agg}, {per_seed}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
