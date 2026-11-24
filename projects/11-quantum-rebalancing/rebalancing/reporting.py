"""Self-contained, escaped HTML reports; no remote scripts or market data."""
from __future__ import annotations

import csv
import html
import json
from pathlib import Path
from statistics import mean, stdev
from scipy.stats import t as student_t

STYLE = """
:root{color-scheme:light;--ink:#16263c;--muted:#526179;--line:#dbe2eb;--accent:#176866}
*{box-sizing:border-box}body{margin:0;background:#f5f7fb;color:var(--ink);font:15px/1.55 system-ui,sans-serif}
main{max-width:1250px;margin:auto;padding:40px 28px}h1{font-size:36px;line-height:1.15;letter-spacing:-1px;margin:10px 0 16px}
h2{font-size:23px;margin:30px 0 12px}h3{font-size:18px}.eyebrow{text-transform:uppercase;letter-spacing:2px;font-size:12px;color:var(--accent);font-weight:700}
.muted,small{color:var(--muted)}.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:16px;margin:25px 0}
.card,article{background:white;border:1px solid var(--line);border-radius:12px;padding:22px;margin-bottom:20px}.value{font-size:28px;font-weight:700}
.scroll{overflow:auto}table{width:100%;border-collapse:collapse;text-align:left;font-size:13px}td,th{padding:11px 9px;border-bottom:1px solid var(--line);white-space:nowrap}th{color:var(--muted);font-size:12px}
.pill{border-radius:12px;background:#e0f0e9;color:#165a43;padding:3px 10px;font-size:12px}.failure{color:#9c3030;background:#ffeded}
code{font-size:12px;overflow-wrap:anywhere}input{padding:12px;border:1px solid var(--line);border-radius:8px;width:100%;max-width:420px}
.bar{display:inline-block;height:10px;background:var(--accent);border-radius:4px;vertical-align:middle;min-width:2px;margin-right:8px}
details{margin-top:16px}summary{cursor:pointer;font-weight:600}footer{margin-top:40px;font-size:12px;color:var(--muted)}
@media(max-width:600px){main{padding:22px 12px}h1{font-size:28px}.card,article{padding:14px}}
"""


def esc(value) -> str:
    return html.escape(str(value), quote=True)


def table(headers, rows) -> str:
    return '<div class="scroll"><table><thead><tr>' + ''.join(f'<th>{esc(h)}</th>' for h in headers) + '</tr></thead><tbody>' + ''.join(
        '<tr>' + ''.join(f'<td>{cell}</td>' for cell in row) + '</tr>' for row in rows) + '</tbody></table></div>'


def document(title: str, content: str) -> str:
    return f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{esc(title)}</title><style>{STYLE}</style></head><body><main><div class="eyebrow">Quantum portfolio research / v1</div><h1>{esc(title)}</h1>{content}
<footer>Fixed-price synthetic snapshots. Simulated orders only. No hardware advantage or live profitability is established.</footer></main></body></html>'''


def write_run_report(runs_directory: str | Path, output: str | Path, *, run_ids: set[str] | None = None) -> Path:
    runs = []
    for path in sorted(Path(runs_directory).rglob("*.json")):
        if run_ids is not None and path.stem not in run_ids:
            continue
        data = json.loads(path.read_text())
        if isinstance(data, dict) and "recommendation_id" in data and "report" in data:
            runs.append(data)
    if not runs:
        raise ValueError("no stored runs found")
    references = {}
    for run in runs:
        key = (run["input_hash"], run["report"]["objective_version"])
        if run["solver_used"] == "classical" and run["status"] == "optimal":
            references[key] = min(references.get(key, float("inf")), run["report"]["objective"])
    content = '<p class="muted">Before/after holdings, execution costs, solver provenance and repair. Tracking weights use initial NAV; actual post-NAV weights are reported separately.</p>'
    content += '<div class="cards">' + ''.join(f'<div class="card"><small>{label}</small><div class="value">{value}</div></div>' for label, value in [
        ("Stored runs", len(runs)), ("Distinct portfolios", len({r['input_hash'] for r in runs})),
        ("Feasible recommendations", sum(r['report']['feasible'] for r in runs)),
        ("Recommendations repaired", sum(r['repair']['applied'] for r in runs))]) + '</div>'
    content += '<label for="filter">Filter by solver or asset</label><br><input id="filter" placeholder="e.g. quantum-sim or SYN01">'
    for run in runs:
        r, m = run["report"], run["solver_metadata"]
        reference = references.get((run["input_hash"], r["objective_version"]))
        gap = f'{max(0, r["objective"] - reference):.8f}' if reference is not None else "No certified reference"
        rows = [[esc(h['symbol']), str(h['before_shares']), str(h['after_shares']), esc(h['side']), str(h['delta_lots']),
                 f"{h['before_weight']:.2%}", f"{h['reference_nav_weight']:.2%}",
                 f"{h['post_nav_weight']:.2%}" if h['post_nav_weight'] is not None else "—",
                 f"{h['target_weight']:.2%}", str(h['fee_cents']), str(h['tax_cents'])] for h in r['holdings']]
        content += f'<article class="run"><h2>{esc(run["solver_requested"])} <span class="pill">{esc(run["status"])}</span></h2>'
        if run.get("fallback_reason"):
            content += f'<p class="failure">Fallback to {esc(run["solver_used"])}: {esc(run["fallback_reason"])}</p>'
        content += f'<p>Objective <b>{r["objective"]:.8f}</b> · absolute gap <b>{gap}</b> · turnover <b>{r["turnover"]:.2%}</b> · planning <b>{run["timings"]["planning_total_s"]:.4f}s</b></p>'
        content += table(["Asset", "Before", "After", "Order", "Δ lots", "Before wt", "Ref-NAV wt", "Post-NAV wt", "Target", "Fee ¢", "Tax ¢"], rows)
        content += f'<p>Cash: {r["cash_before_cents"]:,} → <b>{r["cash_after_cents"]:,} cents</b> · reserve: {r["cash_reserve_cents"]:,} · fees: {r["fees_cents"]:,} · tax: {r["taxes_cents"]:,} · conservation residual: {r["conservation_residual_cents"]}</p>'
        content += '<details><summary>Simulated execution sequence</summary>' + table(
            ["Fill", "Asset", "Side", "Shares", "Cash before ¢", "Cash after ¢"],
            [[str(o.get('sequence', i)), esc(o['symbol']), esc(o['side']), str(o['trade_shares']),
              str(o.get('cash_before_cents', '—')), str(o.get('cash_after_cents', '—'))]
             for i, o in enumerate(r['simulated_orders'], 1)]) + '</details>'
        content += f'<p>Repair: <b>{esc(run["repair"]["applied"])}</b> · moved {run["repair"]["distance_lots"]} lots · {esc(run["repair"]["method"])}</p>'
        if m.get("qubits") is not None:
            content += f'<p>Qubits: {m["qubits"]} · shots: {m["shots"]} · raw feasible measurements: {m["measured_feasible_fraction"]:.1%} · noiseless CPU simulation; optimizer uses exact expectations.</p>'
        timing_rows = [[esc(k), f'{v:.6f}'] for k, v in run['timings'].items()]
        timing_rows += [["quantum / " + esc(k), f'{v:.6f}'] for k, v in m.get('timings', {}).items()]
        content += '<details><summary>Timing and audit provenance</summary>' + table(["Stage (nested totals are not additive)", "Seconds"], timing_rows)
        content += f'<p>Recommendation: <code>{esc(run["recommendation_id"])}</code><br>Input SHA-256: <code>{esc(run["input_hash"])}</code></p></details></article>'
    content += """<script>document.getElementById('filter').addEventListener('input',function(){const q=this.value.toLowerCase();document.querySelectorAll('.run').forEach(e=>e.hidden=!e.textContent.toLowerCase().includes(q));});</script>"""
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(document("Rebalancing run comparison", content))
    return output


def confidence(values: list[float]) -> tuple[float | None, float | None, float | None]:
    if not values:
        return None, None, None
    average = mean(values)
    if len(values) < 2:
        return average, None, None
    half = float(student_t.ppf(0.975, len(values) - 1)) * stdev(values) / len(values)**0.5
    return average, average - half, average + half


def write_benchmark_report(rows: list[dict], output: str | Path) -> list[dict]:
    summaries = []
    for count, solver in sorted({(r['assets'], r['solver']) for r in rows}):
        group = [r for r in rows if r['assets'] == count and r['solver'] == solver]
        success = [r for r in group if r['success']]
        times = confidence([r['end_to_end_s'] for r in success])
        gaps = confidence([r['absolute_gap'] for r in success if r['absolute_gap'] is not None])
        summaries.append({"assets": count, "solver": solver, "attempted": len(group), "succeeded": len(success),
                          "failed": len(group) - len(success), "mean_time_s": times[0], "time_ci95_low": times[1],
                          "time_ci95_high": times[2], "mean_absolute_gap": gaps[0], "gap_ci95_low": gaps[1],
                          "gap_ci95_high": gaps[2], "repaired": sum(r['repair_applied'] for r in success)})
    content = '<p class="muted">Same snapshot, objective, constraints and declared budgets per case. Failures remain in the denominator. Timing includes loading, planning and audit persistence, excludes process startup and report rendering.</p>'
    content += '<p>95% Student-t intervals summarize predeclared seeds on each fixed synthetic snapshot. Three seeds are exploratory; these are not estimates of market performance. QAOA uses exact statevector expectations and final measurement shots. Classical reference gaps appear only for certified MILP optima.</p>'
    maximum = max([r['mean_time_s'] for r in summaries if r['mean_time_s'] is not None] or [1])
    cells = []
    for r in summaries:
        timing = '—' if r['mean_time_s'] is None else f'<span class="bar" style="width:{150*r["mean_time_s"]/maximum:.1f}px"></span>{r["mean_time_s"]:.4f}'
        interval = '—' if r['time_ci95_low'] is None else f'[{r["time_ci95_low"]:.4f}, {r["time_ci95_high"]:.4f}]'
        cells.append([str(r['assets']), esc(r['solver']), f'{r["succeeded"]}/{r["attempted"]}', timing, interval,
                      '—' if r['mean_absolute_gap'] is None else f'{r["mean_absolute_gap"]:.8f}', str(r['repaired']), str(r['failed'])])
    content += table(["Assets", "Solver", "Success", "Mean seconds", "95% time interval", "Mean absolute gap", "Repairs", "Failures"], cells)
    failures = [r for r in rows if not r['success']]
    content += '<h2>Failures and resource limits</h2>' + (table(["Assets", "Seed", "Solver", "Reason"], [[str(r['assets']), str(r['seed']), esc(r['solver']), esc(r['error'])] for r in failures]) if failures else '<p>No failed runs.</p>')
    content += '<h2>Interpretation</h2><p>These small examples verify accounting and solver equivalence. Quantum samples may need classical repair, and a resource refusal is a failed attempt rather than a classical result relabeled as quantum. CPU simulation costs grow exponentially with qubit count. No quantum advantage is claimed.</p>'
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(document("Scaling benchmark", content))
    with output.with_suffix('.summary.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    return summaries

