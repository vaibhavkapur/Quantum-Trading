from __future__ import annotations

from collections import defaultdict
import csv
from html import escape
import json
from pathlib import Path
import numpy as np
from scipy.stats import t
from .experiment import write_csv, write_json

METRICS = ("net_pnl", "net_spread_capture", "sharpe", "max_drawdown", "mean_abs_inventory", "adverse_selection_cost",
           "maker_rebates", "taker_fees", "fill_probability", "quote_ack_p99_ns", "native_quote_p99_ns",
           "policy_update_seconds", "end_to_end_seconds", "max_optimality_gap")


def confidence(values):
    values = np.asarray(values, dtype=float)
    mean = float(np.mean(values))
    half = float(t.ppf(.975, len(values)-1)*np.std(values,ddof=1)/np.sqrt(len(values))) if len(values)>1 else None
    return mean, half


def collect(runs):
    # Repeated executions are not independent seeds. Choose the latest complete
    # trial (including failure) per compatibility/solver/scenario/seed key.
    selected = {}
    manifests = []
    for path in sorted(Path(runs).rglob("manifest.json")):
        manifest = json.loads(path.read_text())
        if manifest.get("schema") != "mm-run-v1":
            continue
        manifests.append(manifest)
        group = tuple(manifest["compatibility"][k] for k in ("config_id", "dataset_id", "source_id"))
        for trial in manifest["trials"]:
            key = (*group, manifest["solver"], trial["scenario"], trial["seed"])
            if key not in selected or manifest["created_utc"] > selected[key][0]:
                selected[key] = (manifest["created_utc"], path.parent, trial)
    groups, failures = defaultdict(list), []
    for key, (_, directory, trial) in selected.items():
        if trial["status"] != "ok":
            failures.append(dict(group=key[:3], solver=key[3], scenario=key[4], seed=key[5], error=trial.get("error", "trial has not completed")))
            continue
        metrics = json.loads((directory / trial["directory"] / "metrics.json").read_text())
        groups[key[:-1]].append(metrics)
    if not selected:
        raise ValueError("no experiment manifests found")
    return groups, failures, manifests


def chart(rows):
    # Standalone SVG, no remote dependencies. Horizontal P&L with 95% seed CIs.
    width, left, right, row_height = 1000, 330, 70, 30
    low = min([0.0, *[r["net_pnl_mean"]-(r["net_pnl_ci95"] or 0) for r in rows]])
    high = max([0.0, *[r["net_pnl_mean"]+(r["net_pnl_ci95"] or 0) for r in rows]])
    span = max(high-low, 0.01)
    low -= span*.08; high += span*.08
    def x(v): return left+(width-left-right)*(v-low)/(high-low)
    height = 90 + row_height*len(rows)
    parts = [f'<svg role="img" aria-label="Out-of-sample net P and L by strategy, with 95 percent confidence intervals" viewBox="0 0 {width} {height}">',
             f'<line x1="{x(0)}" y1="10" x2="{x(0)}" y2="{height-55}" stroke="#adb7c3"/>']
    for i,r in enumerate(rows):
        y=25+i*row_height
        label=f'{r["scenario"]} · {r["solver"]} (n={r["seeds"]})'
        color="#246e61" if r["solver"]=="classical" else ("#74559e" if r["solver"]=="quantum-sim" else "#657384")
        mean, ci=r["net_pnl_mean"],r["net_pnl_ci95"] or 0
        parts += [f'<text x="{left-15}" y="{y+4}" text-anchor="end" font-size="12">{escape(label)}</text>',
                  f'<line x1="{x(mean-ci)}" y1="{y}" x2="{x(mean+ci)}" y2="{y}" stroke="{color}" stroke-width="2"/>',
                  f'<circle cx="{x(mean)}" cy="{y}" r="4" fill="{color}"/>']
    for v in np.linspace(low, high, 5):
        parts.append(f'<text x="{x(v)}" y="{height-28}" text-anchor="middle" font-size="12">{v:.2f}</text>')
    parts.append(f'<text x="{(left+width-right)/2}" y="{height-6}" text-anchor="middle" font-size="13">Net P&amp;L (synthetic currency units)</text></svg>')
    return "".join(parts)


def report(runs, output):
    groups, failures, manifests = collect(runs)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for key, trials in sorted(groups.items()):
        row = dict(config_id=key[0], dataset_id=key[1], source_id=key[2], solver=key[3], scenario=key[4],
                   seeds=len(trials), fallbacks=sum(m["optimizer_fallbacks"] for m in trials),
                   halted_runs=sum(m["halted"] for m in trials))
        for metric in METRICS:
            row[metric+"_mean"], row[metric+"_ci95"] = confidence([m[metric] for m in trials])
        rows.append(row)
    write_csv(output.with_suffix(".csv"), rows)
    write_json(output.with_suffix(".json"), dict(rows=rows, failures=failures))
    by_group = defaultdict(list)
    for row in rows:
        by_group[(row["config_id"],row["dataset_id"],row["source_id"])].append(row)
    body = []
    for index, (key, group_rows) in enumerate(by_group.items()):
        svg = chart(group_rows)
        svg_path = output.parent / f"{output.stem}-pnl-{index+1}.svg"
        svg_path.write_text(svg)
        body.append(f'<h2>Experiment {index+1}</h2><p class="meta">Config {key[0][:12]} · Dataset {key[1][:12]} · Source {key[2][:12]}. Identical costs, splits and budgets within this group.</p>{svg}')
        body.append('<div class="scroll"><table><thead><tr><th>Scenario</th><th>Strategy</th><th>Seeds</th><th>Net P&amp;L ± 95% CI</th><th>Drawdown</th><th>Adverse selection</th><th>Maker rebates</th><th>Quote p99 (µs)</th><th>Policy updates (s)</th><th>Full trial (s)</th><th>Fallbacks</th></tr></thead><tbody>')
        for r in group_rows:
            ci = f' ± {r["net_pnl_ci95"]:.3f}' if r["net_pnl_ci95"] is not None else ' (CI unavailable)'
            cells=[r["scenario"],r["solver"],str(r["seeds"]),f'{r["net_pnl_mean"]:.3f}{ci}',f'{r["max_drawdown_mean"]:.3f}',
                   f'{r["adverse_selection_cost_mean"]:.3f}',f'{r["maker_rebates_mean"]:.3f}',f'{r["quote_ack_p99_ns_mean"]/1000:.2f}',
                   f'{r["policy_update_seconds_mean"]:.3f}',f'{r["end_to_end_seconds_mean"]:.3f}',str(r["fallbacks"])]
            body.append('<tr>'+''.join('<td>'+escape(c)+'</td>' for c in cells)+'</tr>')
        body.append('</tbody></table></div>')
    body.append(f'<h2>Failures and scope</h2><p>{len(failures)} failed trials. Fallbacks are counted in the table and remain part of strategy results.</p>')
    for failure in failures:
        body.append('<p>'+escape(f'{failure["solver"]}/{failure["scenario"]}/{failure["seed"]}: {failure["error"]}')+'</p>')
    body.append('<h2>Measurement protocol</h2><p>Intervals are two-sided Student-t 95% confidence intervals across independent predeclared seeds. Repeated executions of the same seed are not independent samples; the newest trial is used. Small samples and correlated stress scenarios limit inference. Sharpe in CSV is a non-annualized within-window P&amp;L-change statistic.</p><p>Quote p99 is the mean across runs of each run’s p99 Python-to-C++ local quote-and-ack duration, excluding optimizer barriers. It is not exchange latency. Raw trajectories include native and bridge timings. Full trial time includes calibration, validation, test replay, and optimizer waits; manifests additionally record loading, native build/cache, initialization, and full experiment time. Optimizer logs separate objective preparation, state preparation, compilation, iterations, sampling, decoding and communication.</p><p>QAOA uses a noiseless Qiskit statevector, exact expectation values for training and finite-shot final sampling. Every policy cost is enumerated classically before encoding. Exhaustive grid search is therefore a particularly strong baseline; no speedup or quantum advantage is claimed. No QPU, queue fee or network exchange is involved.</p><p>Only synthetic flow is bundled. FIFO queue placement from L2 snapshots is an assumption. Maker rebates and positive adverse-selection losses are separate; markouts at the end of a window are censored. Residual inventory after depth-limited liquidation remains marked to market. Backtests do not establish live profitability.</p>')
    html='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Quantum market-making research</title><style>body{font:15px/1.55 system-ui,sans-serif;color:#203044;background:#f5f6f8;margin:0}main{max-width:1300px;margin:auto;padding:48px 32px}h1{font-size:32px;letter-spacing:-1px;margin-bottom:8px}h2{font-size:20px;margin-top:36px}.meta{color:#647184;font-size:13px}svg{background:white;width:100%;margin:12px 0}table{border-collapse:collapse;width:100%;font-size:12px;background:white}th,td{padding:10px;text-align:right;border-bottom:1px solid #e1e5eb;white-space:nowrap}th:first-child,td:first-child,th:nth-child(2),td:nth-child(2){text-align:left}th{background:#e9edf2}.scroll{overflow:auto}p{max-width:1050px}a{color:#246e61}</style><main><p class="meta">RESEARCH IMPLEMENTATION · CPU SIMULATION</p><h1>Quantum-enhanced market making</h1><p>Out-of-sample policy comparisons under identical inventory limits and execution costs. Means and uncertainty across fixed seeds, with full pipeline timing.</p>'''+''.join(body)+'</main></html>'
    output.write_text(html)
    return output
