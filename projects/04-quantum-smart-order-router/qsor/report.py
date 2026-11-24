"""Aggregate stored runs into an HTML comparison (plus a CSV of the same rows)."""
from __future__ import annotations

import csv
import html
import json
import math
from pathlib import Path

import numpy as np
from scipy import stats

from .experiment import REFERENCE_SOLVERS

ORDER = ["best_price", "pro_rata", "greedy", "greedy_live", "exact", "cpsat", "sa", "random_repair", "qaoa"]

COLUMNS = [  # (metric key, header, format, lower_is_better or None)
    ("is_bps", "Impl. shortfall (bps)", "{:.3f}", True),
    ("fill_rate", "Fill rate", "{:.4f}", False),
    ("price_improvement_bps", "Price improv. (bps)", "{:.3f}", False),
    ("decision_us_p50", "Decision p50 (µs)", "{:.2f}", True),
    ("decision_us_p99", "Decision p99 (µs)", "{:.2f}", True),
    ("decision_us_p999", "Decision p99.9 (µs)", "{:.1f}", True),
    ("o2a_ms_p50", "Order→ack p50 (sim ms)", "{:.3f}", True),
    ("o2a_ms_p99", "Order→ack p99 (sim ms)", "{:.3f}", True),
    ("queue_error_lots_mean", "Queue-pos. error (lots)", "{:.2f}", True),
    ("stale_policy_frequency", "Stale-policy freq.", "{:.4f}", True),
]
SOLVER_COLUMNS = [
    ("solver_wall_ms_mean", "Refresh wall mean (ms)", "{:.2f}"),
    ("solver_wall_ms_p99", "Refresh wall p99 (ms)", "{:.2f}"),
    ("solver_evals_per_problem", "Objective evals / problem", "{:.1f}"),
    ("solver_shots_per_problem", "Shots / problem", "{:.0f}"),
    ("solver_ms_build", "Build (ms)", "{:.1f}"),
    ("solver_ms_encode", "Encode (ms)", "{:.1f}"),
    ("solver_ms_optimize", "Optimize (ms)", "{:.1f}"),
    ("solver_ms_sample", "Sample (ms)", "{:.1f}"),
    ("solver_ms_decode_repair", "Decode+repair (ms)", "{:.1f}"),
    ("solver_fallbacks", "Fallbacks", "{:.1f}"),
    ("solver_overrun_frequency", "Wall > refresh interval", "{:.3f}"),
    ("skipped_refreshes", "Skipped refreshes", "{:.1f}"),
]


def mean_ci(values):
    """Mean and 95% t-interval half-width across seeds."""
    v = np.asarray([x for x in values if x is not None and not (isinstance(x, float) and math.isnan(x))], dtype=float)
    if v.size == 0:
        return float("nan"), float("nan"), 0
    if v.size == 1:
        return float(v[0]), float("nan"), 1
    half = stats.t.ppf(0.975, v.size - 1) * v.std(ddof=1) / math.sqrt(v.size)
    return float(v.mean()), float(half), int(v.size)


def load_runs(runs_root):
    """{experiment: {solver: {"manifest": ..., "seeds": {seed: metrics}}}}"""
    out = {}
    for mpath in sorted(Path(runs_root).glob("*/*/manifest.json")):
        man = json.loads(mpath.read_text())
        seeds = {}
        for seed, info in man["seeds"].items():
            f = mpath.parent / f"seed_{seed}" / "metrics.json"
            if info["status"] == "ok" and f.exists():
                seeds[seed] = json.loads(f.read_text())
        out.setdefault(man["experiment"], {})[man["solver"]] = {"manifest": man, "seeds": seeds}
    return out


def _fmt(fmt, mean, half):
    if math.isnan(mean):
        return "–"
    s = fmt.format(mean)
    return s if math.isnan(half) else f"{s} <span class='ci'>± {fmt.format(half)}</span>"


def _sorted(solvers):
    return sorted(solvers, key=lambda s: (ORDER.index(s) if s in ORDER else 99, s))


def build(runs_root, output) -> Path:
    runs = load_runs(runs_root)
    if not runs:
        raise SystemExit(f"no runs found under {runs_root}")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    parts, csv_rows = [], []
    for exp, solvers in runs.items():
        any_man = next(iter(solvers.values()))["manifest"]
        cfg, calib = any_man["config"], any_man["calibration"]
        ref = next((r for r in REFERENCE_SOLVERS if r in solvers), None)
        parts.append(f"<h2>Experiment <code>{html.escape(exp)}</code></h2>")
        parts.append(
            "<p class='meta'>"
            f"Window: <b>{any_man['window']}</b> (steps {any_man['window_steps'][0]}–{any_man['window_steps'][1]}, "
            "chronologically after the train and validation windows) · "
            f"seeds: {', '.join(map(str, cfg['seeds']))} · "
            f"calibrated fill probabilities {calib['fill_prob']} (train) · "
            f"non-fill penalty {calib['nonfill_penalty']} (validation) · "
            f"policy delay model: {cfg['optimizer']['delay_model']} · "
            f"dataset sha256 <code>{any_man['dataset']['sha256'][:12]}…</code></p>")

        head = "<tr><th>Solver</th><th>Kind</th><th>Seeds ok / failed</th>" + "".join(
            f"<th>{h}</th>" for _, h, _, _ in COLUMNS)
        if ref:
            head += f"<th>ΔIS vs {ref} (bps, paired)</th>"
        rows = [head + "</tr>"]
        for s in _sorted(solvers):
            man, seeds = solvers[s]["manifest"], solvers[s]["seeds"]
            failed = sum(1 for i in man["seeds"].values() if i["status"] != "ok")
            tds = [f"<td><b>{s}</b></td>", f"<td class='kind'>{html.escape(man['solver_kind'])}</td>",
                   f"<td>{len(seeds)} / {failed}</td>"]
            row = {"experiment": exp, "solver": s, "kind": man["solver_kind"], "seeds_ok": len(seeds),
                   "seeds_failed": failed}
            for key, _, fmt, _ in COLUMNS:
                mean, half, _ = mean_ci([m.get(key) for m in seeds.values()])
                tds.append(f"<td>{_fmt(fmt, mean, half)}</td>")
                row[key + "_mean"], row[key + "_ci95"] = mean, half
            if ref:
                common = sorted(set(seeds) & set(solvers[ref]["seeds"]))
                diff = [seeds[k]["is_bps"] - solvers[ref]["seeds"][k]["is_bps"] for k in common]
                mean, half, _ = mean_ci(diff)
                tds.append(f"<td>{'reference' if s == ref else _fmt('{:+.3f}', mean, half)}</td>")
                row["delta_is_vs_ref_mean"], row["delta_is_vs_ref_ci95"] = mean, half
            for key, _, _ in SOLVER_COLUMNS:
                row[key + "_mean"] = mean_ci([m.get(key) for m in seeds.values()])[0]
            row["wall_s_total"] = man["wall_s"]
            csv_rows.append(row)
            rows.append("<tr>" + "".join(tds) + "</tr>")
        parts.append("<h3>Execution quality and latency</h3><div class='scroll'><table>" + "".join(rows) + "</table></div>")

        rows = ["<tr><th>Solver</th>" + "".join(f"<th>{h}</th>" for _, h, _ in SOLVER_COLUMNS)
                + "<th>Run wall, all seeds (s)</th></tr>"]
        for s in _sorted(solvers):
            seeds = solvers[s]["seeds"]
            tds = [f"<td><b>{s}</b></td>"]
            for key, _, fmt in SOLVER_COLUMNS:
                mean, half, _ = mean_ci([m.get(key) for m in seeds.values()])
                tds.append(f"<td>{_fmt(fmt, mean, float('nan'))}</td>")
            tds.append(f"<td>{solvers[s]['manifest']['wall_s']:.1f}</td>")
            rows.append("<tr>" + "".join(tds) + "</tr>")
        parts.append("<h3>Optimizer overhead (per run, mean across seeds)</h3>"
                     "<p class='meta'>Build/encode/optimize/sample/decode are totals per run in ms. "
                     "Evaluations are objective (expectation or energy-difference) evaluations inside the solver.</p>"
                     "<div class='scroll'><table>" + "".join(rows) + "</table></div>")
        v = any_man["versions"]
        parts.append(f"<p class='meta'>Environment: Python {v['python']}, NumPy {v['numpy']}, SciPy {v['scipy']}, "
                     f"OR-Tools {v['ortools']}, Qiskit {v['qiskit']}, {html.escape(v['platform'])}. "
                     f"Hardware: {any_man['hardware']}.</p>")

    with open(output.with_suffix(".csv"), "w", newline="") as f:
        keys = []
        for r in csv_rows:
            keys += [k for k in r if k not in keys]
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(csv_rows)
    output.write_text(PAGE.replace("{{BODY}}", "\n".join(parts)))
    return output


PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Smart order router: solver comparison</title>
<style>
:root { color-scheme: light dark; --fg:#1a1d21; --muted:#5d6670; --line:#d9dde2; --bg:#ffffff; --alt:#f5f6f8; }
@media (prefers-color-scheme: dark) { :root { --fg:#e6e8eb; --muted:#9aa3ad; --line:#353b43; --bg:#15181c; --alt:#1c2025; } }
body { font: 14px/1.5 -apple-system, "Segoe UI", Helvetica, Arial, sans-serif; color: var(--fg); background: var(--bg);
       margin: 0; padding: 32px; }
main { max-width: 1280px; margin: 0 auto; }
h1 { font-size: 22px; margin: 0 0 4px; } h2 { font-size: 17px; margin: 36px 0 6px; } h3 { font-size: 14px; margin: 24px 0 6px; }
p { max-width: 900px; } .meta, .kind, .ci { color: var(--muted); } .meta { font-size: 13px; }
.scroll { overflow-x: auto; }
table { border-collapse: collapse; font-variant-numeric: tabular-nums; white-space: nowrap; }
th, td { padding: 6px 12px; border-bottom: 1px solid var(--line); text-align: right; }
th { font-weight: 600; font-size: 12px; white-space: normal; vertical-align: bottom; min-width: 78px; }
th:first-child, td:first-child, td.kind, th:nth-child(2) { text-align: left; }
td.kind { white-space: normal; min-width: 180px; font-size: 12px; }
tr:nth-child(even) td { background: var(--alt); }
code { font-size: 12px; }
.note { border-left: 3px solid var(--line); padding: 2px 14px; margin: 16px 0; }
</style></head><body><main>
<h1>Smart order router: solver comparison</h1>
<div class="note"><p>Values are mean ± 95% confidence half-width (Student t) across the predeclared seeds.
Every solver sees the same dataset, parent-order flow, cost model, constraints and refresh schedule.
The "quantum" rows are noiseless CPU statevector simulations of small instances; no quantum hardware was used,
and nothing here demonstrates a quantum advantage. Decision latency is wall-clock time of the Python routing
function and is noisy at the tail; order→ack is simulated network time. Results come from a synthetic market
simulator and do not establish live trading performance.</p></div>
{{BODY}}
</main></body></html>
"""
