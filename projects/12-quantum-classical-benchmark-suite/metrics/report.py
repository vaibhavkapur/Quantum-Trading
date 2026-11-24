from __future__ import annotations

import csv
import hashlib
import html
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import pyarrow as pa
import pyarrow.parquet as pq
from plotly.subplots import make_subplots

from benchmarks.adapters.base import STAGES
from experiments.runner import atomic_json, now
from .statistics import mean_ci95, median_ci95, wilson


def read_runs(root: Path) -> tuple[list[dict], list[dict]]:
    files = [root] if root.is_file() else sorted(root.rglob("results.jsonl"))
    if not files:
        raise ValueError(f"no results.jsonl files found under {root}")
    records, sources, seen = [], [], set()
    for path in files:
        manifest_path = path.parent / "manifest.json"
        if not manifest_path.exists():
            raise ValueError(f"run manifest is missing: {manifest_path}")
        manifest = json.loads(manifest_path.read_text())
        raw = path.read_bytes()
        actual_hash = hashlib.sha256(raw).hexdigest()
        expected_hash = manifest.get("results_sha256")
        if expected_hash is not None and expected_hash != actual_hash:
            raise ValueError(f"result checksum mismatch: {path}")
        if manifest.get("status") == "running":
            raise ValueError(f"campaign is still running: {path.parent}; wait until it finishes")
        source_records = []
        entries = {e["instance_id"]: e for e in manifest["dataset_manifest"]["instances"]}
        for number, line in enumerate(raw.decode().splitlines(), 1):
            try:
                row = json.loads(line)
            except ValueError as exc:
                raise ValueError(f"invalid result JSON at {path}:{number}") from exc
            if row.get("schema_version") != 1 or row.get("batch_id") != manifest.get("batch_id"):
                raise ValueError("result schema or batch does not match manifest")
            if row.get("protocol_sha256") != manifest.get("protocol_sha256"):
                raise ValueError("result protocol does not match manifest")
            entry = entries.get(row.get("instance_id"))
            if entry is None or row.get("instance_sha256") != entry["sha256"] or any(row.get(k) != entry[k] for k in ("track", "size")):
                raise ValueError("result instance does not match dataset manifest")
            if row.get("budget") != manifest["config"]["budget"]:
                raise ValueError("result budget does not match predeclared config")
            if row["run_id"] in seen:
                raise ValueError(f"duplicate run ID: {row['run_id']}")
            seen.add(row["run_id"])
            source_records.append(row)
        # Retain unfinished planned attempts as missing instead of silently dropping them.
        observed = Counter((r["instance_id"], r["solver"], r["seed"], r["warmup"]) for r in source_records)
        for index, job in enumerate(manifest["planned_runs"]):
            key = job["instance_id"], job["solver"], job["seed"], job["warmup"]
            if observed[key]:
                observed[key] -= 1
                continue
            entry = next(e for e in manifest["dataset_manifest"]["instances"] if e["instance_id"] == job["instance_id"])
            source_records.append({"schema_version": 1, "run_id": f"{manifest['batch_id']}-missing-{index}",
                                   "batch_id": manifest["batch_id"], "protocol_sha256": manifest["protocol_sha256"],
                                   "instance_sha256": entry["sha256"], "size": entry["size"], "track": entry["track"],
                                   "status": "missing", "error": "planned attempt did not produce a record",
                                   "quality": {"success": False, "feasible": False}, "resources": None,
                                   "timings": {"wall_seconds": None}, **job})
        if any(observed.values()):
            raise ValueError("results contain attempts absent from the predeclared manifest")
        records.extend(source_records)
        sources.append({"file": str(path.resolve()), "sha256": actual_hash, "batch_id": manifest["batch_id"],
                        "campaign_status": manifest["status"], "manifest": manifest})
    return records, sources


def flatten(value: dict, prefix: str = "") -> dict:
    result = {}
    for key, item in value.items():
        name = f"{prefix}.{key}" if prefix else key
        if isinstance(item, dict):
            result.update(flatten(item, name))
        elif isinstance(item, (list, tuple)):
            result[name] = json.dumps(item)
        else:
            result[name] = item
    return result


def summarize(records: list[dict]) -> list[dict]:
    groups = defaultdict(list)
    for row in records:
        if not row["warmup"]:
            # Batch separation prevents repeated seeds from becoming pseudo-independent samples.
            groups[(row["protocol_sha256"], row["instance_sha256"], row["solver"], row["batch_id"])].append(row)
    summaries = []
    for key, rows in sorted(groups.items()):
        protocol, instance_hash, solver, batch = key
        if len({r["seed"] for r in rows}) != len(rows):
            raise ValueError("duplicate measured seeds in a solver/instance batch")
        first = rows[0]
        feasible_rows = [r for r in rows if r["quality"].get("feasible")]
        metric = "absolute_error" if first["track"] == "estimation" else "absolute_gap"
        values = [r["quality"][metric] for r in feasible_rows if r["quality"].get(metric) is not None]
        quality_mean, quality_ci = mean_ci95(values)
        timings = [r["timings"]["wall_seconds"] for r in rows if r["timings"].get("wall_seconds") is not None]
        successes = sum(bool(r["quality"].get("success")) for r in rows)
        objectives = [r["quality"]["objective"] for r in feasible_rows if r["quality"].get("objective") is not None]
        success_probability = successes / len(rows)
        summary = {"protocol_sha256": protocol, "instance_sha256": instance_hash, "instance_id": first["instance_id"],
                   "track": first["track"], "size": first["size"], "solver": solver, "batch_id": batch,
                   "attempts": len(rows), "feasible_runs": len(feasible_rows), "quality_metric": metric,
                   "best_feasible_objective": min(objectives) if objectives else None,
                   "quality_mean_conditional": quality_mean, "quality_mean_ci95": quality_ci,
                   "quality_sample_count": len(values), "successes": successes, "success_probability": success_probability,
                   "success_ci95": wilson(successes, len(rows)), "status_counts": dict(Counter(r["status"] for r in rows)),
                   "raw_feasible_runs": sum(bool(r["quality"].get("raw_feasible")) for r in rows),
                   "repaired_runs": sum(bool(r["quality"].get("repair_applied")) for r in rows),
                   "wall_samples": len(timings), "wall_p50_seconds": float(np.median(timings)) if timings else None,
                   "wall_p95_seconds": float(np.quantile(timings, 0.95)) if timings else None,
                   "wall_p50_ci95": median_ci95(timings)}
        for stage in (*STAGES, "worker_overhead", "startup_ipc_seconds"):
            present = [r["timings"][stage] for r in rows if r["timings"].get(stage) is not None]
            summary[f"{stage}_mean_seconds"] = float(np.mean(present)) if present else None
        for metric in ("shots", "oracle_queries", "logical_qubits", "circuit_depth", "two_qubit_gates", "peak_rss_bytes",
                       "objective_evaluations", "optimizer_evaluations", "queue_seconds", "qpu_cost_usd"):
            present = [r["resources"][metric] for r in rows if r.get("resources") and r["resources"].get(metric) is not None]
            summary[f"{metric}_mean"] = float(np.mean(present)) if present else None
        summaries.append(summary)
    return summaries


def write_csv(path: Path, rows: list[dict]) -> None:
    flat = [flatten(r) for r in rows]
    columns = sorted({key for row in flat for key in row})
    with path.open("w", newline="") as out:
        writer = csv.DictWriter(out, fieldnames=columns)
        writer.writeheader()
        writer.writerows(flat)


def fmt(value) -> str:
    if value is None:
        return "—"
    if isinstance(value, (int, float)):
        return f"{value:.5g}"
    if isinstance(value, list):
        return "[" + ", ".join(fmt(x) for x in value) + "]"
    return html.escape(str(value))


def render_report(runs: Path, output: Path) -> Path:
    records, sources = read_runs(runs)
    summaries = summarize(records)
    if not summaries:
        raise ValueError("no measured attempts found")
    output.parent.mkdir(parents=True, exist_ok=True)
    prefix = output.with_suffix("")
    raw_path, summary_path = prefix.with_suffix(".raw.csv"), prefix.with_suffix(".summary.csv")
    parquet_path = prefix.with_suffix(".parquet")
    write_csv(raw_path, records)
    write_csv(summary_path, summaries)
    flat = [flatten(r) for r in records]
    columns = sorted({k for r in flat for k in r})
    pq.write_table(pa.table({key: [row.get(key) for row in flat] for key in columns}), parquet_path)
    groups = defaultdict(list)
    for summary in summaries:
        groups[(summary["track"], summary["instance_id"], summary["instance_sha256"], summary["protocol_sha256"])].append(summary)
    sections, first_plot = [], True
    fields = [("solver", "Solver"), ("attempts", "Attempts"), ("feasible_runs", "Feasible"),
              ("raw_feasible_runs", "Raw feasible"), ("repaired_runs", "Repaired"), ("best_feasible_objective", "Best objective"),
              ("quality_mean_conditional", "Mean gap / error*"), ("quality_mean_ci95", "Mean 95% CI*"),
              ("success_probability", "Success rate"), ("success_ci95", "Success 95% CI"),
              ("wall_p50_seconds", "p50 seconds"), ("wall_p95_seconds", "p95 seconds"),
              ("wall_p50_ci95", "p50 95% CI"), ("peak_rss_bytes_mean", "Mean peak RSS bytes"),
              ("logical_qubits_mean", "Qubits"), ("circuit_depth_mean", "Depth"), ("shots_mean", "Shots"),
              ("oracle_queries_mean", "Oracle queries"), ("status_counts", "Statuses")]
    for (track, instance_id, instance_hash, protocol), rows in sorted(groups.items()):
        title = f"{track} · {instance_id}"
        metric_title = "Absolute price error" if track == "estimation" else "Absolute objective gap"
        labels = [r["solver"] + " / " + r["batch_id"][-6:] for r in rows]
        figure = make_subplots(rows=1, cols=2, subplot_titles=("Full wall time: all observed attempts", metric_title + ": feasible runs only"))
        figure.add_trace(go.Bar(x=labels, y=[r["wall_p50_seconds"] for r in rows], name="p50 wall seconds"), row=1, col=1)
        figure.add_trace(go.Scatter(x=labels, y=[r["wall_p95_seconds"] for r in rows], name="p95 wall seconds", mode="markers"), row=1, col=1)
        means = [r["quality_mean_conditional"] for r in rows]
        upper = [max(0., r["quality_mean_ci95"][1] - mean) if r["quality_mean_ci95"] else 0 for r, mean in zip(rows, means)]
        lower = [max(0., mean - r["quality_mean_ci95"][0]) if r["quality_mean_ci95"] else 0 for r, mean in zip(rows, means)]
        figure.add_trace(go.Scatter(x=labels, y=means, mode="markers", marker={"size": 9},
                                   error_y={"type": "data", "array": upper, "arrayminus": lower}, name=metric_title), row=1, col=2)
        figure.update_layout(template="plotly_white", height=410, margin={"t": 60, "b": 100}, legend={"orientation": "h"})
        figure.update_yaxes(title_text="seconds", row=1, col=1)
        figure.update_yaxes(title_text="payoff units" if track == "estimation" else "objective units", row=1, col=2)
        chart = figure.to_html(full_html=False, include_plotlyjs=True if first_plot else False,
                               div_id=f"comparison-{len(sections)}", config={"displaylogo": False, "responsive": True})
        first_plot = False
        table = "<table><thead><tr>" + "".join(f"<th>{label}</th>" for _, label in fields) + "</tr></thead><tbody>"
        for row in rows:
            display = row | {"solver": row["solver"] + " / " + row["batch_id"][-6:]}
            table += "<tr>" + "".join(f"<td>{fmt(display[key])}</td>" for key, _ in fields) + "</tr>"
        table += "</tbody></table>"
        stage_figure = go.Figure()
        for stage in (*STAGES, "worker_overhead", "startup_ipc_seconds"):
            stage_figure.add_bar(x=labels, y=[r[f"{stage}_mean_seconds"] for r in rows], name=stage)
        stage_figure.update_layout(barmode="stack", title="Timing components (means of observed stages)",
                                   template="plotly_white", height=400, yaxis_title="seconds", margin={"b": 100})
        stage_chart = stage_figure.to_html(full_html=False, include_plotlyjs=False, div_id=f"timing-{len(sections)}",
                                          config={"displaylogo": False, "responsive": True})
        sections.append(f"<section><h2>{html.escape(title)}</h2><p class='muted'>Instance {instance_hash[:12]} · Protocol {protocol[:12]}</p>"
                        + chart + "<div class='scroll'>" + table + "</div>" + stage_chart + "</section>")
    scaling_groups = defaultdict(list)
    for row in summaries:
        scaling_groups[(row["track"], row["protocol_sha256"])].append(row)
    for (track, protocol), rows in sorted(scaling_groups.items()):
        if len({row["size"] for row in rows}) < 2:
            continue
        figure = make_subplots(rows=1, cols=2, subplot_titles=("Full wall time by problem size", "Peak process memory by problem size"))
        series = defaultdict(list)
        for row in rows:
            series[(row["solver"], row["batch_id"])].append(row)
        for (solver, batch), points in sorted(series.items()):
            points.sort(key=lambda r: r["size"])
            label = solver + " / " + batch[-6:]
            figure.add_trace(go.Scatter(x=[r["size"] for r in points], y=[r["wall_p50_seconds"] for r in points],
                                       mode="lines+markers", name=label, legendgroup=label), row=1, col=1)
            figure.add_trace(go.Scatter(x=[r["size"] for r in points],
                                       y=[r["peak_rss_bytes_mean"] / 2**20 if r["peak_rss_bytes_mean"] is not None else None for r in points],
                                       mode="lines+markers", name=label, legendgroup=label, showlegend=False), row=1, col=2)
        figure.update_layout(template="plotly_white", height=440, legend={"orientation": "h"})
        figure.update_xaxes(title_text="scenarios" if track == "estimation" else "binary variables")
        figure.update_yaxes(title_text="p50 seconds", row=1, col=1)
        figure.update_yaxes(title_text="MiB", row=1, col=2)
        sections.append(f"<section><h2>{html.escape(track)} · Observed size scaling</h2><p class='muted'>Protocol {protocol[:12]}. "
                        "Measured tiny instances only; these points do not establish asymptotic scaling.</p>"
                        + figure.to_html(full_html=False, include_plotlyjs=False, div_id=f"scaling-{track}-{protocol[:12]}",
                                         config={"displaylogo": False, "responsive": True}) + "</section>")
    measured = [r for r in records if not r["warmup"]]
    failures = [r for r in measured if r["status"] != "ok"]
    failure_html = "".join(f"<li>{html.escape(r['instance_id'])} / {html.escape(r['solver'])} / seed {r['seed']}: "
                           f"{html.escape(r['status'])} — {html.escape(str(r.get('error') or r.get('outcome', {}).get('termination', '')))}</li>" for r in failures)
    links = " · ".join(f"<a href='{html.escape(p.name, quote=True)}'>{html.escape(label)}</a>"
                       for p, label in [(raw_path, "Raw CSV"), (summary_path, "Summary CSV"), (parquet_path, "Parquet")])
    document = """<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Quantum vs Classical · Benchmark report</title><style>
body{font:15px/1.55 system-ui,sans-serif;background:#f3f5fa;color:#172038;margin:0}main{max-width:1500px;margin:auto;padding:36px}
h1{font-size:34px;line-height:1.2}h2{font-size:22px}section,.intro{background:white;border:1px solid #dde3ef;border-radius:12px;padding:24px;margin:24px 0}
.muted{color:#59667d;font-size:13px}.scroll{overflow:auto}table{border-collapse:collapse;width:100%;font-size:12px}td,th{padding:9px;border-bottom:1px solid #dce2ec;text-align:left;white-space:nowrap}th{background:#edf2fc}a{color:#1955ab}details{padding:12px}code{overflow-wrap:anywhere}
</style></head><body><main><h1>Quantum vs Classical</h1><p>Reproducible finance benchmarks · CPU-only simulations · No claim of quantum advantage</p>"""
    document += f"<div class='intro'><strong>{len(measured)} measured attempts · {len(failures)} non-OK attempts · {len(groups)} comparison cohorts</strong><p>{links}</p>"
    document += """<p>Each cohort has an identical instance hash, experiment protocol, source version, environment and wall-time limit.
Different objectives and problem sizes have separate plots. Batches remain separate so repeating the same seeds cannot inflate sample size.</p>
<details><summary>Methods, budgets and uncertainty</summary>
<p>Success means a feasible exact optimum within 1e-8 for optimization, or absolute estimation error within the configured tolerance.
Failed, timed-out, infeasible and missing attempts stay in the success denominator. Budget-limited feasible incumbents remain eligible.
Warmups are stored but excluded from statistics. Every measurement starts a new process: interpreter startup, imports, input loading,
encoding, compilation, simulation/solving, decoding, repair and validation are included in wall time. Campaign preflight, warmups,
result persistence and reporting are reported separately in manifests.</p>
<p>*Quality means and Student-t intervals condition on valid feasible outputs. Success intervals use Wilson; median-runtime intervals
use 2,000 deterministic bootstrap resamples. One observation has no estimated uncertainty interval. Small smoke-test seed sets are not
statistically persuasive. p50/p95 include observed failures; missing attempts have no fabricated runtime. Stage means exclude stages
unavailable after hard termination and therefore need not add to the all-attempt runtime.</p>
<p>MILP uses HiGHS with exact binary-product linearization and a node limit; its internal objective evaluation count is unavailable.
Classical candidate scores, QAOA circuit expectations and estimation oracle calls are different work units. Compare wall limits and inspect
all counters. QAE includes generic dense state preparation and counts (2k+1) oracle calls per shot. All circuits are noiseless simulations;
measurement shot noise is included. Reported depth is after basis compilation without physical-device routing or measurement gates.
These tiny explicit distributions also have an exact weighted-sum baseline: input encoding is not a free oracle.
Queue time and QPU charges are zero because no hardware is used; CPU monetary cost is unknown. No hardware scaling, live P&amp;L,
continuous-distribution pricing accuracy or production advantage is inferred.</p></details></div>"""
    document += "".join(sections)
    document += f"<section><h2>Failures and limited runs</h2><ul>{failure_html or '<li>None recorded.</li>'}</ul></section>"
    document += "<section><h2>Reproducibility manifests</h2>"
    for source in sources:
        document += f"<details><summary>{html.escape(source['batch_id'])} · {html.escape(source['campaign_status'])}</summary><pre>" + html.escape(json.dumps(source["manifest"], indent=2)) + "</pre></details>"
    document += "</section></main></body></html>"
    output.write_text(document)
    artifacts = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (output, raw_path, summary_path, parquet_path)}
    atomic_json(prefix.with_suffix(".manifest.json"), {"schema_version": 1, "created_at": now(), "artifacts": artifacts,
                                                     "sources": [{k: v for k, v in s.items() if k != "manifest"} for s in sources],
                                                     "measured_attempts": len(measured), "summary_rows": len(summaries)})
    return output
