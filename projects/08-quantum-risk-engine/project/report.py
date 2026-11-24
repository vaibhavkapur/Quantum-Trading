"""Portable offline HTML/CSV report; no CDN, network calls, or plotting runtime."""
import argparse
import csv
from html import escape
import json
import math
from pathlib import Path


def number(value):
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:,.6g}"
    return escape(str(value))


def table(headers, rows):
    head = "".join(f"<th>{escape(h)}</th>" for h in headers)
    body = "".join("<tr>" + "".join(f"<td>{number(cell)}</td>" for cell in row) + "</tr>" for row in rows)
    return f'<div class="scroll"><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'


def scaling_svg(summary):
    valid = [r for r in summary if r.get("successful", 0) > 0]
    if not valid:
        return ""
    xmin, xmax = min(r["oracle_queries"] for r in valid), max(r["oracle_queries"] for r in valid)
    ymax = max(max(r["mean_error_ci95"] or [r["mean_absolute_error"]]) for r in valid) * 1.1 or 1
    def x(value):
        return 90 + 680*(math.log10(value)-math.log10(xmin))/max(math.log10(xmax)-math.log10(xmin), 1e-9)
    def y(value):
        return 300 - 235*value/ymax
    parts = ['<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 840 390" role="img" aria-label="Mean tail-probability absolute error versus oracle query budget">',
             '<rect width="840" height="390" fill="#182123"/>',
             '<text x="90" y="25" fill="#edf3ed" font-size="16">Tail-probability precision at matched query budgets</text>']
    for j in range(5):
        value = j*ymax/4
        parts += [f'<path d="M90 {y(value)}H770" stroke="#344244"/>',
                  f'<text x="80" y="{y(value)+4}" text-anchor="end" fill="#a9b9b6" font-size="11">{value:.4f}</text>']
    for value in sorted(set(r["oracle_queries"] for r in valid)):
        parts.append(f'<text x="{x(value)}" y="320" text-anchor="middle" fill="#a9b9b6" font-size="11">{value}</text>')
    for method, color, label, legend_x in [("mc_tail", "#91bdce", "Monte Carlo", 90), ("quantum_mlae_tail", "#b9e197", "Quantum MLAE", 270)]:
        rows = sorted([r for r in valid if r["method"] == method], key=lambda r: r["oracle_queries"])
        if not rows:
            continue
        path = " ".join(("M" if i == 0 else "L") + f'{x(r["oracle_queries"])} {y(r["mean_absolute_error"])}' for i, r in enumerate(rows))
        parts += [f'<path d="{path}" stroke="{color}" fill="none" stroke-width="2"/>',
                  f'<text x="{legend_x}" y="48" fill="{color}" font-size="12">{label}</text>']
        for row in rows:
            xx, yy = x(row["oracle_queries"]), y(row["mean_absolute_error"])
            parts.append(f'<circle cx="{xx}" cy="{yy}" r="4" fill="{color}"/>')
            if row["mean_error_ci95"]:
                lo, hi = row["mean_error_ci95"]
                parts.append(f'<path d="M{xx} {y(lo)}V{y(hi)}" stroke="{color}"/>')
    parts += ['<text x="430" y="349" text-anchor="middle" fill="#a9b9b6" font-size="12">Oracle queries / classical samples (log scale)</text>',
              '<text transform="translate(18 190) rotate(-90)" text-anchor="middle" fill="#a9b9b6" font-size="12">Mean absolute probability error</text>',
              '<text x="90" y="377" fill="#a9b9b6" font-size="11">Whiskers: 95% Student-t intervals across predeclared seeds. Source: benchmark.json.</text></svg>']
    return "".join(parts)


def build_report(runs, output, benchmark=None):
    paths = sorted(Path(runs).glob("*.json"))
    if not paths:
        raise ValueError("no run JSON files found")
    records = [json.loads(path.read_text()) for path in paths]
    if any(r.get("schema_version") != 1 or "metrics" not in r for r in records):
        raise ValueError("unsupported or invalid run file")
    # Keep unrelated models/datasets visibly separated instead of averaging them.
    sections, csv_rows = [], []
    for run in records:
        sections.append(f'<section><h2>{escape(run["experiment"])} · {escape(run["solver_requested"])}</h2>'
                        f'<p>Status: {escape(run["status"])} · Seed {run["seed"]} · Confidence {run["alpha"]:.2%} · '
                        f'End-to-end {run["timings"]["end_to_end_seconds"]:.4f} s</p>'
                        f'<p class="small">Run {escape(run["run_id"])}<br>Config {escape(run["config_sha256"])}<br>Dataset {escape(run["data_sha256"])}</p>')
        if run["fallback_reason"]:
            sections.append(f'<p>Explicit classical fallback: {escape(run["fallback_reason"])}</p>')
        sections.append(table(["Estimator", "VaR", "Expected shortfall", "VaR error", "ES error", "Compute (s)", "Samples / shots", "Oracle queries"],
                              [[m["method"], m["var"], m["cvar"], m["var_abs_error"], m["cvar_abs_error"], m["seconds"], m.get("samples"), m.get("oracle_queries")] for m in run["metrics"]]))
        sections.append('<p class="small">Currency units follow portfolio value. Errors are relative to the analytic fitted distribution; historical simulation has different empirical assumptions. Times include each estimator’s scenario generation.</p>')
        for metric in run["metrics"]:
            csv_rows.append({"run_id": run["run_id"], "data_sha256": run["data_sha256"], "config_sha256": run["config_sha256"],
                             **{key: value for key, value in metric.items() if not isinstance(value, (dict, list))}})
        sections.append('<h3>Chronological backtest</h3>')
        sections.append(table(["Method", "Observations", "Exceptions", "Exception rate", "Expected rate", "Exact binomial p", "Kupiec p", "Independence p"],
                              [[b["method"], b["coverage"]["observations"], b["coverage"]["exceptions"], b["coverage"]["exception_rate"], b["coverage"]["expected_exception_rate"], b["coverage"]["exact_binomial_pvalue"], b["coverage"]["kupiec_pvalue"], b["coverage"]["independence_pvalue"]] for b in run["backtests"]]))
        sections.append('<h3>Stress shocks</h3>')
        sections.append(table(["Scenario", "Loss", "Loss / portfolio value"], [[s["name"], s["loss"], s["loss_fraction"]] for s in run["stress"]]))
        sections.append('<h3>Pipeline timing</h3>' + table(["Stage", "Seconds"], [[k,v] for k,v in run["timings"].items() if k.endswith("seconds")]))
        q = run["quantum"]
        if q:
            sections.append('<h3>Quantum error decomposition and resources</h3>')
            sections.append(table(["Metric", "Value"], [[k, q[k]] for k in ["clipped_tail_mass", "discretization_var_error", "discretization_cvar_error", "estimation_var_error", "estimation_cvar_error", "total_shots", "total_oracle_queries", "ambiguous_cdf_steps"]]))
            sections.append(table(["Oracle", "Qubits", "Shots", "Queries", "Max depth", "CX gates", "Build (s)", "Compile (s)", "Simulate (s)", "Decode (s)"],
                                  [[r["kind"], r["resources"]["qubits"], r["resources"]["shots"], r["resources"]["oracle_queries"], r["resources"]["max_circuit_depth"], r["resources"]["max_circuit_cx"], r["timings"]["oracle_build_seconds"], r["timings"]["compile_seconds"], r["timings"]["simulate_and_sample_seconds"], r["timings"]["decode_seconds"]] for r in q["queries"]]))
            sections.append(f'<p class="small">{escape(q["note"])}</p>')
        sections.append('</section>')
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if benchmark is not None and Path(benchmark).exists():
        bench = json.loads(Path(benchmark).read_text())
        matching = any(r["data_sha256"] == bench["data_sha256"] and r["config_sha256"] == bench["config_sha256"] for r in records)
        sections.append('<section><h2>Matched-budget tail estimation</h2>')
        sections.append(f'<p>{escape(bench["protocol"])}</p><p>Benchmark matches a displayed run: {matching}. Seeds: {escape(str(bench["seeds"]))}.</p>')
        plot = scaling_svg(bench["summary"])
        sections.append(plot)
        output.with_name("scaling.svg").write_text(plot)
        sections.append(table(["Method", "Queries", "Success / failed", "Mean absolute error", "RMSE", "Interval coverage", "Mean total (s)", "Max depth"],
                              [[r["method"], r.get("oracle_queries"), f'{r["successful"]} / {r["failed"]}', r.get("mean_absolute_error"), r.get("rmse"), r.get("interval_coverage"), r.get("mean_seconds"), r.get("max_circuit_depth")] for r in bench["summary"]]))
        sections.append(f'<p class="small">{escape(bench["intervals"])}</p><p>Continuous tail probability: {bench["continuous_tail"]:.6g}; discrete target: {bench["exact_discrete_tail"]:.6g}. Quantization error is separate from sampling error.</p></section>')
    style = 'body{font:15px/1.55 system-ui;background:#101718;color:#edf3ed;margin:0}main{max-width:1220px;margin:auto;padding:36px 24px}h1{font-weight:500;font-size:30px}h2{font-size:20px;font-weight:550}h3{font-size:15px;margin-top:26px}p{color:#a9b9b6}section{margin:30px 0;padding:24px;background:#182123;border:1px solid #344244}.small{font-size:12px;overflow-wrap:anywhere}.scroll{overflow-x:auto}table{width:100%;border-collapse:collapse;font-size:12px}td,th{padding:10px;text-align:right;border-bottom:1px solid #344244;font-variant-numeric:tabular-nums}td:first-child,th:first-child{text-align:left}th{color:#b9e197;font-weight:500}svg{max-width:840px;width:100%;display:block;margin:24px auto}a{color:#b9e197}'
    html = '<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Risk Engine · Validation report</title><style>'+style+'</style></head><body><main><h1>Quantum Risk Engine / Validation report</h1><p>One-day synthetic portfolio risk · reproducible CPU experiments · noiseless quantum simulation</p><p>No production or quantum advantage claim. Timings are local measurements; query counts are algorithmic resources. Independent models and datasets are shown separately.</p>'+''.join(sections)+'</main></body></html>'
    output.write_text(html)
    with output.with_suffix(".csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted(set().union(*(row.keys() for row in csv_rows))))
        writer.writeheader()
        writer.writerows(csv_rows)
    return output


def main():
    parser = argparse.ArgumentParser(description="Render stored runs as an offline HTML and CSV validation report")
    parser.add_argument("--runs", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--benchmark", default="reports/benchmark.json")
    args = parser.parse_args()
    print(build_report(args.runs, args.output, args.benchmark))


if __name__ == "__main__":
    main()
