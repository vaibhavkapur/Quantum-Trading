from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import html
import json
import numpy as np
from scipy.stats import t
from .experiment import write_csv, write_json


def seed_interval(rows, extractor):
    """Average folds within each seed before forming a t interval across seeds."""
    seeds = defaultdict(list)
    for row in rows:
        seeds[row["seed"]].append(extractor(row))
    values = np.array([np.mean(value) for value in seeds.values()])
    mean = float(values.mean())
    if len(values) < 2:
        return mean, None, None
    half = float(t.ppf(0.975, len(values)-1) * values.std(ddof=1) / np.sqrt(len(values)))
    return mean, mean - half, mean + half


def summarize(manifests):
    groups, failures = defaultdict(list), []
    seen = set()
    for manifest in manifests:
        for row in manifest["results"]:
            # A repeated run must not masquerade as an independent random seed.
            identity = (manifest["protocol_id"], row["id"])
            if identity in seen:
                raise ValueError("duplicate protocol/model/seed/fold result; pass a directory containing one run per solver")
            seen.add(identity)
            if row["status"] != "ok":
                failures.append({"run": manifest["run_id"], **row})
                continue
            key = (manifest["protocol_id"], row["model"], row["qubits"], row["encoding"], row["shots"])
            groups[key].append(row)
    summaries = []
    for (protocol, model, qubits, encoding, shots), rows in sorted(groups.items(), key=lambda item: str(item[0])):
        summary = {"protocol": protocol, "model": model, "qubits": qubits, "encoding": encoding,
                   "shots": shots, "successful_trials": len(rows)}
        source = next(m for m in manifests if m["protocol_id"] == protocol)
        summary["expected_trials"] = len(source["config"]["seeds"]) * source["config"]["folds"]
        summary["complete"] = summary["successful_trials"] == summary["expected_trials"]
        for name, extract in {
            "balanced_accuracy": lambda r: r["metrics"]["balanced_accuracy"],
            "macro_f1": lambda r: r["metrics"]["macro_f1"],
            "brier_score": lambda r: r["metrics"]["brier_score"],
            "ece": lambda r: r["metrics"]["ece"],
            "net_bps": lambda r: r["trading"]["net_bps"],
            "p99_ms": lambda r: r["latency"]["p99_ms"],
            "throughput_per_second": lambda r: r["latency"]["throughput_per_second"],
            "fit_seconds": lambda r: r["timing"]["fit_seconds"],
            "trial_seconds": lambda r: r["timing"]["trial_total_seconds"],
            "model_bytes": lambda r: r["model_bytes"],
            "measurement_shots": lambda r: r["resources_including_profile"]["measurement_shots"],
        }.items():
            mean, low, high = seed_interval(rows, extract)
            summary[name], summary[f"{name}_ci_low"], summary[f"{name}_ci_high"] = mean, low, high
        summaries.append(summary)
    return summaries, failures


def write_report(runs: str | Path, output: str | Path):
    runs, output = Path(runs), Path(output)
    paths = sorted(runs.rglob("manifest.json"))
    manifests = [json.loads(path.read_text()) for path in paths]
    manifests = [m for m in manifests if "results" in m and "protocol_id" in m]
    if not manifests:
        raise ValueError("no experiment manifests found")
    summaries, failures = summarize(manifests)
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output.with_suffix(".json"), {"summary": summaries, "failures": failures,
                                            "runs": [str(p.resolve()) for p in paths]})
    if summaries:
        write_csv(output.with_suffix(".csv"), summaries)
    esc = lambda value: html.escape(str(value))
    def interval(row):
        if row["balanced_accuracy_ci_low"] is None:
            return "single seed"
        return f'{row["balanced_accuracy_ci_low"]:.3f}–{row["balanced_accuracy_ci_high"]:.3f}'
    sections = []
    for protocol in sorted({m["protocol_id"] for m in manifests}):
        selected = [r for r in summaries if r["protocol"] == protocol]
        rows = "".join(
            f'<tr><td>{esc(r["model"])}<small>{esc(r["encoding"])}, {r["qubits"]} inputs/qubits, '
            f'{"exact" if r["shots"] is None else str(r["shots"])+" shots"}</small></td>'
            f'<td>{r["successful_trials"]}/{r["expected_trials"]}</td>'
            f'<td>{r["balanced_accuracy"]:.3f}<small>{interval(r)}</small></td>'
            f'<td>{r["macro_f1"]:.3f}</td><td>{r["brier_score"]:.3f}</td><td>{r["ece"]:.3f}</td>'
            f'<td>{r["net_bps"]:.2f}</td><td>{r["p99_ms"]:.3f}</td><td>{r["throughput_per_second"]:.0f}</td>'
            f'<td>{r["fit_seconds"]:.3f}</td><td>{r["trial_seconds"]:.3f}</td>'
            f'<td>{r["model_bytes"]/1024:.1f}</td><td>{r["measurement_shots"]:.0f}</td></tr>' for r in selected)
        relevant = [m for m in manifests if m["protocol_id"] == protocol]
        shared = "".join(f'<li>{esc(m["solver"])}: process {m.get("wall_seconds", 0):.3f}s; '
                         f'bootstrap {m["shared_timing"]["bootstrap_seconds"]:.3f}s; '
                         f'data {m["shared_timing"]["data_loading_seconds"]:.3f}s; '
                         f'features/splits {m["shared_timing"]["features_labels_splits_seconds"]:.3f}s; '
                         f'status {esc(m["status"])}</li>' for m in relevant)
        # Lightweight SVG chart remains self-contained and works without plotting dependencies.
        bars = []
        for i, r in enumerate(selected):
            y = 25 + i * 26
            label = f'{r["model"]} / q{r["qubits"]} / {r["encoding"]} / {r["shots"] or "exact"}'
            color = "#5ba7ff" if r["model"] in {"quantum_kernel", "vqc"} else "#64d5b4"
            bars.append(f'<text x="0" y="{y+12}" fill="#c4cee0" font-size="11">{esc(label)}</text>'
                        f'<rect x="330" y="{y}" width="{r["balanced_accuracy"]*480:.1f}" height="16" fill="{color}"/>'
                        f'<text x="{338+r["balanced_accuracy"]*480:.1f}" y="{y+12}" fill="#eef4ff" font-size="11">{r["balanced_accuracy"]:.3f}</text>')
        chart = f'<svg role="img" aria-label="Balanced accuracy by model" viewBox="0 0 880 {len(selected)*26+40}">{"".join(bars)}</svg>'
        sections.append(f'<section><h2>Protocol {esc(protocol)}</h2><p>Only rows within this protocol share data, source, splits and budgets.</p>'
                        f'{chart}<div class="scroll"><table><thead><tr><th>Model</th><th>Trials</th><th>Balanced accuracy / 95% CI</th>'
                        f'<th>Macro F1</th><th>Brier</th><th>ECE</th><th>Net bps / fold</th><th>p99 ms</th><th>Rows/s</th>'
                        f'<th>Fit s</th><th>Trial s</th><th>KiB</th><th>Shots incl. profile</th></tr></thead><tbody>{rows}</tbody></table></div>'
                        f'<ul>{shared}</ul><details><summary>Configuration and provenance</summary><pre>{esc(json.dumps(relevant[0]["config"], indent=2))}</pre>'
                        f'<pre>{esc(json.dumps(relevant[0]["environment"], indent=2))}</pre><p>Dataset SHA-256: {esc(relevant[0]["dataset"]["sha256"])}</p></details></section>')
    failed_html = "".join(f'<li>{esc(r["run"])} / {esc(r["id"])}: {esc(r["status"])} — {esc(r.get("error", ""))}</li>' for r in failures)
    warning_count = sum(bool(r.get("warnings")) for m in manifests for r in m["results"])
    document = f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Order-book prediction benchmark</title><style>
body{{margin:0;background:#0b1220;color:#eef4ff;font:15px system-ui,sans-serif;line-height:1.6}}main{{max-width:1500px;margin:auto;padding:40px 28px}}
h1{{font-size:36px;line-height:1.2;margin:10px 0}}h2{{font-size:19px}}p,li{{color:#c4cee0}}.tag{{color:#64d5b4;font-size:12px;letter-spacing:2px}}
section{{margin:28px 0;padding:22px;background:#111d30;border:1px solid #23344d;border-radius:12px}}.scroll{{overflow:auto}}
table{{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums;font-size:12px}}td,th{{padding:12px;text-align:right;border-bottom:1px solid #27364c;white-space:nowrap}}td:first-child,th:first-child{{text-align:left}}
th{{color:#92a8c6}}small{{display:block;color:#92a8c6;font-size:10px}}pre{{overflow:auto;color:#b1c4df}}svg{{max-width:1000px;width:100%}}a{{color:#64d5b4}}
</style><main><div class="tag">REPRODUCIBLE CPU RESEARCH</div><h1>Quantum ML Order Book Predictor</h1>
<p>Classical and simulated quantum models on identical chronological L2 prediction tasks. Synthetic data; no live-profitability or quantum-advantage claim.</p>
<p>{len(manifests)} runs · {sum(len(m["results"]) for m in manifests)} trials · {len(failures)} failed / over budget · {warning_count} trials with warnings (see raw manifests)</p>
{"".join(sections)}<section><h2>Measurement notes</h2><ul>
<li>Separate training, probability calibration, threshold tuning and test windows. Label endpoints crossing a boundary are purged; embargo removes initial rows from later windows. PCA is fitted only on the shared training sample.</li>
<li>Means average folds within seed. 95% Student-t intervals describe seed variability on this one dataset; they do not quantify market uncertainty. A single seed has no interval. Incomplete groups must not be ranked as complete benchmarks.</li>
<li>p99 measures snapshot conversion, validation, feature generation, PCA, encoding, model inference, calibration and decision. Small samples make tail estimates unstable. Rows/s is serial in-process throughput, not network service capacity.</li>
<li>Net bps sums independent trade returns with one unit, observed bid/ask spreads, delayed fills, fees and adverse slippage. It is not a compounded portfolio return. Test trades do not overlap. Thresholds may select abstention.</li>
<li>Quantum runs use a noiseless local simulator. Finite-shot kernels sample exact overlaps with a binomial distribution and repair the training Gram matrix; VQC shots use PennyLane measurements. Cached simulator states cannot be assumed on hardware.</li>
<li>Trial time includes preprocessing, model initialization, training, calibration, threshold selection, test evaluation, streaming profiling and artifact writes. Shared process/data costs are shown separately. Native training calls can exceed the wall budget before being rejected.</li>
</ul><p><a href="{esc(output.with_suffix('.csv').name)}">Summary CSV</a> · <a href="{esc(output.with_suffix('.json').name)}">Summary JSON and failures</a></p></section>
<section><h2>Failures</h2><ul>{failed_html or '<li>None recorded.</li>'}</ul></section></main></html>'''
    output.write_text(document)
    return output
