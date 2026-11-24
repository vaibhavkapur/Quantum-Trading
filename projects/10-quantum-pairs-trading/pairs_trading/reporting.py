from __future__ import annotations

import html
import json
from pathlib import Path

import numpy as np
import pandas as pd


def bootstrap_interval(values: list[float]) -> tuple[float, float]:
    values = np.asarray(values, float)
    rng = np.random.default_rng(0)
    means = rng.choice(values, size=(2000, len(values)), replace=True).mean(axis=1)
    return tuple(float(x) for x in np.quantile(means, [0.025, 0.975]))


def table(frame: pd.DataFrame, columns: dict[str, str], percent: set[str] | None = None) -> str:
    percent = percent or set()
    header = "".join(f"<th scope='col'>{html.escape(label)}</th>" for label in columns.values())
    rows = []
    for _, row in frame.iterrows():
        cells = []
        for column in columns:
            value = row.get(column)
            if value is None or (isinstance(value, float) and np.isnan(value)):
                formatted = "n/a"
            elif column in percent:
                formatted = f"{float(value):.2%}"
            elif isinstance(value, (float, np.floating)):
                formatted = f"{value:.4f}"
            else:
                formatted = str(value)
            cells.append(f"<td>{html.escape(formatted)}</td>")
        rows.append("<tr>" + "".join(cells) + "</tr>")
    return f"<div class='scroll'><table><thead><tr>{header}</tr></thead><tbody>{''.join(rows)}</tbody></table></div>"


def equity_chart(series: dict[str, pd.DataFrame]) -> str:
    if not series:
        return ""
    colors = ["#2463a0", "#b06715", "#238063", "#925194", "#b34043"]
    values = {name: np.concatenate(([1.0], np.cumprod(1 + frame["return"].to_numpy()))) * 100 for name, frame in series.items()}
    low, high = min(x.min() for x in values.values()), max(x.max() for x in values.values())
    pad = max((high - low) * 0.15, 0.2)
    low, high = low - pad, high + pad
    maximum = max(len(x) for x in values.values()) - 1
    parts = ["<svg viewBox='0 0 1000 340' role='img' aria-label='Out-of-sample equity indexed to 100 by test session'>"]
    for level in np.linspace(low, high, 5):
        y = 280 - (level - low) / (high - low) * 240
        parts.append(f"<line x1='70' y1='{y:.1f}' x2='965' y2='{y:.1f}' stroke='#dee3e8'/><text x='60' y='{y+4:.1f}' text-anchor='end'>{level:.1f}</text>")
    legend = []
    for i, (name, curve) in enumerate(values.items()):
        color = colors[i % len(colors)]
        points = " ".join(f"{70 + t/max(1,maximum)*895:.2f},{280-(value-low)/(high-low)*240:.2f}" for t, value in enumerate(curve))
        parts.append(f"<polyline points='{points}' fill='none' stroke='{color}' stroke-width='2'/>")
        legend.append(f"<span style='color:{color}'>{html.escape(name)}</span>")
    parts.append(f"<text x='70' y='302'>0</text><text x='965' y='302' text-anchor='end'>{maximum}</text><text x='515' y='330' text-anchor='middle'>Out-of-sample trading session</text><text x='18' y='170' transform='rotate(-90 18 170)' text-anchor='middle'>Equity (initial value = 100)</text></svg>")
    return "".join(parts) + "<div class='legend'>" + " · ".join(legend) + "</div>"


def generate_report(runs: Path, output: Path) -> Path:
    files = sorted(runs.rglob("run.json"))
    if not files:
        raise ValueError(f"no run.json artifacts in {runs}")
    latest, manifests, failures = {}, [], []
    for path in files:
        manifest = json.loads((path.parent / "manifest.json").read_text())
        payload = json.loads(path.read_text())
        if manifest["status"] not in ("complete", "completed_with_failures"):
            continue
        manifests.append(manifest)
        for failure in payload["failures"]:
            failures.append({"run_id": manifest["run_id"], **failure})
        for solver in manifest["requested_solvers"]:
            for seed in manifest["seeds"]:
                key = (manifest["comparison_key"], solver, seed)
                stamp = (manifest["created_at_utc"], manifest["run_id"])
                if key not in latest or stamp > latest[key][0]:
                    latest[key] = (stamp, path.parent, manifest, payload)
    if not latest:
        raise ValueError("no finished runs to report")
    output.parent.mkdir(parents=True, exist_ok=True)
    benchmark_rows, fold_rows, sensitivity_rows = [], [], []
    sections = []
    for group in sorted({key[0] for key in latest}):
        seed_rows, records, curves, curve_seeds, group_sensitivity = [], [], {}, {}, []
        group_manifests = {}
        for (key, solver, seed), (_, directory, manifest, payload) in latest.items():
            if key != group:
                continue
            group_manifests[manifest["run_id"]] = manifest
            matching = [s for s in payload["summaries"] if s["solver"] == solver and s["seed"] == seed]
            if not matching:
                continue  # incomplete capital paths are listed as failures, never scored
            summary = dict(matching[0])
            summary["solver"] = solver + (" (fallback)" if summary["fallback_folds"] else "")
            label = summary["solver"]
            seed_rows.append(summary)
            selected = [r for r in payload["records"] if r["solver"] == solver and r["seed"] == seed and r["status"] == "ok"]
            for record in selected:
                row = {"comparison_key": group, "run_id": manifest["run_id"], "solver": label, "seed": seed, "fold": record["fold"],
                       "objective": record["selection"]["objective"], "optimality_gap": record["optimality_gap"], **record["metrics"],
                       **record["timings"], "optimizer_evaluations": record["selection"]["evaluations"],
                       "shots": record["selection"]["details"].get("shots", 0),
                       "decode_objective_evaluations": record["selection"]["details"].get("decode_objective_evaluations", 0),
                       "repair_objective_evaluations": record["selection"]["details"].get("repair_objective_evaluations", 0),
                       **{f"solver_{k}": v for k, v in record["selection"]["timings"].items()}}
                records.append(row)
                fold_rows.append(row)
            daily = pd.read_csv(directory / "daily.csv")
            daily = daily[(daily["solver"] == solver) & (daily["seed"] == seed)].sort_values("timestamp")
            if label not in curve_seeds or seed < curve_seeds[label]:
                curves[label], curve_seeds[label] = daily, seed
            sensitivity = pd.read_csv(directory / "sensitivity.csv")
            sensitivity = sensitivity[(sensitivity["solver"] == solver) & (sensitivity["seed"] == seed)].copy()
            sensitivity["solver"], sensitivity["comparison_key"] = label, group
            group_sensitivity.extend(sensitivity.to_dict("records"))
            sensitivity_rows.extend(sensitivity.to_dict("records"))
        manifest = next(iter(group_manifests.values()))
        sections.append(f"<section><h2>{html.escape(manifest['experiment'])}</h2><p>Comparison {group[:12]} · dataset {manifest['dataset']['dataset_id'][:12]} · {len(manifest['splits'])} folds · seeds {html.escape(str(manifest['seeds']))}</p>")
        if not seed_rows:
            sections.append("<p>All paths in this comparison failed. See failure records below.</p></section>")
            continue
        summary_frame = pd.DataFrame(seed_rows)
        group_benchmarks = []
        for solver, frame in summary_frame.groupby("solver", sort=True):
            lower, upper = bootstrap_interval(frame["sharpe"].tolist())
            row = {"comparison_key": group, "solver": solver, "seeds": len(frame), "sharpe_ci_low": lower, "sharpe_ci_high": upper}
            for column in ("sharpe", "total_return", "max_drawdown", "win_rate", "mean_reversion_success", "trades", "turnover", "max_asset_overlap", "mean_objective_gap", "selection_seconds", "pipeline_seconds", "transaction_costs", "financing_costs"):
                row[column] = float(frame[column].mean()) if frame[column].notna().any() else None
            group_benchmarks.append(row)
            benchmark_rows.append(row)
        sections.append("<h3>Out-of-sample performance, mean across complete seeds</h3>" + table(pd.DataFrame(group_benchmarks),
                        {"solver": "Selector", "seeds": "Seeds", "total_return": "Net return", "sharpe": "Sharpe", "sharpe_ci_low": "Sharpe CI low", "sharpe_ci_high": "Sharpe CI high", "max_drawdown": "Drawdown", "win_rate": "Win rate", "mean_reversion_success": "Reversion success", "trades": "Trades", "turnover": "Turnover / initial NAV", "max_asset_overlap": "Max asset overlap"},
                        {"total_return", "max_drawdown", "win_rate", "mean_reversion_success"}))
        sections.append("<p class='note'>95% percentile bootstrap intervals reflect optimizer seed variation on one synthetic market path. Identical deterministic runs produce zero-width intervals. These are not confidence intervals for future investment returns.</p>")
        sections.append("<h3>Equity after all modeled costs</h3>" + equity_chart(curves))
        first = next(iter(curves.values()))
        sections.append(f"<p class='note'>Source: immutable synthetic data, {html.escape(first.iloc[0]['timestamp'][:10])} through {html.escape(first.iloc[-1]['timestamp'][:10])}. Each line uses that selector's lowest predeclared completed seed ({html.escape(str(curve_seeds))}). Folds compound capital and close positions at each boundary.</p>")
        sections.append("<h3>Selection quality and full pipeline time</h3>" + table(pd.DataFrame(group_benchmarks),
                        {"solver": "Selector", "mean_objective_gap": "Gap to exact optimum (lower is better)", "selection_seconds": "Selection seconds / seed", "pipeline_seconds": "Full fold work seconds / seed", "transaction_costs": "Transaction costs (USD)", "financing_costs": "Carrying costs (USD)"}))
        timings = pd.DataFrame(records).groupby("solver").mean(numeric_only=True).reset_index()
        sections.append("<h3>Mean timing per fold, seconds</h3>" + table(timings, {"solver": "Selector", "screening_seconds": "Screening", "problem_build_seconds": "Problem build", "selection_seconds": "Selection total", "verification_seconds": "Exact verifier", "validation_seconds": "Validation", "backtest_seconds": "Test", "sensitivity_seconds": "Sensitivity", "total_seconds": "Total"}))
        quantum_rows = timings[timings["solver"].str.startswith("quantum-sim")]
        if not quantum_rows.empty:
            sections.append("<h3>Quantum simulation accounting per fold</h3>" + table(quantum_rows, {"solver": "Selector", "solver_encoding_seconds": "Encoding (s)", "solver_state_preparation_seconds": "Energy preparation (s)", "solver_compilation_seconds": "Compilation (s)", "solver_optimizer_seconds": "Optimizer incl. simulation (s)", "solver_sampling_seconds": "Sampling (s)", "solver_repair_decode_seconds": "Repair / decode (s)", "optimizer_evaluations": "Evaluations", "shots": "Final shots"}))
        loads = sum(m["data_loading_seconds"] for m in group_manifests.values())
        wall = sum(m["end_to_end_seconds"] for m in group_manifests.values())
        sections.append(f"<p class='note'>Recorded contributing invocation wall time: {wall:.3f}s, including {loads:.3f}s data loading. Fold totals include screening, selection, exact verification, validation, test and sensitivity. Manifest end-to-end time also includes artifact serialization; Python startup/imports and report generation are outside this timer. Raw manifests retain environment versions and source hashes.</p>")
        sensitivity = pd.DataFrame(group_sensitivity).groupby(["solver", "scenario"])[["total_return", "sharpe", "max_drawdown", "transaction_costs", "financing_costs"]].mean().reset_index()
        sections.append("<h3>Test-window sensitivity, mean per fold and seed</h3><p>Descriptive reruns only; these results never select parameters. Each scenario begins a fold at the base strategy's capital.</p>" + table(sensitivity, {"solver": "Selector", "scenario": "Scenario", "total_return": "Mean fold return", "sharpe": "Mean fold Sharpe", "max_drawdown": "Mean fold drawdown", "transaction_costs": "Transaction costs (USD)", "financing_costs": "Carrying costs (USD)"}, {"total_return", "max_drawdown"}))
        sections.append("</section>")
    if failures:
        sections.append("<section><h2>Failed runs, including superseded attempts</h2>" + table(pd.DataFrame(failures), {"run_id": "Run", "solver": "Selector", "seed": "Seed", "fold": "Fold", "error_type": "Error type", "error": "Error"}) + "</section>")
    pd.DataFrame(benchmark_rows).to_csv(output.with_name(output.stem + "-benchmark.csv"), index=False)
    pd.DataFrame(fold_rows).to_csv(output.with_name(output.stem + "-folds.csv"), index=False)
    pd.DataFrame(sensitivity_rows).to_csv(output.with_name(output.stem + "-sensitivity.csv"), index=False)
    document = """<!doctype html><html lang='en'><meta charset='utf-8'><meta name='viewport' content='width=device-width, initial-scale=1'>
<title>Quantum pairs trading — benchmark</title><style>
:root{font-family:system-ui,sans-serif;color:#1e2934;background:#f7f8fa}body{max-width:1400px;margin:auto;padding:36px 24px}h1{font-size:30px;margin-bottom:8px}h2{font-size:24px}h3{font-size:18px;margin-top:30px}p{line-height:1.55;max-width:1100px}section{background:white;border:1px solid #dde2e7;padding:24px;margin:28px 0}.note{font-size:13px;color:#52616e}.scroll{overflow-x:auto}table{border-collapse:collapse;width:100%;font-size:13px}th,td{padding:11px 10px;text-align:right;border-bottom:1px solid #e3e7eb;font-variant-numeric:tabular-nums}th{background:#edf2f6}th:first-child,td:first-child{text-align:left}svg{width:100%;max-height:400px;font-size:12px;fill:#52616e}.legend{font-size:14px;text-align:center}.badge{display:inline-block;padding:5px 10px;background:#e7eef5;font-size:13px}a{color:#2463a0}
</style><h1>Quantum-optimized pairs trading</h1><span class='badge'>Synthetic research · CPU simulation · no QPU claims</span>
<p>Train-only Engle–Granger screening, identical constrained selection objectives, validation-only entry tuning, and non-overlapping walk-forward tests. Prices, source hashes, budget settings, seeds and failures are recorded for reproducibility.</p>
<p>QAOA uses an exact noiseless expectation during optimization and finite shots for final samples. It is paired with disclosed classical feasibility repair. This experiment does not establish quantum advantage or live profitability.</p>
<p class='note'>Repeated invocations are deduplicated by comparison key, solver and seed, keeping the newest attempt. Different data, code or configurations are displayed in separate comparisons. Failed paths and explicit classical fallbacks are never silently pooled with successful quantum paths.</p>"""
    links = " · ".join(f"<a href='{html.escape(output.stem)}-{suffix}.csv'>{label}</a>" for suffix, label in (("benchmark", "Benchmark CSV"), ("folds", "Raw fold metrics and timings"), ("sensitivity", "Sensitivity CSV")))
    output.write_text(document + "<nav>" + links + "</nav>" + "".join(sections) + "<footer><p>Limitations: designed synthetic mean reversion; fixed universe; uncorrected pairwise screening p-values; no dividends, corporate actions, locate restrictions, liquidity or market impact; close-only stops; forced boundary liquidation; small local statevectors. Cointegration can break. The simulator is not a live trading system.</p></footer></html>")
    return output
