"""Turn stored runs into a self-contained HTML comparison plus CSV tables.

    python -m qmc_options.report --runs runs/ --output reports/comparison.html
"""
from __future__ import annotations

import argparse
import base64
import csv
import html
import io
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
SERIES = {"plain": "#2a78d6", "qmc": "#eb6834", "mlae": "#1baf7a", "iqae": "#eda100"}
LABELS = {"plain": "Classical MC (plain)", "qmc": "Classical randomised QMC",
          "mlae": "QAE: maximum likelihood", "iqae": "QAE: iterative",
          "antithetic": "Classical MC (antithetic)", "control_variate": "Classical MC (control variate)",
          "direct_sampling": "Circuit sampling, no Grover", "dataset": "Stored dataset (plain MC)"}
BLUE_RAMP = ["#86b6ef", "#2a78d6", "#104281"]


# --------------------------------------------------------------------------- helpers

def load_runs(runs_dir: Path) -> Dict[str, Dict[str, Any]]:
    """Latest run per solver found in ``runs_dir``."""
    runs: Dict[str, Dict[str, Any]] = {}
    for path in sorted(runs_dir.glob("*.json")):
        try:
            data = json.loads(path.read_text())
            solver = data["manifest"]["solver"]
        except (ValueError, KeyError, TypeError):
            continue                      # not one of our run files
        current = runs.get(solver)
        if current is None or data["manifest"]["created_utc"] >= current["manifest"]["created_utc"]:
            data["_file"] = path.name
            runs[solver] = data
    return runs


def power_fit(x: Sequence[float], y: Sequence[float]):
    """Least-squares fit of y = c * x**slope on log axes."""
    slope, intercept = np.polyfit(np.log(x), np.log(y), 1)
    return float(np.exp(intercept)), float(slope)


def fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "–"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, (int, np.integer)):
        return f"{int(value):,}"
    if isinstance(value, float):
        if value != value:
            return "–"
        if value != 0 and (abs(value) >= 1e6 or abs(value) < 1e-3):
            return f"{value:.2e}"
        if abs(value) >= 100:
            return f"{value:,.0f}"
        return f"{value:,.{digits}f}"
    return html.escape(str(value))


def duration(seconds: float) -> str:
    for limit, unit, scale in ((1e-3, "µs", 1e6), (1.0, "ms", 1e3), (120.0, "s", 1.0),
                               (7200.0, "min", 1 / 60), (172800.0, "h", 1 / 3600)):
        if seconds < limit:
            return f"{seconds * scale:,.1f} {unit}"
    return f"{seconds / 86400:,.1f} days"


def table(columns: Sequence[str], rows: Iterable[Sequence[Any]]) -> str:
    head = "".join(f"<th>{html.escape(c)}</th>" for c in columns)
    body = "".join("<tr>" + "".join(f"<td>{c if isinstance(c, str) else fmt(c)}</td>" for c in row) + "</tr>"
                   for row in rows)
    return f"<div class='scroll'><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>"


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    flat = [{k: (json.dumps(v) if isinstance(v, (dict, list)) else v) for k, v in row.items()} for row in rows]
    columns = list(dict.fromkeys(key for row in flat for key in row))
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(flat)


def _figure(fig, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=160, facecolor=SURFACE)
    plt.close(fig)
    path.write_bytes(buffer.getvalue())
    return base64.b64encode(buffer.getvalue()).decode()


def _style(ax, xlabel: str, ylabel: str) -> None:
    ax.set_facecolor(SURFACE)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(True, which="major", color=GRID, linewidth=0.8)
    ax.grid(False, which="minor")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED, which="both", length=0, labelsize=9)
    ax.set_xlabel(xlabel, color=MUTED, fontsize=9.5)
    ax.set_ylabel(ylabel, color=MUTED, fontsize=9.5)


# --------------------------------------------------------------------------- aggregation

def classical_summary(run: Dict[str, Any]) -> List[List[Any]]:
    groups = defaultdict(list)
    for row in run["prices"]:
        groups[(row["instrument"], row["method"])].append(row)
    out = []
    for (instrument, method), rows in groups.items():
        ref = run["references"][instrument]
        out.append([instrument, LABELS.get(method, method), float(np.mean([r["price"] for r in rows])),
                    ref["price"], float(np.mean([r["abs_error"] for r in rows])),
                    float(np.mean([r["stderr"] for r in rows])),
                    f"{sum(r['covered'] for r in rows)}/{len(rows)}", rows[0]["n_paths"],
                    duration(float(np.mean([r["wall_time_s"] for r in rows])))])
    return out


def quantum_summary(run: Dict[str, Any]) -> List[List[Any]]:
    groups = defaultdict(list)
    for row in run["prices"]:
        groups[(row["instrument"], row["estimator"])].append(row)
    out = []
    for (instrument, estimator), rows in groups.items():
        out.append([instrument, LABELS.get(estimator, estimator), float(np.mean([r["price"] for r in rows])),
                    rows[0]["discretized_exact_price"], float(np.mean([r["abs_error"] for r in rows])),
                    float(np.mean([r["abs_error_vs_reference"] for r in rows])),
                    f"{sum(r['covered'] for r in rows)}/{len(rows)}",
                    int(np.mean([r["oracle_queries"] for r in rows])),
                    int(max(r["max_grover_power"] for r in rows)),
                    int(max(r["resources"]["max_circuit_cx"] for r in rows)),
                    duration(float(np.mean([r["estimation_time_s"] for r in rows])))])
    return out


def scaling_series(classical: Optional[Dict[str, Any]], quantum: Optional[Dict[str, Any]]):
    """{key: (x, rmse)} for the estimators shown in the convergence figure."""
    series = {}
    if classical:
        for method in ("plain", "qmc"):
            rows = [r for r in classical["error_curve"]["rows"] if r["method"] == method]
            if rows:
                series[method] = ([r["n_paths"] for r in rows], [r["rmse"] for r in rows])
    if quantum:
        for estimator in ("mlae", "iqae"):
            rows = [r for r in quantum["scaling"]["rows"] if r["estimator"] == estimator]
            if rows:
                series[estimator] = ([r["mean_oracle_queries"] for r in rows], [r["rmse"] for r in rows])
    return series


def cost_model(classical: Dict[str, Any], quantum: Dict[str, Any], model: Dict[str, Any]) -> List[List[Any]]:
    """Extrapolate the fitted convergence laws to target errors and convert
    quantum query counts to device time under the stated gate-time assumptions."""
    series = scaling_series(classical, quantum)
    name = quantum["scaling"]["instrument"]
    inst = next(i for i in quantum["instruments"] if i["instrument"] == name)
    q_res = inst["grover_operator"]
    iqae_rows = [r for r in quantum["scaling"]["rows"] if r["estimator"] == "iqae"]
    power_c, power_slope = power_fit([r["rmse"] for r in iqae_rows],
                                     [max(r["mean_max_grover_power"], 0.5) for r in iqae_rows])
    curve = {r["method"]: r for r in classical["error_curve"]["rows"]}      # last (largest) row per method
    rows = []
    for eps in model["target_price_errors"]:
        for key in ("plain", "qmc", "iqae"):
            c, slope = power_fit(*series[key])
            need = (eps / c) ** (1 / slope)
            if key == "iqae":
                cx_total = need * q_res["cx"] / 2                 # Q holds A and its inverse: 2 queries
                k_max = power_c * eps**power_slope
                rows.append([eps, LABELS[key], need, "–",
                             int(inst["a_operator"]["cx"] + k_max * q_res["cx"]),
                             duration(need * q_res["cx_depth"] / 2 * model["physical_two_qubit_gate_s"]),
                             duration(need * q_res["cx_depth"] / 2 * model["logical_cx_s"]),
                             need * q_res["rz"] / 2 * model["t_gates_per_rotation"]])
            else:
                per_path = curve[key]["mean_wall_time_s"] / curve[key]["n_paths"]
                rows.append([eps, LABELS[key], need, duration(need * per_path), "–", "–", "–", "–"])
    return rows


# --------------------------------------------------------------------------- figures

def convergence_figure(series, bias: Optional[float], path: Path) -> str:
    fig, ax = plt.subplots(figsize=(7.6, 4.4), facecolor=SURFACE)
    _style(ax, "Payoff evaluations (classical) or oracle queries (quantum)", "RMSE of price")
    for key, (x, y) in series.items():
        ax.plot(x, y, color=SERIES[key], linewidth=2, marker="o", markersize=4.5, label=LABELS[key])
        ax.annotate(LABELS[key], (x[-1], y[-1]), xytext=(7, 0), textcoords="offset points",
                    color=INK, fontsize=8.5, va="center", zorder=5,
                    bbox={"facecolor": SURFACE, "edgecolor": "none", "pad": 1.5})
    if bias:
        ax.axhline(abs(bias), color=MUTED, linewidth=1, linestyle=(0, (4, 3)), zorder=1)
        ax.annotate(f"discretisation bias of the quantum model ({abs(bias):.3f})", (ax.get_xlim()[0], abs(bias)),
                    xytext=(4, 4), textcoords="offset points", color=MUTED, fontsize=8.5)
    ax.legend(frameon=False, fontsize=8.5, labelcolor=INK, loc="lower left")
    ax.set_xlim(right=ax.get_xlim()[1] * 30)       # room for the direct labels
    fig.tight_layout()
    return _figure(fig, path)


def discretization_figure(rows: List[Dict[str, Any]], path: Path) -> str:
    fig, ax = plt.subplots(figsize=(7.6, 3.8), facecolor=SURFACE)
    _style(ax, "Qubits in the price register", "|discretised price − Black–Scholes|")
    ax.set_xscale("linear")
    cuts = sorted({r["truncation"] for r in rows})
    for cut, color in zip(cuts, BLUE_RAMP):
        pts = [r for r in rows if r["truncation"] == cut]
        x, y = [r["qubits"] for r in pts], [abs(r["bias"]) for r in pts]
        ax.plot(x, y, color=color, linewidth=2, marker="o", markersize=4.5, label=f"cut at ±{cut:g}σ")
        ax.annotate(f"±{cut:g}σ", (x[-1], y[-1]), xytext=(7, 0), textcoords="offset points",
                    color=INK, fontsize=8.5, va="center")
    ax.legend(frameon=False, fontsize=8.5, labelcolor=INK, loc="lower left")
    ax.set_xlim(right=max(r["qubits"] for r in rows) + 0.7)
    fig.tight_layout()
    return _figure(fig, path)


# --------------------------------------------------------------------------- report

CSS = """
body{font:15px/1.5 -apple-system,'Segoe UI',Helvetica,Arial,sans-serif;color:#0b0b0b;background:#fcfcfb;
max-width:1080px;margin:0 auto;padding:32px 24px 64px}
h1{font-size:26px;margin:0 0 4px} h2{font-size:19px;margin:40px 0 8px} p{margin:8px 0}
.sub,.note{color:#52514e} .note{font-size:13.5px}
.scroll{overflow-x:auto;margin:12px 0}
table{border-collapse:collapse;font-size:13px;width:100%;font-variant-numeric:tabular-nums}
th,td{padding:6px 10px;text-align:right;border-bottom:1px solid #e4e3df;white-space:nowrap}
th{color:#52514e;font-weight:600} th:nth-child(-n+2),td:nth-child(-n+2){text-align:left}
img{max-width:100%;height:auto;display:block;margin:12px 0}
.box{border:1px solid #e4e3df;border-radius:8px;padding:12px 16px;margin:16px 0;background:#fff}
code{font-size:12.5px}
"""


def build_report(runs: Dict[str, Dict[str, Any]], out_path: Path, resource_model: Dict[str, Any]) -> str:
    classical, quantum = runs.get("classical"), runs.get("quantum-sim")
    figures = out_path.parent / "figures"
    parts: List[str] = [f"<h1>Classical Monte Carlo vs quantum amplitude estimation</h1>"
                        "<p class='sub'>Option pricing under Black–Scholes. Quantum numbers are noiseless "
                        "local simulation; no QPU was used.</p>"]
    missing = [s for s in ("classical", "quantum-sim") if s not in runs]
    if missing:
        parts.append(f"<div class='box'>No stored run for: {', '.join(missing)}. "
                     "Sections that need it are omitted.</div>")

    series = scaling_series(classical, quantum)
    if series:
        bias = None
        if quantum:
            name = quantum["scaling"]["instrument"]
            bias = next(i["discretization_bias"] for i in quantum["instruments"] if i["instrument"] == name)
        image = convergence_figure(series, bias, figures / "convergence.png")
        fits = [[LABELS[k], power_fit(x, y)[1], x[-1], y[-1]] for k, (x, y) in series.items()]
        parts += ["<h2>Convergence at equal evaluation counts</h2>",
                  "<p>Root-mean-square error over independent repeats. Classical error is measured against the "
                  "closed-form price; quantum error against the exact answer of the discretised circuit, so the "
                  "dashed line (the gap between that discretised model and the true price) comes on top of it.</p>",
                  f"<img alt='RMSE against evaluations for four estimators' src='data:image/png;base64,{image}'>",
                  table(["Estimator", "Fitted log–log slope", "Largest budget", "RMSE at largest budget"], fits),
                  "<p class='note'>Slope −0.5 is Monte Carlo scaling; −1 is the ideal amplitude-estimation "
                  "scaling. One oracle query is one run of the state-preparation and payoff circuit, which is "
                  "far more expensive than one classical payoff evaluation.</p>"]

    if classical:
        parts += ["<h2>Classical prices</h2>",
                  f"<p>{len(classical['manifest']['seeds'])} predeclared seeds per row; coverage counts how often "
                  "the confidence interval contained the reference.</p>",
                  table(["Instrument", "Method", "Mean price", "Reference", "Mean |error|", "Mean std. error",
                         "CI coverage", "Paths", "Mean time"], classical_summary(classical))]
        refs = [[k, v["price"], v["stderr"], v["kind"]] for k, v in classical["references"].items()]
        parts.append(table(["Instrument", "Reference price", "Std. error of reference", "Source"], refs))
        ds = classical["dataset"]
        if ds["status"] == "ok":
            parts += ["<p>Stored sample dataset (hash verified against its manifest):</p>",
                      table(["Instrument", "Price", "Std. error", "|error|", "Reference inside CI", "Paths"],
                            [[r["instrument"], r["price"], r["stderr"], r["abs_error"], r["covered"], r["n_paths"]]
                             for r in ds["rows"]])]
        else:
            parts.append(f"<p class='note'>Dataset check not run: {html.escape(ds['detail'])}.</p>")
        tp = classical["throughput"]
        rows = [["NumPy, 1 thread", tp["numpy"]["paths_per_s"], duration(tp["numpy"]["wall_time_s"])]]
        if tp["cpp"]["status"] == "ok":
            rows.append([f"C++20, {tp['cpp']['threads']} threads", tp["cpp"]["paths_per_s"],
                         duration(tp["cpp"]["wall_time_s"])])
        parts += [f"<p>Throughput on {tp['n_paths']:,} plain paths ({tp['instrument']}):</p>",
                  table(["Kernel", "Paths per second", "Wall time"], rows)]
        if tp["cpp"]["status"] != "ok":
            parts.append(f"<p class='note'>C++ kernel: {html.escape(tp['cpp']['detail'])}.</p>")
        parts += ["<h2>Classical Greeks (finite differences, common random numbers)</h2>",
                  table(["Instrument", "Greek", "Estimate", "Std. error", "Closed form", "Paths"],
                        [[g["instrument"], g["greek"], g["value"], g["stderr"], g["analytic"], g["n_paths"]]
                         for g in classical["greeks"]])]

    if quantum:
        parts += ["<h2>Quantum-simulated prices</h2>",
                  "<p>Coverage and the first error column are against the exact price of the discretised model; "
                  "the second error column is against the continuous-model reference and includes "
                  "discretisation bias.</p>",
                  table(["Instrument", "Estimator", "Mean price", "Discretised exact", "Mean |error| vs discretised",
                         "Mean |error| vs reference", "CI coverage", "Mean oracle queries", "Max Grover power",
                         "Max CX in one circuit", "Mean simulation time"], quantum_summary(quantum)),
                  "<h2>Circuit resources</h2>",
                  "<p>Compiled to CX, RZ, SX, X with all-to-all connectivity. A is state preparation plus the "
                  "payoff rotation; Q is one Grover iteration.</p>",
                  table(["Instrument", "Qubits", "Paths encoded", "Discretisation bias", "A: CX", "A: depth",
                         "Q: CX", "Q: depth", "Simulation", "Build + compile time"],
                        [[i["instrument"], i["total_qubits"], i["basis_states"], i["discretization_bias"],
                          i["a_operator"]["cx"], i["a_operator"]["depth"], i["grover_operator"]["cx"],
                          i["grover_operator"]["depth"], i["simulation_mode"],
                          duration(sum(i["timings"].values()))] for i in quantum["instruments"]])]
        image = discretization_figure(quantum["discretization"]["rows"], figures / "discretization.png")
        parts += ["<h2>Discretisation bias and loading cost</h2>",
                  f"<img alt='Discretisation bias against qubits for three cut-offs' "
                  f"src='data:image/png;base64,{image}'>",
                  table(["Cut-off (σ)", "Qubits", "Discretised price", "Bias", "State-prep CX", "A: CX", "A: depth"],
                        [[f"±{r['truncation']:g}", r["qubits"], r["discretized_exact_price"], r["bias"],
                          r["state_prep_cx"], r["a_operator_cx"], r["a_operator_depth"]]
                         for r in quantum["discretization"]["rows"]]),
                  "<h2>Estimator detail</h2>",
                  table(["Estimator", "Setting", "Mean oracle queries", "Mean max Grover power", "RMSE",
                         "Mean CI half-width", "CI coverage"],
                        [[LABELS[r["estimator"]], ", ".join(f"{k}={v}" for k, v in r["options"].items()),
                          r["mean_oracle_queries"], f"{r['mean_max_grover_power']:.1f}", r["rmse"],
                          r["mean_ci_half_width"], f"{r['coverage']:.0%}"] for r in quantum["scaling"]["rows"]]),
                  "<h2>Sensitivity to noise (toy model)</h2>",
                  "<p>Maximum-likelihood estimation fitted with the noiseless likelihood while each Grover "
                  "iteration depolarises at the given rate. Largest Grover power: "
                  f"{quantum['noise']['max_grover_power']}. This is a model, not a device.</p>",
                  table(["Noise rate per Grover iteration", "RMSE of price", "Bias", "CI coverage"],
                        [[f"{r['noise_rate_per_grover_iteration']:g}", r["rmse"], r["bias"], f"{r['coverage']:.1%}"]
                         for r in quantum["noise"]["rows"]]),
                  "<h2>Quantum delta (two iterative-QAE prices, central difference)</h2>",
                  table(["Relative bump", "Delta", "CI low", "CI high", "Exact delta of discretised model",
                         "Closed form", "Oracle queries"],
                        [[f"{r['rel_bump']:.0%}", r["delta"], r["ci_low"], r["ci_high"],
                          r["discretized_exact_delta"], r["analytic_delta"], r["oracle_queries"]]
                         for r in quantum["delta"]["rows"]])]

    if classical and quantum and {"plain", "qmc", "iqae"} <= set(series):
        parts += ["<h2>Extrapolated cost at a target error</h2>",
                  "<p>Evaluations needed come from the fitted convergence laws above. Classical time uses the "
                  "measured NumPy cost per path. Quantum columns are <b>assumption-driven estimates</b>: "
                  f"sequential two-qubit layers at {resource_model['physical_two_qubit_gate_s']:g} s (physical, no "
                  f"error correction) or {resource_model['logical_cx_s']:g} s (logical), and "
                  f"{resource_model['t_gates_per_rotation']} T gates per rotation. They ignore discretisation "
                  "bias, routing, readout, repetition overhead and the error-correction footprint. Without error "
                  "correction, a circuit with N CX gates only returns signal if the two-qubit error rate is well "
                  "below 1/N.</p>",
                  table(["Target RMSE", "Estimator", "Evaluations / queries", "Measured-rate CPU time",
                         "CX in deepest circuit", "Device time, physical gates", "Device time, logical gates",
                         "T-count estimate"], cost_model(classical, quantum, resource_model))]

    parts.append("<h2>Provenance</h2>")
    parts.append(table(["Solver", "Run file", "Created (UTC)", "Config SHA-256", "Python", "NumPy", "Qiskit",
                        "Total wall time", "Platform"],
                       [[m["solver"], run["_file"], m["created_utc"], m["config_sha256"][:16], m["python"],
                         m["versions"]["numpy"], m["versions"]["qiskit"], duration(m["total_wall_time_s"]),
                         m["platform"]] for run in runs.values() for m in [run["manifest"]]]))
    parts.append("<p class='note'>Research and educational use only. Not investment advice.</p>")
    return (f"<!doctype html><html lang='en'><head><meta charset='utf-8'><title>QMC options comparison</title>"
            f"<meta name='viewport' content='width=device-width,initial-scale=1'><style>{CSS}</style></head>"
            f"<body>{''.join(parts)}</body></html>")


DEFAULT_RESOURCE_MODEL = {"target_price_errors": [0.1, 0.01, 0.001], "physical_two_qubit_gate_s": 1e-7,
                          "logical_cx_s": 1e-4, "t_gates_per_rotation": 50}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--runs", default="runs")
    parser.add_argument("--output", default="reports/comparison.html")
    parser.add_argument("--config", help="config whose resource_model section should be used")
    args = parser.parse_args(argv)
    runs = load_runs(Path(args.runs))
    if not runs:
        parser.error(f"no run files found in {args.runs}; run qmc_options.run first")
    model = dict(DEFAULT_RESOURCE_MODEL)
    if args.config:
        from .config import load_config
        model.update(load_config(args.config).get("resource_model", {}))
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(build_report(runs, out, model))
    for solver, run in runs.items():
        write_csv(out.parent / f"{solver}_prices.csv", run["prices"])
    if "classical" in runs:
        write_csv(out.parent / "classical_error_curve.csv", runs["classical"]["error_curve"]["rows"])
    if "quantum-sim" in runs:
        write_csv(out.parent / "quantum_scaling.csv", runs["quantum-sim"]["scaling"]["rows"])
    print(f"wrote {out} (solvers: {', '.join(sorted(runs))}) and CSV tables in {out.parent}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
