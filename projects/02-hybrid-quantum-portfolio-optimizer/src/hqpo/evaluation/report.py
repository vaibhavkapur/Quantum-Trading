"""Build the comparison dashboard (self-contained HTML) and the benchmark table (Markdown).

The report only *renders* stored runs; it never re-solves anything.
"""
from __future__ import annotations

import html
import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from .metrics import mean_ci

# Fixed entity -> colour map (colour follows the solver, never its rank).
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e0"
SOLVER_STYLE = {
    "miqp":        ("#2a78d6", "solid", "circle"),
    "qaoa":        ("#eb6834", "solid", "diamond"),
    "greedy":      ("#1baf7a", "solid", "square"),
    "sa":          ("#eda100", "solid", "triangle-up"),
    "qubo-sa":     ("#e87ba4", "solid", "cross"),
    "cvxpy-relax": ("#008300", "solid", "x"),
    "cpsat":       ("#6250d6", "solid", "triangle-down"),
    "random-qubo": ("#e34948", "solid", "star"),
    "qaoa-noisy":  ("#eb6834", "dash", "diamond-open"),   # same family as qaoa; dash + open marker
}
NEUTRAL = "#8a8983"
SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
SOLVER_NOTE = {
    "cvxpy-relax": "classical: CVXPY convex relaxation, rounded to lots",
    "miqp": "classical: exact MIQP (CVXPY + SCIP; CP-SAT fallback)",
    "cpsat": "classical: OR-Tools CP-SAT integer model",
    "greedy": "classical: greedy construction + local search",
    "sa": "classical: simulated annealing on lot vectors",
    "qubo-sa": "simulated annealing on the QUBO (annealer proxy, classical)",
    "random-qubo": "control: uniform random bitstrings, best of N",
    "qaoa": "QAOA, noiseless statevector simulation",
    "qaoa-noisy": "QAOA, sampled under an Aer depolarising noise model",
}


def _style(name):
    return SOLVER_STYLE.get(name, (NEUTRAL, "solid", "circle"))


def _layout(fig: go.Figure, title: str, xt: str, yt: str, height: int = 420, **kw) -> go.Figure:
    fig.update_layout(
        title=dict(text=title, font=dict(size=15, color=INK), x=0, xanchor="left"),
        paper_bgcolor=SURFACE, plot_bgcolor=SURFACE, height=height,
        font=dict(family="-apple-system, 'Segoe UI', Helvetica, Arial, sans-serif", size=12, color=INK2),
        margin=dict(l=64, r=24, t=56, b=56), hoverlabel=dict(bgcolor="#ffffff", font_size=12),
        legend=dict(orientation="h", yanchor="bottom", y=-0.32, x=0, font=dict(color=INK2)), **kw)
    fig.update_xaxes(title_text=xt, gridcolor=GRID, zeroline=False, linecolor=GRID, ticks="outside",
                     tickcolor=GRID)
    fig.update_yaxes(title_text=yt, gridcolor=GRID, zeroline=False, linecolor=GRID)
    return fig


# --------------------------------------------------------------------------- loading
def load_runs(runs_dir: Path) -> dict:
    """Collect every experiment under ``runs_dir`` (``<name>/backtest/...``, ``<name>/benchmark``)."""
    runs_dir = Path(runs_dir)
    out = {}
    for exp in sorted(p for p in runs_dir.iterdir() if p.is_dir()):
        item = {"backtests": [], "benchmark": {}, "manifest": None}
        for summ in sorted(exp.glob("backtest/*/*/summary.json")):
            d = summ.parent
            item["backtests"].append({
                "solver": d.parent.name, "seed": d.name,
                "summary": json.loads(summ.read_text()),
                "daily": pd.read_csv(d / "daily.csv", index_col="date", parse_dates=True),
                "rebalances": pd.read_csv(d / "rebalances.csv"),
                "weights": pd.read_csv(d / "weights.csv"),
                "manifest": json.loads((d / "manifest.json").read_text()),
            })
        b = exp / "benchmark"
        for key in ("results", "penalty_sweep", "frontier", "qubit_growth"):
            f = b / f"{key}.csv"
            if f.exists() and f.stat().st_size > 2:
                item["benchmark"][key] = pd.read_csv(f)
        if (b / "manifest.json").exists():
            item["manifest"] = json.loads((b / "manifest.json").read_text())
        elif item["backtests"]:
            item["manifest"] = item["backtests"][0]["manifest"]
        if item["backtests"] or item["benchmark"]:
            out[exp.name] = item
    return out


# --------------------------------------------------------------------------- tables
def benchmark_table(results: pd.DataFrame) -> pd.DataFrame:
    rows = []
    order = {s: i for i, s in enumerate(SOLVER_STYLE)}
    for (n, inst, solver), g in results.groupby(["n", "instance", "solver"], sort=False):
        ok = g[g["feasible"].astype(bool)]
        rows.append({
            "instance": inst, "n": n, "solver": solver, "runs": len(g),
            "feasible": int(len(ok)),
            "skipped/failed": int(g["status"].isin(["skipped", "failed"]).sum()),
            "infeasible pre-repair": int((g["raw_violation"] > 0).sum()),
            "repair failed": int((g["status"] == "infeasible").sum()),
            "median gap %": ok["gap_pct"].median() if len(ok) else np.nan,
            "max gap %": ok["gap_pct"].max() if len(ok) else np.nan,
            "median wall s": g.loc[~g["status"].isin(["skipped"]), "wall_time_s"].median(),
            "median evals": g.loc[~g["status"].isin(["skipped"]), "evals"].median(),
            "qubits": g["required_qubits"].iloc[0],
            "depth": g["depth"].median() if "depth" in g and g["depth"].notna().any() else np.nan,
            "2q gates": g["two_qubit_gates"].median()
            if "two_qubit_gates" in g and g["two_qubit_gates"].notna().any() else np.nan,
            "reference": g["reference_kind"].iloc[0],
        })
    t = pd.DataFrame(rows)
    t["_o"] = t["solver"].map(order).fillna(99)
    return t.sort_values(["n", "instance", "_o"]).drop(columns="_o").reset_index(drop=True)


def backtest_table(backtests: list, split: str) -> pd.DataFrame:
    rows = []
    for solver in dict.fromkeys(b["solver"] for b in backtests):
        runs = [b["summary"][split] for b in backtests if b["solver"] == solver and split in b["summary"]]
        if not runs:
            continue

        def agg(key):
            m, lo, hi = mean_ci([r.get(key, np.nan) for r in runs])
            return m, (hi - lo) / 2

        row = {"solver": solver, "seeds": len(runs)}
        for key, label in [("ann_return", "ann. return"), ("ann_vol", "ann. vol"), ("sharpe", "Sharpe"),
                           ("max_drawdown", "max drawdown"), ("tracking_error", "tracking error"),
                           ("mean_turnover_l1", "turnover / rebalance"), ("total_cost", "total cost"),
                           ("mean_objective", "mean objective"), ("mean_wall_time_s", "solve s / rebalance")]:
            m, half = agg(key)
            row[label] = m
            row[label + " ±"] = half
        for key, label in [("n_rebalances", "rebalances"), ("n_infeasible_pre_repair", "infeasible pre-repair"),
                           ("n_no_portfolio", "no portfolio")]:
            row[label] = int(sum(r.get(key, 0) for r in runs))
        rows.append(row)
    return pd.DataFrame(rows)


def _fmt(v, nd=3):
    if isinstance(v, (float, np.floating)):
        if not np.isfinite(v):
            return "–"
        if v != 0 and abs(v) < 10 ** -nd:
            return f"{v:.1e}"
        return f"{v:.{nd}f}".rstrip("0").rstrip(".") if abs(v) < 1e6 else f"{v:.3g}"
    return html.escape(str(v))


def _html_table(df: pd.DataFrame, pm: bool = False) -> str:
    cols = [c for c in df.columns if not c.endswith(" ±")]
    head = "".join(f"<th>{html.escape(c)}</th>" for c in cols)
    body = []
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            txt = _fmt(r[c])
            if pm and c + " ±" in df.columns and np.isfinite(r[c + " ±"]) and r[c + " ±"] > 0:
                txt += f" <span class='pm'>± {_fmt(r[c + ' ±'])}</span>"
            cells.append(f"<td>{txt}</td>")
        body.append("<tr>" + "".join(cells) + "</tr>")
    return f"<div class='tbl'><table><thead><tr>{head}</tr></thead><tbody>{''.join(body)}</tbody></table></div>"


def to_markdown(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(_fmt(r[c]).replace("–", "-") for c in cols) + " |")
    return "\n".join(lines)


# --------------------------------------------------------------------------- figures
def fig_gap(results: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    for solver in [s for s in SOLVER_STYLE if s in set(results["solver"])]:
        g = results[(results["solver"] == solver) & results["feasible"].astype(bool)]
        if g.empty:
            continue
        xs, ys, lo, hi = [], [], [], []
        for n, gg in g.groupby("n"):
            m, a, b = mean_ci(gg["gap_pct"])
            xs.append(n), ys.append(m), lo.append(m - a), hi.append(b - m)
        color, dash, sym = _style(solver)
        fig.add_trace(go.Scatter(
            x=xs, y=ys, name=solver, mode="lines+markers",
            line=dict(color=color, width=2, dash=dash), marker=dict(size=8, symbol=sym),
            error_y=dict(type="data", symmetric=False, array=hi, arrayminus=lo, thickness=1, width=3),
            hovertemplate=f"{solver}<br>n=%{{x}}<br>gap %{{y:.2f}} %<extra></extra>"))
    _layout(fig, "Objective gap to the reference optimum (0 % = optimal, 100 % = random feasible portfolio)",
            "assets (n)", "normalised gap, % (mean, 95 % CI)")
    fig.update_xaxes(type="log", tickvals=sorted(results["n"].unique()))
    return fig


def fig_time(results: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    ran = results[~results["status"].isin(["skipped"])]
    for solver in [s for s in SOLVER_STYLE if s in set(ran["solver"])]:
        g = ran[ran["solver"] == solver].groupby("n")["wall_time_s"].median()
        color, dash, sym = _style(solver)
        fig.add_trace(go.Scatter(
            x=g.index, y=g.values, name=solver, mode="lines+markers",
            line=dict(color=color, width=2, dash=dash), marker=dict(size=8, symbol=sym),
            hovertemplate=f"{solver}<br>n=%{{x}}<br>%{{y:.3g}} s<extra></extra>"))
    _layout(fig, "End-to-end wall time per solve (model build, optimise, sample, decode, repair)",
            "assets (n)", "median wall time, s")
    fig.update_xaxes(type="log", tickvals=sorted(results["n"].unique()))
    fig.update_yaxes(type="log")
    return fig


def fig_qubits(growth: pd.DataFrame, limit: int = 20) -> go.Figure:
    fig = go.Figure()
    for k, (label, g) in enumerate(growth.groupby("constraints", sort=False)):
        fig.add_trace(go.Scatter(
            x=g["n"], y=g["total"], name=label, mode="lines+markers",
            line=dict(color=SLOTS[k % 8], width=2), marker=dict(size=8),
            customdata=g[["position", "indicator", "slack"]].to_numpy(),
            hovertemplate=(label + "<br>n=%{x}: %{y} qubits<br>position %{customdata[0]}, "
                           "indicator %{customdata[1]}, slack %{customdata[2]}<extra></extra>")))
    fig.add_hline(y=limit, line=dict(color=NEUTRAL, width=1, dash="dot"),
                  annotation_text=f"statevector simulator limit used here ({limit} qubits)",
                  annotation_position="top left", annotation_font_color=INK2)
    _layout(fig, "Qubit requirement of the QUBO encoding (lots = n, position cap = 4 lots)",
            "assets (n)", "logical qubits")
    fig.update_xaxes(type="log", tickvals=sorted(growth["n"].unique()))
    fig.update_yaxes(type="log")
    return fig


def fig_penalty(sweep: pd.DataFrame) -> list[go.Figure]:
    g = sweep.groupby("scale")
    figs = []
    fig = go.Figure()
    for k, (col, label) in enumerate([("ground_state_feasible", "QUBO ground state is feasible"),
                                      ("ground_state_optimal", "QUBO ground state is the true optimum")]):
        y = g[col].mean() * 100
        fig.add_trace(go.Scatter(x=y.index, y=y.values, name=label, mode="lines+markers",
                                 line=dict(color=SLOTS[k], width=2), marker=dict(size=8),
                                 hovertemplate=label + "<br>scale %{x}: %{y:.0f} % of instances<extra></extra>"))
    _layout(fig, "Penalty calibration: is the encoding exact? (brute-force verified)",
            "penalty scale (× gradient bound)", "% of tiny instances")
    fig.update_xaxes(type="log")
    figs.append(fig)
    if "qaoa_sample_feasible_fraction" in sweep:
        fig = go.Figure()
        for k, (col, label) in enumerate([
                ("qaoa_sample_feasible_fraction", "QAOA shots that are feasible"),
                ("qaoa_p_ground_state", "QAOA probability of the QUBO ground state")]):
            y = g[col].mean() * 100
            fig.add_trace(go.Scatter(x=y.index, y=y.values, name=label, mode="lines+markers",
                                     line=dict(color=SLOTS[k + 2], width=2), marker=dict(size=8),
                                     hovertemplate=label + "<br>scale %{x}: %{y:.2f} %<extra></extra>"))
        _layout(fig, "Penalty strength versus QAOA sample quality (noiseless simulation)",
                "penalty scale (× gradient bound)", "% (mean over instances)")
        fig.update_xaxes(type="log")
        fig.update_yaxes(type="log")
        figs.append(fig)
    return figs


def fig_frontier(frontier: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    cont = frontier[frontier["solver"] == "continuous relaxation"].sort_values("risk")
    fig.add_trace(go.Scatter(x=cont["risk"], y=cont["expected_return"], name="continuous relaxation",
                             mode="lines+markers", line=dict(color=NEUTRAL, width=2),
                             marker=dict(size=6), customdata=cont["risk_aversion"],
                             hovertemplate="continuous relaxation<br>λ=%{customdata}<br>risk %{x:.3f}, "
                                           "return %{y:.3f}<extra></extra>"))
    rest = frontier[(frontier["solver"] != "continuous relaxation") & frontier["feasible"].astype(bool)]
    for solver in [s for s in SOLVER_STYLE if s in set(rest["solver"])]:
        g = rest[rest["solver"] == solver].groupby("risk_aversion")[["risk", "expected_return"]].mean()
        color, _, sym = _style(solver)
        fig.add_trace(go.Scatter(
            x=g["risk"], y=g["expected_return"], name=solver, mode="markers",
            marker=dict(size=11, symbol=sym, color=color, line=dict(color=SURFACE, width=2)),
            customdata=g.index,
            hovertemplate=f"{solver}<br>λ=%{{customdata}}<br>risk %{{x:.3f}}, return %{{y:.3f}}<extra></extra>"))
    return _layout(fig, "Efficient frontier: continuous relaxation versus discrete, constrained portfolios",
                   "ex-ante risk (annualised volatility)", "ex-ante expected return")


def fig_equity(backtests: list) -> go.Figure:
    fig = go.Figure()
    bench = None
    boundary = None
    for solver in [s for s in SOLVER_STYLE if s in {b["solver"] for b in backtests}]:
        runs = [b for b in backtests if b["solver"] == solver]
        nav = pd.concat([(1 + b["daily"]["ret"]).cumprod() for b in runs], axis=1).mean(axis=1)
        color, dash, _ = _style(solver)
        fig.add_trace(go.Scatter(x=nav.index, y=nav.values, name=solver, mode="lines",
                                 line=dict(color=color, width=2, dash=dash),
                                 hovertemplate=f"{solver}<br>%{{x|%Y-%m-%d}}<br>NAV %{{y:.3f}}<extra></extra>"))
        if bench is None:
            d = runs[0]["daily"]
            bench = (1 + d["bench"]).cumprod()
            test = d.index[d["split"] == "test"]
            boundary = test[0] if len(test) and (d["split"] == "validation").any() else None
    if bench is not None:
        fig.add_trace(go.Scatter(x=bench.index, y=bench.values, name="equal-weight benchmark", mode="lines",
                                 line=dict(color=NEUTRAL, width=2, dash="dot"),
                                 hovertemplate="equal weight<br>%{x|%Y-%m-%d}<br>NAV %{y:.3f}<extra></extra>"))
    if boundary is not None:
        fig.add_vline(x=boundary.timestamp() * 1000, line=dict(color=NEUTRAL, width=1, dash="dot"),
                      annotation_text="validation | test", annotation_position="top",
                      annotation_font_color=INK2)
    return _layout(fig, "Out-of-sample NAV net of costs (mean across seeds; identical lines overlap)",
                   "", "NAV (start = 1)", hovermode="x unified")


def fig_allocation(backtests: list) -> go.Figure:
    solvers = [s for s in SOLVER_STYLE if s in {b["solver"] for b in backtests}]
    rows, names, assets = [], [], None
    for solver in solvers:
        runs = [b["weights"] for b in backtests if b["solver"] == solver and len(b["weights"])]
        if not runs:
            continue
        assets = [c for c in runs[0].columns if c != "signal_date"]
        rows.append(np.mean([w[assets].mean().to_numpy() for w in runs], axis=0))
        names.append(solver)
    z = np.array(rows) * 100
    fig = go.Figure(go.Heatmap(
        z=z, x=assets, y=names, zmin=0 if z.min() >= 0 else -abs(z).max(), zmax=abs(z).max(),
        colorscale=[[0, "#fcfcfb"], [0.25, "#b7d3f6"], [0.5, "#6da7ec"], [0.75, "#256abf"], [1, "#0d366b"]]
        if z.min() >= 0 else [[0, "#d03b3b"], [0.5, "#f0efec"], [1, "#2a78d6"]],
        text=[[f"{v:.0f}" for v in r] for r in z], texttemplate="%{text}", xgap=2, ygap=2,
        colorbar=dict(title="weight %", thickness=10, outlinewidth=0),
        hovertemplate="%{y} · %{x}<br>mean weight %{z:.1f} %<extra></extra>"))
    _layout(fig, "Allocation comparison: mean weight per asset across all rebalances (%)", "", "",
            height=120 + 38 * len(names))
    fig.update_yaxes(autorange="reversed")
    return fig


def fig_timing(backtests: list) -> go.Figure:
    parts = ["model_build", "compile", "state_prep", "optimize", "sampling", "decode", "repair"]
    solvers = [s for s in SOLVER_STYLE if s in {b["solver"] for b in backtests}]
    fig = go.Figure()
    for k, part in enumerate(parts):
        vals = []
        for solver in solvers:
            frames = [b["rebalances"] for b in backtests if b["solver"] == solver]
            col = f"t_{part}"
            vals.append(float(np.mean([f[col].fillna(0).mean() if col in f else 0.0 for f in frames])))
        if max(vals) <= 0:
            continue
        fig.add_trace(go.Bar(y=solvers, x=vals, name=part.replace("_", " "), orientation="h",
                             marker=dict(color=SLOTS[k], line=dict(color=SURFACE, width=2)),
                             hovertemplate="%{y} · " + part + "<br>%{x:.4f} s<extra></extra>"))
    _layout(fig, "Where the time goes: mean seconds per rebalance by pipeline stage", "seconds", "",
            height=160 + 34 * len(solvers), barmode="stack")
    fig.update_yaxes(autorange="reversed")
    return fig


# --------------------------------------------------------------------------- page
CSS = """
body{margin:0;background:#f4f3ef;color:#0b0b0b;font:15px/1.55 -apple-system,'Segoe UI',Helvetica,Arial,sans-serif}
main{max-width:1120px;margin:0 auto;padding:32px 20px 64px}
h1{font-size:26px;margin:0 0 4px} h2{font-size:19px;margin:44px 0 8px} h3{font-size:15px;margin:22px 0 6px}
p,li{color:#52514e;max-width:86ch} .lede{font-size:16px;color:#0b0b0b}
.card{background:#fcfcfb;border:1px solid #e6e5e0;border-radius:8px;padding:8px;margin:14px 0}
.note{background:#fcfcfb;border-left:3px solid #8a8983;padding:10px 14px;margin:14px 0;border-radius:0 6px 6px 0}
.tbl{overflow-x:auto;background:#fcfcfb;border:1px solid #e6e5e0;border-radius:8px;margin:14px 0}
table{border-collapse:collapse;width:100%;font-size:12.5px;font-variant-numeric:tabular-nums}
th,td{padding:6px 10px;text-align:right;border-bottom:1px solid #e6e5e0;white-space:nowrap}
th{color:#52514e;font-weight:600;position:sticky;top:0;background:#fcfcfb}
td:first-child,th:first-child,td:nth-child(3),th:nth-child(3){text-align:left}
tr:last-child td{border-bottom:0} .pm{color:#8a8983} code{font-size:12.5px;background:#eceae4;padding:1px 5px;border-radius:4px}
dl{display:grid;grid-template-columns:max-content 1fr;gap:2px 16px;font-size:13px;color:#52514e} dt{font-weight:600}
"""


def build_report(runs_dir: Path, output: Path) -> dict:
    data = load_runs(runs_dir)
    if not data:
        raise SystemExit(f"no runs found under {runs_dir}")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    parts, first, written = [], [True], {}

    def plot(fig: go.Figure) -> str:
        inc = "inline" if first[0] else False     # embed plotly.js once: the file works offline
        first[0] = False
        return "<div class='card'>" + fig.to_html(full_html=False, include_plotlyjs=inc,
                                                    config={"displaylogo": False}) + "</div>"

    for name, item in data.items():
        man = item["manifest"] or {}
        parts.append(f"<h2>Experiment <code>{html.escape(name)}</code></h2>")
        pk = man.get("packages", {})
        parts.append("<dl>" + "".join(f"<dt>{k}</dt><dd>{html.escape(str(v))}</dd>" for k, v in [
            ("config", f"{man.get('config_path')} (hash {man.get('config_hash')})"),
            ("code hash", man.get("code_hash")),
            ("machine", f"{man.get('platform')} · {man.get('processor')} · Python {man.get('python')}"),
            ("hardware", man.get("hardware")),
            ("budget per solve", json.dumps(man.get("config", {}).get("budget"))),
            ("seeds", json.dumps(man.get("config", {}).get("seeds"))),
            ("key packages", ", ".join(f"{k} {v}" for k, v in pk.items() if v)),
        ]) + "</dl>")

        bm = item["benchmark"]
        if "results" in bm:
            res = bm["results"]
            parts.append("<h3>Benchmark: synthetic instances, equal budget</h3>")
            parts.append("<p>Each instance is solved by every solver under the same wall-time budget. "
                         "The gap is normalised per instance: 0 % is the reference optimum "
                         "(brute force where enumerable, otherwise a proven-optimal MIP or the best "
                         "solution any solver found) and 100 % is the average random feasible "
                         "portfolio. Solvers that cannot run an instance are listed as skipped, "
                         "not dropped.</p>")
            parts.append(plot(fig_gap(res)))
            parts.append(plot(fig_time(res)))
            table = benchmark_table(res)
            parts.append(_html_table(table))
            md = output.parent / f"benchmark_table_{name}.md"
            md.write_text(f"# Benchmark table: `{name}`\n\n"
                          "Simulation only (no QPU). Gap: 0 % = reference optimum, 100 % = mean random "
                          "feasible portfolio. Generated by `python -m hqpo.report`.\n\n"
                          + to_markdown(table) + "\n")
            res.to_csv(output.parent / f"benchmark_results_{name}.csv", index=False)
            written["benchmark_table"] = str(md)
        if "qubit_growth" in bm:
            limit = man.get("config", {}).get("solver_params", {}).get("qaoa", {}).get("max_qubits", 20)
            parts.append(plot(fig_qubits(bm["qubit_growth"], limit)))
        if "penalty_sweep" in bm:
            parts.append("<h3>Penalty calibration</h3><p>For tiny instances the full QUBO spectrum is "
                         "enumerated, so we can check directly whether the lowest-energy bitstring "
                         "decodes to the true constrained optimum at each penalty strength.</p>")
            parts += [plot(f) for f in fig_penalty(bm["penalty_sweep"])]
        if "frontier" in bm:
            parts.append("<h3>Efficient frontier</h3>")
            parts.append(plot(fig_frontier(bm["frontier"])))

        if item["backtests"]:
            bts = item["backtests"]
            parts.append("<h3>Rolling backtest</h3><p>Signals use prices up to the signal date only and "
                         "trade one session later with linear costs. A backtest on synthetic data "
                         "shows that the pipeline works end to end; it does not establish live "
                         "profitability, and differences between solvers here mostly reflect which "
                         "feasible portfolio each one returned.</p>")
            parts.append(plot(fig_equity(bts)))
            for split in ("validation", "test"):
                t = backtest_table(bts, split)
                if len(t):
                    parts.append(f"<p><b>{split.capitalize()} window</b> (mean ± 95 % CI half-width "
                                 "across seeds)</p>" + _html_table(t, pm=True))
                    t.to_csv(output.parent / f"backtest_{split}_{name}.csv", index=False)
            parts.append(plot(fig_allocation(bts)))
            parts.append(plot(fig_timing(bts)))

    legend = "".join(f"<li><code>{k}</code>: {v}</li>" for k, v in SOLVER_NOTE.items())
    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Hybrid quantum-classical portfolio optimizer: comparison</title><style>{CSS}</style></head>
<body><main>
<h1>Hybrid quantum–classical portfolio optimizer</h1>
<p class="lede">Classical baselines versus QUBO-based solvers on the same discrete portfolio problem,
the same data and the same computation budget.</p>
<div class="note"><b>Read this first.</b> Every quantum result here is a classical simulation on a CPU:
<code>qaoa</code> is a noiseless statevector simulation, <code>qaoa-noisy</code> adds a depolarising noise
model, and <code>qubo-sa</code> is simulated annealing on the QUBO. No QPU was used, so there are no
hardware results, queue times or hardware costs to report. Simulation wall time says nothing about
hardware run time. Nothing here is investment advice.</div>
<ul>{legend}</ul>
{''.join(parts)}
</main></body></html>"""
    output.write_text(page)
    written["html"] = str(output)
    return written
