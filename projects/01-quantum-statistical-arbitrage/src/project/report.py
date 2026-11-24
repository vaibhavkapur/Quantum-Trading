"""CLI: compare stored runs and write a self-contained HTML report.

    python -m project.report --runs runs/ --output reports/comparison.html

Only runs that share an experiment fingerprint (same data, objective,
constraints, splits, seeds and budget) are compared; anything else is listed
as excluded. A summary CSV is written next to the HTML.
"""
from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .backtest.metrics import annualized_return, bootstrap_ci, max_drawdown, sharpe

# Fixed entity -> colour slot mapping (never by rank). Light / dark steps of the same hues.
SOLVER_ORDER = ["qaoa", "cpsat", "greedy", "random_search", "simulated_annealing", "brute_force"]
LABELS = {
    "qaoa": "QAOA (simulator)",
    "cpsat": "CP-SAT",
    "greedy": "Greedy",
    "random_search": "Random search",
    "simulated_annealing": "Simulated annealing",
    "brute_force": "Brute force (reference)",
}
LIGHT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
DARK = ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#0ca30c"]


# --------------------------------------------------------------------------- loading
def load_runs(runs_dir: Path) -> dict:
    study, others, excluded = [], {}, []
    for manifest_path in sorted(runs_dir.glob("*/manifest.json")):
        m = json.loads(manifest_path.read_text())
        m["_dir"] = manifest_path.parent
        fam = m.get("solver_family")
        if fam in ("classical", "quantum-sim"):
            study.append(m)
        elif fam in ("scaling", "l2-replay"):
            others[fam] = m  # sorted glob -> the latest run of each kind wins
    if not study:
        raise SystemExit(f"no classical / quantum-sim runs found under {runs_dir}")
    # Compare the most recent fingerprint; keep the latest run per family.
    latest = max(study, key=lambda m: m["created_utc"])
    chosen: dict[str, dict] = {}
    for m in sorted(study, key=lambda m: m["created_utc"]):
        if m["fingerprint"] == latest["fingerprint"]:
            if m["solver_family"] in chosen:
                excluded.append((chosen[m["solver_family"]]["run_id"], "superseded by a newer run"))
            chosen[m["solver_family"]] = m
        else:
            excluded.append((m["run_id"], "different experiment fingerprint (data / constraints / budget differ)"))
    results = pd.concat([pd.read_parquet(m["_dir"] / "results.parquet") for m in chosen.values()], ignore_index=True)
    returns = pd.concat([pd.read_parquet(m["_dir"] / "returns.parquet") for m in chosen.values()], ignore_index=True)
    return {"manifests": list(chosen.values()), "results": results, "returns": returns, "excluded": excluded, "others": others}


# --------------------------------------------------------------------------- statistics
def _ci(values) -> tuple[float, float, float]:
    v = np.asarray(values, float)
    v = v[np.isfinite(v)]
    if not len(v):
        return float("nan"), float("nan"), float("nan")
    lo, hi = bootstrap_ci(v)
    return float(v.mean()), lo, hi


def summarize(results: pd.DataFrame, returns: pd.DataFrame, capital: float) -> pd.DataFrame:
    rows = []
    for solver in [s for s in SOLVER_ORDER if s in set(results["solver"])]:
        r = results[results["solver"] == solver]
        per_seed = r.groupby("seed")
        gap = _ci(per_seed["norm_gap"].mean())
        popt = _ci(per_seed["is_optimal"].mean())
        raw = r["raw_norm_gap"].to_numpy(float)
        paths = returns[returns["solver"] == solver].groupby("seed")["ret"]
        sh = _ci([sharpe(x.to_numpy()) for _, x in paths])
        rows.append(
            {
                "solver": solver,
                "n_seeds": int(r["seed"].nunique()),
                "n_runs": int(len(r)),
                "n_failed": int(r["failed"].sum()),
                "p_optimal": popt[0], "p_optimal_lo": popt[1], "p_optimal_hi": popt[2],
                "norm_gap": gap[0], "norm_gap_lo": gap[1], "norm_gap_hi": gap[2],
                "raw_feasible_rate": float(r["raw_feasible_rate"].mean()),
                "raw_found_feasible": float(r["raw_found_feasible"].mean()),
                "raw_norm_gap": float(np.nanmean(raw)) if np.isfinite(raw).any() else float("nan"),
                "needed_repair": float(r["repaired"].mean()),
                "n_evals": float(r["n_evals"].mean()),
                "n_shots": float(r["n_shots"].mean()),
                **{c: float(r[c].mean()) for c in ("t_encode", "t_compile", "t_optimize", "t_sample", "t_decode_repair", "t_total")},
                "t_total_max": float(r["t_total"].max()),
                "oos_sharpe": sh[0], "oos_sharpe_lo": sh[1], "oos_sharpe_hi": sh[2],
                "oos_ann_return": float(np.mean([annualized_return(x.to_numpy()) for _, x in paths])),
                "oos_max_dd": float(np.mean([max_drawdown(x.to_numpy()) for _, x in paths])),
                "turnover": float(r["turnover"].mean()),
                "folds_traded": float(r["traded"].mean()),
                "kill_switches": int(r["kill_switch"].sum()),
                "test_half_life": float(np.nanmedian(r["test_half_life"])) if r["test_half_life"].notna().any() else float("nan"),
                "train_half_life": float(np.nanmedian(r["train_half_life"])),
                "total_pnl_pct": float(r.groupby("seed")["test_pnl"].sum().mean() / capital * 100),
            }
        )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- formatting
def esc(x) -> str:
    return html.escape(str(x))


def f(x, digits=3, pct=False) -> str:
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "–"
    x = 0.0 if abs(x) < 0.5 * 10 ** -(digits + (2 if pct else 0)) else float(x)
    return f"{x * 100:.{digits}f}%" if pct else f"{x:.{digits}f}"


def ci(lo, hi, digits=3, pct=False) -> str:
    if not (np.isfinite(lo) and np.isfinite(hi)):
        return '<br><span class="muted">single path</span>'
    return f'<br><span class="muted">[{f(lo, digits, pct)}, {f(hi, digits, pct)}]</span>'


def secs(x: float) -> str:
    if not np.isfinite(x) or x == 0:
        return "–"
    if x < 1e-3:
        return f"{x * 1e6:.0f} µs"
    if x < 1:
        return f"{x * 1e3:.1f} ms"
    return f"{x:.2f} s"


def swatch(solver: str) -> str:
    return f'<span class="sw" style="background:var(--s-{solver})"></span>'


def table(headers: list[str], rows: list[list[str]], numeric_from: int = 1) -> str:
    th = "".join(f'<th class="{"num" if i >= numeric_from else ""}">{h}</th>' for i, h in enumerate(headers))
    body = "".join(
        "<tr>" + "".join(f'<td class="{"num" if i >= numeric_from else ""}">{c}</td>' for i, c in enumerate(r)) + "</tr>" for r in rows
    )
    return f'<div class="tablewrap"><table><thead><tr>{th}</tr></thead><tbody>{body}</tbody></table></div>'


def legend(solvers: list[str]) -> str:
    return '<div class="legend">' + "".join(f"<span>{swatch(s)}{esc(LABELS[s])}</span>" for s in solvers) + "</div>"


# --------------------------------------------------------------------------- charts (inline SVG)
def _ticks(lo: float, hi: float, n: int = 5) -> list[float]:
    if hi <= lo:
        hi = lo + 1.0
    raw = (hi - lo) / n
    mag = 10 ** np.floor(np.log10(raw))
    step = min((m for m in (1, 2, 2.5, 5, 10) if m * mag >= raw), default=10) * mag
    start = np.ceil(lo / step) * step
    return [float(round(v, 10)) for v in np.arange(start, hi + step * 1e-9, step)]


def dot_chart(items: list[dict], x_label: str, log: bool = False, fmt=lambda v: f(v, 3)) -> str:
    """Horizontal dot plot with optional interval. items: {solver, value, lo, hi, tip}."""
    W, row_h, left, right, top = 760, 34, 190, 70, 10
    H = top + row_h * len(items) + 40
    vals = [v for it in items for v in (it["value"], it.get("lo"), it.get("hi")) if v is not None and np.isfinite(v)]
    if log:
        lo, hi = np.log10(max(min(vals), 1e-9)), np.log10(max(vals))
        lo, hi = np.floor(lo), np.ceil(hi)
        ticks = [10.0**e for e in range(int(lo), int(hi) + 1)]
        sx = lambda v: left + (np.log10(max(v, 10.0**lo)) - lo) / (hi - lo) * (W - left - right)  # noqa: E731
    else:
        lo, hi = min(0.0, min(vals)), max(vals) * 1.08 if max(vals) > 0 else 1.0
        ticks = _ticks(lo, hi)
        sx = lambda v: left + (v - lo) / (hi - lo) * (W - left - right)  # noqa: E731
    out = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="{esc(x_label)}">']
    base = top + row_h * len(items)
    for t in ticks:
        x = sx(t)
        out.append(f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{base}" class="grid"/>')
        out.append(f'<text x="{x:.1f}" y="{base + 16}" class="tick" text-anchor="middle">{esc(fmt(t))}</text>')
    out.append(f'<text x="{(left + W - right) / 2:.1f}" y="{H - 4}" class="axis" text-anchor="middle">{esc(x_label)}</text>')
    for k, it in enumerate(items):
        y = top + row_h * k + row_h / 2
        s = it["solver"]
        out.append(f'<text x="{left - 12}" y="{y + 4:.1f}" class="label" text-anchor="end">{esc(LABELS[s])}</text>')
        if it.get("lo") is not None and np.isfinite(it["lo"]) and np.isfinite(it["hi"]):
            out.append(f'<line x1="{sx(it["lo"]):.1f}" y1="{y:.1f}" x2="{sx(it["hi"]):.1f}" y2="{y:.1f}" stroke="var(--s-{s})" stroke-width="2" opacity="0.55"/>')
        x = sx(it["value"])
        out.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="6" fill="var(--s-{s})" stroke="var(--surface)" stroke-width="2"/>')
        end = max(x, sx(it["hi"])) if it.get("hi") is not None and np.isfinite(it["hi"]) else x
        out.append(f'<text x="{end + 12:.1f}" y="{y + 4:.1f}" class="value">{esc(fmt(it["value"]))}</text>')
        out.append(f'<rect x="{left}" y="{y - row_h / 2:.1f}" width="{W - left - right}" height="{row_h}" fill="transparent" data-tip="{esc(it["tip"])}"/>')
    out.append("</svg>")
    return "".join(out)


def line_chart(chart_id: str, x_labels: list[str], series: dict[str, list[float]], y_label: str, fmt: str = "{:.2f}",
               log: bool = False, x_numeric: bool = False) -> str:
    """Multi-series line chart with a crosshair tooltip (data embedded for the hover layer)."""
    W, H, left, right, top, bottom = 760, 320, 62, 20, 14, 44
    n = len(x_labels)
    allv = [v for ys in series.values() for v in ys if v is not None and np.isfinite(v)]
    if log:
        lo, hi = np.floor(np.log10(max(min(allv), 1e-9))), np.ceil(np.log10(max(allv)))
        ticks = [10.0**e for e in range(int(lo), int(hi) + 1)]
        sy = lambda v: top + (1 - (np.log10(max(v, 10.0**lo)) - lo) / (hi - lo)) * (H - top - bottom)  # noqa: E731
    else:
        lo, hi = min(allv + [0.0]), max(allv + [0.0])
        pad = (hi - lo) * 0.06 or 1.0
        lo, hi = lo - pad, hi + pad
        ticks = _ticks(lo, hi)
        sy = lambda v: top + (1 - (v - lo) / (hi - lo)) * (H - top - bottom)  # noqa: E731
    xs = [left + (i / max(n - 1, 1)) * (W - left - right) for i in range(n)]
    out = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="{esc(y_label)}" class="linechart" id="{chart_id}">']
    for t in ticks:
        y = sy(t)
        label = secs(t) if log else fmt.format(t + 0.0 if abs(t) > 1e-12 else 0.0)
        out.append(f'<line x1="{left}" y1="{y:.1f}" x2="{W - right}" y2="{y:.1f}" class="grid"/>')
        out.append(f'<text x="{left - 8}" y="{y + 4:.1f}" class="tick" text-anchor="end">{esc(label)}</text>')
    if not log and lo < 0 < hi:
        out.append(f'<line x1="{left}" y1="{sy(0):.1f}" x2="{W - right}" y2="{sy(0):.1f}" class="zero"/>')
    step = 1 if x_numeric else max(n // 6, 1)
    for i in range(0, n, step):
        out.append(f'<text x="{xs[i]:.1f}" y="{H - bottom + 18}" class="tick" text-anchor="middle">{esc(x_labels[i])}</text>')
    out.append(f'<text x="{left}" y="{H - 6}" class="axis">{esc(y_label)}</text>')
    payload = {"x": x_labels, "xs": [round(x, 1) for x in xs], "series": []}
    for s, ys in series.items():
        pts = [(xs[i], sy(v)) for i, v in enumerate(ys) if v is not None and np.isfinite(v)]
        d = " ".join(f"{'M' if k == 0 else 'L'}{x:.1f},{y:.1f}" for k, (x, y) in enumerate(pts))
        out.append(f'<path d="{d}" fill="none" stroke="var(--s-{s})" stroke-width="2" stroke-linejoin="round"/>')
        if x_numeric:
            out.extend(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="var(--s-{s})" stroke="var(--surface)" stroke-width="2"/>' for x, y in pts)
        payload["series"].append(
            {
                "name": LABELS[s], "key": s,
                "v": [(secs(v) if log else fmt.format(v)) if v is not None and np.isfinite(v) else "–" for v in ys],
                "y": [round(sy(v), 1) if v is not None and np.isfinite(v) else None for v in ys],
            }
        )
    out.append(f'<line class="cross" x1="0" x2="0" y1="{top}" y2="{H - bottom}" visibility="hidden"/>')
    out.append(f'<rect class="hit" x="{left}" y="{top}" width="{W - left - right}" height="{H - top - bottom}" fill="transparent"/>')
    out.append(f'<script type="application/json">{json.dumps(payload)}</script></svg>')
    return "".join(out)


# --------------------------------------------------------------------------- sections
def headline(summary: pd.DataFrame) -> str:
    s = summary.set_index("solver")
    if "qaoa" not in s.index:
        return "Only classical runs are present; run <code>--solver quantum-sim</code> to add the QAOA comparison."
    q = s.loc["qaoa"]
    classical = s.drop(index=["qaoa", "brute_force"], errors="ignore")
    if classical.empty:
        return "Only the quantum-simulator run is present; run <code>--solver classical</code> to add baselines."
    best = classical["norm_gap"].idxmin()
    b = classical.loc[best]
    ratio = q["t_total"] / b["t_total"] if b["t_total"] > 0 else float("nan")
    verdict = (
        "No quantum advantage was observed"
        if q["norm_gap"] >= b["norm_gap"] - 1e-9
        else "QAOA reached a lower mean gap than the best classical baseline on these instances - treat as a lead to re-test, not a result"
    )
    parts = [
        f"<strong>{verdict}.</strong> At an identical budget, simulated QAOA found the optimal basket in "
        f"{f(q['p_optimal'], 0, True)} of fold-seed runs (mean normalised gap {f(q['norm_gap'])}) versus "
        f"{f(b['p_optimal'], 0, True)} for {esc(LABELS[best])} (gap {f(b['norm_gap'])}), while taking "
        f"{secs(q['t_total'])} per solve against {secs(b['t_total'])}"
        + (f" (about {ratio:,.0f}x slower)" if np.isfinite(ratio) and ratio >= 2 else "")
        + "."
    ]
    if "random_search" in s.index:
        r = s.loc["random_search"]
        parts.append(
            f" The budget-matched random control reaches {f(r['p_optimal'], 0, True)} optimal through the same repair step, "
            "so most of QAOA's final quality here comes from classical repair rather than from the circuit."
        )
    return "".join(parts)


def build_html(data: dict) -> tuple[str, pd.DataFrame]:
    results, returns = data["results"], data["returns"]
    m0 = data["manifests"][0]
    cfg = m0["config"]
    capital = float(cfg["risk"]["capital"])
    summary = summarize(results, returns, capital)
    solvers = list(summary["solver"])
    S = summary.set_index("solver")
    sec: list[str] = []

    # ---- protocol
    wf, sel, bud = cfg["walkforward"], cfg["selection"], cfg["budget"]
    folds = results.drop_duplicates("fold").sort_values("fold")
    sec.append(
        "<h2>Protocol</h2>"
        + table(
            ["Item", "Value"],
            [
                ["Experiment fingerprint", f"<code>{esc(m0['fingerprint'])}</code> (shared by all compared runs)"],
                ["Dataset", f"{esc(m0['dataset']['version'])} · {esc(m0['dataset'].get('license', ''))}"],
                ["Walk-forward", f"{len(folds)} folds · train {wf['train_days']} / validation {wf['val_days']} / test {wf['test_days']} bars, step {wf['step_days']}"],
                ["Out-of-sample span", f"{esc(folds['test_start'].iloc[0])} to {esc(folds['test_end'].iloc[-1])}"],
                ["Universe per fold", f"{int(results['n_assets'].min())}–{int(results['n_assets'].max())} assets (point-in-time), {int(results['n_feasible'].min())}–{int(results['n_feasible'].max())} feasible baskets"],
                ["Constraints", f"exactly {sel['basket_size']} assets, at most {sel['max_per_sector']} per sector"],
                ["Budget", f"{bud['max_evals']} scored candidates per stochastic solve · {bud['time_limit_s']} s limit for exact solvers"],
                ["Seeds (declared up front)", ", ".join(str(s) for s in cfg["seeds"])],
                ["Runs", "<br>".join(f"<code>{esc(m['run_id'])}</code> – {esc(m['solver_family'])}, {m['elapsed_s']} s, {m['n_failed']} failed" for m in data["manifests"])],
                ["Quantum hardware", "None. QAOA results are noiseless statevector simulation with sampled shots."],
            ],
            numeric_from=99,
        )
    )
    if data["excluded"]:
        sec.append("<p class='muted'>Excluded runs: " + "; ".join(f"<code>{esc(r)}</code> ({esc(why)})" for r, why in data["excluded"]) + "</p>")

    # ---- selection quality
    items = [
        {"solver": s, "value": S.loc[s, "norm_gap"], "lo": S.loc[s, "norm_gap_lo"], "hi": S.loc[s, "norm_gap_hi"],
         "tip": f"{LABELS[s]}: gap {f(S.loc[s, 'norm_gap'])}, optimal in {f(S.loc[s, 'p_optimal'], 0, True)} of runs"}
        for s in solvers
    ]
    sec.append(
        "<h2>Selection quality at equal budget</h2>"
        "<p>Normalised gap of the returned basket: 0 is the brute-force optimum, 1 is the median feasible basket. "
        "Dots are means over folds and seeds; bars, and the bracketed ranges in the tables, are 95% bootstrap intervals over seeds (deterministic solvers have a single path).</p>"
        + f'<figure>{dot_chart(items, "Normalised optimality gap (lower is better)")}</figure>'
        + table(
            ["Solver", "Seeds", "Runs", "Failed", "P(optimal)", "Norm. gap", "Raw feasible", "Raw-only gap", "Repaired", "Evals", "Shots"],
            [
                [swatch(s) + esc(LABELS[s]), S.loc[s, "n_seeds"], S.loc[s, "n_runs"], S.loc[s, "n_failed"],
                 f(S.loc[s, "p_optimal"], 0, True) + ci(S.loc[s, "p_optimal_lo"], S.loc[s, "p_optimal_hi"], 0, True),
                 f(S.loc[s, "norm_gap"]) + ci(S.loc[s, "norm_gap_lo"], S.loc[s, "norm_gap_hi"]),
                 f(S.loc[s, "raw_feasible_rate"], 1, True), f(S.loc[s, "raw_norm_gap"]), f(S.loc[s, "needed_repair"], 0, True),
                 f"{S.loc[s, 'n_evals']:,.0f}", f"{S.loc[s, 'n_shots']:,.0f}"]
                for s in solvers
            ],
        )
        + "<p class='muted'>Raw feasible = share of sampled candidates satisfying every constraint before repair. "
        "Raw-only gap = gap of the best candidate that was feasible as sampled (runs with none are skipped), which isolates the sampler from the shared greedy repair. "
        "CP-SAT evals are search branches; brute force scores every feasible basket and ignores the budget by design.</p>"
    )

    # ---- timing
    items = [{"solver": s, "value": S.loc[s, "t_total"], "lo": None, "hi": None,
              "tip": f"{LABELS[s]}: mean {secs(S.loc[s, 't_total'])}, max {secs(S.loc[s, 't_total_max'])}"} for s in solvers]
    sec.append(
        "<h2>Wall-clock cost of one solve</h2>"
        "<p>Mean time per solve, log scale. Everything on the path is counted: encoding, compilation, optimiser iterations, sampling, decode and repair.</p>"
        + f'<figure>{dot_chart(items, "Seconds per solve (log scale)", log=True, fmt=secs)}</figure>'
        + table(
            ["Solver", "Encode", "Compile", "Optimise", "Sample", "Decode + repair", "Total (mean)", "Total (max)"],
            [[swatch(s) + esc(LABELS[s])] + [secs(S.loc[s, c]) for c in ("t_encode", "t_compile", "t_optimize", "t_sample", "t_decode_repair", "t_total", "t_total_max")] for s in solvers],
        )
    )
    if "qaoa" in S.index:
        meta = pd.DataFrame([json.loads(x) for x in results.loc[results["solver"] == "qaoa", "meta"]])
        if "qubits" in meta:
            def rng(col, digits=0):
                v = pd.to_numeric(meta[col], errors="coerce").dropna()
                return "–" if v.empty else (f"{v.min():.{digits}f}–{v.max():.{digits}f}" if v.min() != v.max() else f"{v.min():.{digits}f}")
            sec.append(
                "<h3>QAOA circuit and sampling detail</h3>"
                + table(
                    ["Quantity", "Value"],
                    [
                        ["Backend", esc(", ".join(sorted(set(meta["backend"]))))],
                        ["Qubits (one per asset)", rng("qubits")],
                        ["Layers p / shots per iteration / CVaR α", f"{rng('p')} / {rng('shots_per_iter')} / {rng('cvar_alpha', 2)}"],
                        ["Optimiser iterations (COBYLA)", rng("optimizer_iters")],
                        ["Transpiled depth (rz, sx, cx basis, all-to-all)", rng("depth")],
                        ["Two-qubit gates", rng("two_qubit_gates")],
                        ["Total shots per solve", f"{S.loc['qaoa', 'n_shots']:,.0f}"],
                        ["P(optimal state) in final distribution", f"{meta['p_optimal_state'].mean():.5f} (uniform sampling: {meta['uniform_p_optimal_state'].mean():.5f})"],
                        ["P(cardinality satisfied) in final distribution", f(meta["p_cardinality_ok"].mean(), 1, True)],
                    ],
                    numeric_from=99,
                )
                + "<p class='muted'>The two probability rows need the full statevector and exist only in simulation. "
                "Depth assumes all-to-all connectivity and no noise; routing on a real device would increase it.</p>"
            )

    # ---- trading
    curve = returns.groupby(["solver", "ts"])["ret"].mean().unstack("solver").sort_index().cumsum() * 100
    plot_solvers = [s for s in solvers if s != "brute_force" or "cpsat" not in solvers]
    sec.append(
        "<h2>Out-of-sample trading result</h2>"
        "<p>Cumulative P&amp;L after spread, impact, fees and borrow, as a percentage of starting capital, averaged over seeds. "
        "Test windows are chained; folds whose basket failed the validation gate stay flat. "
        + ("Brute force selects the same baskets as CP-SAT and is omitted from the chart. " if "brute_force" not in plot_solvers else "")
        + "<strong>A backtest on synthetic data does not establish live profitability.</strong></p>"
        + legend(plot_solvers)
        + "<figure>"
        + line_chart("equity", [str(t.date()) for t in curve.index], {s: curve[s].tolist() for s in plot_solvers}, "Cumulative P&L, % of capital", "{:.1f}%")
        + "</figure>"
        + table(
            ["Solver", "OOS Sharpe", "Ann. return", "Max DD", "Total P&amp;L", "Turnover /yr", "Folds traded", "Kill switches", "Half-life train → test"],
            [
                [swatch(s) + esc(LABELS[s]), f(S.loc[s, "oos_sharpe"], 2) + ci(S.loc[s, "oos_sharpe_lo"], S.loc[s, "oos_sharpe_hi"], 2),
                 f(S.loc[s, "oos_ann_return"], 2, True), f(S.loc[s, "oos_max_dd"], 2, True), f"{S.loc[s, 'total_pnl_pct']:.2f}%",
                 f(S.loc[s, "turnover"], 2), f(S.loc[s, "folds_traded"], 0, True), S.loc[s, "kill_switches"],
                 f"{f(S.loc[s, 'train_half_life'], 1)} → {f(S.loc[s, 'test_half_life'], 1)} bars"]
                for s in solvers
            ],
        )
        + "<p class='muted'>Sharpe is computed on the chained daily series including flat days. Intervals cover solver-seed variation only, "
        "not sampling error of the return series, which is far larger at this sample size.</p>"
    )

    # ---- per fold
    ref_name = "brute_force" if "brute_force" in solvers else solvers[0]
    ref = results[results["solver"] == ref_name].sort_values("fold")
    rows = []
    for _, r in ref.iterrows():
        same = results[(results["fold"] == r["fold"]) & (results["solver"] != ref_name)]
        agree = (same["basket"] == r["basket"]).mean() if len(same) else float("nan")
        truth = ("yes" if r["truth_all_group_a"] else "no") if "truth_all_group_a" in r else "–"
        rows.append([
            int(r["fold"]), f"{esc(r['test_start'])} → {esc(r['test_end'])}", esc(r["basket"].replace("|", " · ")), truth,
            f(r["val_adf_pvalue"], 3), "pass" if r["gate_pass"] else f"fail ({esc(r['gate_reasons'])})",
            int(r["n_entries"]), f"{r['test_pnl'] / capital * 100:.2f}%", f(agree, 0, True),
        ])
    sec.append(
        f"<h2>Fold by fold ({esc(LABELS[ref_name])} basket)</h2>"
        + table(["Fold", "Test window", "Basket", "Truly cointegrated", "Val. ADF p", "Gate", "Entries", "Test P&amp;L", "Runs agreeing"], rows, numeric_from=99)
        + "<p class='muted'>“Truly cointegrated” uses the synthetic generator's ground truth (all members from the single-trend group). "
        "It is unavailable for real data and never used by any solver.</p>"
    )

    # ---- scaling
    if "scaling" in data["others"]:
        sm = data["others"]["scaling"]
        sc = pd.read_csv(sm["_dir"] / "scaling.csv")
        ss = [s for s in SOLVER_ORDER if s in set(sc["solver"])]
        gap = sc.groupby(["n", "solver"])["norm_gap"].mean().unstack("solver")
        tt = sc.groupby(["n", "solver"])["t_total"].mean().unstack("solver")
        popt = sc.groupby(["n", "solver"])["is_optimal"].mean().unstack("solver")
        ns = [str(n) for n in gap.index]
        sec.append(
            "<h2>Scaling with universe size</h2>"
            f"<p>Planted random instances ({esc(sm['instance_generator'])}), {sm['instances']} instances × {len(sm['seeds'])} seeds per size, "
            f"budget {sm['budget']['max_evals']} candidates. QAOA uses the numpy statevector backend, whose cost grows as 2<sup>N</sup>: "
            "that curve measures classical simulation, not a quantum device.</p>"
            + legend(ss)
            + "<h3>Normalised gap by number of assets</h3><figure>"
            + line_chart("scale-gap", ns, {s: gap[s].tolist() for s in ss}, "Normalised gap vs assets (qubits)", "{:.2f}", x_numeric=True)
            + "</figure><h3>Seconds per solve by number of assets (log scale)</h3><figure>"
            + line_chart("scale-time", ns, {s: tt[s].tolist() for s in ss}, "Seconds per solve vs assets (qubits)", log=True, x_numeric=True)
            + "</figure>"
            + table(
                ["Assets"] + [f"{swatch(s)}{esc(LABELS[s])}" for s in ss],
                [[n] + [f"{f(popt.loc[int(n), s], 0, True)} · {f(gap.loc[int(n), s])} · {secs(tt.loc[int(n), s])}" for s in ss] for n in ns],
            )
            + "<p class='muted'>Cells: P(optimal) · normalised gap · mean time.</p>"
        )

    # ---- L2 engine
    if "l2-replay" in data["others"]:
        lm = data["others"]["l2-replay"]
        rep = json.loads((lm["_dir"] / "engine_report.json").read_text())
        d, o = rep["tick_to_decision_ns"], rep["tick_to_order_ns"]
        sec.append(
            "<h2>Latency-critical path (Rust engine)</h2>"
            "<p>Level-2 quote replay, signal and execution run in a separate Rust binary; hedge weights are fitted in Python on the warm-up segment only. "
            "No quantum component sits on this path.</p>"
            + table(
                ["Measure", "Value"],
                [
                    ["Events replayed", f"{rep['events']:,} ({rep['ticks']:,} ticks; {rep['warmup_ticks']:,} warm-up, {rep['traded_ticks']:,} traded)"],
                    ["Throughput", f"{rep['events_per_second'] / 1e6:.1f} M events/s (replay loop) · file load {rep['load_seconds'] * 1e3:.0f} ms"],
                    ["Tick-to-decision latency", f"p50 {d['p50']} ns · p99 {d['p99']} ns · p99.9 {d['p99_9']} ns · max {d['max']} ns"],
                    ["Tick-to-order latency", f"p50 {o['p50']} ns · p99 {o['p99']} ns · max {o['max']} ns ({rep['tick_to_order_samples']} orders)"],
                    ["Simulated order latency", f"{rep['simulated_order_latency_ns'] / 1e3:.0f} µs between decision and book access"],
                    ["Entries / stop-outs / fills", f"{rep['entries']} / {rep['stops']} / {rep['fills']} (unfilled legs: {rep['unfilled']})"],
                    ["Costs", f"fees {rep['fees']:.2f} · slippage vs mid {rep['slippage_vs_mid']:.2f} on {rep['traded_notional']:,.0f} traded"],
                    ["Net P&amp;L", f"{rep['pnl']:.2f} on {rep['notional']:,.0f} notional per position"],
                ],
                numeric_from=99,
            )
            + "<p class='muted'>Latency is in-process compute time on synthetic data, quantised by the OS clock (about 42 ns on Apple silicon). "
            "It excludes network, kernel and venue time and is not a statement about a live system.</p>"
        )

    sec.append(
        "<h2>Limitations</h2><ul>"
        "<li>Data are synthetic with planted structure; real markets add nonstationarity, borrow constraints and regime change.</li>"
        "<li>Instances are small enough for exact enumeration, so a quantum advantage is not expected and none is claimed.</li>"
        "<li>The objective is a pairwise surrogate: it cannot see cointegration that only appears in triples.</li>"
        "<li>QAOA numbers are noiseless simulation; device noise, connectivity and queueing would only make them worse.</li>"
        "<li>Thresholds and weights in the config were not tuned on the test windows, but multiple-testing risk remains if they are iterated.</li>"
        "</ul>"
    )

    css_light = "".join(f"--s-{s}:{c};" for s, c in zip(SOLVER_ORDER, LIGHT))
    css_dark = "".join(f"--s-{s}:{c};" for s, c in zip(SOLVER_ORDER, DARK))
    dark_vars = f"color-scheme:dark;--surface:#1a1a19;--panel:#222220;--text:#f0efec;--text2:#c3c2b7;--muted:#9b9a90;--grid:#33332f;--line:#3d3d38;{css_dark}"
    page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Quantum-assisted stat arb: solver comparison ({esc(cfg['name'])})</title>
<style>
:root{{color-scheme:light;--surface:#fcfcfb;--panel:#f4f3f0;--text:#0b0b0b;--text2:#52514e;--muted:#6f6e69;--grid:#e6e5e0;--line:#d2d1ca;{css_light}}}
@media (prefers-color-scheme: dark){{:root{{{dark_vars}}}}}
*{{box-sizing:border-box}}
body{{margin:0;background:var(--surface);color:var(--text);font:15px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif}}
main{{max-width:1040px;margin:0 auto;padding:40px 24px 80px}}
h1{{font-size:26px;line-height:1.25;margin:0 0 6px;text-wrap:balance}}
h2{{font-size:19px;margin:44px 0 10px;padding-top:20px;border-top:1px solid var(--line)}}
h3{{font-size:15px;margin:26px 0 8px}}
p{{margin:8px 0;max-width:72ch}} .muted{{color:var(--muted);font-size:13px}}
.sub{{color:var(--text2);margin:0 0 20px}}
.lead{{background:var(--panel);border-left:3px solid var(--s-qaoa);padding:14px 18px;border-radius:4px;max-width:none}}
code{{font:13px ui-monospace,SFMono-Regular,Menlo,monospace;background:var(--panel);padding:1px 5px;border-radius:3px}}
figure{{margin:14px 0}} svg{{width:100%;height:auto;display:block;overflow:visible}}
.grid{{stroke:var(--grid);stroke-width:1}} .zero{{stroke:var(--line);stroke-width:1}} .cross{{stroke:var(--muted);stroke-width:1}}
.tick{{fill:var(--muted);font-size:11px;font-variant-numeric:tabular-nums}} .axis{{fill:var(--text2);font-size:12px}}
.label{{fill:var(--text);font-size:13px}} .value{{fill:var(--text2);font-size:12px;font-variant-numeric:tabular-nums}}
.tablewrap{{overflow-x:auto;margin:12px 0}}
table{{border-collapse:collapse;width:100%;font-size:13px}}
td:first-child{{white-space:nowrap}}
th,td{{text-align:left;padding:7px 10px;border-bottom:1px solid var(--grid);vertical-align:top}}
th{{color:var(--text2);font-weight:600;border-bottom:1px solid var(--line);white-space:nowrap}}
.num{{text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap}}
.sw{{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:7px;vertical-align:baseline}}
.legend{{display:flex;flex-wrap:wrap;gap:6px 18px;font-size:13px;color:var(--text2);margin:10px 0 0}}
#tip{{position:fixed;pointer-events:none;background:var(--text);color:var(--surface);padding:7px 10px;border-radius:5px;font-size:12px;line-height:1.5;display:none;z-index:9;max-width:320px;font-variant-numeric:tabular-nums}}
#tip .sw{{border:1px solid var(--surface)}}
ul{{padding-left:20px;max-width:72ch}}
</style></head><body><main>
<h1>Does a quantum-assisted basket selector beat classical methods at equal budget?</h1>
<p class="sub">Walk-forward comparison on the <code>{esc(cfg['name'])}</code> experiment · generated from {len(data['manifests'])} stored run(s)</p>
<p class="lead">{headline(summary)}</p>
{''.join(sec)}
</main><div id="tip"></div>
<script>
(function(){{
  const tip=document.getElementById('tip');
  const show=(html,e)=>{{tip.innerHTML=html;tip.style.display='block';
    const w=tip.offsetWidth,x=e.clientX+14+w>innerWidth?e.clientX-w-14:e.clientX+14;
    tip.style.left=x+'px';tip.style.top=(e.clientY+14)+'px';}};
  const hide=()=>{{tip.style.display='none';}};
  document.querySelectorAll('[data-tip]').forEach(el=>{{
    el.addEventListener('mousemove',e=>show(el.dataset.tip,e));el.addEventListener('mouseleave',hide);}});
  document.querySelectorAll('svg.linechart').forEach(svg=>{{
    const d=JSON.parse(svg.querySelector('script').textContent),hit=svg.querySelector('.hit'),cross=svg.querySelector('.cross');
    hit.addEventListener('mousemove',e=>{{
      const pt=svg.createSVGPoint();pt.x=e.clientX;pt.y=e.clientY;
      const x=pt.matrixTransform(svg.getScreenCTM().inverse()).x;
      let i=0,best=1e9;d.xs.forEach((v,k)=>{{const dd=Math.abs(v-x);if(dd<best){{best=dd;i=k;}}}});
      cross.setAttribute('x1',d.xs[i]);cross.setAttribute('x2',d.xs[i]);cross.setAttribute('visibility','visible');
      show('<strong>'+d.x[i]+'</strong><br>'+d.series.map(s=>'<span class="sw" style="background:var(--s-'+s.key+')"></span>'+s.name+': '+s.v[i]).join('<br>'),e);
    }});
    hit.addEventListener('mouseleave',()=>{{cross.setAttribute('visibility','hidden');hide();}});
  }});
}})();
</script></body></html>"""
    return page, summary


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    args = ap.parse_args(argv)
    data = load_runs(args.runs)
    page, summary = build_html(data)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(page)
    csv_path = args.output.with_suffix(".csv")
    summary.to_csv(csv_path, index=False)
    print(f"wrote {args.output} and {csv_path} ({len(summary)} solvers, {len(data['results'])} runs)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
