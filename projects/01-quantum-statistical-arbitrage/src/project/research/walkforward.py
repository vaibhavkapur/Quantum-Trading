"""Walk-forward study: fit on train, gate on validation, trade on test, roll forward."""
from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..backtest.engine import ZScoreStrategy, run_backtest
from ..backtest.metrics import annualized_turnover, max_drawdown, sharpe
from ..data.loader import Dataset
from ..solvers.base import Budget, SolverResult
from ..solvers.registry import STOCHASTIC, run_solver
from .hedge import fit_hedge, validation_gate
from .qubo import BasketProblem, build_problem
from .stats import adf_pvalue, half_life, market_betas, pair_stats

DETERMINISTIC_SEED = -1


@dataclass
class Fold:
    index: int
    train: tuple[int, int]
    val: tuple[int, int]
    test: tuple[int, int]


def make_folds(n_bars: int, wf: dict) -> list[Fold]:
    """Chronological, non-overlapping test windows; train < validation < test in every fold."""
    folds, start = [], 0
    span = wf["train_days"] + wf["val_days"] + wf["test_days"]
    while start + span <= n_bars:
        a = start + wf["train_days"]
        b = a + wf["val_days"]
        folds.append(Fold(len(folds), (start, a), (a, b), (b, b + wf["test_days"])))
        start += wf["step_days"]
    if wf["step_days"] < wf["test_days"]:
        raise ValueError("step_days < test_days would trade the same bars twice")
    return folds


@dataclass
class FoldContext:
    fold: Fold
    problem: BasketProblem
    feasible_objectives: np.ndarray  # sorted objective of every feasible basket
    opt_z: np.ndarray
    betas: np.ndarray


def prepare_fold(ds: Dataset, fold: Fold, cfg: dict) -> FoldContext:
    """Build the selection problem from the TRAIN window only."""
    universe = ds.universe(fold.train[0], fold.val[1])
    logp = np.log(ds.close[universe].iloc[fold.train[0] : fold.train[1]].to_numpy())
    stats = pair_stats(logp, hl_cap=cfg["gates"]["max_half_life"])
    betas = market_betas(logp)
    sector_codes = pd.Categorical(ds.assets.loc[universe, "sector"]).codes
    problem = build_problem(
        universe,
        stats,
        betas,
        np.asarray(sector_codes),
        ds.assets.loc[universe, "spread_bps"].to_numpy(float),
        cfg["selection"],
        hl_cap=cfg["gates"]["max_half_life"],
    )
    values = []
    for combo in problem.feasible_baskets():
        idx = list(combo)
        values.append(problem.linear[idx].sum() + np.triu(problem.quad[np.ix_(idx, idx)], 1).sum())
    opt_z, _, _ = problem.brute_force()
    return FoldContext(fold, problem, np.sort(np.array(values)), opt_z, betas)


def evaluate_basket(ds: Dataset, ctx: FoldContext, basket: list[str], cfg: dict) -> tuple[dict, pd.Series]:
    """Hedge ratios (train) -> validation gate (frozen weights) -> test backtest."""
    f = ctx.fold
    close = ds.close[basket]
    logp_train = np.log(close.iloc[f.train[0] : f.train[1]].to_numpy())
    logp_val = np.log(close.iloc[f.val[0] : f.val[1]].to_numpy())
    test_close = close.iloc[f.test[0] : f.test[1]]
    test_open = ds.open[basket].iloc[f.test[0] : f.test[1]]
    flat = pd.Series(0.0, index=test_close.index)

    fit = fit_hedge(logp_train, cfg["hedge"]["method"])
    passed, gate = validation_gate(fit, logp_val, cfg["gates"])
    idx = [ctx.problem.symbols.index(s) for s in basket]
    row = {
        "weights": json.dumps({s: round(float(w), 6) for s, w in zip(basket, fit.weights)}),
        "hedge_method": fit.method,
        "johansen_trace": fit.trace_stat,
        "johansen_crit95": fit.trace_crit95,
        "train_half_life": fit.half_life,
        "train_adf_insample": fit.in_sample_adf,
        "net_exposure": fit.net_exposure,
        "beta_exposure": float(fit.weights @ ctx.betas[idx]),
        "val_adf_pvalue": gate["val_adf_pvalue"],
        "val_half_life": gate["val_half_life"],
        "gate_pass": passed,
        "gate_reasons": ",".join(gate["reasons"]),
    }
    if "true_group" in ds.assets.columns:
        groups = ds.assets.loc[basket, "true_group"]
        row["truth_all_group_a"] = bool((groups == "A").all())

    if not passed:
        row.update({"traded": False, "test_pnl": 0.0, "n_entries": 0, "turnover": 0.0, "kill_switch": False,
                    "cost_total": 0.0, "test_half_life": float("nan"), "test_adf_pvalue": float("nan"),
                    "time_in_market": 0.0, "test_sharpe": 0.0, "test_max_dd": 0.0, "exit_reasons": "{}"})
        return row, flat

    # Signal parameters use train + validation with frozen weights - still strictly before the test window.
    pre = np.concatenate([logp_train, logp_val]) @ fit.weights
    strategy = ZScoreStrategy(fit.weights, float(pre.mean()), float(pre.std(ddof=1)), fit.half_life, cfg["signal"])
    adv = ds.volume[basket].iloc[f.train[0] : f.train[1]].mean().to_numpy(float)
    result = run_backtest(
        test_open,
        test_close,
        strategy,
        ds.assets.loc[basket, "spread_bps"].to_numpy(float),
        ds.assets.loc[basket, "borrow_bps_annual"].to_numpy(float),
        adv,
        cfg["risk"],
        cfg["execution"],
    )
    st = result.stats
    live = test_close.dropna()
    test_spread = np.log(live.to_numpy()) @ fit.weights
    row.update(
        {
            "traded": True,
            "test_pnl": st["pnl"],
            "n_entries": st["n_entries"],
            "turnover": annualized_turnover(st["traded_notional"], cfg["risk"]["capital"], len(test_close)),
            "kill_switch": st["kill_switch"],
            "cost_total": st["fees"] + st["spread_cost"] + st["impact_cost"] + st["borrow_cost"],
            "test_half_life": half_life(test_spread),
            "test_adf_pvalue": adf_pvalue(test_spread)[1],
            "time_in_market": st["time_in_market"],
            "test_sharpe": sharpe(result.returns.to_numpy()),
            "test_max_dd": max_drawdown(result.returns.to_numpy()),
            "exit_reasons": json.dumps(st["exit_reasons"]),
        }
    )
    return row, result.returns


def solver_row(ctx: FoldContext, res: SolverResult, seed: int) -> dict:
    vals, opt = ctx.feasible_objectives, float(ctx.feasible_objectives[0])
    median = float(np.median(vals))
    denom = median - opt
    t = res.timings
    return {
        "solver": res.solver,
        "seed": seed,
        "fold": ctx.fold.index,
        "n_assets": ctx.problem.n,
        "n_feasible": len(vals),
        "basket": "|".join(ctx.problem.decode(res.z)),
        "objective": res.objective,
        "opt_objective": opt,
        "is_optimal": bool(res.objective <= opt + 1e-9),
        # 0 = optimum, 1 = as good as the median feasible basket.
        "norm_gap": float((res.objective - opt) / denom) if denom > 0 else 0.0,
        # Same scale for the best candidate that was feasible as sampled (nan if none was).
        "raw_norm_gap": float((res.raw_best_objective - opt) / denom) if denom > 0 else 0.0,
        "raw_found_feasible": bool(np.isfinite(res.raw_best_objective)),
        "pct_better": float(np.searchsorted(vals, res.objective - 1e-9) / len(vals)),
        "status": res.status,
        "failed": res.status.startswith("fallback"),
        "raw_feasible_rate": res.raw_feasible_rate,
        "repaired": res.repaired,
        "n_evals": res.n_evals,
        "n_shots": res.n_shots,
        "t_encode": t.get("encode", 0.0),
        "t_compile": t.get("compile", 0.0),
        "t_optimize": t.get("optimize", 0.0),
        "t_sample": t.get("sample", 0.0),
        "t_decode_repair": t.get("decode_repair", 0.0),
        "t_total": res.wall_time,
        "meta": json.dumps(res.meta, default=str),
    }


def run_study(ds: Dataset, cfg: dict, solvers: list[str], solver_kwargs: dict | None = None, log=print) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run every (fold, solver, seed). Returns (results, daily_returns)."""
    budget = Budget(cfg["budget"]["max_evals"], cfg["budget"]["time_limit_s"])
    folds = make_folds(len(ds.dates), cfg["walkforward"])
    if not folds:
        raise ValueError("dataset is too short for one train/validation/test fold")
    rows, rets = [], []
    for fold in folds:
        ctx = prepare_fold(ds, fold, cfg)
        cache: dict[tuple, tuple[dict, pd.Series]] = {}
        dates = ds.dates
        for name in solvers:
            seeds = cfg["seeds"] if name in STOCHASTIC else [DETERMINISTIC_SEED]
            for seed in seeds:
                res = run_solver(name, ctx.problem, budget, None if seed == DETERMINISTIC_SEED else seed,
                                 **((solver_kwargs or {}).get(name, {})))
                basket = ctx.problem.decode(res.z)
                key = tuple(basket)
                if key not in cache:
                    cache[key] = evaluate_basket(ds, ctx, basket, cfg)
                trade_row, series = cache[key]
                rows.append(
                    {
                        **solver_row(ctx, res, seed),
                        "train_start": str(dates[fold.train[0]].date()),
                        "test_start": str(dates[fold.test[0]].date()),
                        "test_end": str(dates[fold.test[1] - 1].date()),
                        **trade_row,
                    }
                )
                rets.append(pd.DataFrame({"solver": name, "seed": seed, "fold": fold.index, "ts": series.index, "ret": series.to_numpy()}))
        done = [r for r in rows if r["fold"] == fold.index]
        log(
            f"fold {fold.index}: test {done[0]['test_start']}..{done[0]['test_end']}  N={ctx.problem.n}  "
            + "  ".join(f"{s}:gap={np.mean([r['norm_gap'] for r in done if r['solver'] == s]):.3f}" for s in solvers)
        )
    return pd.DataFrame(rows), pd.concat(rets, ignore_index=True)
