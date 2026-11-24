"""Data preparation, point-in-time estimation, backtest and CLI: deterministic regression tests."""
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from hqpo import benchmark, prepare_data, report, run
from hqpo.config import expand_solvers, load_config
from hqpo.data.estimation import estimate, ledoit_wolf
from hqpo.data.prices import load_dataset, normalize_prices, sha256_file
from hqpo.data.synthetic import generate_price_panel
from hqpo.evaluation.backtest import run_backtest, signal_dates
from hqpo.evaluation.metrics import mean_ci, performance

ROOT = Path(__file__).resolve().parents[1]
REGRESSION_FIRST_ROW = [104.76, 100.42, 102.86, 101.67]   # generate_price_panel(4, 2, 60, seed=3)


@pytest.fixture(scope="module")
def workspace(tmp_path_factory):
    """A throwaway project root with the smoke config, so tests never touch runs/ or data/."""
    root = tmp_path_factory.mktemp("proj")
    (root / "configs").mkdir()
    shutil.copy(ROOT / "configs" / "smoke.yaml", root / "configs" / "smoke.yaml")
    cfg_path = root / "configs" / "smoke.yaml"
    prepare_data.main(["--config", str(cfg_path)])
    return root, cfg_path


def test_synthetic_panel_is_deterministic_regression():
    a, sec = generate_price_panel(n_assets=4, n_sectors=2, n_days=60, seed=3)
    b, _ = generate_price_panel(n_assets=4, n_sectors=2, n_days=60, seed=3)
    pd.testing.assert_frame_equal(a, b)
    assert a.shape == (60, 4) and (a > 0).all().all() and a.index.is_monotonic_increasing
    assert a.index.dayofweek.max() <= 4
    assert sec == {"SYN00": "SEC0", "SYN01": "SEC1", "SYN02": "SEC0", "SYN03": "SEC1"}
    # pinned values: a change here means the dataset (and every stored result) changed
    assert a.iloc[0].round(2).tolist() == pytest.approx(REGRESSION_FIRST_ROW, abs=0.011)
    c, _ = generate_price_panel(n_assets=4, n_sectors=2, n_days=60, seed=4)
    assert not np.allclose(a.to_numpy(), c.to_numpy())


def test_prepared_dataset_matches_manifest_and_detects_tampering(workspace):
    root, cfg_path = workspace
    cfg = load_config(cfg_path)
    path = root / cfg["data"]["path"]
    prices, manifest = load_dataset(path)
    assert manifest["sha256"] == sha256_file(path) and manifest["source"] == "synthetic"
    assert "license" in manifest and manifest["rows"] == len(prices)
    first = manifest["sha256"]
    prepare_data.main(["--config", str(cfg_path)])               # byte-identical on re-run
    assert json.loads((root / "data" / "smoke_prices.manifest.json").read_text())["sha256"] == first
    tampered = root / "data" / "tampered.csv"
    shutil.copy(path, tampered)
    shutil.copy(root / "data" / "smoke_prices.manifest.json", root / "data" / "tampered.manifest.json")
    tampered.write_text(tampered.read_text().replace("100.", "101.", 1))
    with pytest.raises(ValueError, match="manifest"):
        load_dataset(tampered)
    with pytest.raises(FileNotFoundError):
        load_dataset(root / "data" / "missing.csv")


def test_timestamp_normalisation():
    idx = pd.to_datetime(["2024-01-03 21:00", "2024-01-02 21:00", "2024-01-03 21:00", "2024-01-04 21:00",
                          "2024-01-05 21:00"]).tz_localize("America/New_York")
    df = pd.DataFrame({"A": [3.0, 1.0, 3.5, np.nan, 5.0], "B": [1.0, 1.0, 1.0, 1.0, 1.0],
                       "C": [np.nan, 2.0, 2.0, 2.0, 2.0]}, index=idx)
    out = normalize_prices(df)
    assert out.index.tz is None and out.index.is_monotonic_increasing and out.index.is_unique
    assert str(out.index[0].date()) == "2024-01-03"              # 21:00 New York is next day UTC
    assert out["A"].tolist() == [1.0, 3.5, 3.5, 5.0]             # last duplicate wins, gap ffilled
    with pytest.raises(ValueError):
        normalize_prices(pd.DataFrame({"A": [1.0, -1.0]}, index=pd.to_datetime(["2024-01-02", "2024-01-03"])))
    with pytest.raises(ValueError):
        normalize_prices(pd.DataFrame({"A": [np.nan, 1.0]}, index=pd.to_datetime(["2024-01-02", "2024-01-03"])))


def test_estimation_is_point_in_time():
    prices, _ = generate_price_panel(n_assets=4, n_sectors=2, n_days=300, seed=1)
    cfg = {"lookback_days": 120, "covariance": "ledoit_wolf", "expected_return": "shrunk_mean"}
    asof = prices.index[200]
    s1, m1, info = estimate(prices, asof, cfg)
    future = prices.copy()
    future.iloc[201:] *= 5.0                                      # corrupt everything after asof
    s2, m2, _ = estimate(future, asof, cfg)
    assert np.array_equal(s1, s2) and np.array_equal(m1, m2)
    assert info["window_end"] == str(asof.date()) and info["n_obs"] == 120
    assert np.linalg.eigvalsh(s1).min() > 0
    assert np.all(estimate(prices, asof, {**cfg, "expected_return": "zero"})[1] == 0)
    with pytest.raises(ValueError):
        estimate(prices, prices.index[3], cfg)


def test_ledoit_wolf_shrinks_towards_identity(rng):
    X = rng.normal(size=(30, 10))
    cov, k = ledoit_wolf(X)
    S = np.cov(X, rowvar=False, ddof=0)
    assert 0 < k <= 1 and np.trace(cov) == pytest.approx(np.trace(S))
    assert np.linalg.cond(cov) < np.linalg.cond(S)


def test_backtest_regression_and_no_lookahead(workspace):
    root, cfg_path = workspace
    cfg = load_config(cfg_path)
    prices, manifest = load_dataset(root / cfg["data"]["path"])
    a = run_backtest(cfg, prices, manifest["sectors"], "greedy", None)
    b = run_backtest(cfg, prices, manifest["sectors"], "greedy", None)
    pd.testing.assert_frame_equal(a["daily"], b["daily"])
    reb, daily = a["rebalances"], a["daily"]
    assert list(reb["signal_date"]) == signal_dates(cfg, prices)
    assert (pd.to_datetime(reb["trade_date"]) > pd.to_datetime(reb["signal_date"])).all()
    assert reb["signal_date"].min() > pd.Timestamp(cfg["splits"]["train_end"])
    assert set(reb["split"]) == {"validation", "test"} and reb["feasible"].all()
    assert daily.index[0] == reb["trade_date"].iloc[0] and daily["gross_ret"].iloc[0] == 0.0
    assert daily["cost"].iloc[0] == pytest.approx(cfg["backtest"]["cost_bps"] / 1e4)   # cash -> invested
    assert daily["cost"].sum() == pytest.approx(reb["cost"].sum())
    assert (a["weights"].drop(columns="signal_date").sum(axis=1) - 1).abs().max() < 1e-12
    # corrupting prices after the first trade must not change the first decision
    cut = prices.index.get_loc(reb["trade_date"].iloc[0])
    future = prices.copy()
    future.iloc[cut + 1:] *= np.linspace(1, 3, len(prices) - cut - 1)[:, None]
    c = run_backtest(cfg, future, manifest["sectors"], "greedy", None)
    assert a["weights"].iloc[0].equals(c["weights"].iloc[0])
    for split in ("validation", "test"):
        assert a["summary"][split]["n_rebalances"] >= 1 and np.isfinite(a["summary"][split]["sharpe"])


def test_metrics():
    d = pd.DataFrame({"ret": [0.01, -0.02, 0.03, 0.0], "bench": [0.0, 0.0, 0.0, 0.0]})
    m = performance(d)
    assert m["max_drawdown"] == pytest.approx(-0.02) and m["tracking_error"] == pytest.approx(m["ann_vol"])
    assert mean_ci([1.0, 1.0, 1.0]) == (1.0, 1.0, 1.0)
    mean, lo, hi = mean_ci([1.0, 2.0, 3.0, np.nan])
    assert lo < mean == 2.0 < hi


def test_config_groups_and_validation(workspace, tmp_path):
    _, cfg_path = workspace
    cfg = load_config(cfg_path)
    assert expand_solvers(cfg, "classical") == ["cvxpy-relax", "miqp", "greedy", "sa"]
    assert expand_solvers(cfg, "quantum-sim,greedy") == ["qaoa", "greedy"]
    assert "qubo-sa" in expand_solvers(cfg, "all")
    with pytest.raises(ValueError):
        expand_solvers(cfg, "warp-drive")
    bad = tmp_path / "configs" / "bad.yaml"
    bad.parent.mkdir()
    bad.write_text("solvers: {classical: [nope]}\n")
    with pytest.raises(ValueError, match="unknown solver"):
        load_config(bad)


@pytest.mark.slow
def test_cli_end_to_end(workspace, capsys):
    """The README's CLI contract on the reduced config: prepare -> run -> benchmark -> report."""
    root, cfg_path = workspace
    runs = root / "runs"
    run.main(["--config", str(cfg_path), "--solver", "classical", "--runs", str(runs)])
    run.main(["--config", str(cfg_path), "--solver", "quantum-sim", "--runs", str(runs)])
    benchmark.main(["--config", str(cfg_path), "--runs", str(runs)])
    out = root / "reports" / "comparison.html"
    report.main(["--runs", str(runs), "--output", str(out)])

    bt = runs / "smoke" / "backtest"
    assert (bt / "miqp" / "deterministic" / "summary.json").exists()
    assert sorted(p.name for p in (bt / "qaoa").iterdir()) == ["seed-11", "seed-23"]
    man = json.loads((bt / "qaoa" / "seed-11" / "manifest.json").read_text())
    assert man["solver"] == "qaoa" and man["seed"] == 11 and man["dataset"]["sha256"]
    assert man["packages"]["numpy"] and "no QPU" in man["hardware"]

    res = pd.read_csv(runs / "smoke" / "benchmark" / "results.csv")
    small = res[res["n"] == 4]
    assert (small["reference_kind"] == "brute-force").all()
    exact = small[small["solver"].isin(["miqp", "cpsat"])]
    assert exact["gap"].abs().max() < 1e-7
    assert (res.loc[res["feasible"], "gap"] > -1e-7).all()
    big_qaoa = res[(res["n"] == 12) & (res["solver"] == "qaoa")]
    assert (big_qaoa["status"] == "skipped").all()               # reported, not dropped
    sweep = pd.read_csv(runs / "smoke" / "benchmark" / "penalty_sweep.csv")
    assert sweep.loc[sweep["scale"] == 1.0, "ground_state_optimal"].all()

    page = out.read_text()
    assert "No QPU was used" in page and "plotly" in page.lower() and page.count("<table>") >= 3
    assert "| miqp |" in (root / "reports" / "benchmark_table_smoke.md").read_text()
