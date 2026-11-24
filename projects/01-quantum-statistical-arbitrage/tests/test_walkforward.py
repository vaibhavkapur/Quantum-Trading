import json

import numpy as np
import pandas as pd
import pytest

from project.config import load_config, validate, deep_merge, DEFAULTS, experiment_fingerprint
from project.data.loader import load_dataset, Path
from project.prepare_data import prepare
from project.research import walkforward as wf
from project import report, run as run_cli


@pytest.fixture
def ds(cfg):
    prepare(cfg)
    return load_dataset(Path(cfg["_root"]) / cfg["data"]["dir"])


def test_folds_are_chronological_and_tests_do_not_overlap():
    folds = wf.make_folds(1700, {"train_days": 378, "val_days": 126, "test_days": 126, "step_days": 126})
    assert len(folds) == 9
    for f in folds:
        assert f.train[1] == f.val[0] and f.val[1] == f.test[0] and f.train[0] < f.train[1]
    assert all(a.test[1] <= b.test[0] for a, b in zip(folds, folds[1:]))
    assert wf.make_folds(100, {"train_days": 378, "val_days": 126, "test_days": 126, "step_days": 126}) == []
    with pytest.raises(ValueError, match="twice"):
        wf.make_folds(1700, {"train_days": 378, "val_days": 126, "test_days": 126, "step_days": 60})


def test_no_lookahead_selection_and_hedge_ignore_future_prices(ds, cfg):
    """Corrupting every price from the validation window onward must not change the QUBO;
    corrupting the test window must not change hedge weights or the gate."""
    fold = wf.make_folds(len(ds.dates), cfg["walkforward"])[0]
    base = wf.prepare_fold(ds, fold, cfg)
    basket = base.problem.decode(base.opt_z)
    row, _ = wf.evaluate_basket(ds, base, basket, cfg)

    tampered = ds.close.copy()
    tampered.iloc[fold.val[0]:] *= 1.0 + np.linspace(0.1, 3.0, len(tampered) - fold.val[0])[:, None]
    after_val = type(ds)(tampered, ds.open, ds.volume, ds.assets, ds.manifest)
    again = wf.prepare_fold(after_val, fold, cfg)
    np.testing.assert_array_equal(again.problem.quad, base.problem.quad)
    np.testing.assert_array_equal(again.problem.linear, base.problem.linear)

    tampered = ds.close.copy()
    tampered.iloc[fold.test[0]:] *= 1.7
    after_test = type(ds)(tampered, ds.open, ds.volume, ds.assets, ds.manifest)
    row2, _ = wf.evaluate_basket(after_test, wf.prepare_fold(after_test, fold, cfg), basket, cfg)
    for key in ("weights", "train_half_life", "val_adf_pvalue", "gate_pass", "johansen_trace"):
        assert row2[key] == row[key]


def test_study_rows_are_consistent(ds, cfg):
    solvers = ["brute_force", "greedy", "random_search"]
    results, returns = wf.run_study(ds, cfg, solvers, log=lambda *_: None)
    n_folds = len(wf.make_folds(len(ds.dates), cfg["walkforward"]))
    assert len(results) == n_folds * (1 + 1 + len(cfg["seeds"]))
    assert (results.loc[results.solver == "brute_force", "norm_gap"] == 0).all()
    assert (results["norm_gap"] >= -1e-9).all() and results["is_optimal"].dtype == bool
    assert (results.loc[~results["gate_pass"], "test_pnl"] == 0).all()
    assert not results["failed"].any()
    # Same basket in the same fold -> identical trading outcome regardless of which solver found it.
    for _, g in results.groupby(["fold", "basket"]):
        assert g["test_pnl"].nunique() == 1
    per_path = returns.groupby(["solver", "seed"]).size()
    assert (per_path == n_folds * cfg["walkforward"]["test_days"]).all()
    for w in results["weights"]:
        assert sum(abs(v) for v in json.loads(w).values()) == pytest.approx(1.0, abs=1e-4)


def test_study_is_reproducible(ds, cfg):
    a, ra = wf.run_study(ds, cfg, ["greedy", "random_search"], log=lambda *_: None)
    b, rb = wf.run_study(ds, cfg, ["greedy", "random_search"], log=lambda *_: None)
    timing = [c for c in a.columns if c.startswith("t_")]
    pd.testing.assert_frame_equal(a.drop(columns=timing), b.drop(columns=timing))
    pd.testing.assert_frame_equal(ra, rb)


def test_config_validation_and_fingerprint(tmp_path):
    with pytest.raises(ValueError, match="exit_z < entry_z < stop_z"):
        validate(deep_merge(DEFAULTS, {"signal": {"entry_z": 0.1}}))
    with pytest.raises(ValueError, match="seed"):
        validate(deep_merge(DEFAULTS, {"seeds": []}))
    (tmp_path / "configs").mkdir()
    path = tmp_path / "configs" / "x.yaml"
    path.write_text("name: x\nbudget: {max_evals: 8}\n")
    cfg = load_config(path)
    assert cfg["budget"] == {"max_evals": 8, "time_limit_s": 5.0} and cfg["_root"] == str(tmp_path.resolve())
    same = deep_merge(cfg, {"solvers": {"quantum": {"p": 3}}})
    other = deep_merge(cfg, {"budget": {"max_evals": 9}})
    assert experiment_fingerprint(cfg) == experiment_fingerprint(same) != experiment_fingerprint(other)


@pytest.mark.slow
def test_cli_pipeline_end_to_end(cfg, tmp_path):
    prepare(cfg)
    runs = tmp_path / "runs"
    c_dir = run_cli.execute(cfg, "classical", runs, log=lambda *_: None)
    q_dir = run_cli.execute(cfg, "quantum-sim", runs, log=lambda *_: None)
    for d in (c_dir, q_dir):
        manifest = json.loads((d / "manifest.json").read_text())
        assert manifest["fingerprint"] == experiment_fingerprint(cfg) and manifest["n_failed"] == 0
        assert all((d / name).exists() for name in manifest["files"])
        assert manifest["dataset"]["files"]["ohlcv.parquet"]
    assert "none" in json.loads((q_dir / "manifest.json").read_text())["quantum_hardware"]

    # A run with a different budget must be excluded from the comparison, not silently mixed in.
    other = deep_merge(cfg, {"name": "aaa-other", "budget": {"max_evals": 4}, "solvers": {"classical": ["greedy"]}})
    run_cli.execute(other, "classical", runs, log=lambda *_: None)
    c2 = run_cli.execute(cfg, "classical", runs / "later", log=lambda *_: None)  # keep latest fingerprint = cfg
    c2.rename(runs / (c2.name + "-b"))

    out = tmp_path / "reports" / "comparison.html"
    assert report.main(["--runs", str(runs), "--output", str(out)]) == 0
    page = out.read_text()
    assert "QAOA (simulator)" in page and "CP-SAT" in page and "different experiment fingerprint" in page
    summary = pd.read_csv(out.with_suffix(".csv"))
    assert set(summary["solver"]) == {"qaoa", "cpsat", "greedy", "random_search", "simulated_annealing", "brute_force"}
    assert summary.set_index("solver").loc["cpsat", "p_optimal"] == 1.0
    with pytest.raises(ValueError, match="unknown solver family"):
        run_cli.execute(cfg, "annealer", runs)
