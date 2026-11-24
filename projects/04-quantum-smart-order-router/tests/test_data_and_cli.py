import json

import numpy as np
import pytest
import yaml

from conftest import venue_dicts
from qsor.config import load_config
from qsor.core import ASK, BID
from qsor.exchange_sim.data import Dataset, dataset_dir, generate_arrays, prepare


def write_cfg(tmp_path, **over):
    cfg = {"name": "tiny", "venues": venue_dicts(2, fade_coeff=0.5, est_fade_coeff=0.5, order_fee=0.3,
                                                 outage_rate_per_min=2.0),
           "data": {"seed": 3, "steps": 3000}, "seeds": [1, 2],
           "orders": {"rate_per_s": 6.0, "sizes_lots": [2, 4]},
           "optimizer": {"buckets_lots": [2, 4], "calibration_probes": 40, "nonfill_penalty_grid": [1.0, 2.0]},
           "solvers": {"qaoa": {"layers": 1, "maxiter": 8, "shots": 32}}}
    cfg.update(over)
    (tmp_path / "configs").mkdir(exist_ok=True)
    path = tmp_path / "configs" / "tiny.yaml"
    path.write_text(yaml.safe_dump(cfg))
    return path


def test_generation_is_deterministic_and_well_formed(tmp_path):
    cfg = load_config(write_cfg(tmp_path))
    a, b = generate_arrays(cfg), generate_arrays(cfg)
    assert all(np.array_equal(a[k], b[k]) for k in a)
    assert (a["px"][ASK][:, :, 0] > a["px"][BID][:, :, 0]).all()            # never crossed within a venue
    assert (a["px"][BID].max(axis=(1, 2)) < a["px"][ASK].min(axis=(1, 2))).all()  # nor across venues
    assert (np.diff(a["px"][ASK], axis=2) == 1).all() and (np.diff(a["px"][BID], axis=2) == -1).all()
    assert (a["sz"] >= 0).all() and a["status"].any()
    other = generate_arrays(load_config(write_cfg(tmp_path, data={"seed": 4, "steps": 3000})))
    assert not np.array_equal(a["sz"], other["sz"])


def test_generation_regression_golden_values(tmp_path):
    """Pinned against the versions in requirements.txt (NumPy's PCG64 stream)."""
    a = generate_arrays(load_config(write_cfg(tmp_path)))
    got = [int(a["mid"][-1]), int(a["sz"].sum()), int(a["flow"].sum()), int(a["status"].sum())]
    assert got == GOLDEN_DATA


def test_dataset_is_immutable_versioned_and_checksummed(tmp_path):
    path = write_cfg(tmp_path)
    cfg = load_config(path)
    out = prepare(cfg)
    assert out == dataset_dir(cfg) and out.parent == tmp_path / "data"
    stamp = (out / "book.npz").stat().st_mtime_ns
    assert prepare(cfg) == out and (out / "book.npz").stat().st_mtime_ns == stamp   # not rewritten
    ds = Dataset(out)
    s = ds.splits
    assert s["train"][0] == 0 and s["train"][1] == s["validation"][0] and s["validation"][1] == s["test"][0]
    assert s["test"][1] == ds.T == 3000
    assert ds.manifest["license"].startswith("Synthetic")
    other = prepare(load_config(write_cfg(tmp_path, data={"seed": 4, "steps": 3000})))
    assert other != out                                                             # new config -> new version
    with open(out / "book.npz", "ab") as f:
        f.write(b"tamper")
    with pytest.raises(ValueError, match="checksum"):
        Dataset(out)


def test_config_validation(tmp_path):
    with pytest.raises(ValueError, match="unknown venue keys"):
        load_config(write_cfg(tmp_path, venues=[{"name": "A", "latncy_ms": 1}]))
    with pytest.raises(ValueError, match="at least one venue"):
        load_config(write_cfg(tmp_path, venues=[]))


def test_cli_end_to_end(tmp_path, capsys):
    from project import prepare_data, report, run
    from qsor.optimizer import registry

    path = str(write_cfg(tmp_path))
    with pytest.raises(FileNotFoundError, match="prepare_data"):
        run.main(["--config", path, "--solver", "greedy"])
    with pytest.raises(ValueError, match="unknown solver"):
        run.main(["--config", path, "--solver", "dwave"])
    prepare_data.main(["--config", path])
    run.main(["--config", path, "--solver", "classical"])
    run.main(["--config", path, "--solver", "quantum-sim"])
    out = tmp_path / "reports" / "comparison.html"
    report.main(["--runs", str(tmp_path / "runs"), "--output", str(out)])

    exp = next(p for p in (tmp_path / "runs").iterdir() if p.name.startswith("tiny-"))
    solvers = sorted(p.name for p in exp.iterdir())
    assert solvers == sorted(registry.ALIASES["classical"] + registry.ALIASES["quantum-sim"])
    man = json.loads((exp / "qaoa" / "manifest.json").read_text())
    assert man["seeds"] == {"1": man["seeds"]["1"], "2": man["seeds"]["2"]}
    assert all(s["status"] == "ok" for s in man["seeds"].values())
    assert man["calibration"]["fitted_on"] == "train" and man["window"] == "test"
    assert man["dataset"]["sha256"] and "qiskit" in man["versions"]
    for f in ("metrics.json", "parents.csv", "decision_latency_ns.csv", "metrics.prom"):
        assert (exp / "qaoa" / "seed_1" / f).stat().st_size > 0
    assert (exp / "qaoa" / "seed_1" / "messages.fix").read_text().startswith("35=D|")
    assert "sor_decision_latency_microseconds_bucket" in (exp / "qaoa" / "seed_1" / "metrics.prom").read_text()

    page = out.read_text()
    for s in solvers:
        assert f"<b>{s}</b>" in page
    assert "no quantum hardware was used" in page
    table = out.with_suffix(".csv").read_text().splitlines()
    assert len(table) == 1 + len(solvers) and "is_bps_mean" in table[0]

    # same seeds, same data, same objective: stored results are reproducible
    m1 = json.loads((exp / "exact" / "seed_1" / "metrics.json").read_text())
    run.main(["--config", path, "--solver", "exact"])
    m2 = json.loads((exp / "exact" / "seed_1" / "metrics.json").read_text())
    assert m1["is_bps"] == m2["is_bps"] and m1["children"] == m2["children"]
    assert m1["is_bps"] == pytest.approx(GOLDEN_IS_BPS, abs=1e-9)


def test_failed_seed_is_recorded_not_hidden(tmp_path, monkeypatch):
    from project import prepare_data, report, run
    from qsor import experiment

    path = str(write_cfg(tmp_path))
    prepare_data.main(["--config", path])
    experiment.calibrate(load_config(path), experiment.load_for(load_config(path)))
    real = experiment.simulate

    def flaky(cfg, ds, window, solver, seed, model, **kw):
        if seed == 2:
            raise RuntimeError("simulated crash")
        return real(cfg, ds, window, solver, seed, model, **kw)

    monkeypatch.setattr(experiment, "simulate", flaky)
    run.main(["--config", path, "--solver", "greedy"])
    exp = next(p for p in (tmp_path / "runs").iterdir() if p.name.startswith("tiny-"))
    man = json.loads((exp / "greedy" / "manifest.json").read_text())
    assert man["seeds"]["1"]["status"] == "ok" and man["seeds"]["2"]["status"] == "failed"
    out = tmp_path / "r.html"
    report.main(["--runs", str(tmp_path / "runs"), "--output", str(out)])
    assert "<td>1 / 1</td>" in out.read_text()


GOLDEN_DATA = [10011, 461748, 1181, 50]
GOLDEN_IS_BPS = 1.616861216717463
