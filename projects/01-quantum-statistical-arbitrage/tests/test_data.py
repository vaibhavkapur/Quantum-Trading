import json

import numpy as np
import pandas as pd
import pytest

from project.data import loader, synthetic
from project.prepare_data import prepare


def test_generation_is_deterministic():
    a, meta_a = synthetic.generate_daily(300, "2020-01-01", seed=5)
    b, meta_b = synthetic.generate_daily(300, "2020-01-01", seed=5)
    pd.testing.assert_frame_equal(a, b)
    pd.testing.assert_frame_equal(meta_a, meta_b)
    c, _ = synthetic.generate_daily(300, "2020-01-01", seed=6)
    assert not np.allclose(a["close"].to_numpy(), c["close"].to_numpy())


def test_generated_bars_are_valid_utc_ohlcv():
    df, assets = synthetic.generate_daily(400, "2020-01-01", seed=1)
    out = loader.normalize_ohlcv(df)  # raises on bad bars / duplicates
    assert str(out["ts"].dt.tz) == "UTC"
    assert (out["ts"].dt.hour == 21).all()
    assert set(assets["sector"]) == {"TECH", "FIN", "ENERGY", "UTIL"}


def test_membership_changes_through_time():
    df, assets = synthetic.generate_daily(1000, "2018-01-01", seed=1)
    counts = df.groupby("symbol").size()
    assert counts[synthetic.LATE_LISTING[0]] < 1000 and counts[synthetic.DELISTING[0]] < 1000
    assert counts["TEC1"] == 1000
    assert assets.set_index("symbol").loc[synthetic.DELISTING[0], "delisted"] is not pd.NaT


def test_timestamp_normalization_converts_local_time_to_utc():
    naive = pd.Series(pd.to_datetime(["2024-01-15 16:00", "2024-07-15 16:00"]))
    utc = loader.normalize_timestamps(naive, "America/New_York")
    assert list(utc.dt.hour) == [21, 20]  # EST vs EDT
    aware = pd.Series(pd.to_datetime(["2024-01-15 16:00"]).tz_localize("Asia/Tokyo"))
    assert loader.normalize_timestamps(aware).iloc[0] == pd.Timestamp("2024-01-15 07:00", tz="UTC")
    assert str(utc.dtype) == "datetime64[ns, UTC]"


def test_ambiguous_dst_timestamps_raise():
    with pytest.raises(Exception):
        loader.normalize_timestamps(pd.Series(pd.to_datetime(["2024-11-03 01:30"])), "America/New_York")


def _bars(**over):
    row = {"ts": "2024-01-02", "symbol": "X", "open": 10.0, "high": 11.0, "low": 9.0, "close": 10.5, "volume": 100}
    row.update(over)
    return pd.DataFrame([row])


def test_normalize_rejects_bad_input():
    with pytest.raises(ValueError, match="missing columns"):
        loader.normalize_ohlcv(_bars().drop(columns="volume"))
    with pytest.raises(ValueError, match="duplicate"):
        loader.normalize_ohlcv(pd.concat([_bars(), _bars()]))
    with pytest.raises(ValueError, match="non-positive"):
        loader.normalize_ohlcv(_bars(close=-1.0))
    with pytest.raises(ValueError, match="low <= open/close <= high"):
        loader.normalize_ohlcv(_bars(high=10.2))


def test_prepare_writes_versioned_dataset_and_detects_tampering(cfg):
    manifest = prepare(cfg)
    data_dir = loader.Path(cfg["_root"]) / cfg["data"]["dir"]
    assert manifest["version"].startswith("synthetic-v1-seed")
    assert set(manifest["files"]) == {"ohlcv.parquet", "assets.csv", "l2_quotes.csv"}
    ds = loader.load_dataset(data_dir)
    assert ds.close.shape == (900, 14)
    assert prepare(cfg)["files"] == manifest["files"]  # idempotent, no rewrite
    with (data_dir / "assets.csv").open("a") as fh:
        fh.write("EVIL,TECH,A,1,1,,\n")
    with pytest.raises(ValueError, match="immutable"):
        loader.load_dataset(data_dir)


def test_prepare_is_a_deterministic_regression(cfg):
    a = prepare(cfg)
    b = prepare(cfg, force=True)
    assert a["files"]["assets.csv"] == b["files"]["assets.csv"]
    assert a["files"]["l2_quotes.csv"] == b["files"]["l2_quotes.csv"]
    ds = loader.load_dataset(loader.Path(cfg["_root"]) / cfg["data"]["dir"])
    assert json.dumps(a["generator"], sort_keys=True) == json.dumps(b["generator"], sort_keys=True)
    # Golden values: any change to the generator or its RNG stream must be deliberate.
    assert float(ds.close["TEC1"].iloc[-1]) == pytest.approx(226.009932, abs=1e-5)
    assert float(np.nansum(ds.close.to_numpy())) == pytest.approx(1981615.2363, abs=1e-2)


def test_universe_is_point_in_time(cfg):
    prepare(cfg)
    ds = loader.load_dataset(loader.Path(cfg["_root"]) / cfg["data"]["dir"])
    late, gone = synthetic.LATE_LISTING[0], synthetic.DELISTING[0]
    early = ds.universe(0, 200)
    assert late not in early and gone in early            # not yet listed / still alive
    assert late in ds.universe(300, 600)
    assert gone not in ds.universe(600, 900)              # window spans the delisting
    assert gone in ds.universe(400, 700)                  # future delisting must not remove it earlier


def test_l2_stream_is_sorted_and_uncrossed():
    df, info = synthetic.generate_l2(5000, seed=3)
    assert df["ts_ns"].is_monotonic_increasing
    top = df[df["level"] == 0].pivot_table(index=["ts_ns", "sym"], columns="side", values="price")
    assert (top[1] > top[0]).all()
    assert (df["size"] > 0).all() and info["levels"] == 5
