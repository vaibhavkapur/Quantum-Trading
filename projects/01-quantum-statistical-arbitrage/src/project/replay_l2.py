"""CLI: fit hedge weights on the warm-up part of an L2 stream, then replay it in the Rust engine.

    python -m project.replay_l2 --config configs/demo.yaml [--build]

The research layer (Python) only touches the warm-up segment; quote
processing, the signal state machine and execution run in ``engine/`` (Rust).
"""
from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .config import load_config, resolve
from .research.hedge import fit_hedge

L2_COLUMNS = ["ts_ns", "sym", "side", "level", "price", "size"]


def warmup_mids(quotes_path: Path, warmup_frac: float) -> tuple[pd.DataFrame, int]:
    """Top-of-book mids per tick for the warm-up segment, and its last timestamp."""
    df = pd.read_csv(quotes_path, names=L2_COLUMNS)
    ticks = np.sort(df["ts_ns"].unique())
    cutoff = int(ticks[int(len(ticks) * warmup_frac)])
    top = df[(df["level"] == 0) & (df["ts_ns"] <= cutoff)]
    wide = top.pivot_table(index=["ts_ns", "sym"], columns="side", values="price", aggfunc="last")
    mids = ((wide[0] + wide[1]) / 2).unstack("sym").sort_index().ffill().dropna()
    return mids, cutoff


def engine_binary(root: Path, build: bool) -> Path:
    binary = root / "engine" / "target" / "release" / "statarb-engine"
    if build or not binary.exists():
        subprocess.run(["cargo", "build", "--release", "--quiet"], cwd=root / "engine", check=True)
    return binary


def run(cfg: dict, runs_dir: Path, build: bool = False, warmup_frac: float = 0.3, latency_ns: int = 500_000,
        notional: float = 100_000.0, fee_bps: float = 1.0) -> Path:
    root = Path(cfg["_root"])
    quotes = resolve(cfg, cfg["data"]["dir"]) / "l2_quotes.csv"
    if not quotes.exists():
        raise FileNotFoundError(f"{quotes} not found - run project.prepare_data first (synthetic source only)")
    mids, cutoff = warmup_mids(quotes, warmup_frac)
    sample = np.log(mids.iloc[::20].to_numpy())  # thin the ticks so the Johansen fit is not dominated by microstructure noise
    fit = fit_hedge(sample, cfg["hedge"]["method"])
    binary = engine_binary(root, build)

    started = datetime.now(timezone.utc)
    run_id = f"{cfg['name']}-l2-replay-{started.strftime('%Y%m%dT%H%M%SZ')}"
    out = runs_dir / run_id
    out.mkdir(parents=True, exist_ok=False)
    sig = cfg["signal"]
    cmd = [
        str(binary), "replay", "--quotes", str(quotes),
        "--weights", ",".join(f"{w:.10f}" for w in fit.weights),
        "--warmup-until-ns", str(cutoff),
        "--entry", str(sig["entry_z"]), "--exit", str(sig["exit_z"]), "--stop", str(sig["stop_z"]),
        "--notional", str(notional), "--latency-ns", str(latency_ns), "--fee-bps", str(fee_bps),
        "--out", str(out / "engine_report.json"),
    ]
    subprocess.run(cmd, check=True)
    report = json.loads((out / "engine_report.json").read_text())
    manifest = {
        "run_id": run_id,
        "created_utc": started.isoformat(),
        "solver_family": "l2-replay",
        "engine": "statarb-engine (Rust, release build)",
        "command": cmd,
        "hedge": {"method": fit.method, "weights": [float(w) for w in fit.weights], "fit_ticks": int(len(sample)),
                  "warmup_frac": warmup_frac, "warmup_until_ns": cutoff, "train_half_life_samples": fit.half_life},
        "dataset": {"file": str(quotes), "kind": "synthetic L2 (see DATASET.json)"},
        "files": ["engine_report.json"],
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    lat = report["tick_to_decision_ns"]
    print(
        f"run {run_id}: {report['events']} events at {report['events_per_second'] / 1e6:.1f}M/s, "
        f"tick-to-decision p50 {lat['p50']} ns / p99 {lat['p99']} ns, {report['entries']} entries, pnl {report['pnl']:.2f} -> {out}"
    )
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--runs", type=Path, default=None)
    ap.add_argument("--build", action="store_true", help="(re)build the Rust engine first")
    ap.add_argument("--warmup-frac", type=float, default=0.3)
    ap.add_argument("--latency-ns", type=int, default=500_000)
    ap.add_argument("--notional", type=float, default=100_000.0)
    ap.add_argument("--fee-bps", type=float, default=1.0)
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    run(cfg, args.runs or resolve(cfg, "runs"), args.build, args.warmup_frac, args.latency_ns, args.notional, args.fee_bps)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
