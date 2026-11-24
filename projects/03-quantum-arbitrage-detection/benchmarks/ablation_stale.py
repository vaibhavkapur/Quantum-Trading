"""Ablation: what the stale-quote timeout is worth.

    python benchmarks/ablation_stale.py --config configs/demo.yaml --output reports/ablation_stale.csv

Replays every prepared seed with exhaustive enumeration, once with the
configured `graph.stale_timeout_ms` and once with the filter disabled.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
from pathlib import Path

from qarb.adapters.synthetic import dataset_paths
from qarb.config import load_config
from qarb.core.book import read_events
from qarb.metrics import summarize
from qarb.simulation.replay import run_replay

KEYS = ["search_recall", "truth_recall", "false_positives_after_costs", "stale_decoys_flagged", "n_executions",
        "mean_fill_ratio", "expected_profit", "realised_pnl", "residual_exposure"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--output", default="reports/ablation_stale.csv")
    args = ap.parse_args()
    cfg = load_config(args.config)
    rows = []
    for seed in cfg["seeds"]:
        ev_path, man_path = dataset_paths(cfg, seed)
        events, man = list(read_events(ev_path)), json.loads(man_path.read_text())
        for label, timeout in (("filter_on", cfg["graph"]["stale_timeout_ms"]), ("filter_off", None)):
            c = copy.deepcopy(cfg)
            c["graph"]["stale_timeout_ms"] = timeout
            out = run_replay(c, seed, ["enumeration"], events, man)
            m = summarize(out["states"]["enumeration"], out["truth_by_t"], out["decision_times"], man, c)
            rows.append({"seed": seed, "variant": label, "stale_timeout_ms": timeout, **{k: m[k] for k in KEYS}})
            print(f"seed {seed} {label:<10} FP={m['false_positives_after_costs']:.0f} "
                  f"pnl={m['realised_pnl']:.1f} fill={m['mean_fill_ratio']:.2f} residual={m['residual_exposure']:.0f}")
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
