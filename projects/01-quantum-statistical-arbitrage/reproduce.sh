#!/usr/bin/env bash
# One-command, CPU-only reproduction on reduced instance sizes (about a minute).
#   ./reproduce.sh            -> configs/ci.yaml
#   ./reproduce.sh demo       -> configs/demo.yaml (full demo study, a few minutes)
set -euo pipefail
cd "$(dirname "$0")"
NAME="${1:-ci}"
CFG="configs/${NAME}.yaml"
PY="${PYTHON:-python}"
RUNS="runs/${NAME}"

"$PY" -m pytest -q
"$PY" -m project.prepare_data --config "$CFG"
"$PY" -m project.run --config "$CFG" --solver classical --runs "$RUNS"
"$PY" -m project.run --config "$CFG" --solver quantum-sim --runs "$RUNS"
if [ "$NAME" = "ci" ]; then
  "$PY" benchmarks/scaling.py --runs "$RUNS" --quick
else
  "$PY" benchmarks/scaling.py --runs "$RUNS"
fi
if command -v cargo >/dev/null 2>&1; then
  (cd engine && cargo test --release --quiet)
  "$PY" -m project.replay_l2 --config "$CFG" --runs "$RUNS" --build
else
  echo "cargo not found: skipping the Rust replay engine (the report will omit that section)"
fi
"$PY" -m project.report --runs "$RUNS" --output "reports/${NAME}/comparison.html"
