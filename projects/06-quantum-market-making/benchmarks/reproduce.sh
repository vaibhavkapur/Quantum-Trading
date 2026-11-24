#!/usr/bin/env sh
set -eu
cd "$(dirname "$0")/.."
PYTHON=${PYTHON:-python}
"$PYTHON" -m pytest -q
"$PYTHON" -m project.prepare_data --config configs/ci.yaml
"$PYTHON" -m project.run --config configs/ci.yaml --solver all
"$PYTHON" -m project.report --runs runs --output reports/comparison.html
