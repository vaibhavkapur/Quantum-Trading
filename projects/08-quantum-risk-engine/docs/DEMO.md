# Two-minute demo

Activate `.venv` from the project root, then run `python -m uvicorn api.app:app --host 127.0.0.1 --port 8088`.

1. Open `http://127.0.0.1:8088`. Compare five classical estimators on the same three-asset portfolio. Point out confidence level and the distinction between VaR and expected shortfall.
2. Show stress losses and 500 out-of-sample observations. Every forecast uses earlier returns only. Exceptions appear above the forecast line.
3. Select **Classical + quantum simulator**, then **Run analysis**. Show the estimate and circuit-resource summary. A coarse loss grid causes error even with perfect probability estimation.
4. Switch to **Student-t** and rerun. The risk model changes; the Gaussian baseline remains visible for comparison.
5. Use **Export JSON** to retain configuration, metrics, forecasts and resources.
6. Open `reports/comparison.html`. Compare tail-estimation precision at equal query counts, then runtime. The simulator has lower sampling error on this tiny example at the largest budget, while classical computation is faster.

Repeat with `python -m project.reproduce --config configs/demo.yaml`. Raw CSVs, the offline report and a source/artifact checksum manifest appear under `reports/reproduction/`.
