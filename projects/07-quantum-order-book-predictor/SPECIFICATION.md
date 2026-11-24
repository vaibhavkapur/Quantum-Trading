# Quantum ML Order Book Predictor

> **Project specification / proposed implementation** · Short-horizon microstructure forecasting

**Status:** Planned. This README describes what to build; it does **not** claim existing implementations, performance results, or quantum advantage.

## Overview

Compare quantum kernels and small variational circuits with classical models for predicting short-horizon mid-price movement from L2 data.

### Research question

After matched training budgets and full inference overhead, does a quantum model improve out-of-sample classification or trading utility?

## System architecture

```text
Timestamped L2 snapshots → leakage-safe features → chronological splits → PCA/feature reduction → classical and quantum ML training → calibration → inference benchmark → threshold-based trading evaluation.
```

**Design boundary:** Quantum optimization/estimation is a research or asynchronous component. Any latency-critical execution, exchange messaging or risk gate runs classically. Benchmark the *entire* computational pipeline, not just the quantum kernel.

## Mathematical / algorithmic approach

Predict `sign(midprice[t+Δ]−midprice[t])` or a three-class movement label. Compare quantum kernel SVM / small variational classifier with logistic regression, gradient boosting and MLP.

### Classical baselines

Majority class, logistic regression, XGBoost and compact PyTorch MLP; identical inputs and label horizon.

### Quantum approach

Start with **small, verifiable problem instances** on local simulators. Implement QAOA or quantum annealing for the discrete optimization projects; for pricing/risk use amplitude estimation; for ML compare small parameterized circuits or quantum kernels. Use real QPU hardware **only if available**, and publish noise, shot count, queue time and cost separately.

## Data strategy

Use public order-book research datasets if license permits or reproducible crypto L2 snapshots. Ensure snapshots/trades are synchronized without future knowledge.

Use immutable, versioned datasets. Document license terms and timestamp normalization. Use chronological training, validation and test windows where time-series predictions or trading evaluation are involved.

## Technology stack

Python, PennyLane or Qiskit Machine Learning, PyTorch, scikit-learn, XGBoost, Polars, Rust (serving optional).

## Features / deliverables

- Order-book imbalance, microprice, spread, depth and order-flow imbalance
- Purged/embargoed chronological train/validation/test splits
- Data encoding and qubit-count ablations
- Probability calibration and realistic decision thresholds
- Streaming inference and p99 inference-latency profiling

## Proposed repository structure

```text
07-quantum-order-book-predictor/
  README.md
  pyproject.toml                 # add for Python packages
  requirements.txt              # optional pinned environment
  
data/  features/  models/classical/  models/quantum/  serving/  tests/  benchmarks/
```

Create only the directories relevant to the chosen implementation. Include a reproducible sample dataset or a script for generating it, configuration files and a machine-readable experiment manifest.

## Implementation roadmap

- [ ] **M1:** Implement reliable labels, feature pipeline and classical baselines
- [ ] **M2:** Create small stratified research samples without temporal leakage
- [ ] **M3:** Train quantum kernel and VQC alternatives
- [ ] **M4:** Profile encoding/shots/serving overhead
- [ ] **M5:** Run walk-forward and post-cost execution evaluation

## How to run (planned CLI contract)

The following is a **proposed interface**, not an executable command until implemented:

```bash
# 1. Set up the environment using this repository's pinned dependencies
# 2. Generate/download a documented sample dataset
python -m project.prepare_data --config configs/demo.yaml

# 3. Run classical baseline
python -m project.run --config configs/demo.yaml --solver classical

# 4. Run quantum simulator baseline
python -m project.run --config configs/demo.yaml --solver quantum-sim

# 5. Evaluate stored runs with identical constraints and budgets
python -m project.report --runs runs/ --output reports/comparison.html
```

Implement these modules as thin, consistent CLI entry points. Do not present the commands as functioning before the modules exist.

## Benchmark plan

**Primary measures:** Balanced accuracy, macro F1, precision/recall, calibration, net post-cost P&L, inference latency, throughput, model footprint and circuit shots.

**Fair-comparison protocol:**

1. Verify tiny instances against brute force or analytic answers.
2. Compare identical objectives, constraints, data and train/test splits.
3. Set a budget for both approaches (wall time and/or objective evaluations; report both).
4. Record data loading, state preparation, compilation, optimizer iterations, shots, communication, solution repair and result decoding.
5. Repeat stochastic experiments with multiple predeclared seeds; show confidence intervals and failed runs.
6. Report **actual hardware results** separately from noiseless simulation and theoretical asymptotic scaling.
7. For trading P&L, use an out-of-sample cost model and state that backtests do not establish live profitability.

## Testing and acceptance criteria

- Unit tests for objective values, constraints, encodings and input edge cases.
- A small fixture with known optimum or reference price/risk estimate.
- Deterministic regression tests for data preparation, backtesting or simulation.
- Integration tests for solver adapters and failure/fallback behavior.
- A published benchmark table with at least one strong classical baseline and full timing.
- A reproducibility command that a reviewer can run on CPU-only hardware (with reduced instance sizes).

## Known limitations and failure modes

Noise, nonstationarity, class imbalance and large feature-encoding costs can erase apparent gains.

Current quantum devices introduce noise, limited qubit capacity, costly input encoding, circuit-depth constraints and measurement overhead. **Do not assert a production advantage until controlled end-to-end measurements support one.**

## Suggested GitHub portfolio deliverables

- Architecture diagram and rationale for classical/quantum boundaries.
- Reproducible benchmark notebook and raw CSV/Parquet results.
- Clear figures for solution quality, run time, scaling and costs.
- A short technical write-up: hypothesis, experiment, finding, limitations.
- A 1–2 minute demo using synthetic or licensed historical data.

## Further reading / tooling

- [Qiskit documentation](https://docs.quantum.ibm.com/)
- [D-Wave Ocean documentation](https://docs.dwavequantum.com/en/latest/ocean/index.html)
- [Google OR-Tools](https://developers.google.com/optimization)
- [QuantLib](https://www.quantlib.org/) (if applicable)

## Disclaimer

For research and educational use. Not investment advice, not a claim of profitable strategy performance, and not a production trading system. Verify market-data permissions and exchange rules before any live integration.
