# Quantum-Enhanced Market Making Engine

> **Project specification / proposed implementation** · Inventory-aware quoting + event-driven simulator

**Status:** Planned. This README describes what to build; it does **not** claim existing implementations, performance results, or quantum advantage.

## Overview

Design a market maker with C++/Rust event processing and periodically quantum-optimized quote offsets, inventory bands and risk policies.

### Research question

Can hybrid policy optimization improve risk-adjusted market-making P&L compared with simple inventory-skew and classical optimizers?

## System architecture

```text
L2 market-data replay → order book and queue simulation → inventory / alpha / volatility state → periodic optimizer → immutable quote-policy parameters → low-latency bid/ask generation → fills → inventory/risk attribution.
```

**Design boundary:** Quantum optimization/estimation is a research or asynchronous component. Any latency-critical execution, exchange messaging or risk gate runs classically. Benchmark the *entire* computational pipeline, not just the quantum kernel.

## Mathematical / algorithmic approach

Discretize inventory and quoting offsets; maximize expected spread capture minus inventory variance, adverse selection and cancellation costs across a bounded set of scenarios.

### Classical baselines

Fixed-spread quotes, inventory-skew heuristic, Avellaneda–Stoikov-inspired policy, classical grid search / dynamic programming.

### Quantum approach

Start with **small, verifiable problem instances** on local simulators. Implement QAOA or quantum annealing for the discrete optimization projects; for pricing/risk use amplitude estimation; for ML compare small parameterized circuits or quantum kernels. Use real QPU hardware **only if available**, and publish noise, shot count, queue time and cost separately.

## Data strategy

Synthetic Poisson/hawkes-inspired order flow and publicly available historical L2 exchange data for stress scenarios.

Use immutable, versioned datasets. Document license terms and timestamp normalization. Use chronological training, validation and test windows where time-series predictions or trading evaluation are involved.

## Technology stack

C++20 or Rust, Python, Qiskit, OR-Tools, NumPy, Prometheus, Grafana, Docker.

## Features / deliverables

- Price-time-priority matching and queue-position model
- Inventory skew, risk limits and emergency cancel-all
- Maker/taker fees, adverse selection and fill probabilities
- Async optimizer with classical fallback and versioned policies
- Regime and volatility stress testing

## Proposed repository structure

```text
06-quantum-market-making/
  README.md
  pyproject.toml                 # add for Python packages
  requirements.txt              # optional pinned environment
  
lob/  market-maker/  risk/  optimizer/  simulations/  tests/  benchmarks/  reports/
```

Create only the directories relevant to the chosen implementation. Include a reproducible sample dataset or a script for generating it, configuration files and a machine-readable experiment manifest.

## Implementation roadmap

- [ ] **M1:** Build limit order book replay and matching tests
- [ ] **M2:** Implement fixed-spread and inventory-skew market maker
- [ ] **M3:** Formulate small discrete policy optimization
- [ ] **M4:** Integrate QAOA/annealing and classical baseline
- [ ] **M5:** Publish out-of-sample scenario and latency reports

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

**Primary measures:** Net spread capture, P&L, Sharpe, max drawdown, inventory distribution, adverse-selection cost, quote-to-ack p99 latency and policy update time.

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

Simplistic fill models can make market-making strategies look profitable; separate maker rebates from adverse-selection losses.

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
