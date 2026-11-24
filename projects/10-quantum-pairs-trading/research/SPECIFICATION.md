# Quantum-Optimized Pairs Trading System

> **Project specification / proposed implementation** · Cointegration + constrained pair selection

**Status:** Planned. This README describes what to build; it does **not** claim existing implementations, performance results, or quantum advantage.

## Overview

Discover and trade statistically related pairs while using quantum optimization only for constrained selection across many candidate pairs.

### Research question

Does quantum-assisted selection improve diversification and out-of-sample stability versus classical pair-ranking and MILP?

## System architecture

```text
Price history → point-in-time candidate pairs → cointegration tests / hedge ratios → train-only pair statistics → pair-overlap QUBO → classical / QAOA pair selection → z-score trades → backtest and robustness report.
```

**Design boundary:** Quantum optimization/estimation is a research or asynchronous component. Any latency-critical execution, exchange messaging or risk gate runs classically. Benchmark the *entire* computational pipeline, not just the quantum kernel.

## Mathematical / algorithmic approach

Binary pair-choice `z_k` selects candidate pairs subject to capital, security overlap and sector limits; optimize expected residual quality, portfolio correlation and transaction costs. Continuous hedge ratios are estimated classically.

### Classical baselines

Top-K pair ranking, greedy diversified selection, MILP/MIQP and simulated annealing.

### Quantum approach

Start with **small, verifiable problem instances** on local simulators. Implement QAOA or quantum annealing for the discrete optimization projects; for pricing/risk use amplitude estimation; for ML compare small parameterized circuits or quantum kernels. Use real QPU hardware **only if available**, and publish noise, shot count, queue time and cost separately.

## Data strategy

ETFs or adjusted equity daily prices from documented public sources; later add minute bars and conservative fills.

Use immutable, versioned datasets. Document license terms and timestamp normalization. Use chronological training, validation and test windows where time-series predictions or trading evaluation are involved.

## Technology stack

Python, statsmodels, Qiskit, OR-Tools, NumPy, pandas, Rust (execution optional), pytest.

## Features / deliverables

- Engle–Granger cointegration and half-life diagnostics
- Pairs with shared-asset exposure constraints
- Hedge ratio re-estimation without future leakage
- Stop-out, holding-time limits and financing costs
- Classical/quantum pair-selection comparison

## Proposed repository structure

```text
10-quantum-pairs-trading/
  README.md
  pyproject.toml                 # add for Python packages
  requirements.txt              # optional pinned environment
  
data/  research/  selection/  quantum/  execution/  tests/  reports/
```

Create only the directories relevant to the chosen implementation. Include a reproducible sample dataset or a script for generating it, configuration files and a machine-readable experiment manifest.

## Implementation roadmap

- [ ] **M1:** Implement pair testing and vanilla z-score backtester
- [ ] **M2:** Build diversified pair selector with classical baseline
- [ ] **M3:** Encode small constrained pair selection QUBO
- [ ] **M4:** Integrate quantum simulation and verify constraints
- [ ] **M5:** Publish walk-forward and sensitivity analyses

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

**Primary measures:** Out-of-sample Sharpe, mean-reversion success, win rate, drawdown, turnover, exposure overlap and optimizer time.

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

Historical cointegration frequently breaks; avoid screening the entire backtest period before evaluation.

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
