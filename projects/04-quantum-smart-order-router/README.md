# Quantum-Optimized Smart Order Router

> **Project specification / proposed implementation** · Low-latency execution + asynchronous optimization

**Status:** Planned. This README describes what to build; it does **not** claim existing implementations, performance results, or quantum advantage.

## Overview

Split orders across simulated venues using a low-latency classical router whose allocation parameters are periodically optimized with QUBO/QAOA.

### Research question

Does quantum-assisted route allocation reduce implementation shortfall without harming order-to-acknowledgment latency?

## System architecture

```text
Parent order → venue quotes/depth/fees → asynchronous constrained allocation solver → published policy snapshot → deterministic low-latency route decision → simulated venue gateways → fills / rejects / cancel-replace → TCA.
```

**Design boundary:** Quantum optimization/estimation is a research or asynchronous component. Any latency-critical execution, exchange messaging or risk gate runs classically. Benchmark the *entire* computational pipeline, not just the quantum kernel.

## Mathematical / algorithmic approach

Discretize order quantity into lots and minimize spread + fees + estimated market impact + nonfill penalty under conservation, venue caps and participation limits. The microsecond hot path reads precomputed policies only.

### Classical baselines

Best-price-first, pro-rata, greedy liquidity-aware allocation, classical MILP/CP-SAT.

### Quantum approach

Start with **small, verifiable problem instances** on local simulators. Implement QAOA or quantum annealing for the discrete optimization projects; for pricing/risk use amplitude estimation; for ML compare small parameterized circuits or quantum kernels. Use real QPU hardware **only if available**, and publish noise, shot count, queue time and cost separately.

## Data strategy

Replay generated venue-level L2 books with varying fees, latency, queue depth and liquidity. Optional historical multi-venue data subject to licensing.

Use immutable, versioned datasets. Document license terms and timestamp normalization. Use chronological training, validation and test windows where time-series predictions or trading evaluation are involved.

## Technology stack

C++20 or Rust, Python, Qiskit, OR-Tools, gRPC or shared memory, FIX-style messages, Prometheus, Docker.

## Features / deliverables

- Per-venue order books, fees, throttles and connectivity states
- Partial fills, rejections, cancellations and failover
- Optimizer timeout with classical fallback
- Lock-free or immutable policy snapshots with versioning
- Per-decision latency histograms and implementation-shortfall TCA

## Proposed repository structure

```text
04-quantum-smart-order-router/
  README.md
  pyproject.toml                 # add for Python packages
  requirements.txt              # optional pinned environment
  
router/  exchange-sim/  optimizer/  proto/  configs/  tests/  benchmarks/  dashboards/
```

Create only the directories relevant to the chosen implementation. Include a reproducible sample dataset or a script for generating it, configuration files and a machine-readable experiment manifest.

## Implementation roadmap

- [ ] **M1:** Create multi-venue execution simulator
- [ ] **M2:** Implement deterministic smart router and baseline allocation
- [ ] **M3:** Add discrete optimization and QUBO mapping
- [ ] **M4:** Integrate optimizer through asynchronous snapshot updates
- [ ] **M5:** Run stress tests and publish latency + TCA comparisons

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

**Primary measures:** Fill rate, implementation shortfall, price improvement, p50/p99/p99.9 routing latency, queue-position error, solver overhead and stale-policy frequency.

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

Do not put remote QPU calls in the exchange-facing hot path. Model market impact realistically enough that routing improvements are not simulation artifacts.

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
