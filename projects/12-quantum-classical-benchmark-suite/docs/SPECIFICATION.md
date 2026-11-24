# Quantum-vs-Classical Trading Benchmark Suite

> **Project specification / proposed implementation** · Reproducible research + systems performance

**Status:** Planned. This README describes what to build; it does **not** claim existing implementations, performance results, or quantum advantage.

## Overview

Create a fair, extensible benchmarking harness for quantum and classical solvers on representative quant-finance problems, with full end-to-end timing.

### Research question

Which financial problem sizes, encodings and compute budgets yield any measurable advantage—and where do classical solvers dominate?

## System architecture

```text
Versioned problem instances → common objective/constraint specification → solver adapters → warmup + reproducible runs → solution feasibility checks → timing / quality / resource capture → comparison reports.
```

**Design boundary:** Quantum optimization/estimation is a research or asynchronous component. Any latency-critical execution, exchange messaging or risk gate runs classically. Benchmark the *entire* computational pipeline, not just the quantum kernel.

## Mathematical / algorithmic approach

Implement a shared interface for QUBO portfolio selection, constrained cycle/basket selection and separate probabilistic estimation tasks (MC vs QAE). Do not aggregate unrelated objectives into one ranking.

### Classical baselines

Exact brute force on tiny problems, CP-SAT/MILP/MIQP, greedy heuristics, simulated annealing, variance-reduced MC.

### Quantum approach

Start with **small, verifiable problem instances** on local simulators. Implement QAOA or quantum annealing for the discrete optimization projects; for pricing/risk use amplitude estimation; for ML compare small parameterized circuits or quantum kernels. Use real QPU hardware **only if available**, and publish noise, shot count, queue time and cost separately.

## Data strategy

Deterministic synthetic instance generators with known optima where possible; optional versioned market-derived datasets. Log provenance, instance sizes and random seeds.

Use immutable, versioned datasets. Document license terms and timestamp normalization. Use chronological training, validation and test windows where time-series predictions or trading evaluation are involved.

## Technology stack

Python, C++20/Rust (performance kernels), Qiskit, D-Wave Ocean (optional), OR-Tools, CVXPY, Docker, GitHub Actions, Parquet, Plotly.

## Features / deliverables

- Solver adapter interface, timeout and budget policies
- Common feasibility validator and objective evaluator
- Full timing: data prep, encoding, compilation, execution, decoding and repair
- Quantum resource metrics including shots, depth and logical qubits
- Automated plots, statistical uncertainty, CI smoke tests and result manifests

## Proposed repository structure

```text
12-quantum-classical-benchmark-suite/
  README.md
  pyproject.toml                 # add for Python packages
  requirements.txt              # optional pinned environment
  
benchmarks/instances/  benchmarks/adapters/  metrics/  experiments/  reports/  tests/  .github/workflows/
```

Create only the directories relevant to the chosen implementation. Include a reproducible sample dataset or a script for generating it, configuration files and a machine-readable experiment manifest.

## Implementation roadmap

- [ ] **M1:** Define common schemas and synthetic verified instances
- [ ] **M2:** Implement exact and strong classical solver adapters
- [ ] **M3:** Add QAOA/annealing adapters and resource reporting
- [ ] **M4:** Build QAE vs MC track separately
- [ ] **M5:** Automate benchmark matrices and publish transparent results

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

**Primary measures:** Best feasible objective, optimality gap, success probability, full wall time, p50/p95 runtime, memory, circuit depth, shots, cost and reproducibility.

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

Unfair budgets, weak classical baselines, excluding encoding time and cherry-picking successes produce misleading claims.

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
