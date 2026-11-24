# Quantum Arbitrage Detection Engine

> Graph algorithms + executable cross-venue pricing, with a QUBO/QAOA formulation benchmarked against classical search

**Status:** Implemented as a CPU-only research pipeline (milestones M1–M5 below). All quantum results are **noiseless statevector simulations**; no quantum hardware was used. Nothing here claims quantum advantage or a profitable strategy.

## Overview

Detect and size cross-venue currency/crypto arbitrage cycles, then compare graph-search algorithms with a constrained quantum-optimization formulation on identical data, objectives and budgets.

### Research question

Can constrained quantum-search formulations identify economically executable cycles competitively with Bellman–Ford and graph enumeration?

### Short answer from this implementation

Not at the sizes a simulator can reach. On the demo experiment (5 currencies, 3 venues, cycles of length 2–4, 5 seeds):

- Exhaustive enumeration scores all 60 candidate cycles in about 0.05 ms per decision and is complete by construction.
- The QUBO formulation is exact (its minimum was a valid minimum-weight cycle in every instance), and QAOA on it also reached full search recall, but it needed about 65,000 simulated circuit executions and about 270 ms of simulator time per decision. Only 22% of its shots satisfied the cycle constraints; the rest were fixed by a classical greedy repair.
- A uniform random sampler given 5,000 evaluations matched QAOA's recall, so on these instances QAOA did not demonstrate a search benefit.
- Data quality mattered far more than the search algorithm: disabling the stale-quote filter turned a simulated +164 into −952 with identical search.

Details and caveats are in [Results](#results).

## System architecture

```text
quote events (JSONL) ──► symbol normalization ──► currency graph
  exchange_ts / recv_ts     core/symbols.py        graph/build.py: stale filter, fee- and depth-adjusted
                                                   edge ladders, weights w_ij(q) = -log R_ij(q)
                                                        │
        ┌───────────────────────────────────────────────┴──────────────────────────┐
        ▼ classical (graph/cycles.py)                                              ▼ QUBO (quantum/)
  enumeration · Bellman–Ford · bounded DP · random search          fixed-length cycle QUBO ──► QAOA simulator
        │                                                          (also solved by classical simulated annealing)
        └────────────────────────────► candidate currency cycles ◄──── decode + repair
                                              │
                    graph/sizing.py: venue choice, exact liquidity-aware size, cost gate
                                              │
                    simulation/execution.py: IOC legs on the true books after latency
                                              │
                    metrics.py + report.py: recall, false positives, P&L, latency, resources
```

**Design boundary.** Search over the weight matrix is the only step where solvers differ. Graph building, venue selection, sizing, the cost gate, execution and all risk-relevant logic are classical and shared. The quantum path is a research component that falls back to a classical solver if it cannot run (register too large, budget too small, or any error), and the fallback is recorded.

## What is implemented

| Area | Where | Notes |
|---|---|---|
| Canonical symbols | `src/qarb/core/symbols.py` | `BTC-USD`, `XBTUSDT`, `btc_usd` all map to `(BTC, USD)`; alias table; quote-priority orientation |
| Order books, events | `src/qarb/core/book.py` | Full-depth snapshots with exchange and receive timestamps, per-market fees |
| Synthetic data | `src/qarb/adapters/synthetic.py` | Arbitrage-free background by construction, planted profitable cycles, thin-depth decoys, stale-feed decoys |
| Conversion graph | `src/qarb/graph/build.py` | Two edges per book, depth ladders, variable fees, stale-quote timeout, venue restriction, NetworkX export |
| Classical search | `src/qarb/graph/cycles.py` | Bounded-depth enumeration, iterated Bellman–Ford, min-plus bounded DP (shortest-path heuristic), random search |
| Sizing | `src/qarb/graph/sizing.py` | Exact profit-maximizing size on piecewise-linear ladders, capital and per-venue inventory caps, venue-count limit |
| QUBO | `src/qarb/quantum/qubo.py` | Position encoding with one-hot and uniqueness penalties, anchored variant, Ising conversion, decode, repair, brute-force check |
| QAOA | `src/qarb/quantum/qaoa.py`, `qiskit_backend.py` | NumPy statevector engine with shot-based CVaR objective and COBYLA; Qiskit circuit for cross-validation and compiled resource counts |
| Simulated annealing | `src/qarb/quantum/anneal.py` | Classical sampler on the same QUBO (not a quantum result) |
| Replay and execution | `src/qarb/simulation/` | Deterministic replay of asynchronous updates, observed vs true book state, latency, IOC partial fills, inventory |
| CLI | `prepare_data.py`, `run.py`, `report.py` | Thin entry points; one run directory per solver and seed with a machine-readable manifest |

### Mathematical formulation

Each book gives two directed edges with a ladder of levels. For input size `q` the effective executable rate is `R_ij(q)` (after walking depth and paying the taker fee) and the edge cost is `w_ij(q) = -log R_ij(q)`. A cycle with negative total weight is nominally profitable at that size. Search uses weights at a fixed probe notional; every candidate is then sized exactly and must clear a minimum net profit (the cost gate).

**QUBO for cycles of fixed length L.** `x[v,p] = 1` iff currency `v` is at position `p`:

```text
minimize   Σ_p Σ_{u≠v} w̃[u,v] · x[u,p] · x[v,p+1]          (positions cyclic)
         + A · Σ_p (Σ_v x[v,p] − 1)²                        one currency per position
         + A · Σ_v Σ_{p<p'} x[v,p] · x[v,p']                no currency twice
```

Weights are normalized to `|w̃| ≤ 1`, missing edges get weight `L`, and `A = 2L + 4` provably puts every invalid assignment above every valid cycle (derivation in `auto_penalty`; verified by brute force in the tests). The anchored variant fixes position 0 and needs `(N−1)(L−1)` variables instead of `N·L`; it is solved once per anchor currency.

**Cross-venue model.** Cycles are pre-funded: all legs trade simultaneously against inventory already held at each leg's venue, and nothing is transferred between venues inside a cycle. Inventory is not rebalanced during a replay.

## How to run

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt && pip install .

# 1. Generate the synthetic datasets (one per declared seed, with sha256 manifest)
python -m qarb.prepare_data --config configs/demo.yaml

# 2. Classical baselines
python -m qarb.run --config configs/demo.yaml --solver classical

# 3. Quantum simulator (QAOA, noiseless statevector)
python -m qarb.run --config configs/demo.yaml --solver quantum-sim

# 4. Compare stored runs
python -m qarb.report --runs runs/ --output reports/comparison.html

# Tests, and extra benchmarks
pytest
python benchmarks/scaling.py --output reports/scaling.csv
python benchmarks/ablation_stale.py --config configs/demo.yaml --output reports/ablation_stale.csv
```

The install is deliberately not editable (`pip install .`, re-run after changing the code): on macOS, a virtualenv inside a synced folder such as Desktop can have its editable-install `.pth` file flagged hidden, which Python 3.12 then skips, and `import qarb` fails. `pytest` reads `src/` directly and does not need the install.

The spec's placeholder module name `project` is `qarb` here. `--solver` accepts a group from the config (`classical`, `quantum-sim`), `all`, or one solver name. The full demo takes about 90 s on a laptop CPU, most of it QAOA simulation. `configs/tiny.yaml` is a reduced instance that runs end to end in a few seconds.

Each run writes `runs/<config>/<solver>/seed<N>/` containing `manifest.json` (full config, config hash, dataset sha256, package versions, status and traceback if failed, compiled circuit resources), `metrics.json`, `ticks.csv` and `opportunities.csv`. `data/` and `runs/` are regenerated by the commands above and are git-ignored; datasets are deterministic per seed and a run refuses a dataset whose hash does not match its manifest.

## Results

Demo experiment `configs/demo.yaml`: 5 currencies, 3 venues, 60 ticks per seed, 5 planted profitable cycles, 2 thin decoys and 2 stale-feed decoys per seed, seeds `[11, 23, 37, 41, 59]` declared in advance. All 35 runs completed; none failed. Values are mean ± half-width of a 95% Student-t interval over the 5 seeds. Timings are from one Apple-silicon laptop core (Python 3.12, NumPy 2.5, Qiskit 2.6) and will differ elsewhere. The full tables are in `reports/comparison.html` and the raw numbers in `reports/comparison.csv` and `reports/comparison_per_seed.csv`.

| Solver | Planted recall | Search recall | Recall vs true books | Nominal candidates | Rejected by cost gate | False positives after costs | Missed-opportunity rate | Simulated net P&L (USD) | Objective evals / decision | Search p50 (ms) | Search p95 (ms) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| enumeration | 0.96 ± 0.11 | 1.00 | 0.73 ± 0.24 | 92 ± 69 | 9 ± 5 | 0 | 0.35 ± 0.13 | 164 ± 79 | 60 | 0.045 | 0.096 |
| bellman_ford | 0.92 ± 0.14 | 0.66 ± 0.21 | 0.46 ± 0.04 | 52 ± 29 | 4 ± 3 | 0 | 0.54 ± 0.04 | 149 ± 51 | 31 | 0.020 | 0.069 |
| bounded_dp | 0.96 ± 0.11 | 0.72 ± 0.22 | 0.50 ± 0.10 | 58 ± 27 | 6 ± 4 | 0 | 0.43 ± 0.06 | 173 ± 107 | 94 | 0.086 | 0.127 |
| random_search (5,000 samples) | 0.96 ± 0.11 | 1.00 | 0.73 ± 0.24 | 92 ± 69 | 9 ± 5 | 0 | 0.35 ± 0.13 | 164 ± 79 | 5,000 | 15.3 | 25.2 |
| qubo_sa (classical annealing) | 0.96 ± 0.11 | 1.00 | 0.73 ± 0.24 | 92 ± 69 | 9 ± 5 | 0 | 0.35 ± 0.13 | 164 ± 79 | 28,800 | 64.9 | 92.2 |
| naive_mid (mid prices, no costs) | 0.84 ± 0.21 | 0.54 ± 0.25 | 0.36 ± 0.09 | 556 ± 17 | 519 ± 17 | 0 | 0.60 ± 0.06 | 128 ± 52 | 198 | 0.136 | 0.196 |
| **qaoa** (statevector simulation) | 0.96 ± 0.11 | 1.00 (0.99 ± 0.01 without repair) | 0.73 ± 0.24 | 92 ± 69 | 9 ± 5 | 0 | 0.35 ± 0.13 | 164 ± 79 | 65,106 | 267.9 | 404.7 |

*Search recall* is measured against exhaustive search on the same observed books, so it isolates the search algorithm. *Recall vs true books* also counts opportunities that were unobservable because a venue's feed was delayed or filtered as stale. Metric definitions are at the bottom of the HTML report.

**QAOA diagnostics** (2 layers, 30 COBYLA iterations at 128 shots each, 512 final shots, 15 anchored sub-problems per decision):

| | |
|---|---|
| Largest register | 12 qubits |
| Instances where the QUBO minimum is a valid cycle | 100% |
| Shots satisfying all constraints | 22% |
| Distinct sampled bitstrings needing classical repair | 95% |
| P(ground state), optimized vs uniform guess | 0.059 vs 0.034 |
| Time per decision: QUBO build / optimize / sample / decode / repair | 5.8 / 250 / 4.2 / 4.3 / 10.3 ms |
| Compiled circuit, L=4 anchored (CX/RZ/SX/X, all-to-all) | 12 qubits, depth 115, 216 CX |

**What the numbers do and do not show**

- At this size the problem is trivial classically: there are only 60 cycles. Random search with 5,000 evaluations is already complete, so QAOA matching enumeration is not evidence of anything beyond "the pipeline works".
- QAOA's sampled distribution was only modestly better than uniform (1.8× on the ground state), and most of its distinct samples were infeasible. Its recall holds up without repair (0.99) because 512 shots over a register of at most 4,096 states cover the few valid cycles anyway.
- QAOA wall time is simulator time. It says nothing about a device, which would add queueing, compilation to a real topology, noise and readout error. Circuit executions (65k per decision) is the hardware-relevant cost, and it is three orders of magnitude above the classical evaluation counts.
- Iterated Bellman–Ford and the bounded DP are fast but incomplete as enumerators (search recall 0.66 and 0.72). Their P&L is within noise of exhaustive search because they still find the most profitable cycles.
- The mid-price detector flagged 556 nominal cycles per seed, of which 519 failed the cost gate. A negative cycle in mid prices is not executable arbitrage.
- P&L differences between solvers are well inside the confidence intervals. With 5 seeds this experiment cannot rank solvers on profit.

**Stale-quote ablation** (`reports/ablation_stale.csv`, exhaustive search, mean over the 5 seeds):

| Stale filter | False positives after costs | Mean fill ratio | Simulated net P&L | Residual exposure |
|---|---|---|---|---|
| on (300 ms) | 0 | 1.00 | +164 | 0 |
| off | 557 | 0.22 | −952 | 787,896 |

With the filter off, the search "finds" more (recall vs true books rises from 0.73 to 0.96) but trades dead quotes: one leg fails, the others fill, and the book is left with unhedged inventory.

**Scaling** (`reports/scaling.csv`, random instances with planted negative cycles, lengths 2–4, 3 seeds per size, means):

| Currencies | QUBO variables (anchored / unanchored, L=4) | enumeration | bellman_ford | bounded_dp | random_search (5,000) | qubo_sa | qaoa |
|---|---|---|---|---|---|---|---|
| 5 | 12 / 20 | 0.04 ms, recall 1.00 | 0.07 ms, 0.54 | 0.09 ms, 0.60 | 14 ms, 1.00 | 57 ms, 1.00 | 255 ms, 1.00 |
| 6 | 15 / 24 | 0.07 ms, 1.00 | 0.10 ms, 0.38 | 0.09 ms, 0.52 | 14 ms, 1.00 | 74 ms, 1.00 | 823 ms, 1.00 |
| 8 | 21 / 32 | 0.23 ms, 1.00 | 0.27 ms, 0.43 | 0.09 ms, 0.51 | 12 ms, 1.00 | 131 ms, 0.60 | not run (>16 qubits) |
| 12 | 33 / 48 | 1.6 ms, 1.00 | 0.83 ms, 0.29 | 0.12 ms, 0.42 | 13 ms, 0.50 | 302 ms, 0.29 | not run |
| 16 | 45 / 64 | 6.5 ms, 1.00 | 5.4 ms, 0.24 | 0.16 ms, 0.41 | 17 ms, 0.34 | 627 ms, 0.14 | not run |
| 24 | 69 / 96 | 42 ms, 1.00 | 21 ms, 0.24 | 0.28 ms, 0.38 | 20 ms, 0.13 | 1,580 ms, 0.08 | not run |

Enumeration grows as `N^L` but is still 42 ms at 24 currencies. QAOA could not be simulated past 6 currencies, so there is no QAOA data in the regime where enumeration starts to cost anything. Simulated annealing on the QUBO loses recall quickly at fixed effort (32 reads × 60 sweeps), which reflects both the sampler budget and the penalty-dominated landscape.

## Benchmark protocol

1. **Tiny instances verified against brute force.** QUBO minima are compared with exhaustive cycle enumeration, enumeration with NetworkX `simple_cycles`, and Bellman–Ford existence with NetworkX `negative_edge_cycle`.
2. **Identical objectives, constraints and data.** Every solver receives the same weight matrix and length bounds; venue choice, sizing, the cost gate and execution are shared code.
3. **Budgets.** `budget.max_objective_evals` and `budget.max_wall_ms` apply to every solver per decision; both objective evaluations and wall time are reported. Units per solver are defined in `src/qarb/solvers.py`. In the demo the budget (100,000 evaluations) did not bind for any solver; the per-solver cost actually used is what the table reports.
4. **Full timing breakdown.** Graph build, search, sizing, and for QUBO solvers build, optimization, sampling, decoding and repair are recorded separately, along with optimizer iterations and shots.
5. **Seeds and failures.** Seeds are declared in the config; the report shows the number of successful and failed runs per solver and never drops a stored run.
6. **Hardware vs simulation.** There are no hardware results. Compiled circuit resources are reported separately and labelled as not executed.
7. **P&L.** Execution is simulated against the true (undelayed) books after a fixed latency with a taker-fee and depth model. Backtests on synthetic data do not establish live profitability.

## Tests

`pytest` runs 120 tests in about 6 s on CPU: symbol parsing and book validation, depth walking and fees, graph construction and filters, cross-checks of each classical search against NetworkX, analytic and brute-force sizing checks, QUBO energies, penalty sufficiency and brute-force optimality, Ising conversion, NumPy-vs-Qiskit agreement of the QAOA state, a known-optimum QAOA fixture, simulated annealing, dataset determinism, arbitrage-freeness of the background books, planted cycles appearing in ground truth and decoys not, deterministic replay, the stale-filter ablation, solver adapters, fallback behaviour, budget caps, failed-run recording and the three CLIs end to end.

## Roadmap

- [x] **M1:** Graph ingestion and synthetic cycle generator
- [x] **M2:** Bellman–Ford and depth-aware profitability
- [x] **M3:** QUBO for valid fixed-length cycles
- [x] **M4:** Comparison against classical solutions under shared budgets
- [x] **M5:** Stream replay and latency profiling

Not done, relative to the original specification:

- **Python only.** No Rust/C++ core, Kafka/Redpanda transport or Criterion benchmarks. Latency numbers are Python and are not representative of a production path.
- **Synthetic data only.** No archived exchange snapshots; the event format and symbol adapter are ready for them, but no real data, licence review or timestamp reconciliation has been done.
- **No quantum hardware and no noise model.** QAOA is noiseless simulation up to 16 qubits by default.
- **No figures or notebook.** The report is tables plus CSVs.

## Known limitations and failure modes

- A negative cycle in mid prices is not executable arbitrage; fees, depth, staleness and asynchronous fills decide the outcome. The naive baseline and the stale ablation quantify this on synthetic data only.
- Search uses weights at one probe size and at most `venue_top_k` venues per leg. Cycles profitable only at much smaller sizes, or only by splitting across venues, are missed by every solver alike.
- The replay has no market impact beyond consuming the liquidity it trades within a decision, and each opportunity is executed at most once per episode. Inventory is never rebalanced, and transfers, withdrawal limits and settlement delays are not modelled.
- Latency is a fixed configured delay; measured compute time is reported but not fed back into execution timing, so the 270 ms QAOA decisions are not penalized in the simulated P&L. Charging it would make QAOA's economics worse.
- QAOA uses the plain X mixer with penalty terms. Constraint-preserving mixers, parameter transfer between decisions and better initializations are untested here and could change the sampling quality, though not the register-size limit.
- The QUBO penalty `A = 2L + 4` is safe but large relative to the objective, which flattens the useful part of the landscape. Smaller penalties are configurable but lose the guarantee.
- Current quantum devices add noise, limited qubit counts and connectivity, encoding cost, depth limits and measurement overhead that this simulation ignores. **Do not assert a production advantage; these measurements do not support one.**

## Repository structure

```text
configs/            demo.yaml (reported experiment), tiny.yaml (reduced instance)
src/qarb/
  core/             symbols, order books, quote events
  adapters/         synthetic data generator
  graph/            graph build, classical cycle search, sizing
  quantum/          QUBO, QAOA simulator, Qiskit circuits, simulated annealing
  simulation/       replay and execution simulator
  solvers.py        uniform solver interface, budgets, fallback
  metrics.py        run-level metrics
  prepare_data.py   run.py   report.py        CLI entry points
tests/              pytest suite
benchmarks/         scaling.py, ablation_stale.py
reports/            generated comparison report and CSVs
```

## Further reading / tooling

- [Qiskit documentation](https://docs.quantum.ibm.com/)
- [D-Wave Ocean documentation](https://docs.dwavequantum.com/en/latest/ocean/index.html)
- [Google OR-Tools](https://developers.google.com/optimization)

## Disclaimer

For research and educational use. Not investment advice, not a claim of profitable strategy performance, and not a production trading system. Verify market-data permissions and exchange rules before any live integration.
