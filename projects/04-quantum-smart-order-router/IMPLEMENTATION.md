# Quantum-Optimized Smart Order Router — implementation notes and results

This file documents what was built from the specification in [`README.md`](README.md), how to run it, and what was measured. The specification itself is unchanged.

**Status:** Implemented as a Python research simulator (milestones M1–M5). All "quantum" results are **noiseless CPU statevector simulations of instances up to 16 qubits**. No quantum hardware was used, and nothing here shows a quantum advantage — the measured result is the opposite (see [Findings](#findings)).

## Research question

Does quantum-assisted route allocation reduce implementation shortfall without harming order-to-acknowledgment latency?

**Measured answer for this setup: no shortfall reduction, no latency harm.** QAOA (with repair) returns the same allocations as exact enumeration, so shortfall is identical within noise (Δ = −0.000 ± 0.001 bps), while each policy refresh costs about 110 ms instead of 0.2 ms. Routing latency is unaffected because the hot path only reads a precomputed snapshot.

## Architecture

```text
                        ┌──────────────── asynchronous (ms–s) ─────────────────┐
 delayed venue books ─► │ build problems ─► solver (exact / CP-SAT / QAOA / …) │
 (per side, per size    │   per (side, size bucket)      │ timeout / error     │
  bucket)               │                                ▼                     │
                        │                     classical greedy fallback        │
                        └───────────────┬──────────────────────────────────────┘
                                        │ publish immutable PolicySnapshot (version n)
                                        ▼
 parent order ─► hot path: read snapshot ─► scale weights to qty ─► clip to live caps ─► child orders
                 (no solver call; stale/missing snapshot ─► best-price-first)             │
                                                                                          ▼
                              simulated venue gateways: IOC / DAY / cancel / cancel-replace
                              fills · partial fills · rejects (throttle, outage) · quote fade
                                                                                          │
                                           failover + passive leg + deadline sweep ◄──────┘
                                                                                          ▼
                                                                    TCA + latency histograms
```

**Classical/quantum boundary.** The optimizer is the only place a quantum routine can appear, and it only ever produces a policy snapshot. Order entry, caps, failover and messaging are classical and deterministic. The benchmark times the whole optimizer pipeline (problem build, encoding, optimization, sampling, decode + repair), not just the circuit.

## The optimization problem

Quantity is discretized into allocation units (`optimizer.max_units` per order, so 3 venues × 4 units = 12 qubits). For units `x_v` sent to venue `v`:

```text
minimize   Σ_v table_v[x_v]  +  Σ_{a<b} K_ab · x_a · x_b
subject to Σ_v x_v = target            (conservation)
           0 ≤ x_v ≤ cap_v             (displayed depth inside the limit, participation, venue cap)
```

`table_v[k]` = slippage from walking the displayed ladder + taker fee + quadratic impact estimate + non-fill penalty × (1 − calibrated fill probability) + a fixed per-order charge when `k > 0`. `K_ab` is the expected loss from quote fade when one venue's fill prints before the other venue's child order arrives. The fixed charge makes the tables non-convex and the fade term makes the objective non-separable, so the greedy heuristic is not optimal (on the static benchmark it finds the optimum in 50–90% of instances, depending on size).

**QUBO mapping** (`qsor/optimizer/qubo.py`): one-hot variables `y[v,k]` ("venue v gets exactly k units"); tables become linear terms, fade becomes quadratic terms, and one-hot + conservation are penalty terms with a weight large enough to put every infeasible state above every feasible one. Tests check energy = objective on all feasible assignments and ground state = enumerated optimum.

### Solvers (all see the identical problem)

| Name | What it is |
|---|---|
| `best_price`, `pro_rata`, `greedy` | Classical heuristics |
| `exact` | Exhaustive enumeration (ground truth) |
| `cpsat` | OR-Tools CP-SAT (exact, scales past enumeration) |
| `sa` | Simulated annealing on the QUBO (classical) |
| `random_repair` | **Control:** uniform random bitstrings with QAOA's shot budget, same decode + repair |
| `qaoa` | QAOA, X mixer, COBYLA, warm-started angles; built-in NumPy statevector simulator, cross-checked against Qiskit's `Statevector` in the tests |
| `greedy_live` | Reference only: greedy solve on the hot path with the live book, no snapshot |

`--solver classical` runs the first group plus `greedy_live`; `--solver quantum-sim` runs `qaoa`, `sa` and `random_repair`. A single solver name also works.

## How to run

```bash
python3.12 -m venv .venv && .venv/bin/pip install -r requirements.txt
source .venv/bin/activate

python -m pytest -q                                              # 91 tests, a few seconds

python -m project.prepare_data --config configs/demo.yaml        # versioned synthetic dataset
python -m project.run --config configs/demo.yaml --solver classical     # ~15 s
python -m project.run --config configs/demo.yaml --solver quantum-sim   # ~2 min
python -m project.report --runs runs/ --output reports/comparison.html

python benchmarks/static_benchmark.py     # solver quality vs brute force, 4–16 qubits (~10 s)
python benchmarks/hotpath_benchmark.py    # decision latency with the optimizer thread running
```

Quick CPU-only reproducibility check (reduced sizes, a few seconds): use `configs/ci.yaml` in place of `demo.yaml`. `configs/stress.yaml` is the stress scenario. The report covers every experiment found under `runs/`. A `Dockerfile` that runs the tests and the `ci` config is included but has not been built or tested.

Each run writes `runs/<experiment>/<solver>/manifest.json` (config, dataset checksum, calibration, seeds with ok/failed status, library versions) and per seed `metrics.json`, `parents.csv`, `decision_latency_ns.csv`, `metrics.prom` (Prometheus text format) and, for the first seed, `messages.fix` (FIX-style order and execution messages).

## Findings

Outputs: [`reports/comparison.html`](reports/comparison.html) / [`.csv`](reports/comparison.csv), [`benchmarks/results/`](benchmarks/results/). Apple M5, Python 3.12, versions in `requirements.txt`. Mean ± 95% CI across predeclared seeds; test window only (last 30% of the data; fill probabilities fitted on the first 50%, non-fill penalty chosen on the next 20% with the greedy solver, so calibration is the same for every solver). No runs failed.

**Demo scenario** — 3 venues, 5 seeds, ~525 parent orders per seed, policy refreshed every 1 s:

| Solver | Impl. shortfall (bps) | ΔIS vs `exact`, paired (bps) | Decision p50 / p99 / p99.9 (µs) | Refresh compute (ms) |
|---|---|---|---|---|
| best_price | 1.992 ± 0.015 | +0.008 ± 0.020 | 0.99 / 2.2 / 7.0 | 0.07 |
| pro_rata | 2.047 ± 0.014 | +0.062 ± 0.021 | 1.07 / 2.6 / 7.3 | 0.08 |
| greedy | 1.989 ± 0.024 | +0.004 ± 0.013 | 1.05 / 2.6 / 7.3 | 0.10 |
| exact | 1.985 ± 0.024 | reference | 1.04 / 2.4 / 8.7 | 0.19 |
| cpsat | 1.984 ± 0.024 | −0.001 ± 0.001 | 1.42 / 3.4 / 5.8 | 7.1 |
| sa | 1.986 ± 0.022 | +0.002 ± 0.004 | 0.98 / 2.5 / 5.8 | 4.3 |
| random_repair (control) | 1.984 ± 0.024 | −0.000 ± 0.001 | 0.94 / 2.2 / 3.8 | 2.5 |
| **qaoa** (simulated) | 1.984 ± 0.024 | −0.000 ± 0.001 | 1.34 / 3.6 / 8.6 | 109.7 |
| greedy_live (no snapshot) | 1.911 ± 0.019 | −0.074 ± 0.011 | 13.8 / 25.9 / 49.5 | – |

Fill rate was 100% for every solver in this scenario; simulated order→ack is about 2.0 ms p50 for all but `best_price` (0.55 ms, it favours the fastest venue).

What the numbers say:

1. **QAOA matches the exact optimum and adds nothing.** Its allocations coincide with enumeration, at roughly 580× the refresh compute. The `random_repair` control matches too: a 3-venue, 4-unit instance has only about 15 feasible allocations, so 256 shots plus repair find the optimum without any circuit.
2. **The circuit does shift probability toward good states, but not enough to matter here.** Before repair, a QAOA shot is optimal with probability 0.010 at 12 qubits (uniform: 0.00024) and 0.0019 at 16 qubits (uniform: 0.00002); 10–20% of shots are feasible. Simulation cost grows from ~30 ms (12 qubits) to ~275 ms (16 qubits) per problem.
3. **Solution quality barely moves shortfall.** Only `pro_rata` is distinguishable from the exact policy among snapshot solvers. In the stress scenario no solver is distinguishable from another (CIs of ±0.1–0.2 bps on 3 seeds).
4. **Fresh information beats a better solve.** `greedy_live` — a weaker solver on the live book — has the lowest shortfall in the demo (−0.074 ± 0.011 bps) but costs ~13× the decision latency. That is the real trade-off this architecture exposes; the solver choice behind the snapshot is second-order.
5. **Hot-path latency is independent of the solver** by construction: ~1 µs p50 in the simulator for every snapshot solver. With an optimizer thread running in the same interpreter, p99.9 stays under 2 µs but single decisions stalled for up to 7.6 ms (GIL hand-off) in the hot-path benchmark.

**Stress scenario** (3 seeds: 8 orders/s, frequent outages, tight throttles, policy lifetime shorter than the refresh cycle, 60 ms solver budget): ~200 rejects per run with failover, ~20% of fills passive, cancel/replace active, 11% of decisions served by the stale-policy fallback, and QAOA overran its budget on about 118 problems per run, each replaced by the greedy fallback. Fill rate ~96% for all solvers.

## What is implemented

- **Exchange simulator** (`qsor/exchange_sim`): per-venue L2 replay, fees, order and market-data latency, throttles, outages with cancel-on-disconnect, IOC and resting orders with queue position, cancel and cancel/replace, own-fill depletion with replenishment, cross-venue quote fade.
- **Router** (`qsor/router`): immutable versioned `PolicySnapshot`, single-writer `PolicyStore` that never rolls back, deterministic `decide()` with stale-policy fallback.
- **Optimizer** (`qsor/optimizer`): problem builder, solvers above, per-problem timeout and failure fallback, background-thread service.
- **Execution + TCA** (`qsor/sim`): IOC sweeps → failover → passive leg with repricing → deadline sweep; implementation shortfall (slippage + fees + opportunity cost), fill rate, price improvement, decision-latency percentiles, simulated order→ack, queue-position error, stale-policy frequency, solver overhead breakdown.
- **Data** (`python -m project.prepare_data`): seeded synthetic books, written once to `data/<name>-<config hash>/` with SHA-256 and chronological splits; a changed config creates a new version, a modified file fails the checksum.
- **CLI** (`project/`): the three entry points from the specification, as thin wrappers over `qsor`.

## Limitations

- **Synthetic market, and the router's cost model has the same structure as the simulator** (fade, fees, fixed charges). This favours all model-based solvers equally and overstates how well any of them would transfer. Books are exogenous replays: our orders remove liquidity temporarily but never move the price path. Backtests here do not establish live profitability.
- **Python, not C++/Rust.** Latencies are for a CPython function and are useful for comparing designs, not as absolute numbers. Decision latency excludes building the router's view of the books (simulator work). The simulator runs the optimizer inline with a fixed 50 ms publish delay (`delay_model: fixed`) so results are reproducible; `delay_model: measured` uses real solver wall time instead.
- **Tiny instances.** Exhaustive enumeration solves every instance in microseconds, so there is no regime here where a heuristic could win. The one-hot encoding spends one qubit per (venue, unit), and coarse units cost about 0.2–0.3% versus one-lot allocation.
- **No noise model, no QPU, no annealer.** `sa` is classical simulated annealing, not quantum annealing.
- **Not built from the specification's stack list:** gRPC/shared-memory transport, live Prometheus exporter and dashboards (only a scrape-format file per run), historical data, notebook. Queue-position error is small because books barely change within the few-millisecond latencies simulated.

## Disclaimer

For research and educational use. Not investment advice, not a claim of profitable strategy performance, and not a production trading system.
