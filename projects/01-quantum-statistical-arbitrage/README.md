# Quantum-Assisted Statistical Arbitrage Engine

> Hybrid basket discovery + low-latency execution · research prototype

**Status:** Implemented (M1–M5) on synthetic data and a noiseless quantum simulator. The measured result is **negative**: no quantum advantage was observed, and the strategy lost money out of sample on the demo dataset. Nothing here ran on quantum hardware, and nothing here is evidence of live profitability.

## Research question

Does a quantum-assisted discrete basket selector improve **out-of-sample, cost-adjusted** performance or selection quality relative to well-tuned classical methods at equal budgets?

## Answer so far (demo study)

9 walk-forward folds, 13–14 assets, baskets of 4, 5 predeclared seeds, 64 scored candidates per stochastic solve. Full report: [`reports/comparison.html`](reports/comparison.html), raw table: [`reports/comparison.csv`](reports/comparison.csv).

| Solver | P(optimal) | Norm. gap (0 = optimum, 1 = median basket) | Raw feasible samples | Raw-only gap | Time per solve | OOS Sharpe | OOS total P&L |
|---|---|---|---|---|---|---|---|
| QAOA, p=2 (Qiskit statevector) | 78% | 0.035 | 19.9% | 0.466 | 1.08 s (+9,752 shots) | −0.40 | −3.62% |
| CP-SAT (OR-Tools) | 100% | 0.000 | – | – | 12.4 ms | −0.48 | −4.34% |
| Greedy forward selection | 78% | 0.009 | – | – | 56 µs | −0.48 | −4.34% |
| Random search (budget-matched control) | 91% | 0.007 | 8.9% | 0.439 | 1.9 ms | −0.44 | −4.00% |
| Simulated annealing | 31% | 0.158 | 50.7% | 0.525 | 0.6 ms | 0.07 | −0.73% |
| Brute force (reference) | 100% | 0.000 | – | – | 5.1 ms | −0.48 | −4.34% |

What the numbers say:

- **CP-SAT is exact in about 12 ms; simulated QAOA is about 88× slower and optimal in 78% of runs.** These instances have under 900 feasible baskets, so exact enumeration is trivial and no advantage was expected.
- **Most of QAOA's final quality comes from the shared classical repair step.** Uniform random bitstrings pushed through the same repair reach 91% optimal. Looking only at samples that were feasible as drawn ("raw-only gap"), QAOA (0.47) is no better than random (0.44).
- **The circuit does concentrate probability, just not enough to matter.** QAOA puts about 3× more mass on the optimal state than uniform sampling (0.00031 vs 0.00011) and 22% of mass on the right cardinality.
- **Better objective did not mean better P&L.** The exact optimum lost 4.3% out of sample; the worst optimiser (simulated annealing) lost least. With 3 traded folds out of 9 this is noise, but it shows the surrogate objective is the weak link, not the solver.
- **Scaling (8–18 assets, planted instances):** CP-SAT stays optimal in ≤ 34 ms; QAOA drops to 56–89% optimal from 14 assets up while its simulation cost grows as 2^N (1.6 s at 18 qubits). That curve measures classical simulation, not a device.
- **Latency-critical path (Rust):** 400k L2 events replayed at about 190 M events/s, tick-to-decision p50 42 ns / p99 42 ns in-process. Those figures sit at the OS clock resolution (about 42 ns), exclude all network and venue time, and the replayed strategy lost money after costs.

## Architecture

```text
daily OHLCV ─ timestamp normalisation (UTC ns) ─ immutable versioned dataset (sha256)
      │
      ▼  per walk-forward fold, TRAIN window only
pair statistics (ADF strength, stability, residual variance, half-life, beta)
      │
      ▼
binary basket-selection problem ──► QUBO / Ising
      │                                   │
      │ native constraints                │ penalty form
      ▼                                   ▼
greedy · CP-SAT · brute force     random search · simulated annealing · QAOA
      └──────────────┬────────────────────┘      (shared decode + repair)
                     ▼
hedge ratios (Johansen / OLS, train) ─► validation gate (frozen weights) ─► z-score signals
                     ▼
event-driven backtester (TEST window): latency, book-walking fills, fees, borrow, limits, kill switch
                     ▼
runs/<id>/{manifest.json, results.parquet, returns.parquet} ─► reports/comparison.html

L2 quotes ─► Rust engine: book building, spread signal, simulated order latency, fills, latency histogram
```

**Design boundary.** The quantum component is an offline research step that picks which assets go in the basket. Hedge ratios, signals, risk gates and execution are classical; the latency-critical loop is a separate Rust binary with no quantum dependency. Timings cover the whole solve (encode, compile, optimise, sample, decode, repair), not just the circuit.

## Method

Decision variables `z_i ∈ {0,1}`. Base objective (minimise), fitted on the train window only:

```text
f(z) = Σ_i a_i z_i + Σ_{i<j} C_ij z_i z_j
C_ij = − w_coint·strength − w_stab·stability + w_var·(1−R²) + w_hl·half_life
       + w_beta·(β_i−β_j)² + w_sector·[same sector]
a_i  = w_cost · quoted spread
subject to  Σ z_i = K,  at most m assets per sector
```

This is a **pairwise surrogate** for "low residual variance and stable cointegration". Cointegration of the basket is not a quadratic function of `z`; it is tested afterwards (Johansen trace on train, ADF with frozen weights on validation).

QUBO: `E(z) = f(z) + A(Σz − K)²`. Cardinality is encoded exactly. A sector cap of 1 is encoded exactly with pairwise terms; a cap ≥ 2 would need slack qubits, so it is enforced by feasibility checks and a deterministic greedy repair shared by every sampling solver. Tests verify the QUBO, Ising and base-objective forms agree on every bitstring and that the QUBO ground state equals the brute-force optimum.

QAOA: standard X mixer, `p` layers, COBYLA on CVaR shot estimates, one qubit per asset. The Qiskit circuit is transpiled to `rz, sx, cx` to report depth and two-qubit counts (123–132 and 312–364 for 13–14 qubits at p=2, all-to-all), then simulated with `Statevector`; a numpy backend producing the same probabilities (tested to 1e-9) is used for scaling runs.

Fair-comparison protocol as implemented:

1. Tiny instances are checked against brute force and a hand-computed fixture.
2. Runs are compared only if they share an experiment fingerprint (data, objective, constraints, splits, seeds, budget); others are listed as excluded.
3. Budget = scored candidates for stochastic solvers plus a wall-time limit for exact ones; circuit shots are reported separately because QAOA spends them on top.
4. Every stage is timed and stored per run.
5. Seeds are declared in the config; intervals are bootstrap CIs over seeds; failed runs fall back to greedy and stay in the table flagged as failed.
6. Simulator-only diagnostics are labelled as such. There are no hardware results.
7. P&L is out of sample with spread, impact, fees and borrow.

## How to run

Requires Python 3.12. Rust (cargo) is optional and only needed for the L2 replay engine.

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt && pip install --no-deps -e .

# Everything on reduced sizes, CPU only, about 15 seconds: tests, data, both solver families, scaling, engine, report
./reproduce.sh            # writes reports/ci/comparison.html
./reproduce.sh demo       # the full demo study above (about 2 minutes)
```

Individual steps:

```bash
python -m project.prepare_data --config configs/demo.yaml          # versioned synthetic dataset
python -m project.run --config configs/demo.yaml --solver classical
python -m project.run --config configs/demo.yaml --solver quantum-sim
python benchmarks/scaling.py --runs runs/                           # optional scaling study
python -m project.replay_l2 --config configs/demo.yaml --build      # optional Rust L2 replay
python -m project.report --runs runs/ --output reports/comparison.html
pytest -q && (cd engine && cargo test --release)
```

To use your own daily bars, set `data.source: file` with `data.path` (long CSV/Parquet: `ts, symbol, open, high, low, close, volume`), `data.timezone` for naive timestamps, and optionally `data.assets_path` (`symbol, sector, spread_bps, borrow_bps_annual`). Include delisted symbols to avoid survivorship bias and check your data licence; no market data is downloaded or shipped by this repository.

## Repository layout

```text
configs/            demo.yaml (full study), ci.yaml (reduced)
src/project/
  prepare_data.py   run.py   report.py   replay_l2.py        # CLI entry points
  data/             synthetic generator, loader, timestamp normalisation, point-in-time universe
  research/         stats (ADF, half-life, pairs), qubo, hedge, walkforward
  solvers/          greedy, CP-SAT, brute force, random search, simulated annealing, registry + fallback
  quantum/          QAOA (Qiskit and numpy backends)
  backtest/         event-driven engine, metrics
engine/             Rust crate: L2 book, replay, signal, fills, latency (no external crates)
benchmarks/         scaling.py
tests/              96 tests (objective/encoding, solvers, QAOA, stats, data, backtest, leakage, CLI)
reports/            comparison.html + comparison.csv (generated)
runs/               one directory per run with manifest.json (generated, git-ignored)
```

## Roadmap

- [x] **M1:** Pairs/spread baseline and event-driven backtester
- [x] **M2:** Sparse baskets and deterministic classical QUBO optimisation (CP-SAT, brute force, greedy)
- [x] **M3:** QAOA on a simulator, candidate-basket comparison
- [x] **M4:** Rust quote processing, replay and execution costs
- [x] **M5:** Walk-forward study with negative results published

## Deviations from the original specification

- **Data:** synthetic only (latent-factor daily bars with a late listing, a delisting and a regime break; synthetic L2). No real ETF or crypto data is included; a file importer is provided instead.
- **Quantum:** QAOA on a noiseless simulator only. No annealer, no QPU, so no noise, queue-time or cost figures.
- **Sector cap ≥ 2** is enforced by repair, not inside the QUBO (see Method).
- **Language:** Rust was chosen over C++ for the engine.
- **Docker:** a `Dockerfile` is included but has not been built or run.
- **Notebook, demo video and architecture image** from the portfolio list are not produced; the HTML report stands in for the benchmark notebook.

## Known limitations

- Instances are small enough to enumerate, which is exactly the regime where quantum heuristics cannot win. Larger universes exceed what a statevector simulator can hold (limit set at 20 qubits).
- The pairwise surrogate cannot see cointegration that needs three or more assets; the generator plants such a group on purpose and no solver finds it.
- The validation gate rejects 6 of 9 folds, so trading statistics rest on 3 folds and a handful of round trips. Sharpe intervals cover solver-seed variation only.
- Signal and risk parameters were set once and not tuned on test windows; iterating on them would introduce multiple-testing bias.
- Synthetic books and fixed latency are a simplification of real microstructure; borrow availability, halts and partial fills are not modelled.

## Disclaimer

For research and educational use. Not investment advice, not a claim of profitable strategy performance, and not a production trading system. Verify market-data permissions and exchange rules before any live integration.
