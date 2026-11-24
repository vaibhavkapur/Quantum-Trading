# Quantum-Enhanced Market Making Engine

Implemented CPU-only research prototype: a C++20 price-time-priority matching and risk engine, Python L2 replay, five market-making strategies, and asynchronous, versioned QAOA policy optimization. The original proposal is preserved in [SPECIFICATION.md](SPECIFICATION.md).

**Measured result:** all 60 bundled demo trials completed with no optimizer fallbacks. QAOA selected the same policies as exhaustive grid search on these small instances, with additional computation. This is a verification result, **not evidence of quantum advantage or live profitability**.

## Run it

Requires Python **3.12+** for the pinned environment and a C++20 compiler (`c++`, `clang++` or `g++`). The package supports Python 3.10+ with compatible dependency versions. No API keys, QPU account, external market feed or GPU are needed.

Run these commands from the project directory.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install -e .

python -m project.prepare_data --config configs/demo.yaml
python -m project.run --config configs/demo.yaml --solver classical
python -m project.run --config configs/demo.yaml --solver quantum-sim
python -m project.report --runs runs/ --output reports/comparison.html
```

The native library compiles automatically on first use and is cached by source, platform and compiler settings. `CXX` selects a compiler; `MM_BUILD_DIR` selects a writable build directory. Compilation failures propagate with the compiler diagnostics in the subprocess exception.

To include every baseline, use `--solver all`. Individual choices are `fixed-spread`, `inventory-skew`, `avellaneda-stoikov`, `classical` and `quantum-sim`.

The reduced CPU-only reproducibility command runs tests, generates data, executes all five strategies, and regenerates the comparison:

```bash
PYTHON=.venv/bin/python sh benchmarks/reproduce.sh
```

Demo settings use 1,200 frames × four scenarios × three predeclared seeds. CI settings use 180 frames × two scenarios × two seeds. Runs are written to new directories, preserving prior experiments. Run from the project root so the relative config/data/output paths resolve consistently.

## Results and artifacts

- [Comparison report](reports/comparison.html): standalone HTML, with an SVG P&L plot and benchmark table.
- [Raw aggregate results](reports/comparison.csv) and [machine-readable report](reports/comparison.json): means and two-sided Student-t 95% confidence intervals across seeds.
- [Solver scaling](reports/solver_scaling.csv): 2–6 qubits, separate from full replay performance; regenerate with `python benchmarks/solver_scaling.py`.
- [Reproducible notebook](benchmarks/demo.ipynb): invokes the same CLI pipeline, with no hidden data or state.
- `runs/*/manifest.json`: configuration, dataset/source hashes, environment, chronological splits, budgets, initialization and full experiment timing, and trial success/failure records.
- Each trial contains `trajectory.csv`, `fills.csv`, `metrics.json`, `validation.json`, `policies.json`, `optimizer.json` and `metrics.prom`.

The published demo was run on macOS arm64, Python 3.12.14, using the versions in `requirements.txt`. Three seeds are too few for strong statistical conclusions. Both optimized strategies produced mean test P&L of approximately 1.980 in normal flow, 1.560 in trending flow, −1.517 in volatile flow, and 24.513 in the synthetic liquidity-crisis case. The latter is particularly sensitive to the simplified flow model: competing depth is thin and wide, while the maker can improve the spread immediately.

The classical policy-update totals averaged about 0.39–0.56 seconds per trial; QAOA about 0.60–0.69 seconds on this run. Timing varies with the machine and load. See raw manifests and optimizer logs rather than treating these numbers as stable performance guarantees. First-use library loading is included in full trial/experiment timing.

## Architecture

```text
Versioned synthetic L2 + signed market-flow replay
       │
       ▼
C++ FIFO book ── fills ── inventory / cash / fee accounting
       │                         │
       └── observed mid / volatility / risk state
                                 │
                   native quote + hard risk gate
                                 ▲
                   immutable policy (versioned)
                                 ▲
Historical window → background optimizer → publication / classical fallback
                   classical grid or QAOA
```

The latency-critical book, risk limits, quote generation, post-only placement, fees and cancel-all run in C++. Python coordinates research replay and reporting. This is an in-process simulated exchange; there is no exchange network adapter.

`PolicyService.request()` and `poll()` allow nonblocking background optimization. The benchmark uses **offline activation barriers** at predeclared update epochs so replay does not depend on OS thread scheduling. A bounded wait is recorded separately from quote latency and included in full trial time. Solver exceptions, invalid versions, expired results and busy workers use a versioned inventory-skew fallback. Late results are discarded. Deadlines are cooperative between candidate/circuit evaluations; an individual native/library call is not forcibly preempted. The service joins its worker on shutdown.

Qiskit is initialized on the caller thread before creating a worker. This avoids a reproducible native-circuit initialization crash when Qiskit is first loaded from a short-lived worker in the tested environment.

## Matching and risk semantics

- The native order book supports limit and market orders, price-time priority, partial fills, cancellations, FIFO queue-ahead queries and self-trade rejection.
- Native order-level matching has exact FIFO semantics. **L2 replay infers queue placement**: depth increases join the queue tail, reductions cancel the newest external quantity first, and unchanged quantities retain priority. This is not a reconstruction of real exchange order IDs.
- Each replay frame represents the end of an interval. Its market buys/sells consume the previous visible book first; the new mark is then observed. A seeded field fixes buy/sell ordering within the interval.
- When the observed mid changes, remaining maker quotes are cancelled after interval flow, before external depth reconciliation. Depth snapshots themselves never generate fills. Quotes are then refreshed immediately. There is no network delay, hidden liquidity or market impact feedback. The first frame initializes the book and has no preceding flow execution.
- Unchanged quote price and remaining size retain queue priority. Replenishment or repricing loses priority and pays the configured cancellation cost. Quotes are post-only and clamped below/above the visible opposite side.
- Quote prices are `floor(mid − half_spread − inventory × skew)` and `ceil(mid + half_spread − inventory × skew)`, subject to post-only and price bounds.
- Soft inventory bands restrict risk-increasing quote quantity. The independent hard limit reserves worst-case inventory for **all open same-side maker orders**, without netting against opposite-side orders.
- Loss, drawdown and volatility limits are checked on every quote cycle. Emergency cancellation is latched for the remainder of that replay. Inventory limits hold at every fill; loss/drawdown thresholds are observed at quote cycles and can overshoot between cycles.
- End-of-window liquidation consumes actual visible depth at taker prices and fees. Any residual remains marked to the last observed mid and is explicitly reported.

## Policies and optimization objective

Fixed-spread and inventory-skew are simple deterministic baselines. The Avellaneda–Stoikov-inspired baseline uses a finite-horizon reservation-price skew and spread based on trailing volatility, with explicit dimensionless constants. It is an approximation, not a calibration of the original model.

The optimized grid selects half-spread, inventory-skew coefficient and inventory band. Default settings contain **27 candidates**, encoded with **5 qubits**. Each candidate is replayed over the same bounded historical window from flat inventory, with identical execution costs and risk limits. Candidate cost is:

```text
cost = (−net_PnL + adverse_penalty × max(0, signed_adverse_selection_cost)) / frames
       + inventory_penalty × mean(inventory²)
```

Net P&L already includes maker/taker fees, cancellation charges and liquidation. The adverse-selection term is an additional risk penalty, not an accounting subtraction from P&L. Positive markout cost means adverse movement; negative values indicate favorable movement. Markouts beyond the window are censored and counted.

The classical solver enumerates every feasible policy and returns the exact minimum. This is the strong classical baseline for the chosen instance sizes; OR-Tools is unnecessary for this finite grid.

The quantum solver encodes normalized costs in a diagonal Hamiltonian, applies alternating cost phases and X-mixer layers, and trains angles with COBYLA using **exact noiseless statevector expectations**. It samples the final distribution with the configured finite shot count and selects the best feasible measured policy. Padding states receive a strictly larger cost and are discarded during decoding; if all samples are invalid, the service falls back. There is no repair that silently inserts the classical optimum.

**Encoding enumerates the entire grid classically before QAOA.** That work is charged to both solvers and independently logged. This method exists to verify small circuit instances and measure overhead; it cannot demonstrate an efficient quantum oracle or a speedup. It uses a general diagonal Hamiltonian, not a claimed quadratic QUBO reduction.

`max_evaluations` bounds candidate evaluations and separately bounds QAOA expectation evaluations; the configuration must allow the entire grid. Both solvers have the same wall-time ceiling and input windows. Logs distinguish classical objective evaluations from quantum expectations, shots, circuit compilation, state preparation, sampling, decoding and all zero-valued communication/QPU stages. No physical QPU, noise model or hardware cost is included.

## Data, splits and metrics

Bundled data is synthetic and released under **CC0-1.0**. `prepare_data` generates content-addressed directories with immutable JSONL files and SHA-256 manifests. Reusing an existing directory verifies all files and fails on checksum mismatch. No third-party market data is redistributed.

Flow uses seeded Poisson draws with bounded decaying excitation inspired by Hawkes processes; this is not an exact Hawkes sampler. Signed flow is correlated with synthetic price movement. Scenarios are normal, volatile, trend and liquidity crisis. Stress begins only in the test window, leaving the same-seed train and validation prefixes identical across scenarios.

Timestamps are strictly increasing integer nanoseconds from a synthetic UTC epoch; prices are integer ticks and quantities integer lots. Each normalized frame contains `timestamp_ns`, `mid_ticks`, `bids`, `asks`, `market_buy_qty`, `market_sell_qty`, `buy_first`, and `regime`. Real data requires an external adapter with explicit licensing, timestamp normalization and interval-flow attribution; this repository does not claim a verified historical-exchange adapter.

Chronological splits are 50% training, 20% validation and 30% test. Initial calibration uses training only. Validation is an audit with no test-driven hyperparameter selection. Each test run starts flat. Subsequent policy updates use only the trailing window strictly before activation, including already observed test events. This is a walk-forward experiment, not a fixed-policy holdout. Volatility also uses only observed prices.

Metrics include net and gross spread capture, net P&L, fees/rebates, signed adverse-selection markouts, cancellation costs, maximum drawdown, inventory distribution, risk halts and fill counts. Empirical fill probability is the fraction of quoted intervals with at least one maker fill; it is not a calibrated probability for an individual order. Sharpe is a **non-annualized, within-window P&L-change statistic**, including liquidation, not a return-on-capital or investment Sharpe ratio.

Quote-to-ack p99 measures local Python-to-C++ quote generation, risk checks, and immediate simulated order acceptance. Native-only duration is also saved. It excludes data loading, flow replay and optimizer barriers; those appear in full-pipeline timing. It does not measure exchange acknowledgement latency.

Reports separate incompatible config/dataset/source groups. Repeated runs of the same seed do not increase sample size; the latest trial is used, including failed or incomplete trials. Failed runs are listed; fallback results remain in the strategy sample.

## Tests and containers

```bash
python -m pytest -q
clang++ -std=c++20 -O1 -g -fsanitize=address,undefined \
  -fno-omit-frame-pointer tests/native_sanitizer.cpp -o /tmp/mm-native-sanitizer
/tmp/mm-native-sanitizer
```

The implementation passes 52 tests on both macOS and Linux in Docker, covering matching, FIFO depth reconciliation, partial fills, accounting, soft/hard risk limits, post-only quotes, invalid input, deterministic data/P&L, temporal boundaries, QAOA matrix evolution and known optima, solver equality, timeout/failure fallback, stale publication, and end-to-end CLIs/reporting. The C++ sanitizer harness passed 5,000 randomized quote/flow cycles with inventory assertions. The Docker default command also completed all 20 reduced CI trials with networking disabled. See [validation evidence](reports/validation.json).

```bash
docker build -t quantum-market-making:local .
docker run --rm -v "$PWD/runs:/app/runs" -v "$PWD/data:/app/data" \
  quantum-market-making:local
```

Prometheus/Grafana files are included for inspecting **stored experiment results**. The exporter is read-only; it does not start a trading service.

```bash
python -m project.serve_metrics --runs runs --port 9108
# Or start the optional local observability stack:
docker compose up --build
```

The metrics endpoint is `http://127.0.0.1:9108/metrics`; Compose exposes Prometheus on port 9090 and Grafana on 3000, all bound to localhost. Grafana provisions the included dashboard and Prometheus datasource.

## Delivered roadmap

- [x] M1: C++ book, L2 replay and matching tests.
- [x] M2: Fixed-spread, inventory-skew, AS-inspired quotes and classical risk gates.
- [x] M3: Bounded discrete policy objective with exact enumeration.
- [x] M4: Qiskit QAOA, background optimizer, immutable versions and fallback.
- [x] M5: Out-of-sample scenarios, multiple seeds, confidence intervals, raw metrics and latency reports.

The project deliberately has no live exchange messaging or claims of production readiness. Simplified fill/queue models, instantaneous refresh, synthetic regimes and tiny enumerated policy grids limit all conclusions.

References: [Qiskit Statevector](https://quantum.cloud.ibm.com/docs/en/api/qiskit/qiskit.quantum_info.Statevector), [Qiskit DiagonalGate](https://quantum.cloud.ibm.com/docs/en/api/qiskit/qiskit.circuit.library.DiagonalGate), [QAOA paper](https://arxiv.org/abs/1411.4028).
