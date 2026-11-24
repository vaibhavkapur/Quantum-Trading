# Quantum Portfolio Rebalancing Engine

Implemented CPU research prototype for integer-lot, long-only rebalancing with fees, cash reserves, exposure limits, QAOA, and an exact classical reference. All five implementation milestones are delivered. Orders are simulated at fixed prices; there are no exchange or broker connections.

## Run the demo

Use Python 3.12 (the pinned environment was verified on Python 3.12.14). Run these commands from this directory:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install --no-deps --no-build-isolation -e .

python -m project.prepare_data --config configs/demo.yaml
python -m project.run --config configs/demo.yaml --solver classical
python -m project.run --config configs/demo.yaml --solver quantum-sim
python -m project.report --runs runs/ --output reports/comparison.html
python -m pytest -q
```

Open `reports/comparison.html` for before/after holdings, fees, FIFO tax estimates, simulated fill sequence, solver provenance, and stage timings. The report is self-contained and has a solver/asset filter. Dataset and run paths resolve relative to the YAML configuration file, so CLI invocation also works from another working directory when the package is installed.

One command reproduces the dataset, scaling experiment, raw results, confidence intervals, and visual reports:

```bash
python -m project.reproduce --config configs/demo.yaml --quick
# Full predeclared scaling grid:
python -m benchmarks.run --config configs/demo.yaml --output benchmarks/results/new-run
```

`--quick` retains all three seeds and five solvers, uses the first two asset counts, and reduces evaluation/shot budgets. No QPU account, API key, or external data download is needed.

## Delivered milestones

- [x] **M1:** Deterministic integer-cent accounting, fee rounding, FIFO tax estimates, cash conservation, and executable lot domains.
- [x] **M2:** Exact MILP with quadratic-product lifting; exhaustive verification; greedy, convex-relaxation, and annealing baselines.
- [x] **M3:** Small one-hot QUBO and Ising encoding, verified over every bit string of a test instance.
- [x] **M4:** Qiskit QAOA simulation, measured-sample decoding, and a shared classical feasibility repair.
- [x] **M5:** FastAPI, SQLite audit trail, standalone HTML reports, seeded scaling benchmarks, and CPU reproduction.

## Architecture and implementation

```text
Versioned snapshot + configuration
    → integer-cent accounting and local lot domains
    → shared quadratic objective and linear constraints
    → MILP / greedy / convex-round / anneal / QAOA candidate
    → integer-cent risk gate and, if necessary, nearest-lot MILP repair
    → cash-raising sales first, then cash-consuming simulated fills
    → before/after holdings, fees, cash, exposure, audit record and HTML report
```

The quantum optimizer is a research component. Budget checks, repair, order sequencing, storage, and API behavior are classical. Nothing here is a latency-critical exchange execution system.

Code locations:

- `rebalancing/model.py`: validation, integer lots/money, FIFO tax estimates, and objective/report calculations.
- `rebalancing/encoding.py`: common quadratic objective, linear constraints, one-hot QUBO, and Ising conversion.
- `rebalancing/solvers.py`: exact MILP, exhaustive verification, classical candidates, and feasibility repair.
- `rebalancing/quantum.py`: parameterized Qiskit circuits, statevector expectations, final shots, and decoding.
- `rebalancing/engine.py`: common pipeline, explicit fallback, and deterministic recommendation IDs.
- `rebalancing/storage.py`: content-addressed JSON exports and SQLite hash-chain audit.
- `project/`: thin CLI modules; `api/`: local HTTP service; `benchmarks/`: experiment runner.

The chosen stack is NumPy, SciPy/HiGHS, Qiskit, FastAPI, SQLite, PyYAML, and pytest. SciPy supplies the exact MILP and convex optimizer; OR-Tools, CVXPY, D-Wave Ocean, and PostgreSQL are not required. No hardware adapter is implemented.

## Accounting, objective, and controls

Every monetary input is an **integer number of cents**, every share quantity is an integer, and every trade is an integer multiple of its asset's `lot_size`. Existing odd-lot residuals may remain. All assets use the same currency. Missing target weight is the target cash allocation.

For initial NAV `V`, signed lot trade `d_i`, lot size `L_i`, price `p_i`, holdings `h_i`, and fee/tax cost `c_i(d_i)`:

```text
h_post,i = h_i + L_i d_i
cash_post = cash_initial - Σ p_i L_i d_i - Σ c_i(d_i)
V_post = V - Σ c_i(d_i)

objective = Σ (p_i h_post,i / V - target_i)^2
          + (cash_post / V - target_cash)^2
          + turnover_weight × Σ |p_i L_i d_i| / V
          + cost_weight × Σ c_i(d_i) / V
```

**Tracking and all exposure caps use initial NAV as their fixed denominator.** This keeps the objective quadratic despite fees. It is not the nonlinear objective using post-fee NAV. Reports display both reference-NAV and actual post-NAV weights so the distinction is visible. All baselines optimize exactly the same stated objective.

Hard controls include long-only holdings; position and group caps; minimum cash as an amount and/or initial-NAV fraction; per-asset trade-lot and trade-notional limits; and minimum nonzero order notional. Zero orders incur no fee and are exempt from the minimum notional. All constraints apply to the final portfolio. Cash-raising fills are sequenced first to prevent temporary borrowing under the instantaneous-settlement assumption.

Proportional fees are rounded **up** to cents once per asset order, then the fixed fee is added. Optional `tax_lots` cover all existing shares and are consumed FIFO. Positive gains, without loss offsets, are multiplied by `tax_rate_bps` and rounded up to cents. This is a simplified scenario cost, not jurisdiction-specific tax accounting or tax-lot selection optimization.

The risk gate verifies `V = cash_post + holdings_value_post + fees + taxes` exactly in cents. It emits no orders when no feasible portfolio is found. Validations reject duplicate symbols, non-finite values, unknown fields, bad UTC timestamps, invalid tax coverage, and malformed budgets. Explicit resource guards limit assets, trade domains, MILP size, circuit depth, and simulator qubits.

## Solvers and quantum boundary

`--solver classical` uses SciPy's HiGHS MILP. One binary variable selects each allowed order size. Products between different assets are lifted into auxiliary variables with exact binary-product constraints. Within-asset products vanish under one-hot selection. This solves the quadratic objective exactly when HiGHS reports optimality; an incumbent returned at a limit is labeled `feasible_limit`, not optimal. Exhaustive search independently checks small instances in the tests.

`greedy` rounds each asset toward its target while charging turnover and fees. `convex-round` solves the convex hull relaxation over fractional order choices using SLSQP, then rounds. `anneal` performs seeded discrete simulated annealing. All use the common feasibility gate.

`quantum-sim` builds real QAOA circuits with Hadamard preparation, RZ/RZZ cost phases, and RX mixers. Qiskit `Statevector` supplies noiseless expectations for parameter search; only the final distribution is sampled with the configured `shots`. **Optimizer shots are zero**, and circuit execution is CPU simulation. It is not a shot-noisy hardware benchmark.

The QUBO includes the common objective, local trade domains, and one-hot penalties. Its penalty exceeds the objective's binary coefficient range, so its ground state obeys one-hot selection. **Cash and group-cap inequalities are deliberately outside this QUBO**: measured feasible samples are postselected, and otherwise the selected sample is projected and repaired. The result schema explicitly labels this scope, raw measured feasibility, and the selection procedure. This is a hybrid optimizer, not a fully constrained quantum encoding.

Repair minimizes integer lot distance to the proposed order vector with a separate linear MILP. It does not solve the tracking objective as a hidden exact fallback. Its time, changes, and violations are recorded. Infeasibility remains an error. Backend/resource failures also remain errors unless the caller explicitly requests:

```bash
python -m project.run --config configs/demo.yaml --solver quantum-sim --fallback classical
```

Fallbacks record requested/actual solvers and the reason. Benchmarks disable fallback entirely.

## Data and reproducibility

`data/synthetic-v1.json` is the included deterministic, fixed-price snapshot. Its adjacent manifest records the generator version and seed, SHA-256 hashes, UTC timestamp normalization, and **CC0-1.0** synthetic-data terms. Generation is idempotent and refuses to overwrite differing content. Loading verifies the file hash.

There is no fitted predictor, chronological trading backtest, or historical dataset in this implementation; train/validation/test windows do not apply to independent synthetic snapshots. Price impact, partial fills, settlement delays, FX, and corporate actions are outside the model. Liquidity is represented by explicit per-order limits.

`requirements.txt` pins the complete environment used for verification, including test dependencies. The experiment manifest records seeds, budgets, data hashes, source-code hash, package/platform versions, and execution mode. Recommendation IDs hash normalized inputs, solver choices, configuration, objective/implementation version, and chosen lots; they exclude timing. Individual run IDs additionally hash the observed result including timing.

SQLite is the authoritative audit store. JSON exports are immutable and content addressed. `/audit/verify` checks payload and hash-chain consistency. The chain detects edits relative to its stored history; it is not externally signed and cannot prove the absence of wholesale rewriting or tail deletion without an independently saved head hash. The API has no delete or update endpoint.

## Published CPU benchmark

The checked-in experiment uses 2, 3, 4, and 6 assets; solver seeds **7, 19, 41**; the same fixed snapshot per asset count; 60 evaluations for iterative methods; 512 final QAOA shots; depth 1; a 10-second solver budget; and a separate 10-second repair budget. The default QAOA cap is 16 qubits.

- [Visual benchmark table and timing chart](benchmarks/results/demo/comparison.html)
- [Detailed holdings and costs](benchmarks/results/demo/holdings.html)
- [Raw CSV results](benchmarks/results/demo/results.csv) and [JSON results](benchmarks/results/demo/results.json)
- [95% confidence intervals](benchmarks/results/demo/comparison.summary.csv)
- [Machine-readable experiment manifest](benchmarks/results/demo/manifest.json)
- [Audit verification](benchmarks/results/demo/audit-verification.json)

The experiment has **60 attempts: 57 successes and 3 explicit QAOA resource refusals**. Six assets require 18 one-hot qubits and exceed the configured cap. QAOA reaches the certified optimum in the two- and three-asset cases; two of the three four-asset trials have a nonzero gap. These tiny sampled spaces are verification examples, not evidence of quantum advantage. See the raw results for actual runtimes rather than extrapolating simulator performance to hardware.

Timing covers data loading, encoding, solver work, repair, decoding, and audit persistence. Quantum records additionally break out state preparation, compilation, optimizer work, statevector evolution, final measurement, decoding, and zero local queue/communication time. Nested timers must not be added to their parents. Benchmark backend initialization is recorded once in the manifest; per-run measurements use a warmed process. Python process startup and report rendering are excluded. CLI output reports persistence and end-to-end time; the immutable stored record stops before its own persistence.

Wall limits are cooperative and may overrun by one evaluation or native solver call. They are not a process-isolation deadline. Iterative evaluation limits apply to SLSQP, annealing, and QAOA; MILP reports nodes and native optimality gap instead of inventing a comparable objective-evaluation count. All methods have the same nominal solver and repair time budgets. Absolute quality gaps use only certified exact references. Failed attempts remain in the denominator; successful-seed means include Student-t 95% intervals. Three seeds are exploratory and timing is machine dependent.

## API

```bash
python -m uvicorn api.app:app --host 127.0.0.1 --port 8000
```

Interactive API documentation is at `http://127.0.0.1:8000/docs`. The service is intended for local use and has no authentication. `REBALANCING_RUN_DIR` selects its SQLite/export directory; the default is `runs/api`.

```bash
python -c 'import json; print(json.dumps({"portfolio":json.load(open("data/synthetic-v1.json")),"solver":"classical"}))' \
  | curl --json @- http://127.0.0.1:8000/rebalance
```

Endpoints: `GET /health`, `POST /rebalance`, `GET /runs/{run_id}`, and `GET /audit/verify`. Requests accept `portfolio`, `solver`, optional `budget`, and optional `fallback`. Invalid input, infeasibility, and resource limits return structured 422 responses; backend failures return 503. Successful responses include the full audited recommendation and a run ID.

## Verification and references

Tests cover the analytic zero-error fixture, MILP versus exhaustive search, every tiny QUBO/Ising bit string, FIFO costs, local/global constraints, cash conservation and fill sequencing, seeded circuit reproducibility, evaluation limits, failure/fallback behavior, dataset immutability, concurrent audit writes, tamper detection, HTML escaping, the CLI contract, API integration, and benchmark failure retention.

The implementation follows the documented [SciPy MILP interface](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.milp.html) and [Qiskit Statevector interface](https://quantum.cloud.ibm.com/docs/en/api/qiskit/qiskit.quantum_info.Statevector).

For research and education. Not investment advice, a live trading system, a profitable-strategy claim, or a claim of quantum advantage.
