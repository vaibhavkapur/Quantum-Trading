# Quantum-Optimized Pairs Trading System

A working, CPU-reproducible research implementation of cointegration screening,
constrained pair selection and walk-forward trading. Quantum computation is limited
to asynchronous pair selection; signals, accounting and risk gates are classical.

**Implemented:** M1–M5. The original proposal is preserved in
[research/SPECIFICATION.md](research/SPECIFICATION.md). This is a synthetic-data
research demonstration, not a production execution service or evidence of quantum
advantage or profitable live trading.

## Quick start

Use Python 3.11 or newer (the recorded environment uses Python 3.12). From this folder:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-lock.txt

python -m project.prepare_data --config configs/demo.yaml
python -m project.run --config configs/demo.yaml --solver classical
python -m project.run --config configs/demo.yaml --solver quantum-sim
python -m project.report --runs runs/ --output reports/comparison.html
python -m pytest -q
```

The existing `.venv` is ready to use in this workspace. `requirements.txt` pins the
direct dependencies; `requirements-lock.txt` also pins the tested transitive packages.
`pip install -e '.[quantum,test]'` is an alternative development installation with
compatible version ranges. Qiskit is optional for the NumPy backend; OR-Tools is the
strong classical baseline. No API keys, QPU account or external data access are needed.

For a fast CPU-only reproduction, including both quantum adapters' unit tests:

```bash
.venv/bin/python -m project.prepare_data --config configs/ci.yaml
.venv/bin/python -m project.run --config configs/ci.yaml --solver all
.venv/bin/python -m project.report --runs runs/ci --output reports/ci.html
.venv/bin/python -m pytest -q
```

`--solver` also accepts `top-k`, `greedy`, `milp`, or `annealing` individually.
The demo uses Qiskit; the CI configuration uses the independent NumPy statevector
implementation. Both simulate actual QAOA cost and mixer evolution.

## Architecture

```text
Immutable synthetic prices + point-in-time fixed universe
    → chronological train / validation / test folds
    → train-only Engle–Granger, hedge ratio and half-life diagnostics
    → common capital / security / sector-constrained objective
    → ranking / greedy / SCIP MILP / annealing / QAOA
    → classical feasibility gate
    → validation-only entry-threshold choice
    → lagged z-score signals, frozen share hedges and costed test fills
    → raw artifacts, seed intervals, sensitivity results and HTML report
```

The implementation lives in `pairs_trading/`: `data.py`, `research.py`,
`selection.py`, `quantum.py`, `execution.py`, `experiment.py`, and `reporting.py`.
`project/` contains the three thin CLI modules. Config paths inside YAML resolve
relative to the project root, not the caller's working directory.

## Statistical and execution contract

- The demo creates nine fictitious assets in three sectors, with a common random
  walk per sector and independent AR(1) deviations. The process deliberately
  contains mean reversion. All assets exist throughout the sample. Prices are
  positive levels, not log prices; the OLS coefficient is a **share hedge ratio**.
- Data is generated with a fixed seed and stored under `data/versions/<SHA256>/`.
  The loader verifies both price and metadata hashes. It rejects missing,
  nonfinite, nonpositive, duplicate or unsorted observations rather than filling
  them. The generator rejects an attempted overwrite of altered immutable data.
- Generated data is released under CC0-1.0. Metadata records the generator,
  universe, adjustment assumptions and UTC normalization. The 21:00 UTC synthetic
  weekday timestamps do not model exchange holidays or daylight-saving changes.
- Six rolling folds use 252 training, 63 validation and 63 test observations.
  Test windows never overlap. Incomplete trailing windows are excluded and counted
  in the manifest. Each later fold can use earlier test observations only after
  they have become historical data.
- Candidate screening uses only each fold's training window. Engle–Granger tests
  include an intercept and up to four ADF lags selected with AIC. The AR(1)
  half-life is `-log(2)/log(phi)`; fits with `phi` outside `(0,1)` are rejected.
  Perfectly collinear, constant and unsupported negative hedge fits are excluded.
  The six strongest eligible candidates become the common optimization universe.
- Pair quality is a declared training proxy:
  `-log10(max(p, 1e-8))/(1 + half_life/10) - 2*one_way_cost*252/half_life`.
  It is not a forecast of expected returns. Pairwise p-values are uncorrected for
  multiple testing; this limits interpretation of discoveries.
- Entry thresholds `[1.25, 1.75, 2.25]` are chosen on validation Sharpe only, with
  deterministic ties. Test sensitivity results never influence parameter choice.
- A signal at close `t-1` executes at close `t`. Hedge refits use a trailing window
  ending no later than the signal close. Refitting occurs while flat; a live
  trade freezes its fitted model, share quantities and stop basis until exit.
  P&L is computed from actual marked shares, independently of model changes.
- Exit at mean reversion/sign crossing, an extreme z-score, an 8% gross trade
  loss observed at the prior close, 20 holding bars, insolvency, or the known fold
  boundary. There is no same-close exit and re-entry. Boundary liquidation is
  costed, and its trade is not counted as a mean-reversion success.
- Every fill pays commission plus slippage. Shorts pay borrow and longs pay
  financing on prior-close marked notional using elapsed calendar days/365.
  Capital slots reserve gross entry capital; a classical entry gate caps gross
  notional after fees at current equity. Gross exposure can drift between fills.
  Open positions are not continuously rebalanced to maintain a target hedge.

## Common selection objective and QUBO

Minimize `f(x) = -sum(score_i*x_i) + lambda*sum(abs(corr_ij)*x_i*x_j, i<j)`
subject to nonnegative integer linear limits. Correlations are correlations of
training residuals, not test returns.

The default constraints allow at most three pairs, three capital units, one pair
per security and one per sector. A cross-sector pair consumes a slot in each
sector. The problem builder also supports distinct positive integer capital
requirements per candidate. Equivalent constraint rows are deduplicated.

- **Top-K:** visit the quality ranking and admit feasible, objective-improving
  additions, up to the declared limits.
- **Greedy:** choose the best feasible marginal objective improvement each step.
- **MILP:** OR-Tools SCIP with binary product variables and exact linearization of
  every correlation term. The result and optimal bound are recorded.
- **Annealing:** seeded Metropolis bit flips on the same penalized QUBO.
- **QAOA:** uniform superposition, diagonal cost phases and X mixers; COBYLA
  optimizes exact noiseless expectations. Only final decoding uses finite shots.
  The Qiskit circuit is transpiled into H/RZ/RX/CX gates and simulated on CPU.

At-most-one rows become pairwise conflict penalties. Other `a*x <= b` constraints
become `P*(a*x + slack - b)^2`, with bounded binary slack spanning every integer
from zero to `b`. `P = sum(abs(objective coefficients)) + 1`, exceeding the full
objective range, so an infeasible assignment cannot beat the feasible empty set.
All bitstrings of tiny fixtures are checked against their classical objectives.

Every sampled decision is validated classically. Deterministic repair removes
pairs only, choosing the least harmful removal from violated constraints. The
best repaired sampled decision (or empty cash selection) is returned. Sample
feasibility, repairs, decoding evaluations and time are disclosed. The exact
brute-force oracle is evaluated **after selection** for instances up to 16
candidates and never supplies the trading decision.

The local statevector has a configurable qubit cap (16 by default, hard maximum
20). An unavailable backend or exceeded cap fails visibly unless `fallback:
greedy` is explicitly configured. Fallbacks are labeled separately in reports.

## Benchmarks and reproducibility

The checked-in [demo report](reports/comparison.html) contains a benchmark table,
equity chart, full timing breakdown, seed confidence intervals, and cost/entry
sensitivity analysis. Raw snapshots are in `reports/demo-runs/`; report-level
CSV exports sit beside the HTML. [research/benchmark.ipynb](research/benchmark.ipynb)
provides a small reproducible notebook wrapper.
The [recorded findings](research/FINDINGS.md) summarize the measured outcome:
QAOA matched MILP's selections on the demo but required more computation.

The demo predeclares seeds 7, 19 and 41, 60 optimizer evaluations, a five-second
solver wall cap, QAOA depth one, and 512 final shots. Ranking, greedy, annealing
and QAOA observe evaluation caps; MILP uses the shared wall cap because SCIP does
not expose comparable objective-call counts. Wall caps are cooperative for Python
solvers: a single circuit evaluation or final decoding may finish after a cap.
Compilation, encoding and repair remain visible in full selection/pipeline times.

Runs record data loading, screening, encoding, energy preparation, compilation,
optimizer evaluations, simulation, shots, repair/decoding, validation, test,
sensitivity and serialization. Simulation time is nested within optimizer time
and must not be added twice. QPU queue and communication times are zero; hardware
cost is not applicable. Python startup/imports and report generation are outside
the manifest timer. A cold dependency import during selection is included in that
fold's selection total.

`manifest.json` contains the complete resolved experimental configuration, dataset
identity, source hash, dependency versions, host information, split boundaries,
predeclared seeds, timings and status. `run.json` contains candidates, rejected
screening diagnostics, constraints, selected bits, validation decisions, oracle
gaps and refit timestamps. `daily.csv`, `trades.csv`, `summary.csv` and
`sensitivity.csv` preserve the numerical results.

Reports keep different code/data/configuration/dependency combinations separate.
Repeated invocations use the newest attempt for each solver and seed; they do not
inflate sample counts. Failed capital paths are not scored, and the CLI exits
nonzero while preserving failure details. Fallbacks never appear as successful
quantum results. Confidence intervals bootstrap solver seeds on the **same**
synthetic price path, so they measure algorithm variation, not market uncertainty.
Sharpe uses 252 sessions/year, zero risk-free return, and sample daily volatility;
win rate uses closed trades after all costs; turnover is gross traded dollars
relative to initial capital. Cost sensitivities scale fees, slippage, borrow and
financing together. Sensitivity summaries are per-fold descriptive averages.

## Tests and acceptance

`python -m pytest -q` covers deterministic data preparation, immutable hashes,
invalid inputs, known hedge/half-life fixtures, chronological boundaries, a known
selection optimum, all QUBO decision/slack assignments, security/capital/sector
constraints, MILP optimality, stochastic reproducibility, independent statevector
agreement, no-future-data invariance, exact share and cost accounting, all exit
paths, explicit failures/fallbacks, report deduplication and CLI contracts.
The recorded implementation passed 51 tests and all 90 demo fold/seed/solver cases.

- [x] M1: Engle–Granger diagnostics and vanilla z-score backtesting.
- [x] M2: Diversified selectors and strong MILP baseline.
- [x] M3: Constrained pair-selection QUBO with exact tiny-instance verification.
- [x] M4: QAOA simulation and classical post-selection feasibility gate.
- [x] M5: Walk-forward results, sensitivity analyses, raw artifacts and report.

## Limitations and references

Synthetic mean reversion does not establish live profitability. Cointegration can
break. Screening, selection and entry tuning can overfit despite chronological
splits. This implementation has no broker integration, real QPU execution, minute
bars, liquidity/market-impact model, borrow availability, dividends or corporate
actions. Public historical data ingestion is a future extension and requires
point-in-time constituent and licensing metadata. Rust execution is optional in
the original specification and is not required for this daily-close simulator.

The numerical APIs follow the official [statsmodels cointegration documentation](https://www.statsmodels.org/stable/generated/statsmodels.tsa.stattools.coint.html),
[OR-Tools MIP example](https://developers.google.com/optimization/mip/mip_example),
and [Qiskit Statevector documentation](https://quantum.cloud.ibm.com/docs/en/api/qiskit/qiskit.quantum_info.Statevector).

For research and education only. No investment advice, production-readiness claim
or claim of quantum computational advantage.
