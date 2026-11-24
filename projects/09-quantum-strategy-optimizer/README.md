# Quantum Strategy Parameter Optimizer

A runnable research pipeline for discrete strategy and ensemble selection. It enumerates momentum and mean-reversion strategies, constructs a transaction-cost-aware QUBO from **training returns only**, and compares grid search, random search, Optuna TPE, simulated annealing, and Qiskit QAOA on held-out walk-forward periods.

**Implemented:** M1–M5. This is a CPU simulator benchmark and custom daily backtester, not a live trading system or evidence of quantum advantage. The sample is synthetic; licensed historical daily closes can be imported. [The generated benchmark report](reports/benchmark.html) includes every predeclared seed, confidence intervals, failures, timing, and raw data links.

## Quick start

Use Python 3.9–3.12 for the pinned environment (3.11 recommended). The checked-in benchmark ran with Python 3.9.6 and Qiskit 2.2.3, which emits upstream deprecation warnings. Run these commands from the project directory.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-lock.txt
python -m pip install --no-deps --no-build-isolation -e .

# One-command CPU reproduction, with reduced instance sizes:
python -m project.reproduce --config configs/ci.yaml --output reports/cpu-smoke.html

# The original README CLI contract:
python -m project.prepare_data --config configs/demo.yaml
python -m project.run --config configs/demo.yaml --solver classical
python -m project.run --config configs/demo.yaml --solver quantum-sim
python -m project.report --runs runs/ --output reports/comparison.html

python -m pytest -q
```

`--solver all` runs all five search methods in one invocation. Individual names are `grid`, `random`, `tpe`, `annealing`, and `qaoa`. `project.reproduce` reports only its newly created run, making it convenient for repeated experiments. `project.report` rejects duplicate solver/seed records and incompatible experiment cohorts instead of silently pooling them. Pass a single run directory or a directory containing exactly one classical and one quantum run when the default `runs/` contains previous experiments.

Paths inside YAML files resolve relative to the YAML file. Run and report destinations on the command line resolve relative to the working directory. Successful runs return exit status 0; recorded candidate/reference failures return 2; configuration, data, or pipeline failures return 1. Failed and incomplete runs remain inspectable.

## Architecture

```text
Immutable daily closes + provenance manifest
    → causal, content-addressed signal cache
    → rolling outer fold: training | validation | test
        training → fixed-sleeve return/cost matrix → QUBO
                 → grid / random / TPE / annealing / QAOA
                 → separate exhaustive + MILP verification
        validation → choose one training-produced risk-aversion candidate
        test → evaluate the frozen portfolio, once
    → per-candidate traces + daily returns + run manifests
    → checked HTML / CSV / JSON comparison report
```

The quantum boundary is the small binary ensemble-selection problem. Backtesting, parameter selection, costs, risk constraints, and reporting remain classical. There is no exchange connection or latency-critical quantum execution.

## Strategies and execution conventions

The library contains both strategies for every asset/window combination:

- **Momentum:** sign of the trailing close-to-close return over the configured window.
- **Mean reversion:** negative sign of the close's deviation from its trailing moving average.

Signals observed at close `t−2` execute at close `t−1` and earn the close-to-close return ending at `t`. Warm-up signals are zero. No same-close execution or forward-looking indicators are used. Prefix-invariance and future-price perturbation tests enforce causality.

Each selected strategy receives a fixed `1/K` capital allocation, where `K = max_selected`. At most `K` strategies may be active; unused capital remains cash. A binary selection is **not** renormalized by the number selected. This makes the return/covariance objective exactly match the backtest's sleeve accounting. Long and short sleeves are permitted, and aggregate gross exposure is at most one.

Each training, validation, and test window starts flat and liquidates at its final close. Turnover is the absolute position change, including entry and exit. Sleeve costs add independently, even when opposing sleeves trade the same asset. The demo charges 2 bps per unit turnover in training and 3 bps out of sample. The higher out-of-sample rate is predeclared, not fitted on test returns. Borrow costs, financing, market impact, volume constraints, execution uncertainty, and cross-sleeve netting are not modeled.

This is a target-exposure return model, not a share/cash ledger. Turnover measures changes in target exposure and omits rebalancing caused solely by mark-to-market weight drift. The QUBO is exact under these declared accounting assumptions.

## QUBO and constraints

For annualized training means `μ`, annualized sample covariance `Σ`, risk aversion `λ`, and cardinality cost `κ`:

```text
f(z) = −μᵀz + λ zᵀΣz + κ Σᵢzᵢ
zᵢ ∈ {0,1},  Σᵢzᵢ ≤ K
```

Here each return column already incorporates its fixed capital allocation and training transaction costs. Cash (`z = 0`) is feasible. Binary slack `s` encodes the remaining capacity:

```text
E(z,s) = f(z) + P (Σᵢzᵢ + Σⱼ2ʲsⱼ − K)²
P = 2 Σᵢⱼ |Qbaseᵢⱼ| + 1
```

`Qbase` uses the upper-triangular `zᵀQbase z` convention. This penalty exceeds the entire possible range of base-objective differences, so a global penalized optimum is feasible. The Ising transformation uses `zᵢ = (1−Zᵢ)/2`; tests check every bitstring of a tiny instance, its known optimum, and the actual Qiskit cost-layer phases, including little-endian decoding.

Every training objective is independently solved by exhaustive feasible enumeration and a SciPy/HiGHS MILP with exact pair-product linearization. These verification results never initialize a search method or influence validation selection. Their runtime is recorded separately. The implementation intentionally limits the library to 12 strategies and local statevectors to at most 16 qubits; the default guard is 12 qubits.

## Optimizer budgets

All methods receive the same maximum **number of individually scored QUBO candidates** per training objective. Repeated evaluations and infeasible samples count. The default cap is 128, independently for each fold, seed, and risk-aversion candidate.

- Grid enumerates feasible portfolios by cardinality and stops when the domain is exhausted (93 configurations in the demo).
- Random samples feasible portfolios uniformly with replacement.
- Optuna TPE searches binary strategy variables and scores capacity violations with the same QUBO penalty.
- Simulated annealing uses single-bit flips and a predeclared geometric temperature schedule.
- QAOA uses parameterized `RZ`/`RZZ` cost layers and `RX` mixing, Qiskit statevector simulation, finite sampled measurements, and COBYLA angle search. Each measured bitstring consumes one evaluation, including infeasible slack encodings. There is no exhaustive state-energy lookup, repair, classical fallback, or unbudgeted final shot batch.

The demo gives QAOA 16 circuit evaluations with 8 shots each. Its decoded result is the best feasible sample actually observed during that budget. A search with no feasible sample is recorded as failed. Equal candidate caps **do not imply equal compute or wall time**; statevector evolution is exponential. The resource CSV records circuit counts, qubits, layers, gate depth, shots, compilation, preparation, simulation, decoding, scoring, and optimizer/setup time. Queue time, communication, repair, and QPU charges are zero for the local simulator. No physical QPU backend is implemented.

## Nested chronological evaluation

Each outer fold contains one inner training/validation holdout followed by an untouched test period. Training estimates the QUBO separately for each configured risk aversion. Validation net Sharpe chooses among the resulting portfolios, with config-order tie breaking. The selected portfolio is frozen without retraining on validation data; only then is it evaluated on the test period.

Outer training windows roll forward by exactly the test length, so test windows never overlap. Earlier test observations may enter later training windows, as they would in a real chronological process. The feature cache can span the full dataset because every feature is causal. There is no shuffled cross-validation and no tuning on the aggregate reported test result.

The demo uses 504 business-day observations, 252 training days, 63 validation days, 63 test days, three folds, two assets, eight strategies, two risk-aversion values, and seeds `[7, 19, 41]`. The CPU smoke configuration uses one asset, four strategies, three folds, two seeds, and smaller budgets.

## Metrics and interpretation

Reports include held-out net Sharpe, compounded returns, drawdown including initial capital, turnover, costs, training objective gaps, runtime, and adjacent-fold Jaccard selection stability. Machine-readable summaries also contain strategy-selection frequencies.

Sharpe confidence intervals use a deterministic circular moving-block bootstrap of each held-out time series. A one-sided positive-mean test uses a Bartlett/Newey–West HAC standard error, with a conservative Bonferroni factor equal to the predeclared budget across all five methods, seeds, folds, and risk candidates. This implements the specification's multiple-test correction alternative; it is **not** a deflated-Sharpe estimator. Finite samples, normal approximations, regime changes, and repeated use of a research dataset limit inference.

Every seed is reported. A missing test fold invalidates that solver/seed's aggregate series; failures are never filled with zero returns or quietly dropped. Failed risk candidates remain visible even when another candidate permits a valid fold. Cash has zero Sharpe; nonzero constant-return Sharpe is undefined. The plot displays the first predeclared seed, not the best-performing one.

## Data provenance and historical import

`project.prepare_data` creates a deterministic correlated regime-switching synthetic dataset. Its manifest records the generator version, seed, parameters, SHA-256, license (CC0-1.0), price convention, and UTC timestamp normalization. Synthetic business days do not represent an exchange holiday calendar. Existing datasets are checked and never overwritten; use a new path when changing data parameters.

To use licensed adjusted/total-return ETF or futures closes, supply a local CSV with columns `date` followed by the exact configured asset names. Set `data.periods` to its row count and use a fresh `data.path`. No data is downloaded automatically.

```bash
python -m project.prepare_data --config configs/my-data.yaml \
  --source-csv /path/to/adjusted-closes.csv \
  --source-name 'Provider / dataset version / retrieval date' \
  --license-terms 'Your actual license and redistribution restrictions'
```

Imports normalize timestamps to UTC midnight, reject duplicate/unordered dates, missing prices, and nonpositive prices, and record the original file hash and declared license. Naive date labels are interpreted as UTC. `seed` and `start` are generation settings only for imported data. The importer does not infer adjustment factors, stitch futures contracts, or verify that a declared license grants redistribution rights. Do not include restricted vendor data in shared run artifacts.

## Repository and run artifacts

```text
configs/                 demo and reduced CPU configurations
strategy_optimizer/      data, strategies, accounting, QUBO, solvers, metrics, experiments, reporting
project/                 prepare_data, run, report, reproduce CLI modules
tests/                   numerical, causality, solver, failure, and CLI tests
data/                    immutable CSV datasets and provenance manifests
runs/<timestamp-id>/     manifest.json, instances.json, summary.json, daily_returns.csv, cases/*.json
reports/                 HTML reports, summary/resource CSVs, JSON, and published benchmark records
```

Manifests include the resolved config, source hash, all installed distribution versions, platform, seeds, data hash, exact fold boundaries, cache status, and pipeline timing. Candidate traces include every bitstring, objective, penalty energy, and feasibility flag. Report generation verifies artifact checksums and refuses incompatible configurations, datasets, source revisions, or duplicate solver/seed observations.

`requirements.txt` pins direct dependencies. `requirements-lock.txt` records the verified complete environment. Timing includes computation and artifact generation up to the final manifest write; report rendering is a separate command. Shared feature/backtest/reference preparation is amortized within an invocation. Cold and warm cache timings are labeled.

## Acceptance checks

- [x] M1: deterministic momentum/mean-reversion library and cost-aware daily backtests.
- [x] M2: grid, random, Optuna TPE, and simulated annealing baselines.
- [x] M3: faithful QUBO, capacity penalty, Ising mapping, exhaustive and MILP references.
- [x] M4: sampled Qiskit QAOA, strict budgets, feasibility checks, explicit failure records.
- [x] M5: nested walk-forward benchmark, stability, confidence intervals, multiple-test correction, full timing, reports, and raw records.

The tests exercise analytical objective equality, a known optimum, cost accounting, input edge cases, future-data invariance, deterministic optimizer histories, missing dependencies, qubit-limit failures, artifact tampering, cohort rejection, and all original CLI entry points.

## Recorded demo result

The bundled [benchmark](reports/benchmark.html) contains 90 optimizer searches (five methods × three seeds × three folds × two risk settings), with no failed candidates. The complete invocation took 12.87 seconds on this workspace's CPU, including imports, preparation, references, evaluation, and artifact creation. Timing is machine-specific.

Grid and simulated annealing reached the exact training optimum for every validation-selected portfolio. QAOA's mean selected-objective gap was 0.03014, versus 0 for grid/annealing, 0.00173 for random search, and 0.02408 for TPE. QAOA held-out Sharpe varied from −1.52 to 2.49 across the predeclared seeds, compared with 2.37 for the grid solution. This tiny synthetic experiment supports neither a quantum advantage nor an expectation of profitable trading. The complete per-seed results, uncertainty intervals, and costs are in the report; no seed was selected after viewing its test result.

Raw records are preserved under [reports/benchmark-runs](reports/benchmark-runs/). To regenerate the published report without rerunning optimization:

```bash
python -m project.report --runs reports/benchmark-runs --output reports/benchmark.html
```

## Implementation references

- [Qiskit Statevector API](https://quantum.cloud.ibm.com/docs/en/api/qiskit/qiskit.quantum_info.Statevector)
- [Optuna TPESampler](https://optuna.readthedocs.io/en/stable/reference/samplers/generated/optuna.samplers.TPESampler.html)
- [SciPy MILP / HiGHS](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.milp.html)

For research and education. Backtests do not establish live profitability. No production or quantum speedup claim is made.
