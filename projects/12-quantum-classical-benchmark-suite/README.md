# Quantum-vs-Classical Trading Benchmark Suite

A working, CPU-only research harness for fair end-to-end comparisons on small, exactly verified finance problems. **Implemented:** versioned datasets, nine solver adapters, enforced budgets, separate optimization and estimation tracks, full timing/resource capture, uncertainty, and offline reports. No claim of quantum advantage or trading profitability.

The original proposal is retained in [docs/SPECIFICATION.md](docs/SPECIFICATION.md).

## Quick start

Use **Python 3.12** (the pinned NumPy/SciPy environment requires it). Run from this directory; no cloud account, QPU, market-data subscription, or credentials are needed.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
# Optional editable package installation; commands also work directly from this folder.
python -m pip install --no-build-isolation --no-deps -e .

# One-command reduced reproduction: generate, benchmark, verify, report.
python -m project.reproduce --config configs/ci.yaml --output reports/ci.html --strict
python -m pytest -q
```

The original CLI contract is implemented:

```bash
python -m project.prepare_data --config configs/demo.yaml
python -m project.run --config configs/demo.yaml --solver classical
python -m project.run --config configs/demo.yaml --solver quantum-sim
python -m project.report --runs runs/ --output reports/comparison.html
```

Or run the entire demonstration as one campaign:

```bash
python -m project.reproduce --config configs/demo.yaml --output reports/comparison.html --strict
```

`--solver` also accepts `all` or any adapter name below. Config paths resolve data/run directories relative to the project containing `configs/`, independent of the shell working directory. Report output paths are relative to the caller. There are no implicit data downloads or fallback solvers.

The demo runs 84 measured attempts plus 28 discarded warmups: six instances, nine applicable adapters, and seeds 11, 29, 47. The reduced CI configuration runs 28 measurements with two seeds and no warmups. Run times depend on the host; every attempt has a hard 30-second wall limit. Use more predeclared seeds and larger matrices for substantive research.

## Problems and algorithms

**Portfolio selection.** Minimize risk-aversion times selected covariance minus selected expected return, subject to exact cardinality. Each selected asset has equal notional; this is a selection score, not continuous portfolio allocation. Covariance is positive definite. Stored coefficients follow `c @ x + x @ Q @ x`, with strictly upper-triangular `Q` containing each pair coefficient exactly once.

**Constrained basket selection.** Select an exact number of non-overlapping synthetic candidate triangular cycles, maximizing their assigned net values. Shared vertices create conflicts. Candidate cycles are already enumerated; graph discovery, exchange execution and genuine arbitrage detection are outside this benchmark.

Optimization adapters:

- `exact`: complete feasible-subset enumeration, capped at 16 binary variables.
- `milp`: SciPy/HiGHS mixed-integer optimization with exact binary-product linearization, cardinality and conflict constraints. This is the strong classical baseline; it has native time and node limits.
- `greedy`: objective-based incremental feasible construction.
- `annealing`: seeded simulated annealing with cardinality-preserving swaps and exact constraint penalties.
- `qaoa`: parameterized Qiskit QAOA circuits, compiled to `rz/sx/x/cx`, sampled from a noiseless statevector and optimized with COBYLA. All optimizer shots count. There is no extra unbudgeted final sample or exact-solver fallback.

**Estimation.** Estimate a discounted call payoff on a finite lognormal quantile grid. Both approaches receive exactly the same probabilities and normalized payoffs; the reference is their exact weighted expectation, not the continuous Black–Scholes value. Finite-grid bias and tail truncation are outside the sampling-error metric.

Estimation adapters:

- `exact-expectation`: direct weighted sum over the explicit finite distribution. This deliberately makes visible the cheap classical solution to these tiny fixtures.
- `mc`: IID Monte Carlo scenario sampling.
- `mc-antithetic`: antithetic uniforms with uncertainty calculated over independent pair averages.
- `qae`: maximum-likelihood amplitude estimation using actual Qiskit `Q^k A` circuits. `A` prepares every scenario probability and payoff amplitude; the likelihood fitter sees only sampled counts. The known reference answer is never passed to adapters. Generic dense state preparation is included in encoding/compilation and can be exponential in qubit count.

QAE uses Grover powers `[0, 1, 2, 4]` in the demo, with equal shots per circuit reduced as necessary to fit the shot and oracle-query limits. A power-`k` shot costs `2k+1` applications of state preparation or its inverse. Per-estimate QAE intervals are asymptotic profile-likelihood envelopes; they do not guarantee finite-sample coverage. MC intervals use Student-t on samples or pair averages.

## Architecture

```text
validated YAML configuration
  → deterministic synthetic generators + exhaustive/analytic references
  → immutable JSON instances and SHA-256 dataset manifest
  → predeclared, seed-shuffled experiment matrix
  → isolated worker per warmup/measurement
       load + verify → encode → compile → solve/sample → decode → repair → validate
  → append-only JSONL results + environment/source/config manifest
  → separate comparison cohorts → HTML / raw CSV / summary CSV / Parquet
```

`benchmarks/instances/` owns shared objectives, constraints, QUBO/Ising conversion, provenance and checksums. `benchmarks/adapters/` implements the common `solve(problem, context)` protocol. `experiments/` owns configuration, deadlines and manifests. `metrics/` owns statistics and offline Plotly reports. `project/` contains thin CLI entry points.

All orchestration is Python; NumPy, SciPy/HiGHS and Qiskit supply compiled numerical kernels. Custom C++/Rust kernels, OR-Tools, CVXPY and D-Wave are unnecessary for this first small-instance implementation. No hardware adapter or quantum ML track is supplied: neither is required for the selected objective/estimation interface, and neither is simulated as a fictitious hardware result.

## Fair-comparison and budget policy

The common primary budget is **full attempt wall time**, including subprocess launch, interpreter/library setup, input reads and checksum validation, encoding, circuit/optimization-model compilation, execution, decoding, common repair, validation and IPC. The parent kills a worker at its deadline; termination overhead can slightly exceed the configured limit. A late result is recorded as a timeout. Per-stage timers are exclusive, including nested decoding inside QAOA execution.

Campaign preflight, discarded warmups, manifest/result persistence and report construction are outside individual attempt timing. Campaign wall time and preflight duration are recorded separately. Warmups prime OS caches in discarded isolated processes; measured attempts still include cold process startup. These tiny cases often measure startup costs more than numerical compute, which is intentional and visible.

Additional limits are independent: `max_evaluations`, `max_shots`, `max_oracle_queries`, and `mip_nodes`. Classical candidate scores, QAOA circuit expectations and likelihood fits are different evaluation units, explicitly labeled. HiGHS does not expose a comparable objective-evaluation counter, so its value is **null**, never an invented count; it uses the wall/node budgets instead. MC scenario evaluations and QAE oracle applications have their own query counter. Preprocessing, reference generation, and validation are not silently charged as solver objective calls.

A hard timeout has null unavailable stage/resource counters. Errors, infeasible outputs, budget exhaustion and unfinished planned attempts remain in reports. Feasible budget-limited incumbents can succeed; no failure triggers another solver. Optional common repair orders candidates by linear score and greedily enforces cardinality/conflicts. It has no reference access, can fail, and records raw feasibility, original candidate, repair application and repair time separately.

Comparison cohorts require identical instance, protocol, source and environment hashes. Different objectives, sizes, budgets or versions are never pooled. Repeated batches remain separate to avoid treating duplicate seeds as independent samples.

## Data and reproducibility

`project.prepare_data` deterministically derives each instance seed from the declared dataset seed, track and size. Binary fixtures have complete feasible enumeration references; estimation fixtures have analytic finite-distribution expectations. JSON files are immutable: regenerating identical content succeeds; changed content at the same path fails. Use a new dataset directory for a new version. Dataset preparation is capped at 16 binary variables and 256 power-of-two scenarios.

Generated datasets are CC0-1.0 synthetic data, with no third-party market license or historical observations. No timestamp normalization or chronological split is needed for these static tasks. Experiment times are ISO-8601 UTC. Future market/prediction adapters must add licensed, versioned data and chronological evaluation; this implementation makes no backtest or P&L claim.

Every campaign writes `manifest.json` before running: resolved configuration, predeclared jobs/seeds, dataset hashes, source-file hashes, dependency versions, Python/platform/CPU metadata, thread settings and warmup policy. Results are flushed as JSONL after each attempt. The finished manifest contains a result checksum and campaign timing. Interrupted campaigns retain the plan and completed records.

## Reports and sample evidence

Open [reports/comparison.html](reports/comparison.html) after reproduction. The retained demonstration also includes [reports/comparison.summary.csv](reports/comparison.summary.csv), raw CSV/Parquet, a report artifact manifest, and the original campaign in `reports/sample/`. These are measured local CPU results, not projected hardware results.

Reports include best feasible objective, absolute gap or estimation error, raw/repaired feasibility, success rate, p50/p95 full wall time, timing components, memory, logical qubits, compiled depth, shots, oracle queries, status counts, and size-scaling plots. Raw files also retain relative gaps, optimizer evaluations, returned solutions, estimator intervals and resource details.

Optimization success means feasible objective within `1e-8` of the reference. Estimation success uses the declared absolute payoff tolerance. Success proportions use Wilson 95% intervals over **all** planned measured attempts. Conditional feasible-quality means use Student-t intervals. Median wall time uses 2,000 deterministic bootstrap resamples; one observation has no estimated interval. Warmups are excluded. Failed attempts remain in the runtime sample if timed; missing times are not fabricated. Three seeds are demonstration evidence, not an adequate basis for strong statistical claims.

Memory is isolated-process peak RSS, including the interpreter and native libraries. Depth and two-qubit counts describe basis-compiled logical circuits without physical routing or measurement gates. Simulations have no device noise, queue wait or QPU charges; local CPU monetary cost is unknown, not zero. These fixtures do not establish a scaling advantage.

## Validation and automation

```bash
python -m pytest -q
python -m project.reproduce --config configs/ci.yaml --output reports/ci.html --strict
```

Tests exhaustively verify objective/constraint encodings, QUBO-to-Ising energies, MILP agreement with enumeration, real Grover amplification probabilities including endpoint amplitudes, estimator accounting, deterministic seeds, immutable datasets, hard timeout termination, failure handling, interrupted manifests and complete CLI-pipeline reporting. `--strict` returns nonzero for worker errors/timeouts or incorrect exact/MILP results while retaining all evidence.

The [GitHub Actions workflow](../../.github/workflows/quantum-classical-benchmark-suite.yml) runs tests and a CPU reproduction from this project directory and uploads datasets, results, and reports.

```bash
docker build -t quantum-classical-bench .
docker run --rm -v "$PWD/reports:/suite/reports" -v "$PWD/runs:/suite/runs" quantum-classical-bench
```

Local verification covered macOS with Python 3.12 and the Linux Docker image. All 59 tests passed in both environments, and the container completed the 28-attempt strict smoke matrix. Editable installation and an installed wheel were also verified. See the [GitHub Actions run history](https://github.com/vaibhavkapur/Quantum-Trading/actions/workflows/quantum-classical-benchmark-suite.yml) for hosted results.

## Roadmap status and boundaries

- [x] M1: common schemas, immutable synthetic fixtures and verified references.
- [x] M2: exact, MILP, greedy and simulated-annealing classical adapters.
- [x] M3: real QAOA simulator circuits and resource reporting.
- [x] M4: separate QAE, MC, antithetic-MC and exact-expectation track.
- [x] M5: automated matrices, statistical reports, raw artifacts and CPU smoke tests.

Optional market datasets, hardware access/noise experiments, D-Wave, large-scale native kernels and predictive ML are future extensions. Any real-QPU adapter must separately expose noise assumptions, routing, shots, queue/communication time and billing. Latency-critical trading, exchange messages and risk gates remain classical and outside this asynchronous research harness.

## Sources

- [SciPy MILP interface](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.milp.html)
- [IBM QAOA background](https://quantum.cloud.ibm.com/docs/en/tutorials/warm-start-qaoa)
- [Qiskit Finance amplitude-estimation tutorial](https://qiskit-community.github.io/qiskit-finance/tutorials/00_amplitude_estimation.html)

For research and educational use. Not investment advice or a production trading system.
