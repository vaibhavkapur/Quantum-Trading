# Quantum Risk Management Engine

**Implemented research prototype** · One-day portfolio VaR, expected shortfall, stress testing, and a Qiskit amplitude-estimation experiment.

The original [specification](docs/SPECIFICATION.md) is implemented at prototype scope. The engine runs on CPU hardware with synthetic data, an HTTP API, interactive dashboard, and offline reports. The quantum component is a **noiseless classical simulation of quantum circuits**, with no claim of production or runtime advantage.

## Quick start

Run from `08-quantum-risk-engine/`. The pinned environment supports Python 3.9–3.12; bundled results use Python 3.9.6 on macOS arm64. A C++20 compiler is optional; native tests skip when one is unavailable.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install --no-deps -e .

# Data, both solvers, 40 benchmark trials, HTML/CSV/SVG reports and manifest
python -m project.reproduce --config configs/demo.yaml

# Numerical, circuit, API, dataset-integrity and native-adapter checks
python -m pytest -q -W error::RuntimeWarning

# Dashboard and OpenAPI documentation
python -m uvicorn api.app:app --host 127.0.0.1 --port 8088
```

Open [the dashboard](http://127.0.0.1:8088) and [API documentation](http://127.0.0.1:8088/docs) while the server runs. Edit confidence, portfolio value, distribution, and solver; export the result as JSON. The [demo guide](docs/DEMO.md) gives a two-minute walkthrough.

Open `reports/reproduction/comparison.html` directly for an offline report. Published artifacts are also included: [comparison report](reports/comparison.html), [raw benchmark CSV](reports/benchmark.csv), [benchmark JSON](reports/benchmark.json), and [scaling plot](reports/scaling.svg).

For a classical-only environment, install `-e '.[dev]'` instead of the pinned requirements. To add the simulator, install `-e '.[quantum]'`. Missing quantum dependencies fail clearly unless `quantum.fallback: true` is explicitly configured; fallback is always labeled in the result.

## Original CLI contract

All proposed commands are executable:

```bash
python -m project.prepare_data --config configs/demo.yaml
python -m project.run --config configs/demo.yaml --solver classical
python -m project.run --config configs/demo.yaml --solver quantum-sim
python -m benchmarks.run --config configs/demo.yaml
python -m project.report --runs runs/ --output reports/comparison.html
```

Configuration/data/output paths resolve relative to the configuration file's grandparent (the project root). `project.report` paths resolve relative to the working directory. Runs have unique UUID filenames. `project.reproduce --output reports/my-experiment` selects another output directory. Reports retain per-run configuration/data fingerprints rather than pooling incompatible experiments.

## Implemented milestones

- **M1 — Classical risk:** historical simulation; analytic Gaussian VaR/ES; ordinary and antithetic Monte Carlo; Gaussian importance sampling with likelihood weights and effective sample size; analytic Student-t references; input validation and known-answer tests.
- **M2 — Portfolios and scenarios:** signed weights, leverage, zero-return cash, validated positive-semidefinite covariance, correlated Gaussian and multivariate Student-t returns, deterministic seeds, stress shocks and asset-level loss contributions.
- **M3 — Quantum oracle:** discrete loss distributions, Qiskit probability loading, CDF/tail indicator oracles, excess-loss rotation oracles, maximum-likelihood amplitude estimation, approximate VaR bisection and ES.
- **M4 — Comparison:** repeated-seed, matched-query tail benchmarks; exact discrete and continuous references; uncertainty summaries, failed-trial records, raw CSV/JSON, and a [research notebook](benchmarks/research.ipynb).
- **M5 — Delivery:** FastAPI, responsive dashboard, offline reports, staged timing and resource costs, provenance manifest, optional C++20 kernel, and CI configuration.

Optional real QPU execution, ETF downloads and PostgreSQL persistence are not included. No derivatives, exchange connectivity or live risk gates are implemented.

## Definitions and modeling choices

Returns are **one-day simple returns**, expressed as decimals. Portfolio value and losses use the same currency units; the dashboard demo uses USD. Positive values represent losses. For signed asset weights `w` and value `V`:

```text
L = -V × wᵀr
mean(L) = -V × wᵀmean(r)
std(L) = V × sqrt(wᵀ covariance(r) w)
VaR(alpha) = inf {l : P(L <= l) >= alpha}
ES(alpha) = VaR(alpha) + E[max(L - VaR(alpha), 0)] / (1 - alpha)
```

`cvar` and `expected_shortfall` are aliases for integrated-quantile ES. For continuous distributions this equals the conditional tail expectation in the specification. For discrete/empirical distributions it includes only the required fraction of an atom at VaR, avoiding the bias of averaging all observations `>= VaR`. Negative risk estimates are valid for a portfolio that always gains.

Historical risk uses the training observations. Parametric, MC and quantum estimates use moments fitted to the **same training window**, with that fitted distribution as the analytic reference. The reference is not true market risk. Student-t scenarios share a chi-square scale across assets and are rescaled to match the requested covariance; degrees of freedom are configured, not fitted.

The demo has 1,000 synthetic weekdays. Observations `[0,500)` fit the initial model and `[500,1000)` are held out. Each rolling forecast uses the preceding 500 observations and excludes the observation being forecast. There is no parameter search or validation-based tuning. Backtests report strict exceptions, exact binomial coverage, Kupiec coverage, transition counts and an independence p-value when estimable. Gaussian forecasts on Student-t data deliberately expose misspecification.

Importance sampling shifts normal innovations toward losses. Self-normalized likelihood ratios give finite-sample biased quantile/ES estimates; weight mean and effective sample size diagnose poor overlap. It is enabled only for Gaussian scenarios. Antithetic sampling is a variance-reduction candidate, not a guaranteed improvement for tail quantiles.

## Quantum construction

The loss register contains `2**qubits` uniform bins. Analytic Gaussian/Student-t CDF differences give their probabilities. Overflow probability is assigned to the outer bins and reported as `clipped_tail_mass`. Quantization/overflow errors are measured against the continuous reference separately from estimation error.

1. Qiskit `StatePreparation` loads `sqrt(p_i)` into the loss register.
2. A uniformly controlled Y rotation writes an objective `g_i` in `[0,1]` to an objective qubit. CDF/tail indicators use zero or one; excess loss is divided by its maximum.
3. Qiskit builds the Grover operator. Dense statevector simulation applies its actual unitary for powers `[0,1,2,4,8]` by default.
4. Measurements are sampled from the simulated states. A global likelihood grid and local refinement fit `sin²((2k+1)theta)` to the **counts only**; the estimator never receives the exact amplitude.
5. Noisy CDF bisection locates VaR, followed by an excess-loss expectation for ES. Ambiguous CDF steps are counted. Profile-likelihood intervals are asymptotic hulls across grid-resolved modes, not finite-sample or joint VaR/ES guarantees.

Probability loading and payoff lookup grow exponentially with the state register. The simulator is restricted to 1–6 state qubits, at most seven powers, maximum power 32 and 4,096 shots per power; API limits are tighter. The maximum executed circuit is transpiled to `u`/`cx` at optimization level zero. Resources include depth, CX gates, shots, Grover iterations, and `A`/`A⁻¹` calls. Simulation reuses intermediate states, so algorithmic query counts describe the circuits rather than matrix multiplications performed by the simulator.

The method follows [Suzuki et al., amplitude estimation without phase estimation](https://arxiv.org/abs/1904.10246), using Qiskit's [Grover operator](https://quantum.cloud.ibm.com/docs/en/api/qiskit/2.2/qiskit.circuit.library.grover_operator) and [controlled Y rotations](https://quantum.cloud.ibm.com/docs/en/api/qiskit/2.2/qiskit.circuit.library.UCRYGate). Tests compare actual circuit evolution with analytic amplification and exact discrete risk.

## Published measurements

The bundled experiment was run on **24 November 2026**, seed 42, a $1,000,000 portfolio, weights `[0.6,0.3,0.1]`, and 99% confidence. Full tables and timings are in the report.

- Fitted Gaussian reference: VaR **$17,120.09**, ES **$19,689.40**. Ordinary MC with 20,000 samples: **$17,008.50 / $19,552.73**.
- Four-state-qubit quantum estimate: **$16,067.33 / $20,458.51**. The grid alone contributes about **$1,053** of VaR error and **$213** of ES error.
- Historical rolling backtest: **8 exceptions in 500 observations**, versus 5 expected; exact coverage p-value approximately **0.172**.
- At 2,240 queries across seeds `[11,23,37,53,71]`, mean absolute tail-probability error is **0.001047** for MC and **0.000498** for MLAE. Mean pipeline times are about **0.0017 s / 0.0642 s**. The quantum simulator is slower. Maximum tail-circuit depth is **1,370**.

The benchmark estimates the **same strict tail event on the same discrete PMF**. Its exact probability is `0.006210`, versus `0.010000` in the continuous model. That discrepancy is discretization error. MC receives one sample for every quantum `A`/`A⁻¹` call, counted as `shots × sum(2k+1)`. This is matched-query, not matched-wall-time benchmarking; both runtimes are reported.

Timings include data loading, fitting, grid preparation, oracle construction, compilation, simulation/sampling and likelihood decoding. Common loading/fitting/grid costs are charged to both methods. Dataset generation is a one-time offline prerequisite; process startup and output-file writes are outside the timed estimator pipeline. Qiskit import is included in full quantum-run timing but precedes repeated benchmark trials. Queue/communication time and QPU cost are zero because no QPU is used. Five seeds and descriptive Student-t intervals do not establish asymptotic scaling. Failures are retained with errors and cause a nonzero benchmark exit.

## Data and reproducibility

Data are generated locally, without downloads. CSV filenames are SHA-256 content hashes. Versioned manifests record generation parameters, asset order, **CC0-1.0** synthetic-data licensing, units and UTC normalization. NumPy PCG64 and covariance eigendecomposition are seeded. Timestamps are UTC midnight on synthetic weekdays, without an exchange-holiday calendar. Floating-point differences across platforms or NumPy versions can affect content hashes; `requirements.txt` pins the tested environment.

Existing datasets are verified before reuse. Corruption fails instead of silently overwriting files. Changed generation settings create another dataset version; weights/confidence/solver changes do not regenerate returns. No real observations, credentials or licensed feeds are bundled.

Runs record configuration/data hashes, seed, UTC time, platform/package versions, split boundaries and staged timings. `project.reproduce` adds `experiment-manifest.json` with source and artifact hashes. Numerical results are reproducible in the pinned environment; timing, UUIDs and timestamps vary.

## API and C++20 adapter

`GET /health` reports availability. `GET /api/example` returns an example configuration. `POST /api/run` accepts `{ "config": <configuration>, "solver": "classical" | "quantum-sim" }` for a bounded in-memory experiment. Direct endpoints:

- `POST /api/risk/historical`: `losses`, `alpha`, optional probability `weights`.
- `POST /api/risk/gaussian`: `mean_loss`, `std_loss`, `alpha`.
- `POST /api/stress`: `portfolio`, named `shocks` in asset order.
- `POST /api/backtest`: `losses`, `window`, `alpha`, `method` (`historical` or `gaussian`).

Invalid inputs return 422; unavailable quantum backends return 503 unless explicit fallback is enabled. Dashboard rendering uses text nodes and the offline report escapes metadata. This is a localhost research service without authentication, persistence or a production job queue; quantum calls run synchronously on CPU.

The optional C++20 kernel independently computes unweighted empirical VaR/ES via a C ABI:

```bash
cmake -S native -B native/build -DCMAKE_BUILD_TYPE=Release
cmake --build native/build
```

```python
from risk.native import native_risk
native_risk([0, 1, 2, 3], 0.5, "native/build/libquantumrisk.dylib")
# Linux: libquantumrisk.so. Result: VaR=1, ES=2.5.
```

## Repository layout

```text
risk/          Classical estimators, portfolio, backtests, native adapter
scenarios/     Correlated Gaussian/Student-t and importance sampling
quantum/       Discretization, Qiskit oracles, amplitude/risk estimation
project/       Configuration, datasets, CLI, workflow, reports, reproduction
api/           FastAPI service and self-contained dashboard
native/        Optional C++20 kernel and CMake build
configs/       Complete demo configuration
tests/         Numerical, integration, API, circuit and native checks
benchmarks/    Matched-query runner and research notebook
data/          Content-addressed synthetic CSVs and manifests
runs/          Published full experiment records
reports/       HTML/CSV/SVG results and reproduction manifests
docs/          Original specification and demo guide
```

## Limitations

The linear simple-return model omits derivative convexity, volatility dynamics, liquidity liquidation costs and live trading. Extreme Gaussian/Student-t simple returns can imply economically impossible prices. Tail/correlation misspecification dominates numerical precision. Quantum ES errors are amplified by `1/(1-alpha)`; noisy quantile decisions can shift VaR by a whole bin. No finite-sample joint guarantee is claimed.

For research and education only. Not investment advice, not a profitable-strategy claim, and not a production trading system.
