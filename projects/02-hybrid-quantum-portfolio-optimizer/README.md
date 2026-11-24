# Hybrid Quantum–Classical Portfolio Optimizer

> Discrete portfolio construction with real-world constraints, solved by classical baselines and by QUBO-based solvers under the same budget.

**Status:** implemented and tested. Every quantum result in this repository is a **classical simulation on a CPU**. No QPU was used, so there are no hardware results, and nothing here claims quantum advantage.

## Research question

For what asset counts and constraints, if any, does the hybrid method deliver comparable or better feasible portfolios within an equal computation budget?

**Answer from the runs in this repository:** see [Findings](#findings).

## What is implemented

| Area | Contents |
|---|---|
| Problem model | Integer-lot mean-variance problem, long-only and market-neutral, with budget, position limits, cardinality, sector caps and a turnover term (`src/hqpo/models/problem.py`) |
| QUBO | Bounded-coefficient binary encoding, slack and indicator qubits, penalty calibration, Ising mapping, qubit counting (`src/hqpo/models/qubo.py`) |
| Feasibility repair | Deterministic greedy repair with pre- and post-repair objective reporting (`src/hqpo/models/repair.py`) |
| Classical solvers | Brute force, CVXPY relaxation + rounding, exact MIQP (CVXPY + SCIP), OR-Tools CP-SAT, greedy + local search, simulated annealing (`src/hqpo/solvers/classical.py`) |
| Quantum-style solvers | QAOA (noiseless statevector simulation), QAOA sampled under an Aer noise model, simulated annealing on the QUBO via D-Wave Ocean, and a random-sampling control (`src/hqpo/solvers/quantum.py`) |
| Data | Deterministic synthetic price panel, planted instances with analytically known optima, CSV and optional yfinance ingestion, sha256 dataset manifests (`src/hqpo/data/`) |
| Evaluation | Point-in-time rolling backtest, equal-budget benchmark harness, Plotly dashboard, experiment manifests (`src/hqpo/evaluation/`) |
| Tests | 277 tests: encodings, objective values, constraints, solver adapters, fallbacks, data and backtest regressions, CLI end to end (`tests/`) |

## Architecture

```text
price panel (versioned CSV + sha256 manifest)
   │  point-in-time window (dates <= signal date)
   ▼
return and covariance estimation (Ledoit-Wolf, shrunk mean / momentum / zero)
   ▼
PortfolioProblem: integer lots z, objective z'Az + b'z + c0, constraints   ◄── single source of truth
   │                                             │
   │ lot space                                   │ build_qubo: bits, slack, indicators, penalties
   ▼                                             ▼
classical solvers                         QUBO solvers
cvxpy-relax · miqp · cpsat · greedy · sa  qaoa · qaoa-noisy · qubo-sa · random-qubo
   │                                             │ decode bits -> lots
   └──────────────► feasibility repair (if violated) ◄──┘
                         ▼
     weights, pre/post-repair objective, timings, evals, qubits, depth
                         ▼
   backtest (1-session execution lag, linear costs)  ·  benchmark tables  ·  HTML report
```

**Classical/quantum boundary.** Only the optimisation kernel is quantum-style, and it runs asynchronously at rebalance time. Data loading, estimation, decoding, repair, risk evaluation and anything latency-critical are classical. Timings cover the whole pipeline per solve (model build, circuit compilation, state preparation, optimiser loop, sampling, decoding, repair), not just the kernel. Library import time is excluded and recorded separately in each manifest.

## Mathematical formulation

Minimise, over weights on a lot grid,

```text
f(w) = λ wᵀΣw − μᵀw + τ ‖w − w_prev‖²
```

* **Long-only:** `w = z / L` with integer lots `z_i ∈ [0, K]`, `Σ z_i = L`, where `K = ⌊w_max · L⌋`.
* **Market-neutral:** `w = (p − q) · leg / L`, `Σ p_i = Σ q_i = L`, `p_i · q_i = 0`. Net exposure is zero and gross exposure is `2 · leg` (default 1.0).
* **Cardinality:** at most `C` names held. **Sector caps:** `|net sector weight| ≤ cap`.

The turnover term is quadratic on purpose. An L1 term needs auxiliary qubits, and a different turnover model per solver would break the "identical objective" rule. The quadratic form is QUBO-native, so every solver minimises exactly the same function. Realised L1 turnover and linear transaction costs are measured out of sample by the backtester.

### QUBO encoding and qubit growth

| Element | Encoding | Qubits |
|---|---|---|
| Lot variable `z_j ∈ [0, K]` | bounded coefficients `[1, 2, 4, …, rest]` summing to `K`, so the position limit holds by construction | `⌈log₂(K+1)⌉` per asset and leg |
| Budget | `P (Σ z − L)²` | 0 |
| Cardinality | indicator `y_i` per asset, `P x_b (1 − y_i)` per position bit, `P (Σ y − C)²` | `n` if the constraint can bind |
| Sector cap | `P (a·z + s − hi)²` with a bounded binary slack `s` | `⌈log₂(range+1)⌉` per binding sector |
| Long/short exclusivity | `P x_b x_c` for each bit pair of `(p_i, q_i)` | 0 |

Constraints that cannot bind are presolved away and cost no qubits. Measured growth with `L = n` lots and a 4-lot position cap (from `runs/demo/benchmark/qubit_growth.csv`):

| assets | budget + position limit | + cardinality (n/2) | + cardinality + sector caps | market neutral + cardinality |
|---|---|---|---|---|
| 4 | 12 | 16 | 20 | 28 |
| 8 | 24 | 32 | 40 | 56 |
| 16 | 48 | 64 | 76 | 112 |
| 32 | 96 | 128 | 144 | 224 |
| 64 | 192 | 256 | 276 | 448 |
| 128 | 384 | 512 | 536 | 896 |
| 256 | 768 | 1024 | 1052 | 1792 |

### Penalty calibration

`P = scale × G`, where `G = max_j (|b_j| + 2 Σ_k |A_jk| K_k)` bounds how much the objective can change when one lot moves. One unit of violation then costs at least as much as it can gain. The benchmark sweeps `scale` on tiny instances and checks the full QUBO spectrum against brute force:

| scale | QUBO ground state = true optimum | QAOA shots feasible | QAOA P(ground state) |
|---|---|---|---|
| 0.01 | 0 / 12 | 1.2 % | 5.1 % |
| 0.02 | 0 / 12 | 2.7 % | 6.0 % |
| 0.05 | 0 / 12 | 3.7 % | 2.6 % |
| 0.1 | 2 / 12 | 7.0 % | 1.8 % |
| 0.2 | 10 / 12 | 14.4 % | 1.1 % |
| 0.5 | 12 / 12 | 33.0 % | 1.4 % |
| 1 | 12 / 12 | 37.4 % | 1.6 % |
| 2 | 12 / 12 | 40.7 % | 1.6 % |

Twelve tiny instances (four constraint sets × three seeds), QAOA with p = 2 in noiseless simulation. The default `scale: 1.0` is exact on all of them. Below 0.5 the encoding starts returning infeasible ground states. A weaker penalty raises the probability of the QUBO ground state but that ground state is then the wrong portfolio.

## Solvers

| Name | Group | What it is |
|---|---|---|
| `cvxpy-relax` | classical | Convex relaxation in CVXPY/Clarabel (minimum variance when `expected_return: zero`), rounded to lots. Its optimum is a lower bound. |
| `miqp` | classical | Exact mixed-integer QP via CVXPY + SCIP. Falls back to CP-SAT if SCIP is missing or fails. |
| `cpsat` | classical | OR-Tools CP-SAT on the integer model with an integer-scaled objective. |
| `greedy` | classical | Best-marginal lot-by-lot construction plus lot-shift local search. |
| `sa` | classical | Simulated annealing on feasible lot vectors. |
| `brute-force` | reference | Exhaustive enumeration for tiny instances. |
| `qaoa` | quantum-sim | QAOA with a transverse-field mixer, CVaR objective, COBYLA, exact noiseless statevector simulation with finite-shot sampling. Skips instances above the simulator's qubit limit. |
| `qaoa-noisy` | quantum-sim | Same angles, final shots drawn from Qiskit Aer with depolarising and readout noise. |
| `qubo-sa` | anneal-sim | Simulated annealing on the QUBO with D-Wave Ocean's sampler (NumPy fallback). A classical stand-in for an annealer. |
| `random-qubo` | control | Uniform random bitstrings, same shot count and decoding as QAOA. |

Circuit depth and two-qubit gate counts come from transpiling the QAOA circuit with Qiskit to `{rz, sx, x, cx}`, both all-to-all and on a line.

## How to run

Requires Python 3.12. Everything runs on a CPU.

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt && pip install .        # pinned environment
# or: pip install ".[all]"                              # latest compatible versions

# 1. Generate the documented sample dataset (deterministic, writes a sha256 manifest)
python -m hqpo.prepare_data --config configs/demo.yaml

# 2. Classical baselines
python -m hqpo.run --config configs/demo.yaml --solver classical

# 3. Quantum simulator (and the annealing proxy)
python -m hqpo.run --config configs/demo.yaml --solver quantum-sim
python -m hqpo.run --config configs/demo.yaml --solver anneal-sim

# 4. Equal-budget benchmark: scaling, penalty sweep, efficient frontier, qubit growth
python -m hqpo.benchmark --config configs/demo.yaml

# 5. Render all stored runs
python -m hqpo.report --runs runs/ --output reports/comparison.html
```

If you edit the source, either reinstall or run with `PYTHONPATH=src`. Editable installs (`pip install -e .`) can silently stop resolving on macOS when the project sits in a synced folder such as Desktop, because the `.pth` file gets a hidden flag that Python 3.12 skips. `pytest` reads `src/` directly and is unaffected.

**Reviewer command (about a minute, reduced sizes):**

```bash
python -m hqpo.prepare_data --config configs/smoke.yaml && python -m hqpo.run --config configs/smoke.yaml --solver all \
  && python -m hqpo.benchmark --config configs/smoke.yaml && python -m hqpo.report --runs runs/ --output reports/comparison.html
pytest            # 277 tests, about 30 s
```

`--solver` accepts a group (`classical`, `quantum-sim`, `anneal-sim`), a solver name, a comma-separated list, or `all`. Configs: `configs/demo.yaml` (long-only), `configs/market_neutral.yaml`, `configs/smoke.yaml`.

Outputs land in `runs/<config>/backtest/<solver>/<seed>/` (`daily.csv`, `rebalances.csv`, `weights.csv`, `summary.json`, `manifest.json`) and `runs/<config>/benchmark/`. Each manifest records the config and its hash, a hash of the source code, the dataset hash, package versions and the machine. `runs/` is git-ignored; the report step copies the tables into `reports/`.

## Findings

All numbers are from `configs/demo.yaml` and `configs/market_neutral.yaml` on one laptop CPU (Apple silicon, macOS), 3 s wall-time budget per solve (2 s for market-neutral), three instance seeds and three solver seeds. Full tables: [`reports/benchmark_table_demo.md`](reports/benchmark_table_demo.md), [`reports/benchmark_table_market_neutral.md`](reports/benchmark_table_market_neutral.md). Dashboard: [`reports/comparison.html`](reports/comparison.html). Raw rows: `reports/benchmark_results_*.csv`.

**In these experiments the hybrid method never beat a classical baseline, and it only matched them on the smallest instances.**

Long-only benchmark. Gap is normalised per instance: 0 % is the reference optimum, 100 % is the average random feasible portfolio.

| assets | qubits | `miqp` (exact) | `greedy` | `sa` | `qaoa` (noiseless sim) | `random-qubo` control | `qubo-sa` |
|---|---|---|---|---|---|---|---|
| 4 | 12 | 0 % in 0.04 s | 0 % | 0 % | 0 % in 0.23 s | 0 % (max 31 %) | 0 % |
| 6 | 18 | 0 % in 0.03 s | 0 % | 0 % | 17.6 % in 2.7 s | 20.6 % | 0 % |
| 8 | 16 | 0 % in 0.03 s | 0 % | 0 % | 3.7 % in 1.7 s | 21.5 % | 0 % |
| 16 | 73 | 0 % in 0.04 s | 0 % | 0 % | skipped | 113 % | 33 % |
| 32 | 137 | 0 % in 0.28 s | 0 % | 0 % | skipped | 84 % | 58 % |
| 64 | 276 | 0 % in 0.25 s | 0 % | 0 % | skipped | 83 % | 69 % |
| 128 | 544 | 0 % in 1.1 s | 0 % | 0 % | skipped | 67 % | 100 % |

Median gap over 3 instances (× 3 seeds for stochastic solvers). References are brute force up to 8 assets and SCIP-proven optima above.

* **Classical exact solving is easy at this scale.** SCIP proved optimality on every instance up to 128 assets in about a second or less. Greedy plus local search matched it on the median everywhere (worst case 0.7 % at 32 assets).
* **QAOA is exact only at 12 qubits.** At 18 qubits it is statistically level with drawing random bitstrings and keeping the best (17.6 % against 20.6 %). At 16 qubits it is clearly better than that control (3.7 % against 21.5 %) but still short of every classical baseline, and it takes 50 to 100 times longer than SCIP in simulation.
* **What the circuit does achieve** is concentrating probability on feasible portfolios: 30 to 49 % of shots are feasible against 6 to 22 % for random bitstrings. The probability of the exact optimum is small (1.2 % at 12 qubits, 0.03 % at 18, 0.01 % at 16), so the answer comes from best-of-1024-shots selection.
* **The QUBO route degrades with size even with a classical annealer.** `qubo-sa` is exact up to 18 qubits and falls to a 100 % gap, no better than a random feasible portfolio, at 544 qubits, while annealing directly on lot vectors (`sa`) stays at 0 %. The penalty formulation, not the lack of quantum hardware, is the first obstacle.
* **Noise.** At 12 qubits the depolarising model (1 % two-qubit error) cut feasible shots from 49 % to 34 % but the best shot was still optimal. Larger instances were not run under noise (14-qubit limit).
* **Circuit cost.** p = 2 needs depth 94 to 148 and 168 to 480 CX gates with all-to-all connectivity, and 487 to 1,207 CX gates on a line. That is beyond what current devices execute faithfully.
* **Market-neutral** shows the same pattern: QAOA exact at 8 qubits, 16 % gap at 18 qubits (control 23 %), skipped from 89 qubits up.
* **CP-SAT** needs explicit product variables for the quadratic objective and ran out of budget from 32 assets (8 to 25 % gap). It is a weaker baseline than SCIP here.

Backtest (6 synthetic assets, 13 quarterly rebalances, 18 qubits). `miqp`, `greedy`, `sa` and `qubo-sa` returned the identical optimal portfolio at every rebalance. QAOA returned a feasible portfolio every time without needing repair, but a worse one: mean test-window objective 0.094 against 0.077. Its test Sharpe was −0.01 against −0.23 long-only and −0.64 against −0.46 market-neutral. Those differences have opposite signs and come from a suboptimal solver on synthetic data, so they carry no information about skill.

**So, to the research question:** within these experiments there is no asset count or constraint set where the hybrid method is better. It is comparable only at 4 assets and 12 qubits, where everything, including random sampling, is optimal. This is a statement about noiseless simulation of shallow QAOA with penalty encodings and a standard mixer. It does not cover real hardware, deeper circuits or constraint-preserving mixers.

## Fair-comparison protocol, as implemented

1. **Verification on tiny instances.** The QUBO ground state is checked against brute force for every constraint family, and planted instances have analytically known optima (`tests/test_encoding_qubo.py`, `tests/test_problem_repair.py`).
2. **Identical inputs.** Every solver receives the same `PortfolioProblem` object.
3. **Budget.** One wall-time limit per solve is the binding budget. `max_evals` is a second cap in each solver's native unit (circuit evaluations for QAOA, proposed moves for annealing). Both are reported. The noisy Aer sampling step cannot be interrupted, so `qaoa-noisy` can overrun; its time is reported as measured.
4. **Full timing.** Stages are timed separately and summed into end-to-end wall time.
5. **Seeds.** Stochastic solvers run on the seeds declared in the config. Tables show means with 95 % Student-t intervals. Skipped and failed runs stay in the tables.
6. **Hardware.** None was available, so the hardware section is empty by design. Noiseless and noisy simulation are labelled separately.
7. **Backtest.** One-session execution lag, linear costs, chronological validation and test windows. The data is synthetic, so the backtest demonstrates the pipeline and does not establish profitability.

## Data

* `source: synthetic` (default): market + sector + idiosyncratic factor model with a two-state volatility regime, seeded and byte-reproducible. Generated by this repository, no third-party rights.
* `source: csv`: a wide adjusted-close file you supply (`data.raw_path`, optional `data.sectors`, `data.license`).
* `source: yfinance`: optional download of public ETF/equity prices (`pip install -e ".[yfinance]"`). This path is implemented but was not exercised here. Yahoo data is not redistributable, so do not commit the output.

Timestamps are normalised to timezone-naive UTC dates, sorted and de-duplicated. Gaps are forward-filled for at most three sessions and never back-filled. Loading a dataset whose bytes no longer match its manifest hash is refused.

## Repository structure

```text
configs/      demo.yaml, market_neutral.yaml, smoke.yaml
data/         generated price panels and their manifests
src/hqpo/
  models/     problem.py, qubo.py, repair.py
  solvers/    base.py, classical.py, quantum.py
  data/       synthetic.py, prices.py, estimation.py
  evaluation/ backtest.py, benchmark.py, metrics.py, manifest.py, report.py
  prepare_data.py, run.py, benchmark.py, report.py     thin CLI entry points
tests/        277 tests
notebooks/    benchmark_walkthrough.ipynb
reports/      comparison.html, benchmark tables and raw CSV results
```

## Roadmap

- [x] **M1:** Tiny synthetic portfolios solved by brute force to verify the QUBO
- [x] **M2:** CVXPY/MIQP baselines and rolling inputs
- [x] **M3:** QAOA, plus annealing as a simulated proxy
- [x] **M4:** Transaction costs, sector limits and feasibility repair
- [x] **M5:** Scaling results from 4 to 128 assets (QAOA only where it fits the simulator)

Not done: runs on real quantum hardware (no access), real market data in the committed results, constraint-preserving mixers (XY/Grover), warm-started QAOA, and tuning hyper-parameters on the validation window.

## Known limitations

* **Simulation only.** Simulator wall time says nothing about QPU run time, queueing or cost.
* **Qubit ceiling.** The statevector simulator is capped at 20 qubits (14 with noise), which is 4 to 8 assets under these encodings. Beyond that QAOA is reported as skipped.
* **Penalty methods can return infeasible bitstrings.** The share of infeasible shots is reported, and repair fixes the selected sample. Repair is a classical heuristic, so post-repair quality is partly classical work.
* **Best-of-shots selection flatters small instances.** With about a thousand shots on a 12 to 18 qubit problem, random sampling already finds good portfolios. Compare `qaoa` against `random-qubo`, not against nothing.
* **Quadratic turnover proxy** in the objective, not L1.
* **`cpsat` rounds objective coefficients** to integers (relative precision about 1e-7).
* **Synthetic data** has no survivorship, corporate-action or microstructure effects.

## Disclaimer

For research and educational use. Not investment advice, not a claim of profitable strategy performance, and not a production trading system. Verify market-data permissions and exchange rules before any live integration.
