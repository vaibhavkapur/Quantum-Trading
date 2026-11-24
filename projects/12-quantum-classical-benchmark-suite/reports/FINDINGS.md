# Demonstration findings — 24 November 2026

Hypothesis: any apparent quantum benefit must survive identical objectives, feasibility checks, explicit work budgets and full pipeline timing. Optimization and probabilistic estimation are separate experiments.

The retained [campaign manifest](sample/20261124T051444Z-0b1d13fa03/manifest.json) predeclares six synthetic fixtures, nine applicable adapters, seeds 11/29/47, 28 discarded warmups and 84 measured attempts. Every attempt has a 30-second wall limit. All 84 workers completed without operational failures; this does not mean every stochastic estimate met its accuracy target. The [interactive comparison](comparison.html) and [summary CSV](comparison.summary.csv) preserve the complete measurements, uncertainty and resources.

On the six-variable portfolio fixture, exact enumeration, MILP and QAOA all found the verified optimum for every seed. Their full-wall medians were 0.0529, 0.2014 and 0.3442 seconds respectively. Across the four tiny optimization fixtures, all 60 measured optimization attempts returned the exact optimum. This is an easy correctness demonstration, not evidence that these solvers scale equally well.

On the eight-scenario estimation fixture, mean absolute payoff errors were 0.4809 for IID MC, 0.2000 for antithetic MC and 0.1709 for QAE. Their full-wall medians were 0.3662, 0.3696 and 0.5535 seconds. The exact weighted-sum baseline had zero sampling error and a median of 0.0515 seconds with eight scenario queries. The explicit input makes that classical baseline especially important. QAE consumed 452 logical shots and 2,034 oracle applications per run; each MC adapter used 2,048 scenario queries. Logical query counts are not classical CPU operation counts.

The controlled demonstration does not establish a quantum advantage. Measurements include cold process startup and native-library imports; timings vary with host load. Only three seeds and one instance per size are used. Quantum circuits are noiseless simulations with sampled measurement noise, generic dense input encoding and no hardware routing. Likelihood and Student-t intervals are approximate. The finite distribution has discretization bias relative to a continuous option model; no live trading, P&L, noise robustness or asymptotic scaling claim follows.

Validation: 59 tests passed locally on macOS and in the Linux Docker image. The container also completed the 28-attempt strict CI configuration. Editable installation and a wheel-installed MILP pipeline were verified outside the source checkout. The GitHub Actions workflow is included but was not run on a remote repository.

Reproduce from the project directory:

```bash
python -m project.reproduce --config configs/demo.yaml --output reports/comparison.html --strict
```
