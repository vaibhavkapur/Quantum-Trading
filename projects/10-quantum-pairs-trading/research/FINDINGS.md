# Recorded experiment findings

This snapshot answers a narrow question: can a small noiseless QAOA selector match
classical selection quality under the same pair, capital and overlap constraints?

The predeclared demo used six non-overlapping 63-session tests, three solver seeds,
five selectors, 60 optimizer evaluations, depth-one QAOA and 512 final shots.
All 90 solver/seed/fold combinations completed without failure. The data contains
purposefully cointegrated synthetic prices; the universe has nine fictitious assets.

SCIP MILP and QAOA plus deterministic feasibility repair reached the exact
selection optimum in all 18 of their respective fold/seed cases. They produced
the same test trading path: 13.76% net return over 378 out-of-sample
sessions, Sharpe 4.286, and 61 closed trades per seed. These
numbers describe this designed synthetic sample and are not market forecasts.

Average selection time across the complete six-fold path was
0.0320 seconds for MILP versus 0.5674 seconds
for QAOA. Full fold work, including screening, exact verification, validation,
backtesting and sensitivity, averaged 0.1771 seconds for MILP
and 0.7096 seconds for QAOA per seed. The entire invocation,
including all selectors and artifact writing, took 4.159 seconds.
Measurements include cold imports when encountered and are machine-dependent.

There is no evidence of a computational or trading advantage for quantum
selection here. Repair and finite-shot sample search contribute to QAOA's result;
this is not an isolated quantum-kernel achievement. Annealing's weaker training
objective occasionally produced higher test returns, illustrating that the
selection proxy and realized returns need not rank solvers identically.

The report provides bootstrap intervals over solver seeds, per-stage timings,
raw objective gaps and cost/entry perturbations. Seed intervals on one common
price history do not estimate market uncertainty. Sensitivities are descriptive
and did not choose parameters. Real-data robustness, larger problem scaling,
noise, hardware queue costs, and live execution remain untested.

Snapshot: `20261124T050747-c314b073`. See [comparison.html](../reports/comparison.html),
[raw metrics](../reports/comparison-folds.csv), and the
[manifest](../reports/demo-runs/20261124T050747-c314b073/manifest.json).
