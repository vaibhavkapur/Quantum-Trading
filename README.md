# Quantum Trading

Twelve research projects compare classical algorithms with CPU simulations of quantum methods across portfolio construction, trading, derivatives pricing, and risk. Each project includes its own setup instructions, synthetic data or data generator, and benchmark results.

## Getting started

```bash
git clone https://github.com/vaibhavkapur/Quantum-Trading.git
cd Quantum-Trading/projects/02-hybrid-quantum-portfolio-optimizer
```

Follow the README in the selected project directory. Create a separate Python environment for each project and run its commands from that directory so its configuration, data, and report paths resolve correctly.

For a first pass, start with portfolio optimization (02), then use the benchmark suite (12) to see how solution quality and computation time are compared. Statistical arbitrage (01), market making (06), and smart order routing (04) apply those ideas to trading and execution.

## Projects

- [01 · Quantum-Assisted Statistical Arbitrage Engine](projects/01-quantum-statistical-arbitrage/) — Select asset baskets and measure out-of-sample returns after trading costs.
- [02 · Hybrid Quantum–Classical Portfolio Optimizer](projects/02-hybrid-quantum-portfolio-optimizer/) — Compare discrete portfolio solvers under shared position and exposure constraints.
- [03 · Quantum Arbitrage Detection Engine](projects/03-quantum-arbitrage-detection/) — Search for executable cross-venue arbitrage cycles and compare graph and quantum optimization methods.
- [04 · Quantum-Optimized Smart Order Router](projects/04-quantum-smart-order-router/) — Split orders across simulated venues and measure execution costs and routing latency. See the [implementation and results](projects/04-quantum-smart-order-router/IMPLEMENTATION.md).
- [05 · Quantum Monte Carlo Options Pricing Engine](projects/05-quantum-monte-carlo-options/) — Price European, Asian, and barrier options with Monte Carlo and simulated quantum amplitude estimation.
- [06 · Quantum-Enhanced Market Making Engine](projects/06-quantum-market-making/) — Replay order books and compare inventory-aware quoting policies in a C++ matching engine.
- [07 · Quantum ML Order Book Predictor](projects/07-quantum-order-book-predictor/) — Compare classical and quantum models for short-horizon mid-price prediction.
- [08 · Quantum Risk Management Engine](projects/08-quantum-risk-engine/) — Estimate portfolio value at risk and expected shortfall, then stress-test the estimates.
- [09 · Quantum Strategy Parameter Optimizer](projects/09-quantum-strategy-optimizer/) — Select strategy parameters on training data and evaluate them on held-out periods.
- [10 · Quantum-Optimized Pairs Trading System](projects/10-quantum-pairs-trading/) — Screen for cointegration, select pairs under constraints, and backtest them with walk-forward evaluation.
- [11 · Quantum Portfolio Rebalancing Engine](projects/11-quantum-rebalancing/) — Construct integer-lot rebalance orders subject to fees, cash reserves, and exposure limits.
- [12 · Quantum-vs-Classical Trading Benchmark Suite](projects/12-quantum-classical-benchmark-suite/) — Compare solvers on reproducible finance problems with explicit budgets and exact reference solutions.
