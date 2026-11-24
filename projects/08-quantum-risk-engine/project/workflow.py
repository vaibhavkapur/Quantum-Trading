import importlib.metadata
import platform
from time import perf_counter

import numpy as np

from risk.backtest import rolling_backtest
from risk.measures import empirical_risk, gaussian_risk, student_t_risk
from scenarios import correlated_returns, importance_sample


def environment():
    packages = {}
    for package in ("numpy", "scipy", "qiskit", "fastapi", "pydantic"):
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = None
    return {"python": platform.python_version(), "platform": platform.platform(), "packages": packages}


def calculate(config, returns, solver="classical"):
    if solver not in ("classical", "quantum-sim"):
        raise ValueError("unknown solver")
    started = perf_counter()
    portfolio = config.portfolio.build()
    training = returns[:config.data.training_observations]
    fitted_mean = training.mean(axis=0)
    fitted_covariance = np.atleast_2d(np.cov(training, rowvar=False, ddof=1))
    mean_loss, std_loss = portfolio.loss_moments(fitted_mean, fitted_covariance)
    model = config.model
    reference = (gaussian_risk(mean_loss, std_loss, config.alpha) if model.distribution == "gaussian" else
                 student_t_risk(mean_loss, std_loss, model.df, config.alpha))
    fitting_done = perf_counter()
    rows = []

    def add(method, operation, **metadata):
        start = perf_counter()
        result = operation()
        rows.append({"method": method, **result, "seconds": perf_counter()-start,
                     "var_abs_error": abs(result["var"]-reference["var"]),
                     "cvar_abs_error": abs(result["cvar"]-reference["cvar"]), **metadata})

    add("historical", lambda: empirical_risk(portfolio.losses(training), config.alpha), samples=len(training))
    add("parametric_gaussian", lambda: gaussian_risk(mean_loss, std_loss, config.alpha), samples=0)
    for antithetic in (False, True):
        def mc(antithetic=antithetic):
            simulated = correlated_returns(fitted_mean, fitted_covariance, config.simulation.samples, config.seed+1,
                                           model.distribution, model.df, antithetic)
            return empirical_risk(portfolio.losses(simulated), config.alpha)
        add("mc_antithetic" if antithetic else "mc", mc, samples=config.simulation.samples)
    if model.distribution == "gaussian":
        def importance():
            simulated, weights, diagnostics = importance_sample(portfolio, fitted_mean, fitted_covariance,
                                                               config.simulation.samples, config.seed+2,
                                                               config.simulation.importance_shift)
            return {**empirical_risk(portfolio.losses(simulated), config.alpha, weights), "diagnostics": diagnostics}
        add("importance_sampling", importance, samples=config.simulation.samples,
            note="Self-normalized likelihood ratios; finite-sample biased")
    classical_done = perf_counter()
    all_losses = portfolio.losses(returns)
    backtests = [rolling_backtest(all_losses, config.data.training_observations, config.alpha, method)
                 for method in ("historical", "gaussian")]
    backtest_done = perf_counter()
    quantum_result, fallback_reason, effective_solver = None, None, solver
    if solver == "quantum-sim":
        try:
            from quantum.estimation import estimate_risk
            from quantum.distribution import discretize_loss
        except ImportError as error:
            if not config.quantum.fallback:
                raise RuntimeError("quantum backend unavailable; install .[quantum] or explicitly enable quantum.fallback") from error
            effective_solver, fallback_reason = "classical", str(error)
        else:
            settings = config.quantum
            grid_started = perf_counter()
            grid = discretize_loss(mean_loss, std_loss, settings.qubits, settings.support_std, model.distribution, model.df)
            exact = grid.risk(config.alpha)
            grid_seconds = perf_counter()-grid_started
            estimate = estimate_risk(grid, config.alpha, settings.powers, settings.shots, config.seed)
            quantum_result = {**estimate, "exact_discrete_risk": exact,
                              "grid_seconds": grid_seconds, "clipped_tail_mass": grid.clipped_tail_mass,
                              "grid_losses": grid.losses.tolist(), "grid_probabilities": grid.probabilities.tolist(),
                              "discretization_var_error": abs(exact["var"]-reference["var"]),
                              "discretization_cvar_error": abs(exact["cvar"]-reference["cvar"]),
                              "estimation_var_error": abs(estimate["var"]-exact["var"]),
                              "estimation_cvar_error": abs(estimate["cvar"]-exact["cvar"])}
            rows.append({"method": "quantum_mlae", "alpha": config.alpha, "var": estimate["var"], "cvar": estimate["cvar"],
                         "seconds": estimate["total_seconds"]+grid_seconds,
                         "var_abs_error": abs(estimate["var"]-reference["var"]),
                         "cvar_abs_error": abs(estimate["cvar"]-reference["cvar"]),
                         "samples": estimate["total_shots"], "oracle_queries": estimate["total_oracle_queries"]})
    stress = portfolio.stress(config.stress)
    ended = perf_counter()
    return {"solver_requested": solver, "solver_effective": effective_solver, "fallback_reason": fallback_reason,
            "status": "fallback" if fallback_reason else "ok", "alpha": config.alpha,
            "reference": {**reference, "distribution": model.distribution, "note": "Analytic fitted-model reference, not ground-truth market risk"},
            "fit": {"mean": fitted_mean.tolist(), "covariance": fitted_covariance.tolist(), "loss_mean": mean_loss, "loss_std": std_loss,
                    "training_range": [0, len(training)], "test_range": [len(training), len(returns)]},
            "metrics": rows, "backtests": backtests, "stress": stress, "quantum": quantum_result,
            "timings": {"fit_seconds": fitting_done-started, "classical_seconds": classical_done-fitting_done,
                        "backtest_seconds": backtest_done-classical_done, "quantum_and_stress_seconds": ended-backtest_done,
                        "compute_seconds": ended-started},
            "limitations": ["Synthetic one-day linear simple-return risk; no derivatives or multi-period compounding.",
                            "Historical estimates and fitted-model estimates have different sampling assumptions.",
                            "Antithetic sampling is a variance-reduction candidate, not a guaranteed tail-quantile improvement.",
                            "Noiseless classical simulation does not establish quantum advantage."]}
