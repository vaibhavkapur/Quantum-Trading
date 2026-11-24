import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.stats import norm, t

from project.config import fingerprint, load_config
from project.data import load_dataset
from project.workflow import environment
from quantum.distribution import discretize_loss
from quantum.estimation import estimate_amplitude
from risk.measures import gaussian_risk, student_t_risk


def run_benchmark(config, root, output="reports/benchmark"):
    started = perf_counter()
    returns, manifest = load_dataset(config, root)
    loaded = perf_counter()
    train = returns[:config.data.training_observations]
    mu, sigma = config.portfolio.build().loss_moments(train.mean(axis=0), np.atleast_2d(np.cov(train, rowvar=False)))
    fitted = perf_counter()
    reference = (gaussian_risk(mu, sigma, config.alpha) if config.model.distribution == "gaussian" else
                 student_t_risk(mu, sigma, config.model.df, config.alpha))
    threshold = reference["var"]
    grid = discretize_loss(mu, sigma, config.quantum.qubits, config.quantum.support_std,
                           config.model.distribution, config.model.df)
    objective = (grid.losses > threshold).astype(float)
    exact = float(np.dot(grid.probabilities, objective))
    continuous_tail = (1-config.alpha) if sigma > 0 else 0.0
    prepared = perf_counter()
    preparation = {"load_seconds": loaded-started, "fit_seconds": fitted-loaded, "grid_seconds": prepared-fitted}
    rows = []
    for maximum in config.benchmark.powers:
        powers = [0] + [2**i for i in range(int(np.log2(maximum))+1)]
        budget = config.benchmark.shots * sum(2*k+1 for k in powers)
        for seed in config.benchmark.seeds:
            common = {"seed": seed, "max_power": maximum, "oracle_queries": budget,
                      "exact_probability": exact, "continuous_probability": continuous_tail,
                      "discretization_error": abs(exact-continuous_tail), **preparation}
            try:
                result = estimate_amplitude(grid, objective, powers, config.benchmark.shots, seed)
                rows.append({**common, "method": "quantum_mlae_tail", "status": "ok", "error": "",
                             "estimate": result["estimate"], "absolute_error": abs(result["estimate"]-exact),
                             "ci_low": result["ci95"][0], "ci_high": result["ci95"][1],
                             "ci_covers_exact": result["ci95"][0] <= exact <= result["ci95"][1],
                             "seconds": result["timings"]["total_seconds"] + sum(preparation.values()),
                             **result["timings"], **result["resources"]})
            except Exception as error:
                # Preserve failed trials; never silently exclude them from the denominator.
                rows.append({**common, "method": "quantum_mlae_tail", "status": "failed", "error": repr(error)})
            mc_start = perf_counter()
            indices = np.random.default_rng(seed).choice(len(grid.losses), size=budget, p=grid.probabilities)
            count = int(np.sum(grid.losses[indices] > threshold))
            estimate = count / budget
            from scipy.stats import beta
            lo = 0.0 if count == 0 else float(beta.ppf(0.025, count, budget-count+1))
            hi = 1.0 if count == budget else float(beta.ppf(0.975, count+1, budget-count))
            seconds = perf_counter()-mc_start
            rows.append({**common, "method": "mc_tail", "status": "ok", "error": "",
                         "estimate": estimate, "absolute_error": abs(estimate-exact),
                         "ci_low": lo, "ci_high": hi, "ci_covers_exact": lo <= exact <= hi,
                         "shots": budget, "total_seconds": seconds, "seconds": seconds + sum(preparation.values()),
                         "qubits": 0, "max_circuit_depth": 0, "max_circuit_cx": 0})
    summary = []
    for maximum in config.benchmark.powers:
        for method in ("mc_tail", "quantum_mlae_tail"):
            group = [r for r in rows if r["max_power"] == maximum and r["method"] == method]
            valid = [r for r in group if r["status"] == "ok"]
            if not valid:
                summary.append({"method": method, "max_power": maximum, "successful": 0, "failed": len(group)})
                continue
            errors = np.array([r["absolute_error"] for r in valid])
            half = float(t.ppf(0.975, len(valid)-1) * errors.std(ddof=1) / np.sqrt(len(valid))) if len(valid) > 1 else None
            summary.append({"method": method, "max_power": maximum, "oracle_queries": valid[0]["oracle_queries"],
                            "successful": len(valid), "failed": len(group)-len(valid),
                            "mean_absolute_error": float(errors.mean()),
                            "mean_error_ci95": None if half is None else [max(0, float(errors.mean())-half), float(errors.mean())+half],
                            "rmse": float(np.sqrt(np.mean(errors**2))),
                            "interval_coverage": float(np.mean([r["ci_covers_exact"] for r in valid])),
                            "mean_seconds": float(np.mean([r["seconds"] for r in valid])),
                            "max_circuit_depth": max(r["max_circuit_depth"] for r in valid)})
    result = {"schema_version": 1, "created_at": datetime.now(timezone.utc).isoformat(),
              "config_sha256": fingerprint(config.model_dump()), "data_sha256": manifest["sha256"],
              "environment": environment(), "seeds": config.benchmark.seeds, "threshold": threshold,
              "exact_discrete_tail": exact, "continuous_tail": continuous_tail,
              "clipped_tail_mass": grid.clipped_tail_mass, "summary": summary, "trials": rows,
              "wall_seconds": perf_counter()-started,
              "protocol": "Identical discrete loss PMF and strict tail event. MC sample count equals A/A-inverse calls (shots * sum(2k+1)). Common loading/fitting/grid costs charged to both; quantum build/compile/simulation/decode included. No hardware timings.",
              "intervals": "MC: exact binomial; MLAE: asymptotic profile hull. Mean error: Student-t interval across predeclared seeds; small-sample descriptive interval, not a scaling proof."}
    output = Path(root) / output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.with_suffix(".json").write_text(json.dumps(result, indent=2, allow_nan=False))
    fields = sorted(set().union(*(row.keys() for row in rows)))
    with output.with_suffix(".csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return result


def main():
    parser = argparse.ArgumentParser(description="Compare MC and MLAE at identical oracle-query budgets")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", default="reports/benchmark")
    args = parser.parse_args()
    config, root = load_config(args.config)
    result = run_benchmark(config, root, args.output)
    print(f"Recorded {len(result['trials'])} trials in {args.output}.json and .csv")
    if any(row["status"] != "ok" for row in result["trials"]):
        raise SystemExit("Benchmark contains failed trials; see error fields")


if __name__ == "__main__":
    main()
