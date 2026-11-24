"""Run the classical or quantum-simulator experiments and store raw results.

    python -m qmc_options.run --config configs/demo.yaml --solver classical
    python -m qmc_options.run --config configs/demo.yaml --solver quantum-sim
"""
from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from . import analytic, classical_mc
from .config import (instrument_by_name, instruments_from, load_config, manifest, market_from,
                     reference_price, sha256_file, write_json)
from .prepare_data import dataset_key, manifest_path
from .quantum import amplitude_estimation as ae
from .quantum.discretize import discretize
from .quantum.encoding import build_a_operator, build_state_preparation
from .quantum.pricer import estimate_price, prepare, quantum_delta
from .quantum.resources import circuit_resources

SOLVERS = ("classical", "quantum-sim")
CPP_BINARY = Path(__file__).resolve().parent.parent / "pricing-cpp" / "build" / "mc_pricer"


def _score(row: Dict[str, Any], target: float) -> Dict[str, Any]:
    row["abs_error"] = abs(row["price"] - target)
    row["covered"] = bool(row["ci_low"] <= target <= row["ci_high"])
    return row


# --------------------------------------------------------------------------- classical

def _dataset_rows(config, market, instruments, references) -> Dict[str, Any]:
    path = Path(config["data"]["path"])
    meta_path = manifest_path(path)
    if not path.exists() or not meta_path.exists():
        return {"status": "missing", "detail": f"{path} not found; run prepare_data first", "rows": []}
    meta = json.loads(meta_path.read_text())
    if sha256_file(path) != meta["sha256"]:
        return {"status": "hash_mismatch", "detail": f"{path} does not match its manifest", "rows": []}
    rows = []
    with np.load(path) as data:
        for name, option in instruments:
            est = classical_mc.price_from_paths(market, option, data[dataset_key(option.maturity, option.steps)],
                                                config["confidence"])
            rows.append(_score({"instrument": name, **est.to_dict()}, references[name]["price"]))
    return {"status": "ok", "sha256": meta["sha256"], "content_sha256": meta["content_sha256"], "rows": rows}


def _cpp_throughput(market, option, n_paths: int) -> Dict[str, Any]:
    """Optional: time the C++ kernel if it has been built."""
    if not CPP_BINARY.exists():
        return {"status": "not_built", "detail": "build pricing-cpp to include the C++ baseline"}
    try:
        cmd = [str(CPP_BINARY), "--kind", option.kind, "--spot", str(market.spot), "--rate", str(market.rate),
               "--vol", str(market.vol), "--strike", str(option.strike), "--maturity", str(option.maturity),
               "--steps", str(option.steps), "--paths", str(n_paths), "--seed", "1"]
        if option.barrier is not None:
            cmd += ["--barrier", str(option.barrier)]
        out = json.loads(subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=120).stdout)
        return {"status": "ok", **out}
    except (subprocess.SubprocessError, OSError, ValueError) as exc:   # fall back to NumPy-only results
        return {"status": "failed", "detail": str(exc)}


def run_classical(config: Dict[str, Any]) -> Dict[str, Any]:
    market = market_from(config)
    instruments = instruments_from(config)
    spec = config["classical"]
    conf = config["confidence"]
    references = {name: reference_price(config, market, option) for name, option in instruments}

    prices: List[Dict[str, Any]] = []
    for name, option in instruments:
        for method in spec["methods"]:
            for seed in config["seeds"]:
                est = classical_mc.mc_price(market, option, int(spec["n_paths"]), method, seed, conf)
                prices.append(_score({"instrument": name, "seed": seed, **est.to_dict()},
                                     references[name]["price"]))

    curve_spec = spec["error_curve"]
    curve_option = instrument_by_name(config, curve_spec["instrument"])
    curve_ref = references[curve_spec["instrument"]]["price"]
    curve = []
    for method in spec["methods"]:
        for n in curve_spec["n_paths"]:
            runs = [classical_mc.mc_price(market, curve_option, int(n), method, 10_000 + rep, conf)
                    for rep in range(int(curve_spec["repeats"]))]
            errors = np.array([r.price - curve_ref for r in runs])
            curve.append({
                "method": method, "n_paths": runs[0].n_paths, "repeats": len(runs),
                "rmse": float(np.sqrt(np.mean(errors**2))),
                "coverage": float(np.mean([r.ci_low <= curve_ref <= r.ci_high for r in runs])),
                "mean_wall_time_s": float(np.mean([r.wall_time_s for r in runs])),
            })

    greeks = []
    for name, option in instruments:
        exact = (analytic.black_scholes_greeks(market, option.strike, option.maturity,
                                               call=option.kind == "european_call")
                 if not option.path_dependent else {})
        estimates = classical_mc.mc_greeks(market, option, int(spec["greeks"]["n_paths"]),
                                           seed=config["seeds"][0], rel_bump=float(spec["greeks"]["rel_bump"]),
                                           confidence=conf)
        for greek, est in estimates.items():
            greeks.append({"instrument": name, "greek": greek, "value": est.price, "stderr": est.stderr,
                           "ci_low": est.ci_low, "ci_high": est.ci_high, "analytic": exact.get(greek),
                           "n_paths": est.n_paths, "rel_bump": est.extra["rel_bump"]})

    n_big = 2_000_000
    numpy_run = classical_mc.mc_price(market, curve_option, n_big, "plain", seed=1)
    throughput = {
        "instrument": curve_spec["instrument"], "n_paths": n_big,
        "numpy": {"wall_time_s": numpy_run.wall_time_s, "paths_per_s": n_big / numpy_run.wall_time_s,
                  "price": numpy_run.price},
        "cpp": _cpp_throughput(market, curve_option, n_big),
    }
    return {
        "manifest": manifest(config, "classical"),
        "references": references,
        "prices": prices,
        "dataset": _dataset_rows(config, market, instruments, references),
        "error_curve": {"instrument": curve_spec["instrument"], "reference": curve_ref, "rows": curve},
        "greeks": greeks,
        "throughput": throughput,
    }


# --------------------------------------------------------------------------- quantum (simulated)

def _scaling(prepared, spec: Dict[str, Any], alpha: float) -> List[Dict[str, Any]]:
    """Error against oracle queries, repeated over independent seeds. The
    error is measured against the exact amplitude of the same circuit."""
    settings = ([("direct_sampling", {"shots": int(s)}) for s in spec["direct_sampling_shots"]]
                + [("mlae", {"max_power_exponent": int(e), "shots": int(spec["mlae_shots"])})
                   for e in spec["mlae_exponents"]]
                + [("iqae", {"epsilon": float(e), "shots": 100}) for e in spec["iqae_epsilons"]])
    exact = prepared.problem.exact_price
    rows = []
    for estimator, options in settings:
        runs = [estimate_price(prepared, estimator, seed=20_000 + rep, alpha=alpha, **options)
                for rep in range(int(spec["repeats"]))]
        errors = np.array([r["price"] - exact for r in runs])
        rows.append({
            "estimator": estimator, "options": options, "repeats": len(runs),
            "mean_oracle_queries": float(np.mean([r["oracle_queries"] for r in runs])),
            "mean_shots": float(np.mean([r["shots"] for r in runs])),
            "mean_max_grover_power": float(np.mean([r["max_grover_power"] for r in runs])),
            "rmse": float(np.sqrt(np.mean(errors**2))),
            "mean_ci_half_width": float(np.mean([(r["ci_high"] - r["ci_low"]) / 2 for r in runs])),
            "coverage": float(np.mean([r["ci_low"] <= exact <= r["ci_high"] for r in runs])),
        })
    return rows


def _noise_study(amplitude: float, scale: float, spec: Dict[str, Any], alpha: float) -> List[Dict[str, Any]]:
    """MLAE fitted with the noiseless likelihood while the device depolarises."""
    rows = []
    for rate in spec["rates"]:
        estimates, covered = [], []
        for rep in range(int(spec["repeats"])):
            oracle = ae.AnalyticOracle(amplitude, noise_rate=float(rate))
            res = ae.mlae(oracle, np.random.default_rng(30_000 + rep), int(spec["max_power_exponent"]),
                          int(spec["shots"]), alpha)
            estimates.append(res.estimate)
            covered.append(res.ci_low <= amplitude <= res.ci_high)
        errors = scale * (np.array(estimates) - amplitude)
        rows.append({"noise_rate_per_grover_iteration": float(rate), "repeats": len(estimates),
                     "rmse": float(np.sqrt(np.mean(errors**2))), "bias": float(errors.mean()),
                     "coverage": float(np.mean(covered))})
    return rows


def run_quantum_sim(config: Dict[str, Any]) -> Dict[str, Any]:
    market = market_from(config)
    spec = config["quantum"]
    alpha = 1.0 - config["confidence"]

    def qubits_for(option) -> int:
        return int(spec["qubits_per_step"] if option.path_dependent else spec["qubits_european"])

    def truncation_for(option) -> float:
        return float(spec["truncation_path_dependent"] if option.path_dependent else spec["truncation_european"])

    instruments, prices, prepared_by_name = [], [], {}
    for name, option in instruments_from(config):
        reference = reference_price(config, market, option)
        prepared = prepare(market, option, qubits_for(option), truncation_for(option))
        prepared_by_name[name] = prepared
        problem = prepared.problem
        instruments.append({
            "instrument": name, "qubits_per_step": problem.qubits_per_step, "steps": problem.steps,
            "truncation": truncation_for(option),
            "state_qubits": problem.num_state_qubits, "total_qubits": problem.num_state_qubits + 1,
            "basis_states": int(problem.probs.size), "amplitude": problem.amplitude,
            "discretized_exact_price": problem.exact_price, "reference": reference,
            "discretization_bias": problem.exact_price - reference["price"],
            "a_operator": prepared.a_resources, "grover_operator": prepared.q_resources,
            "simulation_mode": prepared.oracle.mode, "timings": prepared.timings,
        })
        for estimator in ("direct_sampling", "mlae", "iqae"):
            for seed in config["seeds"]:
                row = estimate_price(prepared, estimator, seed, alpha, **spec[estimator])
                row["instrument"] = name
                row["abs_error_vs_reference"] = abs(row["price"] - reference["price"])
                prices.append(_score(row, problem.exact_price))

    scaling_name = spec["scaling"]["instrument"]
    scaling_rows = _scaling(prepared_by_name[scaling_name], spec["scaling"], alpha)
    scaling_problem = prepared_by_name[scaling_name].problem

    discretization = []
    scaling_option = instrument_by_name(config, scaling_name)
    scaling_ref = reference_price(config, market, scaling_option)["price"]
    for cut in spec["discretization_truncations"]:
        for qubits in spec["discretization_qubits"]:
            problem = discretize(market, scaling_option, int(qubits), float(cut))
            loader = circuit_resources(build_state_preparation(problem))
            full = circuit_resources(build_a_operator(problem))
            discretization.append({"truncation": float(cut), "qubits": int(qubits),
                                   "discretized_exact_price": problem.exact_price,
                                   "bias": problem.exact_price - scaling_ref,
                                   "state_prep_cx": loader["cx"], "a_operator_cx": full["cx"],
                                   "a_operator_depth": full["depth"]})

    delta_spec = spec["delta"]
    delta_option = instrument_by_name(config, delta_spec["instrument"])
    exact_delta = (analytic.black_scholes_greeks(market, delta_option.strike, delta_option.maturity,
                                                 call=delta_option.kind == "european_call")["delta"]
                   if not delta_option.path_dependent else None)
    deltas = []
    for bump in delta_spec["rel_bumps"]:
        row = quantum_delta(market, delta_option, qubits_for(delta_option), float(bump),
                            seed=config["seeds"][0], truncation=truncation_for(delta_option), alpha=alpha,
                            epsilon=float(delta_spec["epsilon"]))
        row["analytic_delta"] = exact_delta
        deltas.append(row)

    return {
        "manifest": manifest(config, "quantum-sim"),
        "instruments": instruments,
        "prices": prices,
        "scaling": {"instrument": scaling_name, "price_scale": scaling_problem.scale,
                    "discretized_exact_price": scaling_problem.exact_price, "rows": scaling_rows},
        "discretization": {"instrument": scaling_name, "reference": scaling_ref, "rows": discretization},
        "noise": {"instrument": scaling_name, "model": "depolarising per Grover iteration (toy model)",
                  "max_grover_power": 2 ** (int(spec["noise"]["max_power_exponent"]) - 1),
                  "rows": _noise_study(scaling_problem.amplitude, scaling_problem.scale, spec["noise"], alpha)},
        "delta": {"instrument": delta_spec["instrument"], "rows": deltas},
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True)
    parser.add_argument("--solver", required=True, choices=SOLVERS)
    parser.add_argument("--runs-dir", default="runs")
    args = parser.parse_args(argv)
    config = load_config(args.config)
    start = time.perf_counter()
    result = run_classical(config) if args.solver == "classical" else run_quantum_sim(config)
    result["manifest"]["total_wall_time_s"] = time.perf_counter() - start
    out = Path(args.runs_dir) / f"{config['name']}-{args.solver}.json"
    write_json(out, result)
    print(f"{args.solver}: {len(result['prices'])} price estimates in "
          f"{result['manifest']['total_wall_time_s']:.1f}s -> {out}")
    if args.solver == "classical" and result["dataset"]["status"] != "ok":
        print(f"note: dataset check skipped ({result['dataset']['detail']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
