"""Maximum-likelihood amplitude estimation from actual Qiskit A and Q operators.

Circuit simulation is dense and noiseless. Measurements are sampled from the
simulated state, not from the known target probability. No QPU is contacted.
The likelihood fit receives counts only; exact targets are evaluation metadata.
"""
from time import perf_counter

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import StatePreparation, UCRYGate, grover_operator
from qiskit.quantum_info import Operator, Statevector
from scipy.optimize import minimize_scalar
from scipy.special import xlogy
from scipy.stats import chi2

from risk.measures import validate_alpha


def build_oracle(distribution, objective):
    objective = np.asarray(objective, dtype=float)
    if objective.shape != distribution.losses.shape or not np.isfinite(objective).all() or np.any((objective < 0) | (objective > 1)):
        raise ValueError("objective must be a finite vector in [0,1] matching the loss grid")
    n = distribution.qubits
    a = QuantumCircuit(n+1, name="A")
    a.append(StatePreparation(np.sqrt(distribution.probabilities)), range(n))
    a.append(UCRYGate((2*np.arcsin(np.sqrt(objective))).tolist()), [n] + list(range(n)))
    good = QuantumCircuit(n+1)
    good.z(n)
    q = grover_operator(good, state_preparation=a)
    return a, q


def validate_budget(powers, shots):
    if not powers or powers[0] != 0 or len(powers) > 7 or len(set(powers)) != len(powers):
        raise ValueError("powers must start with zero and contain at most 7 distinct entries")
    if any(isinstance(k, bool) or not isinstance(k, int) or not 0 <= k <= 32 for k in powers):
        raise ValueError("Grover powers must be integers in [0,32]")
    if powers != sorted(powers) or not isinstance(shots, int) or isinstance(shots, bool) or not 1 <= shots <= 4096:
        raise ValueError("powers must increase and shots must be an integer in [1,4096]")


def fit_amplitude(powers, shots, counts):
    """Global likelihood grid + local refinement; interval is an asymptotic profile hull.

    The hull includes all grid-resolved likelihood modes, not just a Fisher
    interval around one mode. It is not a finite-sample coverage guarantee.
    """
    multipliers = 2*np.asarray(powers) + 1
    counts = np.asarray(counts)

    def nll(theta):
        probabilities = np.clip(np.sin(np.multiply.outer(theta, multipliers))**2, 1e-15, 1-1e-15)
        return -np.sum(xlogy(counts, probabilities) + xlogy(shots-counts, 1-probabilities), axis=-1)

    grid = np.linspace(0, np.pi/2, max(4097, 512*int(multipliers.max())+1))
    values = nll(grid)
    i = int(values.argmin())
    fitted = minimize_scalar(nll, bounds=(grid[max(i-1, 0)], grid[min(i+1, len(grid)-1)]), method="bounded",
                             options={"xatol": 1e-14})
    theta = min([0.0, float(fitted.x), float(grid[i]), np.pi/2], key=lambda x: float(nll(x)))
    selected = np.flatnonzero(values <= float(nll(theta)) + chi2.ppf(0.95, 1)/2)
    lo = grid[max(0, int(selected[0])-1)] if selected.size else theta
    hi = grid[min(len(grid)-1, int(selected[-1])+1)] if selected.size else theta
    return {"estimate": float(np.sin(theta)**2), "ci95": [float(np.sin(lo)**2), float(np.sin(hi)**2)],
            "interval_method": "asymptotic profile-likelihood hull; no finite-sample guarantee"}


def estimate_amplitude(distribution, objective, powers=(0, 1, 2, 4, 8), shots=128, seed=42):
    powers = list(powers)
    validate_budget(powers, shots)
    started = perf_counter()
    a, q = build_oracle(distribution, objective)
    built = perf_counter()
    # Compile the largest executed circuit to a fixed basis for auditable costs.
    longest = a.copy()
    for _ in range(max(powers)):
        longest.compose(q, inplace=True)
    compiled = transpile(longest, basis_gates=["u", "cx"], optimization_level=0, seed_transpiler=seed)
    compiled_at = perf_counter()
    state = np.asarray(Statevector(a).data)
    operator = Operator(q).data
    rng = np.random.default_rng(seed)
    previous, counts = 0, []
    for power in powers:
        for _ in range(power-previous):
            state = np.einsum("ij,j->i", operator, state)
        probability = float(np.clip(np.sum(np.abs(state[len(state)//2:])**2), 0, 1))
        counts.append(int(rng.binomial(shots, probability)))
        previous = power
    sampled_at = perf_counter()
    result = fit_amplitude(powers, shots, counts)
    ended = perf_counter()
    return {**result, "method": "maximum_likelihood_amplitude_estimation",
            "backend": "qiskit_dense_statevector_noiseless", "seed": seed,
            "schedule": [{"power": k, "shots": shots, "ones": c} for k, c in zip(powers, counts)],
            "resources": {"qubits": a.num_qubits, "shots": shots * len(powers),
                          "oracle_queries": shots * sum(2*k+1 for k in powers),
                          "grover_iterations": shots * sum(powers), "max_power": max(powers),
                          "max_circuit_depth": int(compiled.depth()),
                          "max_circuit_cx": int(compiled.count_ops().get("cx", 0)),
                          "state_preparation_amplitudes": len(distribution.losses),
                          "hardware": False, "noise_model": "none", "queue_seconds": 0.0,
                          "communication_seconds": 0.0, "qpu_cost_usd": 0.0},
            "timings": {"oracle_build_seconds": built-started, "compile_seconds": compiled_at-built,
                        "simulate_and_sample_seconds": sampled_at-compiled_at,
                        "decode_seconds": ended-sampled_at, "total_seconds": ended-started}}


def estimate_risk(distribution, alpha=0.99, powers=(0, 1, 2, 4, 8), shots=128, seed=42):
    """Noisy CDF bisection followed by a scaled excess-loss expectation oracle."""
    validate_alpha(alpha)
    validate_budget(list(powers), shots)
    started = perf_counter()
    lo, hi, queries = 0, len(distribution.losses)-1, []
    while lo < hi:
        middle = (lo+hi)//2
        threshold = float(distribution.losses[middle])
        result = estimate_amplitude(distribution, (distribution.losses <= threshold).astype(float),
                                    powers, shots, seed+len(queries))
        queries.append({"kind": "cdf", "threshold": threshold, **result})
        if result["estimate"] >= alpha:
            hi = middle
        else:
            lo = middle+1
    var = float(distribution.losses[lo])
    scale = float(distribution.losses[-1]-var)
    cvar = var
    if scale > 0:
        excess = np.maximum(distribution.losses-var, 0)/scale
        result = estimate_amplitude(distribution, excess, powers, shots, seed+len(queries))
        queries.append({"kind": "excess_loss", "threshold": var, "scale": scale, **result})
        cvar += scale*result["estimate"]/(1-alpha)
    return {"alpha": alpha, "var": var, "cvar": float(cvar), "expected_shortfall": float(cvar),
            "queries": queries,
            "ambiguous_cdf_steps": sum(q["ci95"][0] <= alpha <= q["ci95"][1] for q in queries if q["kind"] == "cdf"),
            "total_shots": sum(q["resources"]["shots"] for q in queries),
            "total_oracle_queries": sum(q["resources"]["oracle_queries"] for q in queries),
            "total_seconds": perf_counter()-started,
            "note": "Approximate noisy quantile search; ES errors are amplified by 1/(1-alpha). No joint risk interval."}
