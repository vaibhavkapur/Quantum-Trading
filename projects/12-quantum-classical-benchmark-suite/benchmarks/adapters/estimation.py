from __future__ import annotations

import numpy as np

from .base import BudgetExceeded, Context


def amplitude_circuits(problem: dict, powers: list[int]) -> list[QuantumCircuit]:
    """Construct actual Q^k A circuits from all scenario probabilities and payoffs."""
    from qiskit import QuantumCircuit
    from qiskit.circuit.library import StatePreparation

    p, f = np.asarray(problem["probabilities"]), np.asarray(problem["payoffs"])
    amplitudes = np.r_[np.sqrt(p * (1 - f)), np.sqrt(p * f)]
    width = int(np.log2(problem["size"])) + 1
    a = QuantumCircuit(width)
    a.append(StatePreparation(amplitudes), range(width))
    q = QuantumCircuit(width)
    q.z(width - 1)  # Reflect good states, i.e. payoff ancilla = 1.
    q.compose(a.inverse(), inplace=True)
    q.x(range(width))
    q.h(width - 1)
    q.mcx(list(range(width - 1)), width - 1)
    q.h(width - 1)
    q.x(range(width))  # Reflect |0...0>; global sign does not affect measurements.
    q.compose(a, inplace=True)
    circuits = []
    for power in powers:
        circuit = a.copy()
        for _ in range(power):
            circuit.compose(q, inplace=True)
        circuits.append(circuit)
    return circuits


def maximum_likelihood(schedule: list[dict]) -> tuple[float, list[float]]:
    """Global grid plus local polishing; CI is an asymptotic profile-likelihood envelope."""
    from scipy.optimize import minimize_scalar
    from scipy.stats import chi2

    multipliers = np.array([2 * row["power"] + 1 for row in schedule])
    shots, hits = np.array([row["shots"] for row in schedule]), np.array([row["hits"] for row in schedule])

    def nll(theta):
        probs = np.clip(np.sin(np.outer(np.atleast_1d(theta), multipliers)) ** 2, 1e-14, 1 - 1e-14)
        return -(hits * np.log(probs) + (shots - hits) * np.log1p(-probs)).sum(axis=-1)

    grid = np.linspace(0, np.pi / 2, max(4097, int(multipliers.max()) * 512 + 1))
    values = nll(grid)
    index = int(values.argmin())
    lo, hi = grid[max(0, index - 1)], grid[min(len(grid) - 1, index + 1)]
    polished = minimize_scalar(lambda x: float(nll(x)[0]), bounds=(lo, hi), method="bounded",
                               options={"xatol": 1e-12})
    theta = min([float(polished.x), float(grid[index]), 0.0, np.pi / 2], key=lambda x: float(nll(x)[0]))
    accepted = np.flatnonzero(values <= float(nll(theta)[0]) + chi2.ppf(0.95, 1) / 2)
    if not len(accepted):
        accepted = np.array([index])
    bounds = [grid[max(0, accepted[0] - 1)], grid[min(len(grid) - 1, accepted[-1] + 1)]]
    return float(np.sin(theta) ** 2), [float(np.sin(x) ** 2) for x in bounds]


class EstimationAdapter:
    def __init__(self, name: str):
        self.name = name

    def solve(self, problem: dict, context: Context) -> dict:
        if self.name == "exact-expectation":
            with context.stage("encoding"):
                probabilities, payoffs = np.asarray(problem["probabilities"]), np.asarray(problem["payoffs"])
            with context.stage("execution"):
                context.charge(evaluations=1, queries=problem["size"])
                estimate = float(probabilities @ payoffs * problem["scale"])
            context.resources["evaluation_unit"] = "weighted_distribution_sum"
            return {"estimate": estimate, "estimate_ci95": [estimate, estimate],
                    "interval_method": "exact finite-distribution sum; no sampling uncertainty",
                    "termination": "all scenarios evaluated exactly"}
        return self.qae(problem, context) if self.name == "qae" else self.mc(problem, context)

    def mc(self, p: dict, ctx: Context) -> dict:
        with ctx.stage("setup"):
            from scipy.stats import t

        with ctx.stage("encoding"):
            cdf = np.cumsum(p["probabilities"])
            cdf[-1] = 1.0
            payoffs = np.array(p["payoffs"]) * p["scale"]
        antithetic = self.name == "mc-antithetic"
        n = ctx.budget.max_oracle_queries
        if antithetic:
            n -= n % 2
        if n < 2:
            raise BudgetExceeded("at least two scenario queries required for MC uncertainty")
        ctx.resources.update(evaluation_unit="scenario_payoff", objective_evaluations=0)
        with ctx.stage("execution"):
            # Query count is distinct from optimizer/candidate objective evaluations.
            ctx.charge(queries=n)
            uniforms = ctx.rng.random(n // 2 if antithetic else n)
            first = payoffs[np.searchsorted(cdf, uniforms, side="right")]
            if antithetic:
                second = payoffs[np.searchsorted(cdf, np.minimum(1 - uniforms, np.nextafter(1., 0.)), side="right")]
                observations = (first + second) / 2
            else:
                observations = first
        with ctx.stage("decoding"):
            estimate = float(observations.mean())
            ci = None
            if len(observations) > 1:
                half = float(t.ppf(0.975, len(observations) - 1) * observations.std(ddof=1) / np.sqrt(len(observations)))
                ci = [max(0., estimate - half), min(p["scale"], estimate + half)]
            return {"estimate": estimate, "estimate_ci95": ci, "independent_observations": len(observations),
                    "interval_method": "Student-t on independent pair means" if antithetic else "Student-t on IID payoffs",
                    "termination": "query budget sampled"}

    def qae(self, p: dict, ctx: Context) -> dict:
        with ctx.stage("setup"):
            from qiskit import transpile
            from qiskit.quantum_info import Statevector

        powers = ctx.algorithm["qae_powers"]
        shots = min(ctx.algorithm["shots_per_circuit"], ctx.budget.max_shots // len(powers),
                    ctx.budget.max_oracle_queries // sum(2 * k + 1 for k in powers))
        if shots < 1:
            raise BudgetExceeded("QAE schedule cannot fit shot/oracle query budget")
        with ctx.stage("encoding"):
            circuits = amplitude_circuits(p, powers)
        with ctx.stage("compilation"):
            compiled = [transpile(c, basis_gates=["u", "cx"], optimization_level=0, seed_transpiler=ctx.seed)
                        for c in circuits]
        ctx.resources.update(logical_qubits=compiled[0].num_qubits,
                             circuit_depth=max(c.depth() for c in compiled),
                             two_qubit_gates=max(c.count_ops().get("cx", 0) for c in compiled),
                             noise_model="none; noiseless statevector with finite measurement shots",
                             hardware_kind="quantum_simulator_cpu", evaluation_unit="likelihood_fit",
                             state_preparation="generic dense StatePreparation; exponential input encoding")
        schedule = []
        with ctx.stage("execution"):
            for power, circuit in zip(powers, compiled):
                ctx.charge(shots=shots, queries=shots * (2 * power + 1))
                probabilities = Statevector.from_instruction(circuit).probabilities()
                probability = float(probabilities[p["size"]:].sum())
                hits = int(ctx.rng.binomial(shots, np.clip(probability, 0, 1)))
                schedule.append({"power": power, "shots": shots, "hits": hits})
        with ctx.stage("decoding"):
            ctx.charge(evaluations=1)
            amplitude, ci = maximum_likelihood(schedule)
        return {"estimate": amplitude * p["scale"], "estimate_ci95": [v * p["scale"] for v in ci],
                "interval_method": "asymptotic 95% profile-likelihood envelope; not guaranteed finite-sample coverage",
                "schedule": schedule, "termination": "MLAE schedule completed"}
