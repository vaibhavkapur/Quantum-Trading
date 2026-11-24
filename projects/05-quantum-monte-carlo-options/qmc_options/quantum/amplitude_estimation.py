"""Amplitude estimation without phase estimation: MLAE and IQAE.

Estimators only talk to an oracle through ``sample(k, shots)``, which returns
how many of ``shots`` measurements of Q^k A|0> found the objective qubit in
|1>. The oracle keeps the books: shots, Grover iterations, and oracle queries
(applications of A or its inverse, 2k+1 per shot).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator, Statevector
from scipy.optimize import minimize_scalar
from scipy.stats import beta, norm

from .encoding import build_grover_operator


class AmplitudeOracle:
    """Base class: subclasses implement ``probability(k)``."""

    def __init__(self) -> None:
        self.shots = 0
        self.queries = 0
        self.grover_iterations = 0
        self.max_power = 0

    def probability(self, k: int) -> float:
        raise NotImplementedError

    def sample(self, k: int, shots: int, rng: np.random.Generator) -> int:
        if k < 0 or shots < 1:
            raise ValueError("need k >= 0 and shots >= 1")
        self.shots += shots
        self.queries += shots * (2 * k + 1)
        self.grover_iterations += shots * k
        self.max_power = max(self.max_power, k)
        return int(rng.binomial(shots, min(max(self.probability(k), 0.0), 1.0)))

    def usage(self) -> Dict[str, int]:
        return {"shots": self.shots, "oracle_queries": self.queries,
                "grover_iterations": self.grover_iterations, "max_grover_power": self.max_power}


class AnalyticOracle(AmplitudeOracle):
    """P(1) = sin^2((2k+1) theta) for a known amplitude. ``noise_rate`` is a
    toy depolarising model: each Grover iteration keeps the signal with
    probability exp(-noise_rate), otherwise the outcome is a fair coin."""

    def __init__(self, amplitude: float, noise_rate: float = 0.0) -> None:
        super().__init__()
        if not 0.0 <= amplitude <= 1.0:
            raise ValueError("amplitude must lie in [0, 1]")
        self.theta = float(np.arcsin(np.sqrt(amplitude)))
        self.noise_rate = noise_rate

    def probability(self, k: int) -> float:
        ideal = float(np.sin((2 * k + 1) * self.theta) ** 2)
        return 0.5 + np.exp(-self.noise_rate * k) * (ideal - 0.5)


class CircuitOracle(AmplitudeOracle):
    """Noiseless simulation of the actual circuits.

    ``mode="unitary"`` turns the A and Q circuits into dense unitaries and
    applies Q gate-for-gate k times. ``mode="reflection"`` simulates only the
    A circuit and applies Q through the identity A S_0 A^dagger = I - 2|psi><psi|
    with |psi> = A|0>; it is the same operator (the tests check this against
    the Q circuit) at O(2**n) cost per iteration instead of a dense matrix.
    ``mode="auto"`` picks "unitary" up to ``UNITARY_MAX_QUBITS`` qubits.
    """

    UNITARY_MAX_QUBITS = 7

    def __init__(self, a_operator: QuantumCircuit, grover_operator: Optional[QuantumCircuit] = None,
                 mode: str = "auto") -> None:
        super().__init__()
        if mode not in ("auto", "unitary", "reflection"):
            raise ValueError(f"unknown mode {mode!r}")
        if mode == "auto":
            mode = "unitary" if a_operator.num_qubits <= self.UNITARY_MAX_QUBITS else "reflection"
        self.mode = mode
        self._initial = np.asarray(Statevector(a_operator).data).copy()
        self._good = (np.arange(self._initial.size) >> (a_operator.num_qubits - 1)) & 1 == 1
        self._q = None
        if mode == "unitary":
            self._q = Operator(grover_operator or build_grover_operator(a_operator)).data
        self._state_k = 0
        self._state = self._initial
        self._cache: Dict[int, float] = {}

    def _apply_grover(self, state: np.ndarray) -> np.ndarray:
        if self._q is not None:
            return self._q @ state
        flipped = np.where(self._good, -state, state)                       # S_chi
        return 2 * np.vdot(self._initial, flipped) * self._initial - flipped  # -(I - 2|psi><psi|)

    def probability(self, k: int) -> float:
        if k not in self._cache:
            if k < self._state_k:
                self._state_k, self._state = 0, self._initial
            while self._state_k < k:
                self._state = self._apply_grover(self._state)
                self._state_k += 1
            self._cache[k] = float(np.sum(np.abs(self._state[self._good]) ** 2))
        return self._cache[k]


@dataclass
class AEResult:
    method: str
    estimate: float
    ci_low: float
    ci_high: float
    shots: int
    oracle_queries: int
    grover_iterations: int
    max_grover_power: int
    schedule: List[Tuple[int, int, int]] = field(default_factory=list)  # (k, shots, ones)

    def to_dict(self) -> dict:
        return asdict(self)


def _result(method: str, estimate: float, lo: float, hi: float, oracle: AmplitudeOracle,
            before: Dict[str, int], schedule: List[Tuple[int, int, int]]) -> AEResult:
    used = {key: value - before[key] for key, value in oracle.usage().items() if key != "max_grover_power"}
    return AEResult(method, float(estimate), float(max(lo, 0.0)), float(min(hi, 1.0)),
                    max_grover_power=max((k for k, _, _ in schedule), default=0),
                    schedule=schedule, **used)


def clopper_pearson(ones: int, shots: int, alpha: float) -> Tuple[float, float]:
    lo = 0.0 if ones == 0 else float(beta.ppf(alpha / 2, ones, shots - ones + 1))
    hi = 1.0 if ones == shots else float(beta.ppf(1 - alpha / 2, ones + 1, shots - ones))
    return lo, hi


def direct_sampling(oracle: AmplitudeOracle, shots: int, rng: np.random.Generator,
                    alpha: float = 0.05) -> AEResult:
    """Measure A|0> only (k = 0). This is classical Monte Carlo run on the
    circuit and has the same 1/sqrt(shots) convergence."""
    before = oracle.usage()
    ones = oracle.sample(0, shots, rng)
    lo, hi = clopper_pearson(ones, shots, alpha)
    return _result("direct_sampling", ones / shots, lo, hi, oracle, before, [(0, shots, ones)])


def mlae(oracle: AmplitudeOracle, rng: np.random.Generator, max_power_exponent: int = 5,
         shots: int = 100, alpha: float = 0.05,
         powers: Optional[Sequence[int]] = None) -> AEResult:
    """Maximum-likelihood amplitude estimation (Suzuki et al. 2020) with the
    exponential schedule k = 0, 1, 2, 4, ..., 2**(max_power_exponent - 1).
    The interval is the Fisher-information (asymptotic normal) interval."""
    if powers is None:
        powers = [0] + [2**j for j in range(max_power_exponent)]
    before = oracle.usage()
    schedule = [(int(k), shots, oracle.sample(int(k), shots, rng)) for k in powers]
    ks = np.array([s[0] for s in schedule])
    ns = np.array([s[1] for s in schedule])
    hs = np.array([s[2] for s in schedule])
    mult = 2 * ks + 1

    def neg_log_likelihood(theta):
        p = np.clip(np.sin(np.multiply.outer(theta, mult)) ** 2, 1e-12, 1 - 1e-12)
        return -(hs * np.log(p) + (ns - hs) * np.log1p(-p)).sum(axis=-1)

    # The likelihood oscillates on the scale 1/(2 k_max + 1): grid search, then polish.
    grid = np.linspace(0.0, np.pi / 2, max(2000, 200 * int(mult.max())) + 1)
    best = int(np.argmin(neg_log_likelihood(grid)))
    lo_t, hi_t = grid[max(best - 1, 0)], grid[min(best + 1, grid.size - 1)]
    theta = minimize_scalar(neg_log_likelihood, bounds=(lo_t, hi_t), method="bounded",
                            options={"xatol": 1e-12}).x
    a = float(np.sin(theta) ** 2)
    a_safe = min(max(a, 1e-9), 1 - 1e-9)
    fisher = float((ns * mult**2).sum() / (a_safe * (1 - a_safe)))
    half = norm.ppf(1 - alpha / 2) / np.sqrt(fisher)
    return _result("mlae", a, a - half, a + half, oracle, before, schedule)


def _find_next_k(k: int, upper_half: bool, theta_l: float, theta_u: float, min_ratio: float = 2.0):
    """Largest k' whose scaled interval (4k'+2)[theta_l, theta_u] stays inside
    one half-circle, so that the measured probability can be inverted."""
    old_scaling = 4 * k + 2
    max_scaling = int(1 / (2 * (theta_u - theta_l)))
    scaling = max_scaling - (max_scaling - 2) % 4
    while scaling >= min_ratio * old_scaling:
        t_min = scaling * theta_l - int(scaling * theta_l)
        t_max = scaling * theta_u - int(scaling * theta_u)
        if t_min <= t_max <= 0.5:
            return (scaling - 2) // 4, True
        if t_max >= t_min >= 0.5:
            return (scaling - 2) // 4, False
        scaling -= 4
    return k, upper_half


def iqae(oracle: AmplitudeOracle, rng: np.random.Generator, epsilon: float = 0.01,
         alpha: float = 0.05, shots: int = 100, max_rounds_cap: int = 10_000) -> AEResult:
    """Iterative QAE (Grinko et al. 2021) with Clopper-Pearson intervals.
    Returns an interval of half-width <= epsilon that contains the amplitude
    with probability >= 1 - alpha."""
    if not 0 < epsilon <= 0.5:
        raise ValueError("epsilon must lie in (0, 0.5]")
    before = oracle.usage()
    max_rounds = int(np.log(2.0 * np.pi / 8 / epsilon) / np.log(2.0)) + 1
    theta_l, theta_u = 0.0, 0.25      # in units of 2*pi
    k, upper_half = 0, True
    ones_at_k = shots_at_k = 0
    schedule: List[Tuple[int, int, int]] = []
    rounds = 0
    while theta_u - theta_l > epsilon / np.pi:
        rounds += 1
        if rounds > max_rounds_cap:
            raise RuntimeError("IQAE did not converge")
        next_k, upper_half = _find_next_k(k, upper_half, theta_l, theta_u)
        if next_k != k or not schedule:
            ones_at_k = shots_at_k = 0      # a new power starts a fresh sample
        k = next_k
        ones = oracle.sample(k, shots, rng)
        schedule.append((k, shots, ones))
        ones_at_k += ones
        shots_at_k += shots
        a_min, a_max = clopper_pearson(ones_at_k, shots_at_k, alpha / max_rounds)
        if upper_half:
            t_min = np.arccos(1 - 2 * a_min) / (2 * np.pi)
            t_max = np.arccos(1 - 2 * a_max) / (2 * np.pi)
        else:
            t_min = 1 - np.arccos(1 - 2 * a_max) / (2 * np.pi)
            t_max = 1 - np.arccos(1 - 2 * a_min) / (2 * np.pi)
        scaling = 4 * k + 2
        theta_l, theta_u = ((int(scaling * theta_l) + t_min) / scaling,
                            (int(scaling * theta_u) + t_max) / scaling)
    a_l, a_u = np.sin(2 * np.pi * theta_l) ** 2, np.sin(2 * np.pi * theta_u) ** 2
    return _result("iqae", 0.5 * (a_l + a_u), a_l, a_u, oracle, before, schedule)
