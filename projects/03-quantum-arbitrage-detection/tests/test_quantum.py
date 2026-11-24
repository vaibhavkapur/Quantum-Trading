import itertools
import math

import numpy as np
import pytest

from qarb.graph.cycles import canonical, cycle_weight, enumerate_cycles
from qarb.quantum.anneal import simulated_annealing
from qarb.quantum.qaoa import cvar, qaoa_probabilities, run_qaoa
from qarb.quantum.qubo import (
    all_bitstrings,
    argmin_is_valid_optimum,
    auto_penalty,
    best_valid_cycle,
    build_cycle_qubo,
    energy_vector,
    index_to_bits,
    to_ising,
)

from conftest import random_weights


def best_cycle(W, L, anchor=None):
    cands = [(w, c) for c, w in enumerate_cycles(W, L, L)[0] if anchor is None or anchor in c]
    return min(cands) if cands else (math.inf, None)


@pytest.mark.parametrize("anchor", [None, 0, 2])
@pytest.mark.parametrize("L", [2, 3, 4])
def test_valid_cycle_energy_equals_normalized_weight(L, anchor):
    W = random_weights(np.random.default_rng(L), 5)
    q = build_cycle_qubo(W, L, anchor)
    assert q.n_vars == (5 * L if anchor is None else 4 * (L - 1))
    for c, w in enumerate_cycles(W, L, L)[0]:
        if anchor is not None and anchor not in c:
            continue
        x = q.encode(c)
        assert q.energy(x) == pytest.approx(w / q.scale, abs=1e-9)
        assert q.decode(x, W) == c


@pytest.mark.parametrize("seed", range(5))
@pytest.mark.parametrize("L,anchor,n", [(2, None, 4), (3, None, 4), (3, 0, 5), (4, 1, 5), (4, None, 4), (2, 3, 6)])
def test_qubo_minimum_is_best_cycle_bruteforce(seed, L, anchor, n):
    W = random_weights(np.random.default_rng(100 * seed + L), n, neg_edges=seed % 3)
    q = build_cycle_qubo(W, L, anchor)
    E = energy_vector(q)
    w_best, c_best = best_cycle(W, L, anchor)
    k = int(np.argmin(E))
    assert q.decode(index_to_bits(k, q.n_vars), W) is not None
    assert E[k] * q.scale == pytest.approx(w_best, abs=1e-9)
    assert argmin_is_valid_optimum(q, W, E)
    found, e = best_valid_cycle(q, W, E)
    assert cycle_weight(W, found) == pytest.approx(w_best) and e == pytest.approx(E[k])


def test_every_infeasible_assignment_is_above_every_feasible_one():
    W = random_weights(np.random.default_rng(3), 4, neg_edges=3)
    for L, anchor in ((3, None), (3, 0), (4, 0)):
        q = build_cycle_qubo(W, L, anchor)
        E = energy_vector(q)
        bits = all_bitstrings(q.n_vars)
        feasible = np.array([q.decode(b, W) is not None for b in bits])
        assert feasible.any() and E[~feasible].min() > E[feasible].max()
        assert q.penalty == auto_penalty(L)


def test_missing_edges_never_decode_and_sparse_minimum_is_valid():
    W = random_weights(np.random.default_rng(8), 5, density=0.55, neg_edges=3)
    q = build_cycle_qubo(W, 3, 0)
    E = energy_vector(q)
    for k in range(2**q.n_vars):
        c = q.decode(index_to_bits(k, q.n_vars), W)
        assert c is None or math.isfinite(cycle_weight(W, c))
    w_best, _ = best_cycle(W, 3, 0)
    if math.isfinite(w_best) and w_best < 0:
        assert argmin_is_valid_optimum(q, W, E)


def test_energy_vector_matches_direct_evaluation_and_ising():
    W = random_weights(np.random.default_rng(4), 4)
    q = build_cycle_qubo(W, 3, 1)
    E = energy_vector(q, chunk=7)
    bits = all_bitstrings(q.n_vars)
    h, J, const = to_ising(q.Q, q.offset)
    for k in (0, 1, 5, 17, 2**q.n_vars - 1):
        assert E[k] == pytest.approx(q.energy(bits[k]))
        z = 1 - 2 * bits[k].astype(float)
        assert const + h @ z + z @ J @ z == pytest.approx(E[k])


def test_repair_returns_valid_cycles():
    W = random_weights(np.random.default_rng(5), 5)
    q = build_cycle_qubo(W, 4, 0)
    rng = np.random.default_rng(0)
    for _ in range(50):
        x = rng.integers(0, 2, q.n_vars)
        c = q.repair(x, W)
        assert c is not None and len(set(c)) == 4 and 0 in c
    best = min((w, c) for c, w in enumerate_cycles(W, 4, 4)[0] if 0 in c)[1]
    assert q.repair(q.encode(best), W) == best  # a feasible sample is left unchanged


def test_invalid_length_rejected():
    W = random_weights(np.random.default_rng(0), 3)
    with pytest.raises(ValueError):
        build_cycle_qubo(W, 4)
    with pytest.raises(ValueError):
        build_cycle_qubo(W, 1)


def test_qaoa_probabilities_normalized_and_zero_angles_uniform():
    E = np.random.default_rng(0).normal(size=2**5)
    p = qaoa_probabilities(E, np.zeros(2), np.zeros(2))
    assert p == pytest.approx(np.full(32, 1 / 32))
    p = qaoa_probabilities(E, np.array([0.3, 0.5]), np.array([0.4, 0.2]))
    assert p.sum() == pytest.approx(1.0) and p.std() > 0


def test_cvar():
    e = np.array([3.0, 1.0, 2.0, 4.0])
    assert cvar(e, 1.0) == pytest.approx(2.5)
    assert cvar(e, 0.5) == pytest.approx(1.5)
    assert cvar(e, 0.5, np.array([0.1, 0.4, 0.4, 0.1])) == pytest.approx((0.4 * 1 + 0.1 * 2) / 0.5)


def test_numpy_engine_matches_qiskit_circuit():
    from qarb.quantum.qiskit_backend import circuit_metrics, probabilities_fn

    W = random_weights(np.random.default_rng(6), 4)
    q = build_cycle_qubo(W, 3, 0)  # 6 qubits
    E = energy_vector(q)
    E_phase = (E - E.mean()) / E.std()
    gammas, betas = np.array([0.37, 0.81]), np.array([0.55, 0.23])
    ours = qaoa_probabilities(E_phase, gammas, betas)
    theirs = probabilities_fn(q, 2)(gammas, betas)
    assert ours == pytest.approx(theirs, abs=1e-9)
    m = circuit_metrics(q, 2)
    assert m["n_qubits"] == 6 and m["depth"] > 0 and m["two_qubit_gates"] > 0


def test_qaoa_concentrates_on_the_optimum_fixture():
    """Known-optimum fixture: one planted negative triangle among positive edges."""
    W = np.full((4, 4), 0.002)
    np.fill_diagonal(W, np.inf)
    W[0, 1] = W[1, 2] = W[2, 0] = -0.003
    q = build_cycle_qubo(W, 3, 0)
    E = energy_vector(q)
    res = run_qaoa(E, np.random.default_rng(2), layers=2, maxiter=150, expectation="exact", final_shots=1024)
    uniform = float(np.isclose(E, E.min()).mean())
    assert res.p_ground > 5 * uniform
    top = max(res.counts, key=res.counts.get)
    assert q.decode(index_to_bits(top, q.n_vars), W) == (0, 1, 2)
    assert res.circuit_executions == 1024  # exact expectation consumes no optimizer shots


def test_qaoa_shot_accounting_and_determinism():
    W = random_weights(np.random.default_rng(7), 4)
    E = energy_vector(build_cycle_qubo(W, 3, 0))
    a = run_qaoa(E, np.random.default_rng(1), 1, 12, 32, 64)
    b = run_qaoa(E, np.random.default_rng(1), 1, 12, 32, 64)
    assert a.counts == b.counts and np.allclose(a.params, b.params)
    assert a.circuit_executions == a.optimizer_iters * 32 + 64 and sum(a.counts.values()) == 64


def test_simulated_annealing_finds_qubo_minimum():
    W = random_weights(np.random.default_rng(9), 5, neg_edges=3)
    q = build_cycle_qubo(W, 3, None)  # 15 variables
    states, energies = simulated_annealing(q.Q, np.random.default_rng(0), reads=40, sweeps=120)
    E = energy_vector(q)
    for x, e in zip(states[:5], energies[:5]):
        assert e + q.offset == pytest.approx(q.energy(x))
    assert energies.min() + q.offset == pytest.approx(E.min(), abs=1e-9)
