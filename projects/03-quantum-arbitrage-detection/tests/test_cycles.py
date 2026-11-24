import math

import networkx as nx
import numpy as np
import pytest

from qarb.graph.cycles import (
    bellman_ford_cycles,
    bellman_ford_negative_cycle,
    bounded_dp_cycles,
    canonical,
    cycle_weight,
    enumerate_cycles,
    random_cycles,
)

from conftest import random_weights


def nx_cycles(W, max_len):
    g = nx.DiGraph()
    n = W.shape[0]
    g.add_nodes_from(range(n))
    for u in range(n):
        for v in range(n):
            if u != v and math.isfinite(W[u, v]):
                g.add_edge(u, v, weight=W[u, v])
    return {canonical(tuple(c)) for c in nx.simple_cycles(g, length_bound=max_len) if len(c) >= 2}, g


@pytest.mark.parametrize("seed", range(6))
@pytest.mark.parametrize("density", [1.0, 0.6])
def test_enumeration_matches_networkx(seed, density):
    W = random_weights(np.random.default_rng(seed), 6, density)
    ours, evals, truncated = enumerate_cycles(W, 2, 4)
    ref, _ = nx_cycles(W, 4)
    assert {c for c, _ in ours} == ref and evals == len(ref) and not truncated
    for c, w in ours:
        assert w == pytest.approx(cycle_weight(W, c))


def test_enumeration_budget_truncates():
    W = random_weights(np.random.default_rng(0), 6)
    ours, evals, truncated = enumerate_cycles(W, 2, 4, max_evals=10)
    assert truncated and evals == 10 and len(ours) == 10


@pytest.mark.parametrize("seed", range(10))
def test_bellman_ford_agrees_with_networkx_on_existence(seed):
    rng = np.random.default_rng(seed)
    W = random_weights(rng, 6, density=0.7, neg_edges=int(rng.integers(0, 3)))
    cycle, relax = bellman_ford_negative_cycle(W)
    _, g = nx_cycles(W, 6)
    assert (cycle is not None) == nx.negative_edge_cycle(g)
    if cycle is not None:
        assert cycle_weight(W, cycle) < 0 and len(set(cycle)) == len(cycle)
    assert relax > 0


def test_no_negative_cycle_in_positive_graph():
    W = np.full((5, 5), 0.001)
    np.fill_diagonal(W, np.inf)
    assert bellman_ford_negative_cycle(W)[0] is None
    assert bellman_ford_cycles(W)[0] == []
    assert bounded_dp_cycles(W)[0] == []


def test_heuristics_return_only_true_negative_cycles_and_find_the_best():
    for seed in range(8):
        W = random_weights(np.random.default_rng(seed), 6, neg_edges=3)
        truth = {c: w for c, w in enumerate_cycles(W, 2, 4)[0] if w < -1e-12}
        for solver in (bellman_ford_cycles, bounded_dp_cycles):
            found, stats = solver(W, 2, 4)
            for c, w in found:
                assert c in truth and w == pytest.approx(truth[c])
            assert stats["relaxations"] > 0
        if truth:
            best = min(truth, key=truth.get)
            assert best in dict(bounded_dp_cycles(W, 2, 4)[0])  # the DP always recovers the global best
            assert bellman_ford_cycles(W, 2, 6)[0]  # BF finds at least one when one exists


def test_bellman_ford_respects_length_bound():
    # only negative cycle has length 5
    n = 5
    W = np.full((n, n), 1.0)
    np.fill_diagonal(W, np.inf)
    for i in range(n):
        W[i, (i + 1) % n] = -0.1
    found, stats = bellman_ford_cycles(W, 2, 4)
    assert found == [] and stats["rejected_length"] == 1
    assert [c for c, _ in bellman_ford_cycles(W, 2, 5)[0]] == [(0, 1, 2, 3, 4)]


def test_random_cycles_valid_and_deterministic():
    W = random_weights(np.random.default_rng(1), 6, density=0.7)
    a, n = random_cycles(W, np.random.default_rng(9), 200, 2, 4)
    b, _ = random_cycles(W, np.random.default_rng(9), 200, 2, 4)
    assert a == b and n == 200
    ref, _ = nx_cycles(W, 4)
    assert {c for c, _ in a} <= ref
