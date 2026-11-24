"""Classical cycle search on a dense weight matrix W (inf = no edge).

All routines return simple directed cycles as tuples of node indices in
canonical rotation (smallest index first), without repeating the start.
"""

from __future__ import annotations

import math

import numpy as np

EPS = 1e-12
Cycle = tuple[int, ...]


def canonical(cycle: Cycle) -> Cycle:
    k = min(range(len(cycle)), key=lambda i: cycle[i])
    return tuple(cycle[k:]) + tuple(cycle[:k])


def cycle_weight(W: np.ndarray, cycle: Cycle) -> float:
    return float(sum(W[cycle[i], cycle[(i + 1) % len(cycle)]] for i in range(len(cycle))))


def is_valid_cycle(W: np.ndarray, cycle: Cycle, min_len: int = 2, max_len: int | None = None) -> bool:
    n = len(cycle)
    if n < min_len or (max_len is not None and n > max_len) or len(set(cycle)) != n:
        return False
    return math.isfinite(cycle_weight(W, cycle))


def enumerate_cycles(
    W: np.ndarray, min_len: int = 2, max_len: int = 4, max_evals: int | None = None
) -> tuple[list[tuple[Cycle, float]], int, bool]:
    """Exhaustive bounded-depth enumeration (the brute-force reference).

    Returns (cycles with weights, number of cycles scored, truncated flag).
    Each cycle is generated once, rooted at its smallest node.
    """
    n = W.shape[0]
    finite = np.isfinite(W)
    out: list[tuple[Cycle, float]] = []
    evals = 0
    truncated = False

    def dfs(path: list[int], weight: float) -> bool:
        nonlocal evals, truncated
        start, last = path[0], path[-1]
        if len(path) >= min_len and finite[last, start]:
            if max_evals is not None and evals >= max_evals:
                truncated = True
                return False
            evals += 1
            out.append((tuple(path), weight + W[last, start]))
        if len(path) < max_len:
            for v in range(start + 1, n):
                if finite[last, v] and v not in path:
                    path.append(v)
                    ok = dfs(path, weight + W[last, v])
                    path.pop()
                    if not ok:
                        return False
        return True

    for s in range(n):
        if not dfs([s], 0.0):
            break
    return out, evals, truncated


def bellman_ford_negative_cycle(W: np.ndarray) -> tuple[Cycle | None, int]:
    """One negative cycle via Bellman-Ford from a virtual source, or None.

    Returns (cycle, number of edge relaxations performed).
    """
    n = W.shape[0]
    edges = [(u, v, float(W[u, v])) for u in range(n) for v in range(n) if u != v and math.isfinite(W[u, v])]
    dist = [0.0] * n
    pred = [-1] * n
    relaxations = 0
    last = -1
    for _ in range(n):
        last = -1
        for u, v, w in edges:
            relaxations += 1
            if dist[u] + w < dist[v] - EPS:
                dist[v] = dist[u] + w
                pred[v] = u
                last = v
        if last < 0:
            return None, relaxations
    # `last` was relaxed on the n-th pass: walk back n steps to land inside a cycle.
    x = last
    for _ in range(n):
        x = pred[x]
    cyc = [x]
    y = pred[x]
    while y != x:
        cyc.append(y)
        y = pred[y]
    cyc.reverse()
    cycle = canonical(tuple(cyc))
    if cycle_weight(W, cycle) >= -EPS:  # guard against float-noise artefacts
        return None, relaxations
    return cycle, relaxations


def bellman_ford_cycles(
    W: np.ndarray, min_len: int = 2, max_len: int = 4, max_relaxations: int | None = None
) -> tuple[list[tuple[Cycle, float]], dict]:
    """Repeated Bellman-Ford: find a negative cycle, delete its costliest edge, repeat.

    Bellman-Ford is a detector, not an enumerator. Deleting the *least*
    attractive edge of each found cycle keeps strongly negative edges available
    to other cycles, but cycles sharing the deleted edge can still be missed,
    and cycles longer than `max_len` are discarded (path-length restriction).
    """
    W = W.copy()
    found: dict[Cycle, float] = {}
    stats = {"relaxations": 0, "bf_runs": 0, "rejected_length": 0, "truncated": False}
    while True:
        if max_relaxations is not None and stats["relaxations"] >= max_relaxations:
            stats["truncated"] = True
            break
        cycle, r = bellman_ford_negative_cycle(W)
        stats["relaxations"] += r
        stats["bf_runs"] += 1
        if cycle is None:
            break
        if min_len <= len(cycle) <= max_len:
            found[cycle] = cycle_weight(W, cycle)
        else:
            stats["rejected_length"] += 1
        k = max(range(len(cycle)), key=lambda i: W[cycle[i], cycle[(i + 1) % len(cycle)]])
        W[cycle[k], cycle[(k + 1) % len(cycle)]] = np.inf
    return sorted(found.items()), stats


def _decompose_closed_walk(walk: list[int]) -> list[Cycle]:
    """Split a closed walk (start repeated at the end) into simple cycles."""
    cycles: list[Cycle] = []
    stack: list[int] = []
    pos: dict[int, int] = {}
    for v in walk:
        if v in pos:
            i = pos[v]
            cyc = stack[i:]
            cycles.append(canonical(tuple(cyc)))
            for u in stack[i + 1:]:
                del pos[u]
            del stack[i + 1:]
        else:
            pos[v] = len(stack)
            stack.append(v)
    return cycles


def bounded_dp_cycles(W: np.ndarray, min_len: int = 2, max_len: int = 4) -> tuple[list[tuple[Cycle, float]], dict]:
    """Length-bounded shortest-path heuristic (min-plus dynamic programme).

    D_k[s, v] = cheapest walk of exactly k edges from s to v. For every start s
    and length k the cheapest closed walk D_k[s, s] is reconstructed and split
    into simple cycles. Polynomial, O(max_len * N^3), but it keeps only the
    best closed walk per (start, length), so it is not exhaustive.
    """
    n = W.shape[0]
    Wd = W.copy()
    np.fill_diagonal(Wd, np.inf)
    D = Wd.copy()
    preds: list[np.ndarray] = [np.tile(np.arange(n)[:, None], (1, n))]  # P_1[s, v] = s
    found: dict[Cycle, float] = {}
    stats = {"relaxations": 0, "closed_walks": 0}
    for k in range(1, max_len + 1):
        if k > 1:
            cand = D[:, :, None] + Wd[None, :, :]  # [s, u, v]
            stats["relaxations"] += n * n * n
            P = np.argmin(cand, axis=1)
            D = np.take_along_axis(cand, P[:, None, :], axis=1)[:, 0, :]
            preds.append(P)
        if k < min_len:
            continue
        for s in range(n):
            if not (D[s, s] < -EPS):
                continue
            stats["closed_walks"] += 1
            walk = [s]
            v = s
            for j in range(k, 0, -1):
                v = int(preds[j - 1][s, v])
                walk.append(v)
            walk.reverse()
            for cyc in _decompose_closed_walk(walk):
                if min_len <= len(cyc) <= max_len:
                    w = cycle_weight(Wd, cyc)
                    if w < -EPS:
                        found[cyc] = w
    return sorted(found.items()), stats


def random_cycles(
    W: np.ndarray, rng: np.random.Generator, n_samples: int, min_len: int = 2, max_len: int = 4
) -> tuple[list[tuple[Cycle, float]], int]:
    """Uniform random simple cycles: the equal-budget sanity baseline."""
    n = W.shape[0]
    hi = min(max_len, n)
    found: dict[Cycle, float] = {}
    if hi < min_len:
        return [], 0
    for _ in range(n_samples):
        L = int(rng.integers(min_len, hi + 1))
        cyc = canonical(tuple(int(x) for x in rng.permutation(n)[:L]))
        if cyc not in found:
            found[cyc] = cycle_weight(W, cyc)
    return sorted((c, w) for c, w in found.items() if math.isfinite(w)), n_samples
