# Hot-path decision latency

200000 decisions per row, wall clock (`perf_counter_ns`), single CPython process. Microseconds.

| Scenario | p50 | p99 | p99.9 | max | Policies published |
|---|---|---|---|---|---|
| snapshot router, optimizer idle | 0.46 | 0.62 | 0.92 | 27 | 1 |
| snapshot router, `exact` optimizer thread solving continuously | 0.50 | 0.67 | 1.46 | 7630 | 1384 |
| snapshot router, `qaoa` optimizer thread solving continuously | 0.42 | 0.62 | 1.38 | 627 | 3 |
| live greedy solve on the hot path (no snapshot) | 10.75 | 20.79 | 37.54 | 288 | – |

Caps are randomized per decision; the snapshot router never calls a solver. With the optimizer on a thread of the same interpreter, the tail reflects GIL hand-offs, which is why a production layout would put the optimizer in a separate process (or the router in native code) and share snapshots through shared memory.
