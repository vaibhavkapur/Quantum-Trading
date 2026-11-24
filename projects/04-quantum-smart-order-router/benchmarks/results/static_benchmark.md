# Static solver benchmark

20 random instances per size, seeds `[venues, units, instance]`. Optimum from exhaustive enumeration. Gap is the cost excess as a percentage of the feasible cost range (worst - best). QAOA: 2 layers, 40 COBYLA iterations, 256 shots, noiseless statevector simulation on CPU. `random_repair` draws 256 uniform bitstrings and uses the same decode + repair.

| Venues x units | Qubits | Feasible allocs | Solver | Optimal found | Mean gap % | Wall ms | Evals | Failed |
|---|---|---|---|---|---|---|---|---|
| 2 x 2 | 4.0 | 3 | exact | 20/20 | 0.00 | 0.028 | 0 | 0 |
| 2 x 2 | 4.0 | 3 | best_price | 16/20 | 15.30 | 0.018 | 0 | 0 |
| 2 x 2 | 4.0 | 3 | pro_rata | 5/20 | 62.51 | 0.016 | 0 | 0 |
| 2 x 2 | 4.0 | 3 | greedy | 14/20 | 14.99 | 0.020 | 0 | 0 |
| 2 x 2 | 4.0 | 3 | cpsat | 20/20 | 0.00 | 10.945 | 0 | 0 |
| 2 x 2 | 4.0 | 3 | sa | 20/20 | 0.00 | 0.634 | 480 | 0 |
| 2 x 2 | 4.0 | 3 | random_repair | 20/20 | 0.00 | 0.102 | 0 | 0 |
| 2 x 2 | 4.0 | 3 | qaoa | 20/20 | 0.00 | 14.931 | 41 | 0 |
| 2 x 4 | 8.0 | 5 | exact | 20/20 | 0.00 | 0.032 | 0 | 0 |
| 2 x 4 | 8.0 | 5 | best_price | 14/20 | 17.26 | 0.026 | 0 | 0 |
| 2 x 4 | 8.0 | 5 | pro_rata | 1/20 | 57.15 | 0.016 | 0 | 0 |
| 2 x 4 | 8.0 | 5 | greedy | 18/20 | 0.85 | 0.019 | 0 | 0 |
| 2 x 4 | 8.0 | 5 | cpsat | 20/20 | 0.00 | 1.244 | 0 | 0 |
| 2 x 4 | 8.0 | 5 | sa | 20/20 | 0.00 | 1.230 | 960 | 0 |
| 2 x 4 | 8.0 | 5 | random_repair | 20/20 | 0.00 | 1.010 | 0 | 0 |
| 2 x 4 | 8.0 | 5 | qaoa | 20/20 | 0.00 | 18.572 | 41 | 0 |
| 3 x 3 | 9.0 | 10 | exact | 20/20 | 0.00 | 0.033 | 0 | 0 |
| 3 x 3 | 9.0 | 10 | best_price | 11/20 | 13.58 | 0.020 | 0 | 0 |
| 3 x 3 | 9.0 | 10 | pro_rata | 0/20 | 68.94 | 0.015 | 0 | 0 |
| 3 x 3 | 9.0 | 10 | greedy | 17/20 | 2.94 | 0.020 | 0 | 0 |
| 3 x 3 | 9.0 | 10 | cpsat | 20/20 | 0.00 | 1.879 | 0 | 0 |
| 3 x 3 | 9.0 | 10 | sa | 20/20 | 0.00 | 1.059 | 1080 | 0 |
| 3 x 3 | 9.0 | 10 | random_repair | 20/20 | 0.00 | 1.944 | 0 | 0 |
| 3 x 3 | 9.0 | 10 | qaoa | 20/20 | 0.00 | 17.201 | 41 | 0 |
| 3 x 4 | 12.0 | 15 | exact | 20/20 | 0.00 | 0.033 | 0 | 0 |
| 3 x 4 | 12.0 | 15 | best_price | 14/20 | 21.94 | 0.018 | 0 | 0 |
| 3 x 4 | 12.0 | 15 | pro_rata | 0/20 | 55.88 | 0.015 | 0 | 0 |
| 3 x 4 | 12.0 | 15 | greedy | 18/20 | 2.39 | 0.022 | 0 | 0 |
| 3 x 4 | 12.0 | 15 | cpsat | 20/20 | 0.00 | 2.247 | 0 | 0 |
| 3 x 4 | 12.0 | 15 | sa | 19/20 | 0.58 | 1.274 | 1440 | 0 |
| 3 x 4 | 12.0 | 15 | random_repair | 20/20 | 0.00 | 3.448 | 0 | 0 |
| 3 x 4 | 12.0 | 15 | qaoa | 20/20 | 0.00 | 31.361 | 41 | 0 |
| 4 x 3 | 12.0 | 20 | exact | 20/20 | 0.00 | 0.041 | 0 | 0 |
| 4 x 3 | 12.0 | 20 | best_price | 4/20 | 28.92 | 0.019 | 0 | 0 |
| 4 x 3 | 12.0 | 20 | pro_rata | 0/20 | 66.58 | 0.014 | 0 | 0 |
| 4 x 3 | 12.0 | 20 | greedy | 14/20 | 3.79 | 0.022 | 0 | 0 |
| 4 x 3 | 12.0 | 20 | cpsat | 20/20 | 0.00 | 2.352 | 0 | 0 |
| 4 x 3 | 12.0 | 20 | sa | 18/20 | 1.01 | 1.349 | 1440 | 0 |
| 4 x 3 | 12.0 | 20 | random_repair | 20/20 | 0.00 | 4.956 | 0 | 0 |
| 4 x 3 | 12.0 | 20 | qaoa | 20/20 | 0.00 | 31.426 | 41 | 0 |
| 4 x 4 | 16.0 | 35 | exact | 20/20 | 0.00 | 0.059 | 0 | 0 |
| 4 x 4 | 16.0 | 35 | best_price | 11/20 | 14.13 | 0.024 | 0 | 0 |
| 4 x 4 | 16.0 | 35 | pro_rata | 0/20 | 53.39 | 0.017 | 0 | 0 |
| 4 x 4 | 16.0 | 35 | greedy | 10/20 | 8.41 | 0.030 | 0 | 0 |
| 4 x 4 | 16.0 | 35 | cpsat | 20/20 | 0.00 | 3.513 | 0 | 0 |
| 4 x 4 | 16.0 | 35 | sa | 15/20 | 2.57 | 1.687 | 1920 | 0 |
| 4 x 4 | 16.0 | 35 | random_repair | 20/20 | 0.00 | 7.880 | 0 | 0 |
| 4 x 4 | 16.0 | 35 | qaoa | 20/20 | 0.00 | 275.447 | 41 | 0 |

## QAOA sampling quality before repair

Probability that one measured bitstring is feasible / optimal, against a uniformly random bitstring.

| Venues x units | Qubits | P(feasible) QAOA | P(feasible) uniform | P(optimal) QAOA | P(optimal) uniform | Optimize ms | Sample ms | Decode+repair ms |
|---|---|---|---|---|---|---|---|---|
| 2 x 2 | 4.0 | 0.4270 | 0.1875 | 0.1540 | 0.06250 | 14.6 | 0.07 | 0.07 |
| 2 x 4 | 8.0 | 0.1278 | 0.0195 | 0.0285 | 0.00391 | 17.8 | 0.17 | 0.26 |
| 3 x 3 | 9.0 | 0.2024 | 0.0195 | 0.0230 | 0.00195 | 16.5 | 0.13 | 0.35 |
| 3 x 4 | 12.0 | 0.1247 | 0.0037 | 0.0101 | 0.00024 | 29.2 | 0.48 | 0.62 |
| 4 x 3 | 12.0 | 0.2009 | 0.0049 | 0.0120 | 0.00024 | 29.3 | 0.47 | 0.67 |
| 4 x 4 | 16.0 | 0.1077 | 0.0005 | 0.0019 | 0.00002 | 244.9 | 6.20 | 1.07 |

## Cost of coarse allocation units

Excess cost of the best allocation at the QUBO's granularity over the best one-lot allocation.

| Venues x units | Mean granularity gap % |
|---|---|
| 2 x 2 | 0.00 |
| 2 x 4 | 0.21 |
| 3 x 3 | 0.27 |
| 3 x 4 | 0.01 |
| 4 x 3 | 0.25 |
| 4 x 4 | 0.26 |
