import numpy as np
import pytest

from project.config import DEFAULTS, deep_merge
from project.research.qubo import BasketProblem


@pytest.fixture
def tiny_problem() -> BasketProblem:
    """6 assets, K=3, known optimum {0, 1, 2} (objective -2.9, verified by hand below)."""
    C = np.zeros((6, 6))
    for i, j, v in [(0, 1, -1.0), (0, 2, -1.0), (1, 2, -1.0), (3, 4, -0.8), (4, 5, -0.8), (0, 3, 0.5), (2, 5, 0.3)]:
        C[i, j] = C[j, i] = v
    return BasketProblem(
        symbols=list("ABCDEF"),
        linear=np.array([0.0, 0.1, 0.0, 0.0, 0.0, 0.0]),
        quad=C,
        sectors=np.array([0, 1, 2, 0, 1, 2]),
        basket_size=3,
        max_per_sector=1,
    )


def random_problem(n: int, seed: int, k: int = 3, max_per_sector: int = 2, n_sectors: int = 3) -> BasketProblem:
    rng = np.random.default_rng(seed)
    C = rng.normal(0, 1, (n, n))
    C = (C + C.T) / 2
    np.fill_diagonal(C, 0.0)
    return BasketProblem(
        symbols=[f"S{i}" for i in range(n)],
        linear=rng.normal(0, 0.2, n),
        quad=C,
        sectors=np.arange(n) % n_sectors,
        basket_size=k,
        max_per_sector=max_per_sector,
    )


@pytest.fixture
def cfg(tmp_path) -> dict:
    c = deep_merge(
        DEFAULTS,
        {
            "name": "test",
            "data": {"dir": "data/test", "n_days": 900, "l2": {"n_events": 20000, "seed": 7}},
            "walkforward": {"train_days": 300, "val_days": 100, "test_days": 100, "step_days": 200},
            "budget": {"max_evals": 16, "time_limit_s": 2.0},
            "solvers": {"classical": ["brute_force", "greedy", "cpsat", "random_search", "simulated_annealing"],
                        "quantum": {"backend": "numpy", "p": 1, "shots": 64, "maxiter": 8, "cvar_alpha": 0.5}},
            "seeds": [1, 2],
        },
    )
    c["_root"] = str(tmp_path)
    c["_path"] = str(tmp_path / "configs" / "test.yaml")
    return c
