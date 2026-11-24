import pytest
import yaml

from qmc_options.models import Market, Option


@pytest.fixture
def market():
    return Market(spot=100.0, rate=0.05, vol=0.2)


@pytest.fixture
def call():
    return Option("european_call", strike=100.0, maturity=1.0)


@pytest.fixture
def tiny_config(tmp_path):
    """A reduced-size copy of the demo experiment, written inside tmp_path."""
    config = {
        "name": "tiny",
        "market": {"spot": 100.0, "rate": 0.05, "vol": 0.2},
        "confidence": 0.95,
        "seeds": [1, 2],
        "instruments": [
            {"name": "call", "kind": "european_call", "strike": 100.0, "maturity": 1.0},
            {"name": "asian", "kind": "asian_call", "strike": 100.0, "maturity": 1.0, "steps": 2},
            {"name": "barrier", "kind": "barrier_up_out_call", "strike": 100.0, "maturity": 1.0,
             "steps": 2, "barrier": 130.0},
        ],
        "data": {"path": str(tmp_path / "data" / "paths.npz"), "n_paths": 2000, "seed": 5},
        "reference": {"n_paths": 16384, "seed": 9},
        "classical": {
            "n_paths": 4000, "methods": ["plain", "antithetic", "control_variate", "qmc"],
            "greeks": {"n_paths": 4000, "rel_bump": 0.01},
            "error_curve": {"instrument": "call", "n_paths": [256, 1024, 4096], "repeats": 6},
        },
        "quantum": {
            "qubits_european": 3, "truncation_european": 3.0,
            "qubits_per_step": 2, "truncation_path_dependent": 3.0,
            "iqae": {"epsilon": 0.02, "shots": 50},
            "mlae": {"max_power_exponent": 3, "shots": 50},
            "direct_sampling": {"shots": 500},
            "scaling": {"instrument": "call", "repeats": 5, "direct_sampling_shots": [100, 400],
                        "mlae_exponents": [1, 3], "mlae_shots": 32, "iqae_epsilons": [0.05, 0.02, 0.01]},
            "discretization_qubits": [2, 3], "discretization_truncations": [3.0],
            "noise": {"rates": [0.0, 0.05], "max_power_exponent": 3, "shots": 50, "repeats": 5},
            "delta": {"instrument": "call", "rel_bumps": [0.05], "epsilon": 0.02},
        },
    }
    path = tmp_path / "tiny.yaml"
    path.write_text(yaml.safe_dump(config))
    return path
