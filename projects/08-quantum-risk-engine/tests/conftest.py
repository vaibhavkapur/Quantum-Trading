from pathlib import Path

import pytest

from project.config import load_config


@pytest.fixture
def config():
    config, _ = load_config(Path(__file__).parents[1] / "configs/demo.yaml")
    config.data.observations = 80
    config.data.training_observations = 40
    config.simulation.samples = 500
    config.quantum.qubits = 2
    config.quantum.shots = 32
    config.quantum.powers = [0, 1]
    config.benchmark.seeds = [11, 23]
    config.benchmark.powers = [1]
    config.benchmark.shots = 16
    return config
