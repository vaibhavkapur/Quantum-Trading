from pathlib import Path

import pytest

from strategy_optimizer.config import load_config


@pytest.fixture
def config(tmp_path):
    result = load_config(Path(__file__).parents[1] / "configs/ci.yaml")
    result["data"]["path"] = str(tmp_path / "prices.csv")
    result["data"]["periods"] = 112
    result["output"]["runs"] = str(tmp_path / "runs")
    result["optimization"]["seeds"] = [7]
    result["report"]["bootstrap_samples"] = 40
    return result

