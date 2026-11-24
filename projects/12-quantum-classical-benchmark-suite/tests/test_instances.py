import copy
import itertools
import json

import numpy as np
import pytest

from benchmarks.instances.dataset import load_manifest, prepare
from benchmarks.instances.generators import generate
from benchmarks.instances.schema import (digest, encode_qubo, exact_reference, feasible,
                                         objective, qubo_to_ising, repair, validate_instance, validate_problem)
from experiments.config import validate_config


@pytest.mark.parametrize("track,size", [("portfolio", 4), ("basket", 6), ("estimation", 8)])
def test_deterministic_verified_instances(track, size):
    first, second = generate(track, size, 19), generate(track, size, 19)
    assert digest(first) == digest(second)
    validate_instance(first)
    if track != "estimation":
        _, optimum = exact_reference(first["problem"])
        assert optimum == first["reference"]["value"]


def test_portfolio_objective_matches_finance_model():
    p = generate("portfolio", 6, 9)["problem"]
    x = np.array([1, 0, 1, 0, 1, 0])
    model = p["model"]
    expected = model["risk_aversion"] * (x @ np.asarray(model["covariance"]) @ x) - np.asarray(model["expected_returns"]) @ x
    assert objective(p, x) == pytest.approx(expected)


@pytest.mark.parametrize("track", ["portfolio", "basket"])
def test_qubo_and_ising_match_all_bitstrings_and_preserve_optimum(track):
    instance = generate(track, 6, 19)
    p = instance["problem"]
    c, q, offset, penalty = encode_qubo(p)
    h, j, constant = qubo_to_ising(c, q, offset)
    energies = []
    for bits in itertools.product([0, 1], repeat=p["size"]):
        x = np.array(bits)
        energy = c @ x + x @ q @ x + offset
        expected = objective(p, bits) + penalty * ((x.sum() - p["cardinality"]) ** 2 + sum(x[i] * x[k] for i, k in p["conflicts"]))
        z = 1 - 2 * x
        assert energy == pytest.approx(expected)
        assert h @ z + z @ j @ z + constant == pytest.approx(expected)
        energies.append((energy, bits))
    optimum, bits = min(energies)
    assert feasible(p, bits)
    assert optimum == pytest.approx(instance["reference"]["value"])


def test_constraint_rejection_and_repair():
    p = generate("portfolio", 4, 0)["problem"]
    for invalid in (None, [], [1, 0], [0.5, 0.5, 1, 0], [1, 1, 1, 1], [float("nan"), 0, 1, 0]):
        assert not feasible(p, invalid)
    assert feasible(p, repair(p, [1, 1, 1, 1]))
    basket = generate("basket", 6, 0)["problem"]
    i, j = basket["conflicts"][0]
    x = [int(k in (i, j)) for k in range(6)]
    assert not feasible(basket, x)
    assert {tuple(pair) for pair in basket["conflicts"]} == {
        (i, j) for i in range(6) for j in range(i + 1, 6)
        if set(basket["model"]["candidate_cycles"][i]) & set(basket["model"]["candidate_cycles"][j])}


def test_input_shape_finiteness_and_reference_validation():
    instance = generate("portfolio", 4, 2)
    instance["problem"]["quadratic"][0][0] = 1
    with pytest.raises(ValueError, match="triangular"):
        validate_instance(instance)
    instance = generate("portfolio", 4, 2)
    instance["reference"]["value"] += 1
    with pytest.raises(ValueError, match="reference value"):
        validate_instance(instance)
    p = generate("estimation", 4, 2)["problem"]
    p["probabilities"][0] = -1
    with pytest.raises(ValueError, match="probabilities"):
        validate_problem(p)


def test_immutable_data_and_tamper_detection(tmp_path, config):
    manifest_path = prepare(config, tmp_path)
    original = manifest_path.read_bytes()
    assert prepare(config, tmp_path).read_bytes() == original
    manifest = load_manifest(manifest_path, config)
    path = manifest_path.parent / manifest["instances"][0]["file"]
    instance = json.loads(path.read_text())
    instance["provenance"]["seed"] += 1
    path.write_text(json.dumps(instance))
    with pytest.raises(ValueError, match="checksum"):
        load_manifest(manifest_path, config)
    with pytest.raises(ValueError, match="immutable"):
        prepare(config, tmp_path)


@pytest.mark.parametrize("change", [
    {"seeds": [1, 1]}, {"seeds": []}, {"warmups": -1}, {"budget": {"wall_seconds": 0}},
    {"budget": {"wall_seconds": float("nan")}}, {"budget": {"max_shots": -1}},
    {"instances": [{"track": "estimation", "sizes": [3]}]}, {"algorithm": {"qae_powers": [1, 2]}},
    {"algorithm": {"qae_powers": [0, 2, 1]}}, {"algorithm": {"repair": "yes"}},
    {"solvers": ["unknown"]}, {"typo": 1}, {"success_tolerance": -1},
])
def test_config_rejects_invalid_values(config, change):
    with pytest.raises(ValueError):
        validate_config(config | change)
