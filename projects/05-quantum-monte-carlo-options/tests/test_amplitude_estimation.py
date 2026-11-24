import numpy as np
import pytest

from qmc_options.quantum import amplitude_estimation as ae


def test_clopper_pearson_edges():
    assert ae.clopper_pearson(0, 50, 0.05)[0] == 0.0
    assert ae.clopper_pearson(50, 50, 0.05)[1] == 1.0
    lo, hi = ae.clopper_pearson(20, 100, 0.05)
    assert lo < 0.2 < hi and hi - lo < 0.2


def test_query_accounting():
    oracle = ae.AnalyticOracle(0.3)
    rng = np.random.default_rng(0)
    res = ae.direct_sampling(oracle, 500, rng)
    assert (res.shots, res.oracle_queries, res.grover_iterations, res.max_grover_power) == (500, 500, 0, 0)
    res = ae.mlae(oracle, rng, max_power_exponent=3, shots=10)          # k = 0, 1, 2, 4
    assert res.shots == 40
    assert res.oracle_queries == 10 * (1 + 3 + 5 + 9)
    assert res.grover_iterations == 10 * 7 and res.max_grover_power == 4
    assert oracle.usage()["oracle_queries"] == 500 + 180           # the oracle keeps the running total
    with pytest.raises(ValueError):
        oracle.sample(-1, 10, rng)


@pytest.mark.parametrize("amplitude", [0.02, 0.13, 0.5, 0.81])
def test_mlae_recovers_the_amplitude(amplitude):
    estimates = [ae.mlae(ae.AnalyticOracle(amplitude), np.random.default_rng(s), 6, 100).estimate
                 for s in range(20)]
    assert np.sqrt(np.mean((np.array(estimates) - amplitude) ** 2)) < 0.004


@pytest.mark.parametrize("amplitude", [0.02, 0.13, 0.5, 0.81])
@pytest.mark.parametrize("epsilon", [0.02, 0.002])
def test_iqae_interval_width_and_coverage(amplitude, epsilon):
    """The half-width guarantee is deterministic; containment is a 95% guarantee."""
    covered = 0
    for seed in range(200):
        res = ae.iqae(ae.AnalyticOracle(amplitude), np.random.default_rng(seed), epsilon=epsilon, alpha=0.05)
        assert (res.ci_high - res.ci_low) / 2 <= epsilon * 1.0001
        covered += res.ci_low <= amplitude <= res.ci_high
    assert covered >= 190


@pytest.mark.parametrize("amplitude", [0.0, 1.0])
def test_extreme_amplitudes(amplitude):
    rng = np.random.default_rng(1)
    assert ae.iqae(ae.AnalyticOracle(amplitude), rng, epsilon=0.01).estimate == pytest.approx(amplitude, abs=0.01)
    assert ae.mlae(ae.AnalyticOracle(amplitude), rng, 4, 50).estimate == pytest.approx(amplitude, abs=0.01)


def test_quadratic_query_advantage_in_the_ideal_setting():
    """Tightening epsilon tenfold should cost IQAE far less than the 100x of sampling."""
    def queries(eps):
        return np.mean([ae.iqae(ae.AnalyticOracle(0.2), np.random.default_rng(s), epsilon=eps).oracle_queries
                        for s in range(10)])
    assert queries(0.001) / queries(0.01) < 30


def test_noise_destroys_the_estimate():
    clean = ae.mlae(ae.AnalyticOracle(0.2), np.random.default_rng(0), 6, 200)
    noisy = ae.mlae(ae.AnalyticOracle(0.2, noise_rate=0.2), np.random.default_rng(0), 6, 200)
    assert abs(clean.estimate - 0.2) < 0.005 < abs(noisy.estimate - 0.2)
    assert ae.AnalyticOracle(0.2, noise_rate=5.0).probability(50) == pytest.approx(0.5, abs=1e-6)


def test_invalid_arguments():
    with pytest.raises(ValueError):
        ae.AnalyticOracle(1.5)
    with pytest.raises(ValueError):
        ae.iqae(ae.AnalyticOracle(0.2), np.random.default_rng(0), epsilon=0.0)
