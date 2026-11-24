"""M1: the QUBO is verified against brute force on tiny instances."""
import itertools

import numpy as np
import pytest

from hqpo.models import build_qubo, bounded_coefficients, qubit_count
from hqpo.models.qubo import encode_int, penalty_strength
from hqpo.solvers.classical import brute_force_optimum
from hqpo.solvers.quantum import qaoa_state

from conftest import make


@pytest.mark.parametrize("ub", range(0, 40))
def test_bounded_coefficients_cover_exact_range(ub):
    coefs = bounded_coefficients(ub)
    sums = {sum(c for c, b in zip(coefs, bits) if b)
            for bits in itertools.product([0, 1], repeat=len(coefs))}
    assert sums == set(range(ub + 1))            # every value reachable, nothing above ub
    assert len(coefs) == (ub.bit_length() if ub else 0)
    for v in range(ub + 1):
        assert sum(c * b for c, b in zip(coefs, encode_int(v, ub))) == v


def test_encode_int_rejects_out_of_range():
    with pytest.raises(ValueError):
        encode_int(6, 5)


def test_qubo_energy_equals_objective_on_feasible_points(tiny):
    qubo = build_qubo(tiny)
    z, f, _ = brute_force_optimum(tiny)
    x = qubo.encode(z)
    assert np.array_equal(qubo.decode(x), z)
    assert qubo.energy(x) == pytest.approx(f, abs=1e-10)       # zero penalty when feasible


def test_qubo_ground_state_is_the_constrained_optimum(tiny):
    qubo = build_qubo(tiny)                                     # default penalty calibration
    E = qubo.energies_all()
    z_qubo = qubo.decode(qubo.index_to_bits(int(np.argmin(E))))
    z, f, _ = brute_force_optimum(tiny)
    assert tiny.is_feasible(z_qubo)
    assert tiny.objective(z_qubo) == pytest.approx(f, abs=1e-10)
    assert E.min() == pytest.approx(f, abs=1e-10)


def test_infeasible_bitstrings_never_beat_the_optimum(tiny):
    """Penalty >= 0 everywhere, and strictly positive exactly on infeasible decodes
    (up to indicator/slack bits being set inconsistently)."""
    qubo = build_qubo(tiny)
    E = qubo.energies_all()
    idx = np.arange(len(E))
    Z = qubo.index_to_bits(idx) @ qubo.T.T
    pen = E - tiny.objective_batch(Z)
    assert pen.min() > -1e-9
    infeasible = tiny.total_violation_batch(Z) > 0
    assert np.all(pen[infeasible] >= qubo.penalty - 1e-9)


def test_energies_all_matches_direct_evaluation(tiny, rng):
    qubo = build_qubo(tiny)
    E = qubo.energies_all()
    for idx in rng.integers(0, len(E), 50):
        assert E[idx] == pytest.approx(qubo.energy(qubo.index_to_bits(int(idx))), abs=1e-10)


def test_ising_mapping_matches_qubo(tiny, rng):
    qubo = build_qubo(tiny)
    h, J, const = qubo.to_ising()
    for _ in range(30):
        x = rng.integers(0, 2, qubo.n_qubits)
        s = 1 - 2 * x
        assert h @ s + s @ J @ s + const == pytest.approx(qubo.energy(x), abs=1e-9)


def test_ising_mapping_agrees_with_qiskit_optimization():
    qo = pytest.importorskip("qiskit_optimization")
    p = make(4, lots=4, max_weight=0.5, cardinality=3)
    qubo = build_qubo(p)
    qp = qo.QuadraticProgram()
    for i in range(qubo.n_qubits):
        qp.binary_var(f"x{i}")
    n = qubo.n_qubits
    qp.minimize(constant=qubo.offset, linear=np.diag(qubo.Q),
                quadratic={(i, j): qubo.Q[i, j] for i in range(n) for j in range(i + 1, n)})
    op, offset = qp.to_ising()
    h, J, const = qubo.to_ising()
    assert offset == pytest.approx(const, abs=1e-9)
    diag = np.real(op.to_matrix().diagonal()) + offset       # little-endian, like energies_all
    assert np.allclose(diag, qubo.energies_all(), atol=1e-8)


def test_too_small_penalty_is_detected_as_invalid():
    p = make(5, lots=4, max_weight=0.5, cardinality=3, seed=5)
    qubo = build_qubo(p, scale=1e-4)
    z = qubo.decode(qubo.index_to_bits(int(np.argmin(qubo.energies_all()))))
    assert not p.is_feasible(z)                  # documents why calibration matters


def test_penalty_methods():
    p = make(5, lots=4, max_weight=0.5)
    assert penalty_strength(p, "fixed", value=2.0, scale=0.5) == 1.0
    assert penalty_strength(p, "range") >= penalty_strength(p, "gradient_bound") > 0
    with pytest.raises(ValueError):
        penalty_strength(p, "nope")
    with pytest.raises(ValueError):
        penalty_strength(p, "fixed")


def test_qubit_count_matches_built_model_and_presolve():
    p = make(6, lots=4, max_weight=0.5, cardinality=3)
    assert qubit_count(p) == build_qubo(p).qubit_breakdown() == {
        "position": 12, "indicator": 6, "slack": 0, "total": 18}
    # cardinality that cannot bind costs no qubits; neither does a sector cap that cannot bind
    assert qubit_count(make(6, lots=4, max_weight=0.5, cardinality=4))["indicator"] == 0
    assert qubit_count(make(6, lots=4, max_weight=0.5, sector_cap=1.0))["slack"] == 0
    assert qubit_count(make(6, lots=4, max_weight=0.5, sector_cap=0.5))["slack"] > 0


def test_qaoa_kernel_matches_qiskit_statevector():
    pytest.importorskip("qiskit")
    from qiskit.quantum_info import Statevector

    from hqpo.solvers.quantum import build_qaoa_circuit

    p = make(3, lots=2, max_weight=1.0)
    qubo = build_qubo(p)
    h, J, const = qubo.to_ising()
    gammas, betas = [0.31, 0.57], [0.44, 0.13]
    mine = np.abs(qaoa_state(qubo.energies_all(), gammas, betas)) ** 2
    theirs = Statevector(build_qaoa_circuit(h, J, gammas, betas)).probabilities()
    assert np.allclose(mine, theirs, atol=1e-10)
    assert mine.sum() == pytest.approx(1.0)
