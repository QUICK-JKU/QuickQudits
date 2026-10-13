import numpy as np
import pytest

from quickqudits import QuantumCircuit
from quickqudits.gates import apply_CNOT_reshape_idx


def _reference_cnot(n, d, control, target, dagger, dtype):
    """Build the full operator directly from computational-basis digit values."""
    size = d**n
    control_stride = d ** (n - 1 - control)
    target_stride = d ** (n - 1 - target)
    matrix = np.zeros((size, size), dtype=dtype)
    sign = -1 if dagger else 1
    for column in range(size):
        c = (column // control_stride) % d
        t = (column // target_stride) % d
        output_t = (t + sign * c) % d
        row = column + (output_t - t) * target_stride
        matrix[row, column] = 1
    return matrix


@pytest.mark.parametrize("d", [2, 3, 4])
@pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
@pytest.mark.parametrize("dagger", [False, True])
@pytest.mark.parametrize(
    "control,target", [(0, 1), (1, 0), (0, 3), (3, 0), (1, 3), (3, 1)]
)
def test_cnot_matches_independent_full_matrix(d, dtype, dagger, control, target):
    n = 4
    rng = np.random.default_rng(531 + 10 * d + control * n + target)
    psi = (rng.normal(size=d**n) + 1j * rng.normal(size=d**n)).astype(dtype)
    original = psi.copy()

    actual = apply_CNOT_reshape_idx(psi, n, d, (control, target), dagger)
    expected = _reference_cnot(n, d, control, target, dagger, dtype) @ original

    assert actual.dtype == dtype
    assert not np.shares_memory(actual, psi)
    np.testing.assert_array_equal(psi, original)
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("d", [2, 3, 4])
@pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
@pytest.mark.parametrize("control,target", [(0, 2), (2, 0)])
def test_cnot_then_dagger_restores_state_through_circuit(d, dtype, control, target):
    rng = np.random.default_rng(716 + d + control)
    psi = (rng.normal(size=d**3) + 1j * rng.normal(size=d**3)).astype(dtype)
    psi /= np.linalg.norm(psi)
    original = psi.copy()
    circuit = QuantumCircuit(3, d)
    circuit.CNOT(control, target)
    circuit.CNOTdag(control, target)

    actual, _ = circuit.execute(psi0=psi)

    np.testing.assert_allclose(actual, original, atol=2e-6 if dtype == np.complex64 else 1e-14)
    np.testing.assert_array_equal(psi, original)


def test_cnot_accepts_noncontiguous_readonly_input_without_aliasing():
    rng = np.random.default_rng(812)
    storage = (rng.normal(size=2 * 3**3) + 1j * rng.normal(size=2 * 3**3))
    psi = storage[::2]
    psi.flags.writeable = False
    original = psi.copy()

    actual = apply_CNOT_reshape_idx(psi, 3, 3, (2, 0))
    expected = _reference_cnot(3, 3, 2, 0, False, psi.dtype) @ original

    assert not np.shares_memory(actual, storage)
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(psi, original)
