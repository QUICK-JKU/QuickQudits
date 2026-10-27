import numpy as np
import pytest

from quickqudits import backend, gates
from quickqudits.predefined_gates import H_gate


def _fourier_reference(psi, n, d, target, dagger):
    """Apply the Fourier phase to basis columns without using tensor contraction."""
    stride = d ** (n - 1 - target)
    result = np.zeros(psi.size, dtype=np.complex128)
    sign = -1 if dagger else 1
    for column, amplitude in enumerate(psi):
        digit = (column // stride) % d
        for row_digit in range(d):
            row = column + (row_digit - digit) * stride
            result[row] += amplitude * np.exp(sign * 2j * np.pi * row_digit * digit / d) / np.sqrt(d)
    return result


@pytest.mark.parametrize("d", [2, 3, 4])
@pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
@pytest.mark.parametrize("target", [0, 1, 2])
@pytest.mark.parametrize("dagger", [False, True])
def test_h_matches_independent_phase_reference(d, dtype, target, dagger):
    backend.set_precision("single" if dtype == np.complex64 else "double")
    try:
        rng = np.random.default_rng(1820 + d + target)
        psi = (rng.normal(size=d**3) + 1j * rng.normal(size=d**3)).astype(dtype)
        original = psi.copy()
        actual = H_gate.apply(psi, 3, d, (target,), dagger)
        expected = _fourier_reference(original, 3, d, target, dagger)
        tol = 2e-6 if dtype == np.complex64 else 2e-14
        np.testing.assert_allclose(actual, expected, rtol=tol, atol=tol)
        np.testing.assert_array_equal(psi, original)
        assert actual.dtype == dtype
        assert not np.shares_memory(actual, psi)
        restored = H_gate.apply(actual, 3, d, (target,), not dagger)
        np.testing.assert_allclose(restored, original, rtol=tol, atol=tol)
    finally:
        backend.set_precision("double")


def test_qutrit_h_squared_is_not_identity():
    backend.set_precision("double")
    psi = np.array([0, 1, 0], dtype=np.complex128)
    twice = H_gate.apply(H_gate.apply(psi, 1, 3, (0,)), 1, 3, (0,))
    np.testing.assert_allclose(twice, [0, 0, 1], atol=1e-14)
