"""
Density-matrix utilities.

This module provides helpers for constructing and validating qudit density matrices,
computing purity, and extracting computational-basis measurement probabilities.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from .. import backend

__all__ = [
    "get_zero_density_matrix",
    "is_valid_density_matrix",
    "purity",
    "density_matrix_to_probvector",
]


def get_zero_density_matrix(n: int, d: int) -> np.ndarray:
    """
    Construct the density matrix for the pure state |0...0⟩⟨0...0|.

    Parameters
    ----------
    n : int
        Number of qudits.
    d : int
        Dimension.

    Returns
    -------
    np.ndarray
        Density matrix of shape (d**n, d**n), dtype backend.COMPLEX_DTYPE.
    """
    size = d**n
    rho = np.zeros((size, size), dtype=backend.COMPLEX_DTYPE)
    rho[0, 0] = 1.0
    return rho


def is_valid_density_matrix(
    rho: np.ndarray,
    tol: Optional[float] = None,
    psd_tol: Optional[float] = None,
) -> tuple[bool, str]:
    """
    Validate a density matrix.

    Checks:
    1. square 2D array
    2. Hermitian within `tol`
    3. trace ≈ 1 within `tol`
    4. PSD: min eigenvalue >= -psd_tol

    If `tol`/`psd_tol` are None, they are chosen based on dtype:
    - float64/complex128: tol=1e-6,  psd_tol=1e-12
    - float32/complex64:  tol=1e-5,  psd_tol=1e-6

    Parameters
    ----------
    rho : np.ndarray
        Candidate density matrix.
    tol : Optional[float], optional
        Tolerance for Hermiticity and trace checks. If None, chosen from dtype.
    psd_tol : Optional[float], optional
        Tolerance for PSD eigenvalue check. If None, chosen from dtype.

    Returns
    -------
    (bool, str)
        (is_valid, message). If valid, message is "ok". Otherwise message explains
        the first failing check.
    """
    rho = np.asarray(rho)

    if rho.ndim != 2 or rho.shape[0] != rho.shape[1]:
        return False, f"rho must be a square 2D array, got shape {rho.shape}"

    real_dtype = np.real(rho).dtype
    is_32 = real_dtype == np.float32

    if tol is None:
        tol = 1e-5 if is_32 else 1e-6
    if psd_tol is None:
        psd_tol = 1e-6 if is_32 else 1e-12

    if not np.allclose(rho, rho.conj().T, atol=tol):
        return False, "rho is not Hermitian within tolerance"

    tr = np.trace(rho)
    if not (np.isclose(tr.real, 1.0, atol=tol) and np.isclose(tr.imag, 0.0, atol=tol)):
        return False, f"trace(rho) must be 1 (real); got {tr!r}"

    evals = np.linalg.eigvalsh(rho)
    min_eval = float(np.min(evals))
    if min_eval < -psd_tol:
        return False, f"rho is not PSD: min eigenvalue = {min_eval} < {-psd_tol}"

    return True, "ok"


def purity(rho: np.ndarray) -> float:
    """
    Compute the purity Tr(rho^2).

    Parameters
    ----------
    rho : np.ndarray
        Density matrix.

    Returns
    -------
    float
        Purity Tr(rho^2).

    Raises
    ------
    ValueError
        If `rho` fails `is_valid_density_matrix(rho)`.
    """
    ok, msg = is_valid_density_matrix(rho)
    if not ok:
        raise ValueError(f"Input is not a valid density matrix: {msg}")

    return float(np.real(np.trace(rho @ rho)))


def density_matrix_to_probvector(rho: np.ndarray) -> np.ndarray:
    """
    Extract the computational-basis measurement probability vector from a density matrix.

    The probability of measuring basis state |k⟩ is rho[k, k].

    Parameters
    ----------
    rho : np.ndarray
        Density matrix.

    Returns
    -------
    np.ndarray
        1D array of length d**n containing probabilities (real-valued).

    Raises
    ------
    ValueError
        If `rho` fails `is_valid_density_matrix(rho)`.
    """
    ok, msg = is_valid_density_matrix(rho)
    if not ok:
        raise ValueError(f"Input is not a valid density matrix: {msg}")

    return np.real(np.diag(rho))
