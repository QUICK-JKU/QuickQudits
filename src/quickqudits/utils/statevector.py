"""
Statevector utilities.

This module contains helper functions for working with qudit statevectors:
computational-basis measurement, reshaping/endianness conversion, projection
onto measurement outcomes (with partial trace), and simple comparison metrics.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from .. import backend

__all__ = [
    "measureall",
    "flip_endianness_statevector",
    "nonzero_dary_indices",
    "get_zero_vector",
    "entangle_input",
    "collapse_universal_to_basis",
    "get_projected_state",
    "get_projected_state_multi",
    "compare_statevectors",
    "norm_diff",
    "apply_unitary",
]


def measureall(
    psi: np.ndarray, n: int, d: int, seed: int = None
) -> tuple[np.ndarray, int]:
    """
    Perform a global computational-basis measurement on |psi⟩ ∈ C^{d^n}.

    Parameters
    ----------
    psi : np.ndarray
        Flat statevector of length d**n.
    n : int
        Number of qudits.
    d : int
        Dimension.
    seed : int
        Seed for reproducibility.

    Returns
    -------
    collapsed : np.ndarray
        Collapsed statevector (a basis vector with amplitude 1).
    k : int
        Linear index in [0, d**n - 1] of the measured basis state.

    Raises
    ------
    ValueError
        If `psi` is not a flat vector of length d**n, or if normalization fails.
    """
    rng = np.random.default_rng(seed)
    size = d**n
    if psi.shape != (size,):
        raise ValueError(f"psi must be a flat vector of length {size}")

    psi = psi.astype(backend.COMPLEX_DTYPE)
    probs = (np.abs(psi) ** 2).astype(backend.FLOAT_DTYPE)
    probs /= np.sum(probs)

    k = int(rng.choice(np.arange(size, dtype=np.int64), p=probs))

    collapsed = np.zeros_like(psi, dtype=backend.COMPLEX_DTYPE)
    collapsed[k] = 1

    return collapsed, k


def flip_endianness_statevector(psi: np.ndarray, n: int, d: int) -> np.ndarray:
    """
    Reverse the qudit axis order of a length d**n statevector.

    Interprets `psi` as shape (d,)*n with axes (q0, …, q{n-1}), reverses to
    (q{n-1}, …, q0), then flattens back to 1D. Applying this twice returns the
    original vector.

    Parameters
    ----------
    psi : np.ndarray
        1D statevector of length d**n.
    n : int
        Number of qudits.
    d : int
        Dimension.

    Returns
    -------
    np.ndarray
        1D array of length d**n (same dtype as `psi`).

    Raises
    ------
    ValueError
        If `psi.size != d**n`.
    """
    psi = np.asarray(psi)
    if psi.size != d**n:
        raise ValueError(f"Expected length {d**n}, got {psi.size}")
    return psi.reshape([d] * n).transpose(tuple(reversed(range(n)))).reshape(-1)


def nonzero_dary_indices(
    vector: np.ndarray,
    d: int,
    tol: float = 1e-12,
) -> list[tuple[int, ...]]:
    """
    Return the non-zero positions of a statevector/probability vector in base-d (d-ary) form.

    Parameters
    ----------
    vector : np.ndarray
        Either a statevector (complex amplitudes) or a probability vector (real, >=0),
        of length d**n for some integer n.
    d : int
        Dimension.
    tol : float, optional
        Threshold below which entries are treated as zero.
        For statevectors this applies to |amp|**2; for probability vectors to p directly.

    Returns
    -------
    list[tuple[int, ...]]
        A list of n-length tuples with digits in base-d (most-significant digit first),
        corresponding to the indices with non-zero probability.

    Raises
    ------
    ValueError
        If `vector` is not 1D, empty, or its length is not a power of `d`,
        or if `d < 2`.
    """
    arr = np.asarray(vector)
    if arr.ndim != 1:
        raise ValueError(f"`vector` must be 1D, got shape {arr.shape}.")

    if d < 2:
        raise ValueError(f"`d` must be >= 2, got {d}.")

    L = arr.size

    if L == 0:
        raise ValueError("`vector` must be non-empty.")

    n_float = np.log(L) / np.log(d)
    n = int(round(n_float))
    if d**n != L:
        raise ValueError(
            f"Length {L} is not a power of d={d}. Expected d**n for some integer n."
        )

    if np.iscomplexobj(arr):
        mask = (arr.real**2 + arr.imag**2) > tol
    else:
        mask = arr > tol

    idxs = np.nonzero(mask)[0]
    if idxs.size == 0:
        return []

    powers = d ** np.arange(n - 1, -1, -1, dtype=np.int64)
    results = []
    for idx in idxs:
        rem = int(idx)
        digits = [0] * n
        for k in range(n):
            q, rem = divmod(rem, int(powers[k]))
            digits[k] = q
        results.append(tuple(digits))

    return results


def get_zero_vector(n: int, d: int) -> np.ndarray:
    """
    Construct the computational |0...0⟩ statevector.

    Parameters
    ----------
    n : int
        Number of qudits.
    d : int
        Dimension.

    Returns
    -------
    np.ndarray
        Statevector of shape (d**n,), dtype backend.COMPLEX_DTYPE.
    """
    size = d**n
    psi = np.zeros(size, dtype=backend.COMPLEX_DTYPE)
    psi[0] = 1
    return psi


def entangle_input(n: int, d: int) -> np.ndarray:
    """
    Construct a maximally entangled input state on 2n qudits.

    The state is:
        (1/sqrt(d^n)) * sum_{i=0}^{d^n-1} |i⟩|i⟩

    Parameters
    ----------
    n : int
        Number of qudits per register (total qudits = 2n).
    d : int
        Dimension.

    Returns
    -------
    np.ndarray
        Statevector of shape (d**(2n),), dtype backend.COMPLEX_DTYPE.
    """
    total_qudits = 2 * n
    size = d**total_qudits
    num_basis_states = d**n

    val = backend.COMPLEX_DTYPE(1) / np.sqrt(
        np.array(num_basis_states, dtype=backend.FLOAT_DTYPE)
    )
    psi = np.zeros(size, dtype=backend.COMPLEX_DTYPE)

    for i in range(num_basis_states):
        index = i * num_basis_states + i
        psi[index] = val

    return psi


def collapse_universal_to_basis(
    vec_2n: np.ndarray,
    n: int,
    d: int,
    basis_index: int = 0,
    normalize: bool = True,
) -> np.ndarray:
    """
    Extract U|basis_index⟩ from (U ⊗ I)|Φ⟩ for a universal (2n-qudit) mat-vec run.

    Parameters
    ----------
    vec_2n : np.ndarray
        Flat vector of length d**(2n), representing (U ⊗ I)|Φ⟩.
    n : int
        Number of qudits per register.
    d : int
        Dimension.
    basis_index : int, optional
        Which |i⟩ on the right register to select (default 0 corresponds to |0...0⟩).
    normalize : bool, optional
        If True, renormalize the extracted vector.

    Returns
    -------
    np.ndarray
        Length d**n vector proportional to U|basis_index⟩ (up to global phase).
    """
    D = d**n
    M = vec_2n.reshape(D, D)  # rows = left (output), cols = right
    psi = np.sqrt(D) * M[:, basis_index]  # column i = U|i⟩ / sqrt(D)
    if normalize:
        nrm = np.linalg.norm(psi)
        if nrm != 0:
            psi = psi / nrm
    return psi.astype(backend.COMPLEX_DTYPE, copy=False)


def get_projected_state(
    psi: np.ndarray,
    n: int,
    d: int,
    q_idx: int,
    outcome: int,
) -> Optional[np.ndarray]:
    """
    Project a statevector onto a specific measurement outcome for a single qudit
    and trace out that qudit.

    This function performs the equivalent of applying the projector P = |k><k|_q
    (where k is the outcome) and then discarding the measured qudit q from the
    system. Reduces the system size from n to n-1 qudits.

    Parameters
    ----------
    psi : np.ndarray
        The full statevector of the system (length d**n).
    n : int
        Total number of qudits in the system.
    d : int
        Dimension of each qudit.
    q_idx : int
        The index of the qudit being measured (0 <= q_idx < n).
    outcome : int
        The measurement result obtained (0 <= outcome < d).

    Returns
    -------
    np.ndarray or None
        The normalized statevector of the remaining n-1 qudits.
        Returns None if the norm of the projected state is near zero
        (i.e., the outcome has probability 0).

    Raises
    ------
    ValueError
        If `psi` has the wrong length, or if `q_idx`/`outcome` are out of range.
    """
    psi = np.asarray(psi)

    psi_reshaped = psi.reshape([d] * n)

    slicer = [slice(None)] * n
    slicer[q_idx] = outcome

    psi_reduced = psi_reshaped[tuple(slicer)]

    psi_reduced = psi_reduced.flatten()

    norm = np.linalg.norm(psi_reduced)

    if norm < 1e-10:
        return None

    return (psi_reduced / norm).astype(backend.COMPLEX_DTYPE)


def get_projected_state_multi(
    psi: np.ndarray,
    n: int,
    d: int,
    q_indices: Sequence[int],
    outcomes: Sequence[int],
) -> Optional[np.ndarray]:
    """
    Project onto outcomes for multiple qudits simultaneously and trace them out.

    Parameters
    ----------
    psi : np.ndarray
        Flat statevector of length d**n.
    n : int
        Total number of qudits.
    d : int
        Dimension.
    q_indices : Sequence[int]
        Indices of measured qudits.
    outcomes : Sequence[int]
        Corresponding outcomes (same length as `q_indices`).

    Returns
    -------
    Optional[np.ndarray]
        Normalized reduced statevector for the remaining qudits, or None if the
        projected state has near-zero norm.

    Raises
    ------
    ValueError
        If lengths mismatch, indices/outcomes are out of range, or `psi` length is invalid.
    """
    psi = np.asarray(psi)

    psi_tensor = psi.reshape([d] * n)

    slicer = [slice(None)] * n

    for q, val in zip(q_indices, outcomes):
        slicer[q] = val

    psi_reduced = psi_tensor[tuple(slicer)]

    psi_reduced = psi_reduced.flatten()

    norm = np.linalg.norm(psi_reduced)
    if norm < 1e-10:
        return None

    return (psi_reduced / norm).astype(backend.COMPLEX_DTYPE)


def compare_statevectors(
    psi1: np.ndarray, psi2: np.ndarray, atol: float = 1e-8
) -> bool:
    """
    Check whether two statevectors are equal up to a global phase.

    Parameters
    ----------
    psi1, psi2 : np.ndarray
        Statevectors to compare (flattened internally).
    atol : float, optional
        Absolute tolerance for the phase-invariant overlap test.

    Returns
    -------
    bool
        True if the statevectors represent the same physical state up to global phase.

    Raises
    ------
    ValueError
        If either statevector contains NaN/Inf or has near-zero norm.
    """
    a = np.asarray(psi1).reshape(-1)
    b = np.asarray(psi2).reshape(-1)

    if a.shape != b.shape:
        return False

    if np.isnan(a).any() or np.isnan(b).any() or np.isinf(a).any() or np.isinf(b).any():
        raise ValueError("Statevector contains NaN or Inf, cannot compare.")

    is_32 = a.dtype in (np.float32, np.complex64) or b.dtype in (
        np.float32,
        np.complex64,
    )
    eps_floor = 1e-6 if is_32 else atol
    atol = max(atol, eps_floor)

    n1 = np.linalg.norm(a)
    n2 = np.linalg.norm(b)
    if n1 <= atol or n2 <= atol:
        raise ValueError(
            "One or both statevectors have near-zero norm and cannot be compared."
        )

    u1 = a / n1
    u2 = b / n2

    fid = np.abs(np.vdot(u1, u2))

    return np.isclose(fid, 1.0, atol=atol, rtol=0.0)


def norm_diff(psi1: np.ndarray, psi2: np.ndarray) -> float:
    """
    Compute ||psi1 - e^{iφ} psi2|| with the phase chosen to best align the vectors.

    This function removes an (estimated) global phase between the vectors by using
    the overlap ⟨psi2|psi1⟩. If the overlap magnitude is too small, phase extraction
    becomes unstable; in that case the raw difference ||psi1 - psi2|| is returned.

    Parameters
    ----------
    psi1, psi2 : np.ndarray
        Statevectors to compare (flattened internally). Must have the same shape.

    Returns
    -------
    float
        Phase-aligned difference norm.

    Raises
    ------
    ValueError
        If shapes mismatch or if either statevector contains NaN/Inf.
    """
    a = np.asarray(psi1).reshape(-1)
    b = np.asarray(psi2).reshape(-1)

    if a.shape != b.shape:
        raise ValueError(
            f"State vectors must have the same shape, but got {a.shape} and {b.shape}"
        )

    if np.isnan(a).any() or np.isnan(b).any() or np.isinf(a).any() or np.isinf(b).any():
        raise ValueError("Statevector contains NaN or Inf, cannot compare.")

    overlap = np.vdot(b, a)

    is_32 = a.dtype in (np.float32, np.complex64) or b.dtype in (
        np.float32,
        np.complex64,
    )
    atol = 1e-6 if is_32 else 1e-12

    if np.abs(overlap) <= atol:
        return float(np.linalg.norm(a - b))

    phase_factor = overlap / np.abs(overlap)

    return float(np.linalg.norm(a - b * phase_factor))


def apply_unitary(
    unitary: np.ndarray,
    n: int,
    d: int,
    psi0: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Apply a (d**n x d**n) unitary to an initial statevector.

    Parameters
    ----------
    unitary : np.ndarray
        Unitary matrix of shape (d**n, d**n).
    n : int
        Number of qudits.
    d : int
        Dimension.
    psi0 : Optional[np.ndarray], optional
        Initial statevector of length d**n. If None, uses |0...0⟩.

    Returns
    -------
    np.ndarray
        Final statevector, dtype backend.COMPLEX_DTYPE.

    Raises
    ------
    ValueError
        If `unitary` shape is not (d**n, d**n) or if `psi0` has the wrong length.
    """
    size = d**n

    if unitary.shape != (size, size):
        raise ValueError(
            f"Unitary matrix must have shape ({size}, {size}), but got {unitary.shape}"
        )

    psi = (
        np.zeros(size, dtype=backend.COMPLEX_DTYPE)
        if psi0 is None
        else psi0.copy().astype(backend.COMPLEX_DTYPE)
    )
    if psi0 is None:
        psi[0] = 1

    unitary = unitary.astype(backend.COMPLEX_DTYPE)
    final_psi = unitary @ psi

    return final_psi
