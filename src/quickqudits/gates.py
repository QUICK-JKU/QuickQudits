"""
Numerical matrix definitions for standard qudit gates.

This module provides functions to generate the unitary matrices for Clifford
gates (X, Z, H, CNOT, etc.) for an arbitrary dimension `d`. It employs
caching to avoid recomputing matrices for dimensions that have already
been used.
"""

from typing import Callable

import numpy as np

from . import backend

# ===========================================================================
# Caching
# ===========================================================================


# Matrix cache: stores (gate_name, dimension) -> np.ndarray
MATRICES: dict[tuple[str, int], np.ndarray] = {}


def _cached(name: str, d: int, builder: Callable[[], np.ndarray]) -> np.ndarray:
    """
    Retrieve a cached matrix or build it if not present.

    Parameters
    ----------
    name : str
        The unique name of the gate (e.g., "X", "CNOT").
    d : int
        The dimension of the qudit(s).
    builder : Callable[[], np.ndarray]
        A function that constructs the matrix if it is not found in the cache.

    Returns
    -------
    np.ndarray
        The unitary matrix for the requested gate and dimension.
    """
    key = (name, int(d))
    mat = MATRICES.get(key)
    if mat is None:
        mat = builder()
        MATRICES[key] = mat
    return mat


def clear_cache() -> None:
    """
    Clear all cached gate matrices.

    This rebuilds matrices on the next access.
    """
    MATRICES.clear()


# ===========================================================================
# Phases
# ===========================================================================


def omega(d: int) -> np.complex128:
    """
    Return the d-th root of unity: ω = exp(2πi / d).

    Parameters
    ----------
    d : int
        Dimension.

    Returns
    -------
    np.complex128
        The complex phase ω.
    """
    return np.exp(2j * np.pi / d)


def tau(d: int) -> np.complex128:
    """
    Return τ = exp(πi (d^2 + 1) / d).

    Parameters
    ----------
    d : int
        Dimension.

    Returns
    -------
    np.complex128
        The complex phase τ.
    """
    return np.exp(1j * np.pi * (d**2 + 1) / d)


# ===========================================================================
# 1-qudit Gates
# ===========================================================================


def I(d: int) -> np.ndarray:
    """
    Construct the d-dimensional Identity matrix.

    Parameters
    ----------
    d : int
        Dimension of the qudit.

    Returns
    -------
    np.ndarray
        The d x d identity matrix.
    """
    return _cached("I", d, lambda: np.eye(d, dtype=np.bool_))


def X(d: int) -> np.ndarray:
    """
    Construct the d-dimensional Pauli X (Shift) matrix.

    Action: X|k> = |k+1 mod d>.

    Parameters
    ----------
    d : int
        Dimension of the qudit.

    Returns
    -------
    np.ndarray
        The d x d X matrix.
    """
    return _cached("X", d, lambda: np.roll(np.eye(d, dtype=np.bool_), -1, axis=1))


def Y(d: int) -> np.ndarray:
    """
    Construct the d-dimensional Pauli Y matrix.

    Defined as Y = τ^{-1} X @ Z.

    Parameters
    ----------
    d : int
        Dimension of the qudit.

    Returns
    -------
    np.ndarray
        The d x d Y matrix.
    """
    return _cached(
        "Y",
        d,
        lambda: W(d, 1, 1),
    )


def Z(d: int) -> np.ndarray:
    """
    Construct the d-dimensional Pauli Z (Clock) matrix.

    Action: Z|k> = ω^k |k>.

    Parameters
    ----------
    d : int
        Dimension of the qudit.

    Returns
    -------
    np.ndarray
        The d x d Z matrix.
    """
    return _cached(
        "Z",
        d,
        lambda: np.diag(
            np.power(
                omega(d).astype(backend.COMPLEX_DTYPE),
                np.arange(d, dtype=backend.INT_DTYPE),
            )
        ).astype(backend.COMPLEX_DTYPE),
    )


def H(d: int) -> np.ndarray:
    """
    Construct the d-dimensional Hadamard (Fourier) matrix.

    Elements: H_{jk} = ω^{jk} / sqrt(d).

    Parameters
    ----------
    d : int
        Dimension of the qudit.

    Returns
    -------
    np.ndarray
        The d x d Hadamard matrix.
    """

    def _build():
        j, k = np.indices((d, d), dtype=backend.INT_DTYPE)
        return (np.power(omega(d), j * k) / np.sqrt(d)).astype(backend.COMPLEX_DTYPE)

    return _cached("H", d, _build)


def S(d: int) -> np.ndarray:
    """
    Construct the d-dimensional Phase (S) matrix.

    Action: S|k> = τ^{k(k-1)}|k> (odd d) or τ^{k^2}|k> (even d).

    Parameters
    ----------
    d : int
        Dimension of the qudit.

    Returns
    -------
    np.ndarray
        The d x d S matrix.
    """

    def _build():
        j = np.arange(d, dtype=np.int32)
        if d % 2:
            diag = np.power(omega(d), (j * (j - 1) // 2) % d)
        else:
            diag = np.power(tau(d), (j * j) % (2 * d))
        return np.diag(diag).astype(backend.COMPLEX_DTYPE)

    return _cached("S", d, _build)


# ===========================================================================
# 2-qudit Gates
# ===========================================================================


def _perm_matrix(size: int, perm: np.ndarray) -> np.ndarray:
    """
    Helper to create a permutation matrix from an index array.

    Parameters
    ----------
    size : int
        The size of the square matrix (size x size).
    perm : np.ndarray
        An array of integers representing the permutation.
        P[i, j] = 1 if perm[i] == j.

    Returns
    -------
    np.ndarray
        The resulting permutation matrix.
    """
    P = np.zeros((size, size), dtype=np.bool_)
    P[perm, np.arange(size, dtype=np.int32)] = 1
    return P


def CNOT(d: int) -> np.ndarray:
    """
    Construct the Generalized CNOT (SUM) matrix.

    Action: |c, t> -> |c, (t+c) mod d>.

    Parameters
    ----------
    d : int
        Dimension of the qudits.

    Returns
    -------
    np.ndarray
        The d^2 x d^2 CNOT matrix.
    """

    def _build():
        n = d * d
        idx = np.arange(n, dtype=backend.INT_DTYPE)
        c, t = divmod(idx, d)
        perm = c * d + (t + c) % d
        return _perm_matrix(n, perm)

    return _cached("CNOT_mat", d, _build)


def CZ(d: int) -> np.ndarray:
    """
    Construct the Generalized CZ matrix.

    Action: |c, t> -> ω^{c*t} |c, t>.

    Parameters
    ----------
    d : int
        Dimension of the qudits.

    Returns
    -------
    np.ndarray
        The d^2 x d^2 CZ matrix.
    """
    return _cached(
        "CZ",
        d,
        lambda: np.diag(
            np.exp(
                2j * np.pi / d * (np.arange(d)[:, None] * np.arange(d)[None, :])
            ).ravel()
        ),
    )


def SWAP(d: int) -> np.ndarray:
    """
    Construct the SWAP matrix.

    Action: |q1, q2> -> |q2, q1>.

    Parameters
    ----------
    d : int
        Dimension of the qudits.

    Returns
    -------
    np.ndarray
        The d^2 x d^2 SWAP matrix.
    """

    def _build():
        n = d * d
        idx = np.arange(n, dtype=np.int32)
        q1, q2 = divmod(idx, d)
        perm = q2 * d + q1
        return _perm_matrix(n, perm)

    return _cached("SWAP", d, _build)


# ===========================================================================
# Optimized Application Functions (Reshape + Indexing)
# ===========================================================================


def apply_CNOT_reshape_idx(
    psi: np.ndarray,
    n: int,
    d: int,
    qudits: tuple[int, int],
    dagger: bool = False,
) -> np.ndarray:
    """
    Apply CNOT to any pair of qudits using strided statevector views.

    Parameters
    ----------
    psi : np.ndarray
        The statevector of length d^n.
    n : int
        Number of qudits.
    d : int
        Dimension of qudits.
    qudits : tuple[int, int]
        Indices (control, target).
    dagger : bool, default=False
        If True, applies CNOT†.

    Returns
    -------
    np.ndarray
        The transformed statevector.
    """
    control, target = qudits
    shape = (d,) * n
    source = np.moveaxis(psi.reshape(shape), (control, target), (0, 1))
    out = np.empty(psi.shape, dtype=psi.dtype)
    destination = np.moveaxis(out.reshape(shape), (control, target), (0, 1))

    # Qudit 0 is the most significant digit. Each assignment copies the
    # amplitudes for fixed control and target digits; all other axes remain
    # strided views, so no full-size permutation/index array is needed.
    sign = -1 if dagger else 1
    for c in range(d):
        for t in range(d):
            destination[c, (t + sign * c) % d] = source[c, t]

    return out


def apply_CZ_reshape_idx(
    psi: np.ndarray,
    n: int,
    d: int,
    qudits: tuple[int, int],
    dagger: bool = False,
) -> np.ndarray:
    """
    Apply a CZ gate using reshaping and broadcasting.

    Parameters
    ----------
    psi : np.ndarray
        The statevector of length d^n.
    n : int
        Number of qudits.
    d : int
        Dimension of qudits.
    qudits : tuple[int, int]
        Indices (control, target).
    dagger : bool, default=False
        If True, applies CZ†.

    Returns
    -------
    np.ndarray
        The ransformed statevector.
    """
    i, j = qudits
    psi_nd = psi.reshape([d] * n)

    axes = [i, j] + [k for k in range(n) if k not in (i, j)]
    tpsi = np.transpose(psi_nd, axes)

    idx = np.arange(d, dtype=backend.INT_DTYPE)
    sign = -1 if dagger else 1
    phase2d = np.exp(sign * 2j * np.pi * (idx[:, None] * idx[None, :]) / d)

    tpsi *= phase2d.reshape(d, d, *([1] * (n - 2)))

    inv_axes = np.argsort(axes)
    return np.transpose(tpsi, inv_axes).reshape(-1)


def apply_SWAP_reshape_idx(
    psi: np.ndarray,
    n: int,
    d: int,
    qudits: tuple[int, int],
    dagger: bool = False,
) -> np.ndarray:
    """
    Apply a SWAP gate by transposing axes.

    Parameters
    ----------
    psi : np.ndarray
        Statevector.
    n : int
        Number of qudits.
    d : int
        Dimension.
    qudits : tuple[int, int]
        Indices (q1, q2).
    dagger : bool
        Ignored (SWAP is Hermitian).

    Returns
    -------
    np.ndarray
        The transformed statevector.
    """
    q1, q2 = qudits
    if q1 == q2:
        return psi

    psi_nd = psi.reshape([d] * n)
    axes = list(range(n))

    axes[q1], axes[q2] = axes[q2], axes[q1]

    out = np.transpose(psi_nd, axes)
    return out.reshape(d**n)


# ===========================================================================
# Weyl Operators
# ===========================================================================


def W(d: int, a: int, b: int) -> np.ndarray:
    """
    Construct the Weyl operator W_{a,b} = τ^{-ab} X^a Z^b.

    Parameters
    ----------
    d : int
        Dimension.
    a : int
        X exponent (shift).
    b : int
        Z exponent (phase).

    Returns
    -------
    np.ndarray
        The d x d Weyl matrix.
    """
    a %= d
    b %= d

    def _build():
        phase = np.power(tau(d), -a * b)
        xa = np.linalg.matrix_power(X(d), a)
        zb = np.linalg.matrix_power(Z(d), b)
        return (phase * (xa @ zb)).astype(backend.COMPLEX_DTYPE)

    return _cached(f"W_{a}_{b}", d, _build)


def W_dag(d: int, a: int, b: int) -> np.ndarray:
    """
    Construct the dagger of W_{a,b}, which is W_{-a,-b}.

    Parameters
    ----------
    d : int
        Dimension.
    a : int
        X exponent.
    b : int
        Z exponent.

    Returns
    -------
    np.ndarray
        The dagger matrix.
    """
    a %= d
    b %= d
    return _cached(f"W_{a}_{b}†", d, lambda: W(d, -a, -b))


def apply_W_reshape_idx(
    psi: np.ndarray,
    n: int,
    d: int,
    qudits: tuple[int, ...],
    dagger: bool = False,
    a: int = 0,
    b: int = 0,
) -> np.ndarray:
    """
    Apply Weyl W_{a,b} to a single qudit using optimized indexing.

    Action: W_{a,b} = τ^{-ab} X^a Z^b

    Parameters
    ----------
    psi : np.ndarray
        Statevector.
    n : int
        Number of qudits.
    d : int
        Dimension.
    qudits : tuple[int, ...]
        Index of the target qudit (single-element tuple).
    dagger : bool
        If True, applies W_{-a, -b}.
    a, b : int
        Weyl indices.

    Returns
    -------
    np.ndarray
        The transformed statevector.
    """
    j = qudits[0]
    a_orig, b_orig = a, b

    # For dagger, W_{a,b}^† = W_{-a,-b}
    if dagger:
        a = -a_orig
        b = -b_orig

    a %= d
    b %= d

    # Reshape statevector
    psi_nd = psi.reshape([d] * n)

    # Global phase τ^{-ab}
    phase_global = np.power(tau(d), -a * b)
    psi_nd = psi_nd * phase_global

    # Apply Z^b (diagonal phasing)
    if b != 0:
        phases = np.power(omega(d), b * np.arange(d, dtype=backend.INT_DTYPE))
        # Reshape to broadcast correctly along axis j
        phase_shape = [1] * n
        phase_shape[j] = d
        psi_nd = psi_nd * phases.reshape(phase_shape)

    # Apply X^a (cyclic shift)
    if a != 0:
        psi_nd = np.roll(psi_nd, shift=a, axis=j)

    return psi_nd.reshape(d**n)


# ===========================================================================
# Daggers
# ===========================================================================


def _dag(m: np.ndarray) -> np.ndarray:
    """
    Helper to compute the conjugate transpose of a matrix.

    Parameters
    ----------
    m : np.ndarray
        Input matrix.

    Returns
    -------
    np.ndarray
        Conjugate transpose of m.
    """
    return np.conjugate(m.T)


def I_dag(d: int) -> np.ndarray:
    """
    Return the dagger of I (which is I).

    Parameters
    ----------
    d : int
        Dimension.

    Returns
    -------
    np.ndarray
        Identity matrix.
    """
    return I(d)


def X_dag(d: int) -> np.ndarray:
    """
    Return the dagger of X.

    Parameters
    ----------
    d : int
        Dimension.

    Returns
    -------
    np.ndarray
        X dagger matrix.
    """
    return _cached("X†", d, lambda: _dag(X(d)))


def Y_dag(d: int) -> np.ndarray:
    """
    Return the dagger of Y.

    Parameters
    ----------
    d : int
        Dimension.

    Returns
    -------
    np.ndarray
        Y dagger matrix.
    """
    return _cached("Y†", d, lambda: _dag(Y(d)))


def Z_dag(d: int) -> np.ndarray:
    """
    Return the dagger of Z.

    Parameters
    ----------
    d : int
        Dimension.

    Returns
    -------
    np.ndarray
        Z dagger matrix.
    """
    return _cached("Z†", d, lambda: _dag(Z(d)))


def H_dag(d: int) -> np.ndarray:
    """
    Return the dagger of H.

    Parameters
    ----------
    d : int
        Dimension.

    Returns
    -------
    np.ndarray
        H dagger matrix.
    """
    return _cached("H†", d, lambda: _dag(H(d)))


def S_dag(d: int) -> np.ndarray:
    """
    Return the dagger of S.

    Parameters
    ----------
    d : int
        Dimension.

    Returns
    -------
    np.ndarray
        S dagger matrix.
    """
    return _cached("S†", d, lambda: _dag(S(d)))


def CNOT_dag(d: int) -> np.ndarray:
    """
    Return the dagger of CNOT.

    Parameters
    ----------
    d : int
        Dimension.

    Returns
    -------
    np.ndarray
        CNOT dagger matrix.
    """
    return _cached("CNOT†", d, lambda: _dag(CNOT(d)))


def CZ_dag(d: int) -> np.ndarray:
    """
    Return the dagger of CZ.

    Parameters
    ----------
    d : int
        Dimension.

    Returns
    -------
    np.ndarray
        CZ dagger matrix.
    """
    return _cached("CZ†", d, lambda: _dag(CZ(d)))


def SWAP_dag(d: int) -> np.ndarray:
    """
    Return the dagger of SWAP (same as SWAP).

    Parameters
    ----------
    d : int
        Dimension.

    Returns
    -------
    np.ndarray
        SWAP dagger matrix.
    """
    return SWAP(d)
