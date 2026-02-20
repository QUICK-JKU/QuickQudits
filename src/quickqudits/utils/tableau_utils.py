"""
Tableau utilities.

This module contains helper functions related to the `Tableau` representation:
- Numba engine warmup (JIT compilation trigger)
- Consistency checks for full tableaus (symplectic commutation structure)
"""

from __future__ import annotations

import numpy as np

from ..circuit import QuantumCircuit
from ..predefined_gates import CNOT_gate, H_gate, S_gate, X_gate, Z_gate
from ..tableau import Tableau

__all__ = [
    "warmup_numba",
    "check_symplectic_structure",
]

# ===========================================================================
# Numba Warmup
# ===========================================================================


def warmup_numba(d: int, *, raise_on_fail: bool = False) -> bool:
    """
    Trigger JIT compilation of the Numba tableau kernels by running a tiny circuit.

    This is a convenience function to avoid JIT overhead on the first “real” call
    to `Tableau.apply_circuit(..., engine="numba")`.

    Parameters
    ----------
    d : int
        Qudit dimension.
    raise_on_fail : bool, optional
        If True, raise RuntimeError when the Numba engine is unavailable.
        If False (default), return False on failure.

    Returns
    -------
    bool
        True if warmup succeeded, False otherwise.

    Raises
    ------
    ValueError
        If `d < 2`.
    RuntimeError
        If warmup fails and `raise_on_fail=True`.
    """
    if d < 2:
        raise ValueError(f"d must be >= 2, got {d}.")

    qc = QuantumCircuit(2, d, name="numba_warmup")
    for g in (H_gate, S_gate, X_gate, Z_gate, CNOT_gate):
        if g.n_qudits == 1:
            qc.append(g, 0)
        else:
            qc.append(g, (0, 1))

    try:
        tab = Tableau(2, d, full=False)
        tab.apply_circuit(qc, engine="numba")

        tab = Tableau(2, d, full=True)
        tab.apply_circuit(qc, engine="numba")
    except Exception as e:
        if raise_on_fail:
            raise RuntimeError(f"Numba warmup failed: {e}") from e
        return False

    return True


# ===========================================================================
# Tableau Validity Checks
# ===========================================================================


def check_symplectic_structure(tab: Tableau, canonical: bool = True) -> bool:
    """
    Verify that the tableau satisfies symplectic commutation constraints.

    Checks:
    1. Stabilizers commute: <S_i, S_j> = 0
    2. Destabilizers commute: <D_i, D_j> = 0
    3. Canonical pairing: <S_i, D_j> = -δ_ij (mod d) (if canonical=True)
                          or <S_i, D_j> = 0 for i!=j, ≠ 0 for i=j (if canonical=False)

    where <A, B> is the symplectic inner product (A_x B_z^T - A_z B_x^T) % d.

    Parameters
    ----------
    tab : Tableau
        A full tableau.
    canonical : bool, default=True
        If True, enforces <S_i, D_j> = -δ_ij (mod d). If False, allows any non-zero value.

    Returns
    -------
    bool
        True if all symplectic constraints are met.

    Raises
    ------
    ValueError
        If Tableau is not full.
    """
    if not tab.full:
        raise ValueError(
            "check_symplectic_structure requires a full tableau (full=True)."
        )

    n, d = tab.n, tab.d

    Dx, Dz = tab.X[:n], tab.Z[:n]
    Sx, Sz = tab.X[n:], tab.Z[n:]

    def symplectic_product(Ax, Az, Bx, Bz):
        Ax = np.asarray(Ax, dtype=np.int64)
        Az = np.asarray(Az, dtype=np.int64)
        Bx = np.asarray(Bx, dtype=np.int64)
        Bz = np.asarray(Bz, dtype=np.int64)
        return (Ax @ Bz.T - Az @ Bx.T) % d

    # 1. & 2. Check S-S and D-D commutativity (must be 0)
    if np.any(symplectic_product(Sx, Sz, Sx, Sz) != 0):
        return False
    if np.any(symplectic_product(Dx, Dz, Dx, Dz) != 0):
        return False

    # 3. Check S-D pairing
    C_SD = symplectic_product(Sx, Sz, Dx, Dz)

    if canonical:
        I = np.eye(n, dtype=C_SD.dtype) % d
        return np.array_equal(C_SD, (-I) % d)

    diag = np.diag(C_SD)
    if np.any(diag == 0):
        return False

    C_off = C_SD.copy()
    C_off[np.diag_indices(n)] = 0

    return not np.any(C_off != 0)
