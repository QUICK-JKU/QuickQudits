"""
Entanglement utilities for stabilizer tableaus.

This module provides Schmidt-rank computations for qudit stabilizer states
in prime dimension d. It includes:
- Schmidt exponent k and Schmidt rank d**k across a bipartition A|B
- The per-qudit Schmidt-rank vector (single-qudit vs rest)
"""

from __future__ import annotations

from typing import Union

import numpy as np

from ..tableau import Tableau

__all__ = [
    "schmidt_rank",
    "schmidt_rank_vector",
]

# ===========================================================================
# Schmidt Rank & Rank Vector
# ===========================================================================


def _rank_from_rref(M: np.ndarray, d: int) -> int:
    """
    Compute the rank of a matrix over Z_d using RREF (prime d).

    Parameters
    ----------
    M : np.ndarray
        Matrix to rank (entries interpreted modulo d).
    d : int
        Prime modulus.

    Returns
    -------
    int
        Rank of M over Z_d.

    Raises
    ------
    ValueError
        If d < 2.
    """
    if d < 2:
        raise ValueError(f"d must be >= 2, got {d}.")

    rref = Tableau._rref_mod_d(M.copy(), d)
    rank = np.sum(np.any(rref != 0, axis=1))

    return int(rank)


def _get_stabilizer_blocks(tab: Tableau) -> tuple[np.ndarray, np.ndarray]:
    """
    Return the stabilizer X and Z blocks of a tableau.


    Parameters
    ----------
    tab : Tableau
        Tableau instance.

    Returns
    -------
    (np.ndarray, np.ndarray)
        (X_stab, Z_stab).
    """
    n = tab.n

    if tab.full:
        X_stab = tab._X[n:]
        Z_stab = tab._Z[n:]
    else:
        X_stab = tab._X
        Z_stab = tab._Z

    return X_stab, Z_stab


def schmidt_rank(
    tab: Tableau,
    A: Union[set[int], list[int], tuple[int, ...]],
    B: Union[set[int], list[int], tuple[int, ...]],
) -> tuple[int, int]:
    """
    Compute the Schmidt exponent k and Schmidt rank d^k across partition A | B.

    Given a pure stabilizer state (n stabilizers on n qudits),
    the Schmidt rank across bipartition A|B is d^k.

    Partition requirements:
    - A and B must be disjoint sets of qudit indices.
    - A | B must cover all qudits {0,1,...,n-1}.

    This function:
      - extracts stabilizer generators,
      - restricts them to subsystem A and B,
      - computes ranks r_A and r_B via RREF mod d,
      - uses the formula k = (r_A + r_B - n) // 2.

    Parameters
    ----------
    tab : Tableau
        The stabilizer tableau (full or stabilizer-only).
    A : Union[set[int], list[int], tuple[int, ...]]
        Indices of qudits making subsystem A.
    B : Union[set[int], list[int], tuple[int, ...]]
        Indices of qudits making subsystem B.

    Returns
    -------
    (k, d^k) : tuple[int, int]
        k   = number of d-dimensional Bell pairs across A|B
        d^k = Schmidt rank across this bipartition

    Raises
    ------
    ValueError
        If A or B contain duplicate indices, if A and B overlap, if A|B does not
        cover all qudits.
    """
    if isinstance(A, (list, tuple)) and len(set(A)) != len(A):
        raise ValueError("A contains duplicate indices.")
    if isinstance(B, (list, tuple)) and len(set(B)) != len(B):
        raise ValueError("B contains duplicate indices.")

    n = tab.n
    d = tab.d

    # validate partitions
    A_set = set(A)
    B_set = set(B)

    if A_set & B_set:
        raise ValueError(f"Partitions A and B overlap: A & B = {A_set & B_set}")

    all_indices = set(range(n))
    if A_set | B_set != all_indices:
        raise ValueError(
            f"A|B must cover all qudits. Got A|B = {A_set|B_set}, expected {all_indices}"
        )

    X_stab, Z_stab = _get_stabilizer_blocks(tab)

    # build matrices X_A|Z_A and X_B|Z_B
    A = list(A)
    B = list(B)

    X_A = X_stab[:, A]
    Z_A = Z_stab[:, A]
    M_A = np.concatenate([X_A, Z_A], axis=1)

    X_B = X_stab[:, B]
    Z_B = Z_stab[:, B]
    M_B = np.concatenate([X_B, Z_B], axis=1)

    # compute ranks
    r_A = _rank_from_rref(M_A, d)
    r_B = _rank_from_rref(M_B, d)

    # k = number of Bell pairs
    k = (r_A + r_B - n) // 2
    rank = d**k

    return int(k), int(rank)


def schmidt_rank_vector(tab: Tableau) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the Schmidt rank vector for a stabilizer state.

    For each qudit i, computes the Schmidt rank across partition:
        A = {i}
        B = all other qudits

    Returns two arrays:
      - k_vec[i] = number of d-dimensional Bell pairs between qudit i and the rest
      - r_vec[i] = d^(k_vec[i]) = actual Schmidt rank for qudit i vs rest

    Parameters
    ----------
    tab : Tableau
        The stabilizer tableau.

    Returns
    -------
    (k_vec, r_vec) : tuple[np.ndarray, np.ndarray]
        k_vec : np.ndarray of shape (n,)
            Schmidt exponents for each qudit.
        r_vec : np.ndarray of shape (n,)
            Schmidt ranks = d^k_i for each qudit.
    """
    n = tab.n
    all_qudits = set(range(n))

    k_vec = []
    r_vec = []

    for i in range(n):
        A = {i}
        B = all_qudits - A
        k_i, r_i = schmidt_rank(tab, A, B)
        k_vec.append(k_i)
        r_vec.append(r_i)

    return np.array(k_vec), np.array(r_vec)
