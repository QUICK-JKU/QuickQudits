"""
Input/Output utilities for saving and loading Tableaus.

This module provides functions to serialize `Tableau` objects to disk using
NumPy's compressed `.npz` format and reconstruct them.
"""

from __future__ import annotations

import numpy as np

from ..tableau import Tableau

__all__ = [
    "save_tableau",
    "load_tableau",
]

SCHEMA = 1


def save_tableau(tab: Tableau, path: str) -> None:
    """
    Save a Tableau to a binary file in .npz format.

    Parameters
    ----------
    tab : Tableau
        The tableau object to save.
    path : str
        The destination file path. If the extension '.npz' is missing,
        it will be appended automatically.
    """
    if not path.endswith(".npz"):
        path += ".npz"

    tab.normalize()

    X, Z, T = tab.X, tab.Z, tab.tau_exp

    np.savez_compressed(
        path,
        schema=np.int64(SCHEMA),
        n=np.int64(tab.n),
        d=np.int64(tab.d),
        full=np.int8(1 if tab.full else 0),
        X=X,
        Z=Z,
        tau=T,
    )


def load_tableau(path: str) -> Tableau:
    """
    Load a Tableau from a .npz file.

    Parameters
    ----------
    path : str
        The file path to load from. If the extension '.npz' is missing,
        it will be appended automatically.

    Returns
    -------
    Tableau
        The reconstructed tableau object.
    """
    if not path.endswith(".npz"):
        path += ".npz"

    with np.load(path, allow_pickle=False) as f:
        n = int(f["n"])
        d = int(f["d"])
        full = bool(int(f["full"]))

        X, Z, T = f["X"], f["Z"], f["tau"]

    t = Tableau.from_rows(n, d, X, Z, T, full=full)
    t.normalize()
    return t
