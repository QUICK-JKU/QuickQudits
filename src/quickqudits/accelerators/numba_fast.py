"""
Numba-accelerated tableau update kernels.

This module provides a small set of JIT-compiled functions that apply packed
Clifford/Weyl gates to a stabilizer tableau represented by three arrays:

- X : (2n, n) array of X exponents (mod d)
- Z : (2n, n) array of Z exponents (mod d)
- tau_exp : (2n,) array of τ-exponents (mod 2d)

The kernels operate on *packed circuits* ``G`` of shape (L, 4)
(L=number of ops after packing), where each row encodes a single operation:

    G[k] = [gate_id, q1, q2, dag]
"""

from __future__ import annotations

import numpy as np
from numba import njit

GATE_I = 0
GATE_X = 1
GATE_Y = 2
GATE_Z = 3
GATE_H = 4
GATE_S = 5
GATE_CNOT = 6
GATE_CZ = 7
GATE_WEYL = 8
GATE_SWAP = 9

NAME_TO_ID = {
    "I": GATE_I,
    "X": GATE_X,
    "Y": GATE_Y,
    "Z": GATE_Z,
    "H": GATE_H,
    "S": GATE_S,
    "CNOT": GATE_CNOT,
    "CZ": GATE_CZ,
    "SWAP": GATE_SWAP,
}

ID_TO_NAME = {v: k for k, v in NAME_TO_ID.items()}
ID_TO_NAME[GATE_WEYL] = "W_ab"


def pack_circuit_from_qc(qc) -> np.ndarray:
    """
    Pack circuit operations into an ``int64`` gate array for Numba kernels.

    The returned array has shape (L, 4) and dtype ``np.int64``, with rows

        [gate_id, q1, q2, dag]

    where the meaning of q2 and dag depends on the gate type (see below).

    Encoding conventions
    --------------------
    Single-qudit gates:
        - q1 is the target qudit index
        - q2 is set to -1
        - dag is 0 (False) or 1 (True), except for packed Weyl gates (see below)

    Two-qudit gates (CNOT, CZ, SWAP):
        - q1 and q2 are the two qudit indices
        - dag is 0/1 for dagger flag

    Weyl gates W_ab:
        Packed as:
            [GATE_WEYL, j, a, b]
        where ``j`` is the target qudit, and ``a, b`` are integer parameters.
        If the original gate was marked as dagger, the packer negates ``a`` and ``b``
        before storing them.


    Parameters
    ----------
    qc : QuantumCircuit
        An iterable of operations producing triples ``(gate, qudits, dagger)``.
        - ``gate`` should have attribute ``name`` for named gates, and may have
          attributes ``a`` and ``b`` for realized Weyl gates.
        - ``qudits`` is an indexable sequence of qudit indices.
        - ``dagger`` is a boolean for daggered operations.

    Returns
    -------
    np.ndarray
        Packed operations as an array of shape (L, 4), dtype ``np.int64``.
        If no operations are packable, returns an empty array of shape (0, 4).

    Notes
    -----
    Gates with unknown/unsupported names are skipped.
    """
    ops = list(qc)
    packed_ops = []
    for gate, qudits, dagger in ops:
        # logic to handle realized Weyl gates
        if gate.a is not None and gate.b is not None:
            # This is a pre-realized W_ab gate,
            # pack it as [GATE_WEYL, qudit, a, b]
            final_a, final_b = gate.a, gate.b
            if dagger:
                # the dagger is encoded by negating the parameters
                final_a = -final_a
                final_b = -final_b
            packed_ops.append([GATE_WEYL, qudits[0], final_a, final_b])
        else:
            name = str(getattr(gate, "name", "")).upper()
            gid = NAME_TO_ID.get(name, -1)
            if gid < 0:
                continue

            q1 = int(qudits[0])
            q2 = int(qudits[1]) if len(qudits) > 1 else -1
            dag_flag = 1 if dagger else 0
            packed_ops.append([gid, q1, q2, dag_flag])

    if not packed_ops:
        return np.empty((0, 4), dtype=np.int64)

    return np.array(packed_ops, dtype=np.int64)


@njit(cache=True)
def nb_apply_gate(
    X: np.ndarray,
    Z: np.ndarray,
    tau_exp: np.ndarray,
    gid: int,
    q1: int,
    q2: int,
    dag: int,
    mod_d: int,
    mod_2d: int,
) -> None:
    """
    Apply a single packed gate in-place to ``(X, Z, tau_exp)``.

    Parameters
    ----------
    X : np.ndarray
        X exponent tableau, shape (2n, n), interpreted mod ``mod_d``.
    Z : np.ndarray
        Z exponent tableau, shape (2n, n), interpreted mod ``mod_d``.
    tau_exp : np.ndarray
        τ-exponent vector, shape (2n,), interpreted mod ``mod_2d``.
    gid : int
        Gate ID (one of the GATE_* constants).
    q1 : int
        First qudit index (target or control, depending on gate).
    q2 : int
        Second qudit index for two-qudit gates; for Weyl gates this is ``a``.
    dag : int
        Dagger flag (0/1) for non-Weyl gates; for Weyl gates this is ``b``.
    mod_d : int
        Modulus d.
    mod_2d : int
        Modulus 2d.

    Returns
    -------
    None
        Operates in-place.
    """
    dir_ = 1 if dag == 0 else -1
    even = (mod_d % 2) == 0

    if gid == GATE_I:
        return

    elif gid == GATE_X:
        tau_exp[:] = (tau_exp - dir_ * 2 * Z[:, q1]) % mod_2d

    elif gid == GATE_Z:
        tau_exp[:] = (tau_exp + dir_ * 2 * X[:, q1]) % mod_2d

    elif gid == GATE_Y:
        tau_exp[:] = (tau_exp + dir_ * 2 * (X[:, q1] - Z[:, q1])) % mod_2d

    elif gid == GATE_H:
        x_old = X[:, q1].copy()
        z_old = Z[:, q1].copy()
        X[:, q1] = (-dir_ * z_old) % mod_d
        Z[:, q1] = (dir_ * x_old) % mod_d
        tau_exp[:] = (tau_exp + 2 * (X[:, q1] * Z[:, q1])) % mod_2d

    elif gid == GATE_S:
        xj = X[:, q1]
        Z[:, q1] = (Z[:, q1] + dir_ * xj) % mod_d
        if even:
            tau_exp[:] = (tau_exp + dir_ * (xj * xj)) % mod_2d
        else:
            tau_exp[:] = (tau_exp + dir_ * (xj * (xj - 1))) % mod_2d

    elif gid == GATE_CNOT:
        X[:, q2] = (X[:, q2] + dir_ * X[:, q1]) % mod_d
        Z[:, q1] = (Z[:, q1] - dir_ * Z[:, q2]) % mod_d

    elif gid == GATE_CZ:
        Z[:, q1] = (Z[:, q1] + dir_ * X[:, q2]) % mod_d
        Z[:, q2] = (Z[:, q2] + dir_ * X[:, q1]) % mod_d
        tau_exp[:] = (tau_exp + 2 * dir_ * (X[:, q1] * X[:, q2])) % mod_2d

    elif gid == GATE_WEYL:
        a = q2
        b = dag
        tau_exp[:] = (tau_exp + 2 * b * X[:, q1] - 2 * a * Z[:, q1]) % mod_2d

    elif gid == GATE_SWAP:
        if q1 != q2:
            tmp = X[:, q1].copy()
            X[:, q1] = X[:, q2]
            X[:, q2] = tmp

            tmp = Z[:, q1].copy()
            Z[:, q1] = Z[:, q2]
            Z[:, q2] = tmp


@njit(cache=True)
def nb_apply_circuit(
    X: np.ndarray,
    Z: np.ndarray,
    tau_exp: np.ndarray,
    G: np.ndarray,
    mod_d: int,
    mod_2d: int,
) -> None:
    """
    Apply a packed circuit ``G`` in-place to ``(X, Z, tau_exp)``.

    Parameters
    ----------
    X : np.ndarray
        X exponent tableau, shape (2n, n), interpreted mod ``mod_d``.
    Z : np.ndarray
        Z exponent tableau, shape (2n, n), interpreted mod ``mod_d``.
    tau_exp : np.ndarray
        τ-exponent vector, shape (2n,), interpreted mod ``mod_2d``.
    G : np.ndarray
        Packed circuit array of shape (L, 4), dtype int64. Each row is a packed gate.
    mod_d : int
        Modulus d.
    mod_2d : int
        Modulus 2d.

    Returns
    -------
    None
        Operates in-place.
    """
    L = G.shape[0]
    for k in range(L):
        nb_apply_gate(X, Z, tau_exp, G[k, 0], G[k, 1], G[k, 2], G[k, 3], mod_d, mod_2d)
