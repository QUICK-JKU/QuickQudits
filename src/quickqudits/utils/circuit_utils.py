"""
Circuit construction utilities.

This module contains helper functions that build specific circuits used
throughout the library, such as universal input-state preparation.
"""

from __future__ import annotations

from ..circuit import QuantumCircuit

__all__ = [
    "univ_input_state_circuit",
    "encoding_circuit",
]

# ===========================================================================
# Circuit Generation Functions
# ===========================================================================


def univ_input_state_circuit(n: int, d: int) -> QuantumCircuit:
    """
    Construct a universal input state circuit on 2n qudits of dimension d.

    The circuit prepares n maximally entangled pairs (generalized Bell states)
    between the first n and the last n qudits. For each i in [0, n-1], it applies:

      1. A Hadamard gate H on qudit i (creating superposition).
      2. A controlled-X (CNOT) from qudit i (control) to qudit i+n (target),
         entangling the registers.

    Parameters
    ----------
    n : int
        Number of entangled pairs (half the total number of qudits).
        The total system size is 2n qudits.
    d : int
        Dimension.

    Returns
    -------
    QuantumCircuit
        A circuit on 2n qudits of dimension d that prepares the universal input state.

    Raises
    ------
    ValueError
        If `n < 1` or `d < 2`.
    """
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}.")
    if d < 2:
        raise ValueError(f"d must be >= 2, got {d}.")

    qc_univ = QuantumCircuit(2 * n, d, name="universal_input_state")
    for i in range(n):
        qc_univ.H(i)
        qc_univ.CX(i, i + n)
    return qc_univ


def encoding_circuit(n: int, d: int, base: int = 2) -> QuantumCircuit:
    """
    Creates the circuit on fig. 3 from this article:
    http://doi.org/10.1088/0253-6102/55/5/11

    Places H on qudits from 0 to n-2.

    Places base^i (i from 0 to n-2) CZ gates from qudit n-2 (control) to n-1 (target),
    n-2 to n, ..., 0 to n.

    The total number of gates in the circuit:
    total_gates = n-1 + (1-base^(n-1))/(1-base)

    Parameters
    ----------
    n : int
        Number of qudits. Must be >= 2.
    d : int
        Qudit dimension. Must be >= 2.
    base : int, optional
        Base which is raised to powers 0...(n-2).

    Returns
    -------
        QuantumCircuit

    Raises
    ------
    ValueError
        If `n < 2`, `d < 2`, or `base < 1`.
    """
    if n < 2:
        raise ValueError(f"n must be >= 2, got {n}.")
    if d < 2:
        raise ValueError(f"d must be >= 2, got {d}.")
    if base < 1:
        raise ValueError(f"base must be >= 1, got {base}.")

    qc = QuantumCircuit(n, d, name="encoding_circuit")

    for i in range(n - 2, -1, -1):
        qc.H(i)

        reps = base ** (n - 2 - i)
        for _ in range(reps):
            qc.CZ(i, n - 1)

    return qc
