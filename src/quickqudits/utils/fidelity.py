"""
Fidelity estimation utilities.

This module provides a fast Monte-Carlo estimator for either:
- entanglement fidelity (default) or
- state fidelity on the input |0...0⟩ (stabilizers-only tableau).
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from ..circuit import QuantumCircuit
from ..gate_class import NoiseChannel
from ..tableau import Tableau

__all__ = ["fidelity_np", "fidelity_pf"]

# ===========================================================================
# Fidelity
# ===========================================================================


def fidelity_np(
    qc: QuantumCircuit,
    shots: int,
    entanglement_fidelity: bool = True,
    seed: Optional[int] = None,
) -> float:
    """
    Computes the computational basis fidelity of a noisy circuit using Monte Carlo sampling.

    The fidelity is defined as the probability of measuring the all-zero state |0...0>
    after applying the circuit U.

    By default (entanglement_fidelity=True), this computes the Entanglement Fidelity.
    In this case Tableau is initialized with full=True (destabilizers + stabilizers).

    If entanglement_fidelity=False, this computes the State Fidelity on the
    input state |0...0>.
    In this case Tableau is initialized with full=False (stabilizers only).

    The algorithm in this function:
    1. Runs the forward circuit once to cache the X/Z columns (frames) at every noise location.
    2. Samples noise and sums phase errors (delta_tau) for all shots.
    3. Checks if the total accumulated phase error on the stabilizers
    (and destabilizers if full=True) is zero.

    When entanglement_fidelity=True, this method is mathematically equivalent to the
    standard Entanglement Fidelity calculation (via Choi-Jamiolkowski isomorphism
    on 2n qudits), but there's no need to apply the universal input state circuit.

    Parameters
    ----------
    qc : QuantumCircuit
        The noisy quantum circuit to simulate. Must contain NoiseChannel gates.
    shots : int
        Number of Monte Carlo samples to use for the estimation.
    entanglement_fidelity : bool, optional
        If True (default), computes Entanglement Fidelity.
        If False, computes State Fidelity on |0...0>.
    seed : int or None, optional
        Seed for the random number generator.

    Returns
    -------
    float
        The estimated fidelity (0.0 to 1.0).

    Raises
    ------
    ValueError
        If `shots <= 0`.
    """
    if shots <= 0:
        raise ValueError("shots must be positive.")

    n = qc.n
    d = qc.d
    rng = np.random.default_rng(seed)

    if entanglement_fidelity:
        tab = Tableau(n, d, full=True)
        num_rows = 2 * n
    else:
        tab = Tableau(n, d, full=False)
        num_rows = n

    noise_frames = []  # Stores (qudit_index, X_col, Z_col, model_id)

    # Clean run + record noise_frames
    for gate, qudits, dagger in qc.ops:
        if isinstance(gate, NoiseChannel):
            q = qudits[0]
            X_col = tab.X[:, q].copy()
            Z_col = tab.Z[:, q].copy()
            noise_frames.append((q, X_col, Z_col, gate.model_id))
        else:
            tab.apply_gate(gate, qudits, dagger)

    if not noise_frames:
        return 1.0

    # Monte Carlo simulation
    delta_tau = np.zeros((shots, num_rows), dtype=tab._tau_dtype)
    mod_2d = tab.mod_2d

    for q, X_col, Z_col, model_id in noise_frames:
        probs = qc.noise_registry[model_id].reshape(-1)

        # Sample 'shots' indices [0...d^2-1]
        flat_choices = rng.choice(d * d, size=shots, p=probs)

        # Convert to Weyl coeffs (a, b)
        a_samples, b_samples = np.unravel_index(flat_choices, (d, d))

        # Calculate phase contribution: + 2 * (b * X - a * Z)
        contrib = (
            2 * b_samples[:, None] * X_col[None, :]
            - 2 * a_samples[:, None] * Z_col[None, :]
        )

        delta_tau = (delta_tau + contrib) % mod_2d

    # Check for phase errors
    has_error = np.any(delta_tau != 0, axis=1)

    success_count = np.sum(~has_error)

    return float(success_count) / shots


def fidelity_pf(
    qc: QuantumCircuit,
    shots: int,
    entanglement_fidelity: bool = True,
    seed: Optional[int] = None,
) -> float:
    """
    Computes the computational basis fidelity of a noisy circuit using Pauli-frame Monte Carlo sampling.

    This estimator propagates `shots` Pauli frames through the circuit. Each NoiseChannel samples a
    Weyl operator W(a,b)=X^a Z^b and multiplies it into the frame; Clifford gates conjugate the frame.
    At the end, the frame represents the effective Weyl error acting after the (noiseless) circuit.

    Success criteria:
    - State fidelity on |0...0> (entanglement_fidelity=False): success iff the final effective error
      is Z-type, i.e. all X-exponents are zero.
    - Entanglement fidelity (entanglement_fidelity=True): success iff the final effective error is
      the identity (up to global phase), i.e. all X- and Z-exponents are zero.

    Parameters
    ----------
    qc : QuantumCircuit
        The noisy quantum circuit to simulate. Must contain NoiseChannel gates.
    shots : int
        Number of Monte Carlo samples to use for the estimation.
    entanglement_fidelity : bool, optional
        If True (default), computes entanglement fidelity (identity-error probability).
        If False, computes state fidelity on |0...0> (Z-type-error probability).
    seed : int or None, optional
        Seed for the random number generator.

    Returns
    -------
    float
        The estimated fidelity (0.0 to 1.0).

    Raises
    ------
    ValueError
        If `shots <= 0`.
    """
    if shots <= 0:
        raise ValueError("shots must be positive.")

    n = qc.n
    d = qc.d
    rng = np.random.default_rng(seed)

    x = np.zeros((shots, n), dtype=np.int64)
    z = np.zeros((shots, n), dtype=np.int64)

    saw_noise = False

    for gate, qudits, dagger in qc._ops:
        if isinstance(gate, NoiseChannel):
            saw_noise = True
            q = int(qudits[0])
            probs = qc.noise_registry.get(gate.model_id)
            if probs is None:
                probs = qc.noise_registry[gate.model_id]
            probs = np.asarray(probs, dtype=float).reshape(-1)
            Tableau._pf_apply_noise_channel(x, z, q=q, probs=probs, d=d, rng=rng)
            continue

        key = Tableau._pf_gate_key(gate)
        if key == "I":
            continue
        Tableau._pf_apply_gate(x, z, key=key, qudits=qudits, dagger=dagger, d=d)

    if not saw_noise:
        return 1.0

    if entanglement_fidelity:
        success = np.all(x == 0, axis=1) & np.all(z == 0, axis=1)
    else:
        success = np.all(x == 0, axis=1)

    return float(np.mean(success))
