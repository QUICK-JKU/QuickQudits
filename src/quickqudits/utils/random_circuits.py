"""
Random circuit generation utilities.

This module provides helpers to generate random qudit circuits with bounded
total gate count (size) and bounded parallel depth (layers). Gates are sampled
from a weighted gateset.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from ..circuit import QuantumCircuit
from ..predefined_gates import (
    CNOT_gate,
    CZ_gate,
    H_gate,
    I_gate,
    S_gate,
    SWAP_gate,
    X_gate,
    Y_gate,
    Z_gate,
)

__all__ = [
    "random_circuit",
]

# ===========================================================================
# Random Circuit Generation
# ===========================================================================


def _parse_gate_key(key: str) -> tuple[str, bool]:
    """
    Parse a gate name key possibly suffixed by "dag" (case-insensitive).

    Examples
    --------
    "X" -> ("X", False)
    "xdag" -> ("X", True)
    "CNOTdag" -> ("CNOT", True)
    "cx" -> ("CNOT", False)

    Parameters
    ----------
    key : str
        Gate name, optionally ending with "dag" (any case).

    Returns
    -------
    (str, bool)
        (BASE_NAME, is_dagger).
    """
    k = key.strip()
    kl = k.lower()
    is_dagger = kl.endswith("dag")
    base = kl[:-3] if is_dagger else kl
    base = base.upper()
    if base == "CX":
        base = "CNOT"
    return base, is_dagger


def _prepare_weighted_gates(
    gateset: dict[str, float],
    num_qudits: int,
) -> list[tuple[object, bool, float]]:
    """
    Validate, filter, and normalize a weighted gateset.

    Rules
    -----
    - Unknown gate names raise ValueError.
    - Negative probabilities raise ValueError.
    - Empty / all-zero weights raise ValueError.
    - If num_qudits == 1: drop 2-qudit gates; error if none remain.
    - Remaining weights are normalized to sum to 1.

    Parameters
    ----------
    gateset : dict[str, float]
        Mapping from gate-name keys to weights. Keys can include a "dag" suffix
        (case-insensitive), e.g. "Hdag", "CNOTDAG".
    num_qudits : int
        Number of qudits in the circuit.

    Returns
    -------
    list[tuple[object, bool, float]]
        List of (gate_object, is_dagger, prob) with normalized probabilities.

    Raises
    ------
    ValueError
        If gateset is empty, contains unknown gates, contains negative weights,
        or has no usable gates after filtering.
    """
    if not gateset:
        raise ValueError("gateset is empty.")

    _BASE_GATE = {
        "I": I_gate,
        "X": X_gate,
        "Y": Y_gate,
        "Z": Z_gate,
        "H": H_gate,
        "S": S_gate,
        "CNOT": CNOT_gate,
        "CX": CNOT_gate,
        "CZ": CZ_gate,
        "SWAP": SWAP_gate,
    }

    unknown, negative = [], []
    items, total = [], 0.0
    for name, p in gateset.items():
        base, is_dag = _parse_gate_key(name)
        if base not in _BASE_GATE:
            unknown.append(name)
            continue
        p = float(p)
        if p < 0:
            negative.append((name, p))
            continue
        if p == 0:
            continue
        gate = _BASE_GATE[base]
        items.append((gate, is_dag, p))
        total += p

    if unknown:
        raise ValueError(f"Unknown gate names in gateset: {unknown}")
    if negative:
        raise ValueError(f"Negative probabilities in gateset: {negative}")
    if total == 0 or not items:
        raise ValueError("All gate probabilities are zero.")

    if num_qudits == 1:
        oneq = [(g, d, p) for (g, d, p) in items if g.n_qudits == 1]
        if not oneq:
            raise ValueError("num_qudits=1 but gateset contains only 2-qudit gates.")
        s = sum(p for (_, _, p) in oneq)
        return [(g, d, p / s) for (g, d, p) in oneq]

    s = sum(p for (_, _, p) in items)
    return [(g, d, p / s) for (g, d, p) in items]


def random_circuit(
    num_qudits: int,
    dim: int,
    max_size: int = 10,
    max_depth: int = 10,
    gateset: Optional[dict[str, float]] = None,
    adjacent_only: bool = False,
    seed: Optional[int] = None,
) -> QuantumCircuit:
    """
    Build a random QuantumCircuit with up to `max_size` total gates and up to
    `max_depth` *true* circuit layers (parallel time steps).

    The circuit will satisfy:
        qc.size  ≤ max_size
        qc.depth ≤ max_depth

    Supported gates: I, H, S, X, Y, Z, CNOT, CZ, SWAP, and their daggers.

    Gates are sampled from a probability dict like:
      {"X": p1, "Xdag": p2, "H": p3, "CNOT": p4, "CNOTdag": p5, ...}

      Example:
      gateset = {
        "I": 1.0,
        "H": 1.0, "S": 1.0, "Hdag": 1.0, "Sdag": 1.0,
        "X": 1.0, "Y": 1.0, "Z": 1.0,
        "Xdag": 1.0, "Ydag": 1.0, "Zdag": 1.0,
        "CNOT": 2.5, "CNOTdag": 2.5,
        "CZ": 2.5, "CZdag": 2.5, "SWAP": 2.5
      }

    Probabilities are normalized automatically (after filtering out invalid/2-qudit
    gates when num_qudits==1).

    _DEFAULT_GATESET = {
      "I": 1.0,
      "H": 1.0, "S": 1.0, "Hdag": 1.0, "Sdag": 1.0,
      "X": 1.0, "Y": 1.0, "Z": 1.0,
      "Xdag": 1.0, "Ydag": 1.0, "Zdag": 1.0,
      "CNOT": 1.0, "CNOTdag": 1.0,
      "CZ": 1.0, "CZdag": 1.0,
    }

    The circuit stores operations as (Gate, qudits_tuple, dagger_bool).
    `adjacent_only=True` means 2-qudit gates (e.g., CNOT/CZ) are applied only to
    neighboring qudits; the control/target orientation is randomized.

    Parameters
    ----------
    num_qudits : int
        Number of qudits (wires).
    dim : int
        Dimension.
    max_size : int, optional
        Maximum total number of gates appended.
    max_depth : int, optional
        Maximum number of parallel layers to construct.
    gateset : Optional[dict[str, float]], optional
        Gate-name weights. Keys can include a "dag" suffix (case-insensitive).
        If None, a default Clifford-like gateset is used.
    adjacent_only : bool, optional
        If True, restrict 2-qudit gates to adjacent pairs; orientation is randomized.
    seed : Optional[int], optional
        Seed for reproducible randomness.

    Returns
    -------
    QuantumCircuit
        Randomly generated circuit with depth <= max_depth and size <= max_size.

    Raises
    ------
    ValueError
        If num_qudits < 1, dim < 2, max_size < 0, max_depth < 0, or the gateset is invalid.
    """
    if num_qudits < 1:
        raise ValueError(f"num_qudits must be >= 1, got {num_qudits}.")
    if dim < 2:
        raise ValueError(f"dim must be >= 2, got {dim}.")
    if max_size < 0:
        raise ValueError(f"max_size must be >= 0, got {max_size}.")
    if max_depth < 0:
        raise ValueError(f"max_depth must be >= 0, got {max_depth}.")

    rng = np.random.default_rng(seed)

    _DEFAULT_GATESET = {
        "I": 1.0,
        "H": 1.0,
        "S": 1.0,
        "Hdag": 1.0,
        "Sdag": 1.0,
        "X": 1.0,
        "Y": 1.0,
        "Z": 1.0,
        "Xdag": 1.0,
        "Ydag": 1.0,
        "Zdag": 1.0,
        "CNOT": 1.0,
        "CNOTdag": 1.0,
        "CZ": 1.0,
        "CZdag": 1.0,
    }

    weighted = _prepare_weighted_gates(gateset or _DEFAULT_GATESET, num_qudits)
    gate_objs = [g for (g, d, p) in weighted]
    dag_flags = [d for (g, d, p) in weighted]
    probs = np.array([p for (g, d, p) in weighted], dtype=np.float64)
    probs /= probs.sum()

    qc = QuantumCircuit(num_qudits, dim, name="random_circuit")
    count = 0

    # Build up to max_depth parallel layers
    for _ in range(max_depth):
        if count >= max_size:
            break

        used = set()  # qudits occupied in this layer
        layer_gates = int(
            rng.integers(1, num_qudits + 1)
        )  # attempt this many placements

        for _ in range(layer_gates):
            if count >= max_size:
                break

            idx = int(rng.choice(len(gate_objs), p=probs))
            gate = gate_objs[idx]
            is_dag = bool(dag_flags[idx])

            if gate.n_qudits == 1:
                free = [q for q in range(num_qudits) if q not in used]
                if not free:
                    break
                q = int(rng.choice(free))
                qc.append(gate, (q,), dagger=is_dag)
                used.add(q)
                count += 1
                continue

            # 2-qudit gate
            if num_qudits < 2:
                continue  # should not happen if _prepare_weighted_gates filtered properly

            if adjacent_only:
                pairs = [
                    (i, i + 1)
                    for i in range(num_qudits - 1)
                    if i not in used and (i + 1) not in used
                ]
                if not pairs:
                    break
                i, j = pairs[int(rng.integers(len(pairs)))]
                # randomize orientation (control/target) when it matters
                if int(rng.integers(0, 2)) == 1:
                    i, j = j, i
            else:
                free = [q for q in range(num_qudits) if q not in used]
                if len(free) < 2:
                    break
                i, j = rng.choice(free, size=2, replace=False)
                i, j = int(i), int(j)

            qc.append(gate, (i, j), dagger=is_dag)
            used.update((i, j))
            count += 1

    return qc
