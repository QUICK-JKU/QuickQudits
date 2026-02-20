"""
Circuit input/output utilities.

This module provides serialization helpers for :class:`quickqudits.circuit.QuantumCircuit`
using NumPy's compressed ``.npz`` format.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np

from .. import gates, predefined_gates
from ..circuit import QuantumCircuit
from ..gate_class import Gate, NoiseChannel

__all__ = ["save_circuit", "load_circuit"]

SCHEMA = 2

_FIXED_GATES: dict[str, Gate] = {
    "I": predefined_gates.I_gate,
    "X": predefined_gates.X_gate,
    "Y": predefined_gates.Y_gate,
    "Z": predefined_gates.Z_gate,
    "H": predefined_gates.H_gate,
    "S": predefined_gates.S_gate,
    "CNOT": predefined_gates.CNOT_gate,
    "CZ": predefined_gates.CZ_gate,
    "SWAP": predefined_gates.SWAP_gate,
    "M": predefined_gates.M_gate,
    "P": predefined_gates.P_gate,
    "GROUP:U(φ)": predefined_gates.GROUP_gate,
}


def save_circuit(c: QuantumCircuit, path: str) -> None:
    """
    Serialize a ``QuantumCircuit`` to disk.

    The circuit is stored as a compressed ``.npz`` archive containing:
    - global circuit metadata (number of qudits, dimension, name)
    - the full noise registry (custom noise models indexed by ``model_id``)
    - an ordered list of circuit operations (gates and noise channels)

    Noise channels are stored indirectly via their ``model_id``. The actual
    noise models are saved once in the noise registry, ensuring deduplication
    and exact reconstruction of custom noise behavior. Noise visualization
    colors are stored per ``model_id`` and restored on load.

    Parameters
    ----------
    c : QuantumCircuit
        The circuit instance to serialize.
    path : str
        Output file path. If the ``.npz`` suffix is missing, it is appended
        automatically.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If the circuit contains an unknown gate name, or a Weyl gate without
        parameters, or inconsistent colors for the same ``model_id``, or a
        noise color that is not an RGBA tuple.
    """
    if not path.endswith(".npz"):
        path += ".npz"

    ops: list[dict[str, Any]] = []
    model_color: dict[int, tuple[float, float, float, float]] = {}

    for g, q, dag in c.ops:
        if isinstance(g, NoiseChannel):
            mid = int(g.model_id)

            if g.color is None:
                col = (1.0, 0.0, 0.0, 1.0)
            else:
                col = g.color
                if not (isinstance(col, tuple) and len(col) == 4):
                    raise ValueError(
                        f"NoiseChannel.color must be an RGBA tuple, got {col!r}"
                    )
                col = (float(col[0]), float(col[1]), float(col[2]), float(col[3]))

            if mid in model_color and model_color[mid] != col:
                raise ValueError(
                    f"Inconsistent colors for noise model_id={mid}: "
                    f"{model_color[mid]!r} vs {col!r}"
                )
            model_color.setdefault(mid, col)
            ops.append({"t": "noise", "q": list(q), "model_id": mid})
            continue

        if not isinstance(g, Gate):
            raise ValueError(f"Unsupported op gate type: {type(g)!r}")

        if g.name == "NOISE_CHANNEL":
            raise ValueError(
                "Found Gate('NOISE_CHANNEL') placeholder in program ops. "
                "Use NoiseChannel instances created via QuantumCircuit noise helpers."
            )

        if g.name.startswith("W_"):
            if g.a is None or g.b is None:
                raise ValueError("W gate requires parameters a and b.")
            ops.append(
                {
                    "t": "gate",
                    "name": f"W_{g.a},{g.b}",
                    "q": list(q),
                    "dag": bool(dag),
                    "a": int(g.a),
                    "b": int(g.b),
                }
            )
            continue

        if g.name not in _FIXED_GATES:
            raise ValueError(f"Unknown gate name {g.name!r} cannot be serialized.")

        ops.append({"t": "gate", "name": g.name, "q": list(q), "dag": bool(dag)})

    noise_ids = np.array(sorted(c.noise_registry.keys()), dtype=np.int64)

    if noise_ids.size:
        noise_probs = np.stack(
            [c.noise_registry[int(i)] for i in noise_ids], axis=0
        ).astype(np.float64, copy=False)
        noise_rgba = np.array(
            [model_color.get(int(i), (1.0, 0.0, 0.0, 1.0)) for i in noise_ids],
            dtype=np.float64,
        )
        if noise_rgba.ndim != 2 or noise_rgba.shape[1] != 4:
            raise ValueError("noise_rgba must have shape (K, 4).")
    else:
        noise_probs = np.zeros((0, c.d * c.d), dtype=np.float64)
        noise_rgba = np.zeros((0, 4), dtype=np.float64)

    op_json = json.dumps(ops, separators=(",", ":"), ensure_ascii=False)

    np.savez_compressed(
        path,
        schema=np.int64(SCHEMA),
        n=np.int64(c.n),
        d=np.int64(c.d),
        name=np.array("" if c.name is None else str(c.name), dtype="U256"),
        noise_ids=noise_ids,
        noise_probs=noise_probs,
        noise_rgba=noise_rgba,
        op_json=np.array(op_json, dtype=np.str_),
    )


def load_circuit(path: str) -> QuantumCircuit:
    """
    Load a ``QuantumCircuit`` from disk.

    Parameters
    ----------
    path : str
        Path to a serialized circuit file. If the ``.npz`` suffix is missing,
        it is appended automatically.

    Returns
    -------
    QuantumCircuit
        A newly constructed circuit equivalent to the serialized one.

    Raises
    ------
    KeyError
        If required fields are missing from the archive.
    ValueError
        If the serialization schema version is unsupported, or if an unknown
        gate is encountered.
    """
    if not path.endswith(".npz"):
        path += ".npz"

    with np.load(path, allow_pickle=False) as f:
        schema = int(f["schema"])
        if schema != SCHEMA:
            raise ValueError(f"Unsupported circuit schema {schema}, expected {SCHEMA}.")

        n = int(f["n"])
        d = int(f["d"])
        name = str(f["name"])
        noise_ids = f["noise_ids"].astype(np.int64, copy=False)
        noise_probs = f["noise_probs"].astype(np.float64, copy=False)
        noise_rgba = f["noise_rgba"].astype(np.float64, copy=False)
        ops = json.loads(str(f["op_json"]))

    c = QuantumCircuit(n, d, name=None if name == "" else name)

    for i, mid in enumerate(noise_ids.tolist()):
        c.noise_registry[int(mid)] = noise_probs[i].astype(np.float64, copy=False)

    c._next_noise_model_id = (
        (max(c.noise_registry.keys()) + 1) if c.noise_registry else 0
    )
    c._noise_model_cache.clear()
    c._kraus_operator_cache.clear()

    color_map = {
        int(mid): (
            float(noise_rgba[i, 0]),
            float(noise_rgba[i, 1]),
            float(noise_rgba[i, 2]),
            float(noise_rgba[i, 3]),
        )
        for i, mid in enumerate(noise_ids.tolist())
    }

    for op in ops:
        t = op.get("t")
        if t == "noise":
            mid = int(op["model_id"])
            g = NoiseChannel(
                model_id=mid, color=color_map.get(mid, (1.0, 0.0, 0.0, 1.0))
            )
            c.ops.append((g, tuple(op["q"]), False))
            continue

        if t == "gate":
            name_g = str(op["name"])
            if name_g.startswith("W_"):
                a = op.get("a")
                b = op.get("b")
                if a is None or b is None:
                    raise ValueError("Serialized W gate missing parameters a/b.")
                a = int(a)
                b = int(b)
                g = Gate(
                    f"W_{a},{b}",
                    1,
                    None,
                    None,
                    apply_fn=gates.apply_W_reshape_idx,
                    a=a,
                    b=b,
                )
            else:
                if name_g not in _FIXED_GATES:
                    raise ValueError(
                        f"Unknown gate name {name_g!r} in serialized circuit."
                    )
                g = _FIXED_GATES[name_g]

            c.ops.append((g, tuple(op["q"]), bool(op.get("dag", False))))
            continue

        raise ValueError(f"Unknown op type {t!r} in serialized circuit.")

    return c
