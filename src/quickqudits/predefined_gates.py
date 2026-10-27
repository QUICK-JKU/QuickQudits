"""
Definitions of gate instances.

This module provides pre-instantiated `Gate` objects for one- and two-qudit
operations (I, X, Y, Z, H, S, CNOT, CZ, SWAP) that can be directly appended to
a QuantumCircuit.
"""

from . import gates
from .gate_class import Gate

__all__ = [
    "I_gate",
    "X_gate",
    "Y_gate",
    "Z_gate",
    "H_gate",
    "S_gate",
    "CNOT_gate",
    "CZ_gate",
    "SWAP_gate",
    "W_gate",
    "M_gate",
    "P_gate",
    "GROUP_gate",
    "NOISE_gate",
]

# ===========================================================================
# Predefined gate instances
# ===========================================================================

# 1-Qudit Gates
I_gate = Gate("I", 1, gates.I, gates.I_dag)
X_gate = Gate("X", 1, gates.X, gates.X_dag)
Y_gate = Gate("Y", 1, gates.Y, gates.Y_dag)
Z_gate = Gate("Z", 1, gates.Z, gates.Z_dag)
H_gate = Gate("H", 1, gates.H, gates.H_dag, apply_fn=gates.apply_H_matmul)
S_gate = Gate("S", 1, gates.S, gates.S_dag)

# 2-Qudit Gates
CNOT_gate = Gate(
    "CNOT", 2, gates.CNOT, gates.CNOT_dag, apply_fn=gates.apply_CNOT_reshape_idx
)
CZ_gate = Gate("CZ", 2, gates.CZ, gates.CZ_dag, apply_fn=gates.apply_CZ_reshape_idx)
SWAP_gate = Gate(
    "SWAP", 2, gates.SWAP, gates.SWAP_dag, apply_fn=gates.apply_SWAP_reshape_idx
)

# Special / Placeholder Gates
# W_gate is a template for Weyl operators; specific instances (W_a,b) are created
# dynamically by QuantumCircuit.W(q, a, b).
W_gate = Gate(
    "W_{a,b}",
    1,
    None,
    None,
    apply_fn=gates.apply_W_reshape_idx,
    a=None,
    b=None,
)

# Measurement placeholder
M_gate = Gate("M", 1, None, None)

# Identity placeholder (visual only)
P_gate = Gate("P", 1, None, None)

# Group placeholder (visual only, for boxing multiple qudits)
GROUP_gate = Gate("GROUP:U(φ)", 2, None, None)

# Noise placeholder (visual only, default color red)
NOISE_gate = Gate("NOISE_CHANNEL", 1, None, None, color="#FF0000")
