"""
Quantum Circuit definition and simulation logic.

This module provides the `QuantumCircuit` class, which acts as a container for
quantum gates and handles the simulation of statevectors and density matrices.
"""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING, Generator, Literal, Optional, Union

import numpy as np
from numba.cuda.cudadrv import ndarray

from . import backend, gates
from .backend import COMPLEX_DTYPE
from .gate_class import Gate, NoiseChannel
from .predefined_gates import (
    CNOT_gate,
    CZ_gate,
    GROUP_gate,
    H_gate,
    I_gate,
    M_gate,
    P_gate,
    S_gate,
    SWAP_gate,
    W_gate,
    X_gate,
    Y_gate,
    Z_gate,
)

if TYPE_CHECKING:
    pass


# ===========================================================================
# QuantumCircuit Class
# ===========================================================================


class QuantumCircuit:
    """
    A quantum circuit container.

    Stores a sequence of operations and provides methods for simulation (statevector
    and density matrix evolution) and noise modeling.

    Sets up the basic circuit properties (qudits, dimension) and
    initializes the internal storage for operations. It also establishes
    the noise registry system used to manage noise models.

    Operations are stored in the self._ops list as tuples
    (Gate, qudits (one or two indices), dagger).
    self._ops : list[tuple[Gate, tuple[int, ...], bool]]

    Parameters
    ----------
    num_qudits : int
        The number of qudits in the system.
    dim : int
        The dimension (d) of each qudit.
    name : str or None, optional
        An optional name for the circuit.

    Notes
    -----
    **Noise Registry System:**
    This class uses a registry to store noise models.
    Instead of storing a full probability array for every noise gate instance,
    unique noise models are stored once in `self.noise_registry` and assigned
    a unique integer ID.

    - `self.noise_registry`: Maps model IDs to normalized probability arrays.
    - `self._noise_model_cache`: Caches unique models to prevent duplication.
    - `self._kraus_operator_cache`: Caches computed Kraus operators during
        simulation to avoid re-calculation.
    """

    def __init__(self, num_qudits: int, dim: int, name: Optional[str] = None):
        self.n = num_qudits
        self.d = dim
        self.name = name
        # Store tuples: (Gate, indices, dagger_flag)
        self._ops: list[tuple[Gate, tuple[int, ...], bool]] = []

        # Store states for mid-circuit-measurements
        self._stored_psi = None
        # Current gate index
        self._gate_idx = 0

        # --- Noise Registry System ---
        # The key is a unique model ID (int).
        # The value is the normalized probability array.
        self.noise_registry: dict[int, np.ndarray] = {}
        # A cache to find existing models to avoid storing duplicate arrays.
        # The key is a hashable tuple of the probability array.
        self._noise_model_cache: dict[tuple, int] = {}
        # A counter to generate new model IDs.
        self._next_noise_model_id = 0

        # --- Cache for Kraus Operators ---
        self._kraus_operator_cache: dict[int, list[np.ndarray]] = {}

    # -----------------------------------------------------------------------
    # Properties
    # -----------------------------------------------------------------------

    @property
    def ops(self) -> list[tuple[Gate, tuple[int, ...], bool]]:
        """
        Get the list of operations in the circuit.

        Returns
        -------
        list[tuple[Gate, tuple[int, ...], bool]]
            The internal list of operations.
        """
        return self._ops

    @property
    def size(self) -> int:
        """
        Get the total number of gates (excluding measurements).

        Returns
        -------
        int
            Count of unitary/noise gates.
        """
        return sum(1 for gate, _, _ in self._ops if gate.name != "M")

    @property
    def depth(self) -> int:
        """
        Calculate the circuit depth (number of parallel layers).

        Measurements are not counted in the depth calculation.

        Returns
        -------
        int
            The circuit depth.
        """
        cols = self._asap_schedule_packed()
        return sum(1 for col in cols if any(g.name != "M" for g, _, _ in col))

    # -----------------------------------------------------------------------
    # Basic Utils
    # -----------------------------------------------------------------------

    def __repr__(self) -> str:
        """
        Return a string representation of the circuit instance.

        Returns
        -------
        str
            String containing the circuit's dimensions and size.
        """
        return (
            f"QuantumCircuit(num_qudits={self.n}, dim={self.d}, "
            f"name={self.name!r}, size={self.size}, depth={self.depth})"
        )

    def __str__(self) -> str:
        """
        Return a string description of the circuit.

        Returns
        -------
        str
            Description including name, dimensions, size, depth.
        """
        return (
            f"<QuantumCircuit: "
            f"name={self.name!r}, n={self.n}, d={self.d}, size={self.size}, depth={self.depth}>"
        )

    def __getitem__(self, idx: int) -> tuple[Gate, tuple[int, ...], bool]:
        """
        Return the operation at the specified index.

        Parameters
        ----------
        idx : int
            The index of the operation.

        Returns
        -------
        tuple[Gate, tuple[int, ...], bool]
            The operation tuple (gate, qudits, dagger_flag).
        """
        return self._ops[idx]

    def __iter__(self) -> Generator[tuple[Gate, tuple[int, ...], bool], None, None]:
        """
        Iterate over the operations in the circuit.

        Yields
        ------
        tuple[Gate, tuple[int, ...], bool]
            Operations in the order they were appended.
        """
        yield from self._ops

    def __len__(self) -> int:
        """
        Return the number of operations in the circuit.

        Returns
        -------
        int
            Total count of operations.
        """
        return len(self._ops)

    def __copy__(self) -> QuantumCircuit:
        """
        Create a shallow copy of the circuit using the copy module.

        Returns
        -------
        QuantumCircuit
            A new circuit instance with copied operations and noise registry.
        """
        new = QuantumCircuit(self.n, self.d, name=self.name)
        new._ops = list(self._ops)

        new.noise_registry = self.noise_registry.copy()
        new._noise_model_cache = self._noise_model_cache.copy()
        new._next_noise_model_id = self._next_noise_model_id

        return new

    def copy(self) -> QuantumCircuit:
        """
        Create a shallow copy of the circuit.

        Returns
        -------
        QuantumCircuit
            A new circuit instance with copied operations and noise registry.
        """
        return copy.copy(self)

    def change_dim(self, new_d: int) -> None:
        """
        Change the dimension of the qudits in the circuit.

        Parameters
        ----------
        new_d : int
            The new dimension.
        """
        self.d = new_d

    # -----------------------------------------------------------------------
    # Scheduling Helpers
    # -----------------------------------------------------------------------

    def _asap_schedule(self) -> list[list[tuple[Gate, tuple[int, ...], bool]]]:
        """
        Generate an ASAP schedule for visualization.

        Each gate is placed in the earliest column where its qudits are free.
        Measurement gates are grouped in a final column.

        Returns
        -------
        list[list[tuple]]
            A list of columns for visualization.
        """
        unitary_ops = [op for op in self._ops if op[0].name != "M"]
        measurement_ops = [op for op in self._ops if op[0].name == "M"]

        next_free = {q: 0 for q in range(self.n)}
        cols: list[list[tuple]] = []
        latest_col_used = -1

        for gate, qudits, dagger in unitary_ops:
            lo, hi = min(qudits), max(qudits)
            span = range(lo, hi + 1)
            col = max(next_free[q] for q in span)

            latest_col_used = max(latest_col_used, col)

            while len(cols) <= col:
                cols.append([])
            cols[col].append((gate, qudits, dagger))

            for q in span:
                next_free[q] = col + 1

        if measurement_ops:
            final_col_idx = latest_col_used + 1
            while len(cols) <= final_col_idx:
                cols.append([])
            cols[final_col_idx].extend(measurement_ops)

        return cols

    def _asap_schedule_packed(self) -> list[list[tuple[Gate, tuple[int, ...], bool]]]:
        """
        Helper to schedule gates purely for depth calculation.

        Returns
        -------
        list[list[tuple]]
            A list of layers, where each layer is a list of operations.
        """
        unitary_ops = [op for op in self._ops if op[0].name != "M"]
        measurement_ops = [op for op in self._ops if op[0].name == "M"]

        next_free = {q: 0 for q in range(self.n)}
        cols, latest = [], -1

        for gate, qudits, dagger in unitary_ops:
            qs = set(qudits)  # <- only touched wires
            col = max(next_free[q] for q in qs)
            latest = max(latest, col)
            while len(cols) <= col:
                cols.append([])
            cols[col].append((gate, qudits, dagger))
            for q in qs:
                next_free[q] = col + 1

        if measurement_ops:
            i = latest + 1
            while len(cols) <= i:
                cols.append([])
            cols[i].extend(measurement_ops)
        return cols

    def _logical_schedule(self) -> list[list[tuple[Gate, tuple[int, ...], bool]]]:
        """
        Generate a logical schedule (program order).

        Returns
        -------
        list[list[tuple]]
            A list of columns, where each column contains a single operation.
        """
        return [[(gate, qudits, dagger)] for gate, qudits, dagger in self._ops]

    # -----------------------------------------------------------------------
    # Gate Manipulation
    # -----------------------------------------------------------------------

    def append(
        self,
        gate: Gate,
        qudits: Union[int, tuple[int, ...]],
        dagger: bool = False,
    ) -> None:
        """
        Append a gate to the end of the circuit.

        Parameters
        ----------
        gate : Gate
            The gate object to apply.
        qudits : int or tuple[int, ...]
            The index or indices of the target qudits.
        dagger : bool, default=False
            If True, apply the inverse of the gate.

        Raises
        ------
        IndexError
            If qudit indices are out of range.
        ValueError
            If the number of qudits does not match the gate's requirement.
        """
        if type(qudits) is int:
            qudits = (qudits,)
        if any(q < 0 or q >= self.n for q in qudits):
            raise IndexError("qudit index out of range")
        if len(qudits) != gate.n_qudits:
            raise ValueError(
                f"Gate {gate.name} expects {gate.n_qudits} qudits, got {len(qudits)}"
            )
        if gate.n_qudits == 2 and qudits[0] == qudits[1]:
            raise ValueError(
                f"Cannot apply {gate.name} to the same qudit (#{qudits[0]}) twice"
            )
        self._ops.append((gate, qudits, dagger))

    def insert(
        self,
        gate: Gate,
        qudits: Union[int, tuple[int, ...]],
        column: int,
        dagger: bool = False,
    ) -> None:
        """
        Insert a gate at a specific ASAP-scheduled column index.

        This method is layout-aware: it shifts subsequent gates in the visual
        schedule to the right.

        Parameters
        ----------
        gate : Gate
            The gate to insert.
        qudits : int or tuple[int, ...]
            Target qudits.
        column : int
            The zero-based column index.
        dagger : bool, default=False
            If True, apply the dagger version.

        Raises
        ------
        ValueError
            If columns are negative or qudit count mismatches.
        IndexError
            If qudit index is out of range.
        """
        qts = (qudits,) if isinstance(qudits, int) else qudits
        if len(qts) != gate.n_qudits:
            raise ValueError(
                f"Gate {gate.name} expects {gate.n_qudits} qudits, got {len(qts)}"
            )
        if any(q < 0 or q >= self.n for q in qts):
            raise IndexError("qudit index out of range")
        if column < 0:
            raise ValueError(f"Column must be >=0, got {column}")

        cols = self._asap_schedule()

        while len(cols) <= column:
            cols.append([])

        cols.insert(column, [(gate, qts, dagger)])

        new_ops: list[tuple[Gate, tuple[int, ...], bool]] = []
        for col_ops in cols:
            new_ops.extend(col_ops)

        self._ops = new_ops

    def replace(
        self,
        gate: Gate,
        qudits: Union[int, tuple[int, ...]],
        column: int,
        dagger: bool = False,
    ) -> None:
        """
        Replace an existing gate or insert a new one at the given ASAP-scheduled column.

        Parameters
        ----------
        gate : Gate
            The new gate instance.
        qudits : int or tuple[int, ...]
            Target qudits.
        column : int
            Zero-based ASAP-scheduled column index.
        dagger : bool, default=False
            If True, apply dagger version.

        Raises
        ------
        ValueError
            If column is out of range or gate dimensions mismatch.
        IndexError
            If any qudit index is out of range.
        """
        if isinstance(qudits, int):
            qts = (qudits,)
        else:
            qts = qudits
        if len(qts) != gate.n_qudits:
            raise ValueError(
                f"Gate {gate.name} expects {gate.n_qudits} qudits, got {len(qts)}"
            )
        for q in qts:
            if not (0 <= q < self.n):
                raise IndexError(f"Qudit index {q} out of range 0..{self.n-1}")
        if column < 0:
            raise ValueError(f"Column index must be ≥0, got {column}")

        cols = self._asap_schedule()
        if column >= len(cols):
            raise ValueError(f"Column {column} is out of range (0..{len(cols)-1})")

        col_ops = cols[column]
        removed_indices = [
            idx
            for idx, (_, existing_qidx, _) in enumerate(col_ops)
            if any(q in existing_qidx for q in qts)
        ]
        if removed_indices:
            insert_pos = removed_indices[0]
            for idx in reversed(removed_indices):
                col_ops.pop(idx)
        else:
            insert_pos = len(col_ops)

        col_ops.insert(insert_pos, (gate, tuple(qts), dagger))

        new_ops: list[tuple[Gate, tuple[int, ...], bool]] = []
        for ops in cols:
            new_ops.extend(ops)
        self._ops = new_ops

    def delete(self, qudit: int, column: int) -> None:
        """
        Delete the gate at the specified ASAP-scheduled column on the given qudit.
        If the qudit is part of a multi-qudit gate, that entire gate is deleted.

        Parameters
        ----------
        qudit : int
            Wire index to target.
        column : int
            Zero-based ASAP-scheduled column index.

        Raises
        ------
        ValueError
            If column invalid or no gate at that position.
        IndexError
            If qudit index out of range.
        """
        if not (0 <= qudit < self.n):
            raise IndexError(f"Qudit index {qudit} out of range 0..{self.n-1}")
        if column < 0:
            raise ValueError(f"Column index must be ≥0, got {column}")

        cols = self._asap_schedule()
        if column >= len(cols):
            raise ValueError(f"Column {column} is out of range 0..{len(cols)-1}")

        col_ops = cols[column]
        for idx, (_, qidx, _) in enumerate(col_ops):
            if qudit in qidx:
                col_ops.pop(idx)
                new_ops: list[tuple[Gate, tuple[int, ...], bool]] = []
                for ops in cols:
                    new_ops.extend(ops)
                self._ops = new_ops
                return
        raise ValueError(f"No gate to delete at column {column} on qudit {qudit}")

    # -----------------------------------------------------------------------
    # Op-List Utilities (indexing on self._ops)
    # -----------------------------------------------------------------------

    def find(self, name: str) -> list[tuple[int, tuple[int, ...]]]:
        """
        Return all indices i and qudits qs such that self._ops[i][0].name == name.
        Matching is exact (case-sensitive).

        Parameters
        ----------
        name : str
            Name of the gate to find.

        Returns
        -------
        list[tuple[int, tuple[int, ...]]]
            List of tuples (index, qudit(s)) where Gate name matches the provided name.

        NOTE: This operates on program order (the raw op list), not on ASAP columns.
        """
        return [(i, qs) for i, (g, qs, _) in enumerate(self._ops) if g.name == name]

    def insert_at_position(
        self,
        pos: int,
        gate: Gate,
        qudits: Union[int, tuple[int, ...]],
        dagger: bool = False,
    ) -> None:
        """
        Insert a gate at a specific list index in the operations list.

        Parameters
        ----------
        pos : int
            The list index at which to insert.
        gate : Gate
            The gate to insert.
        qudits : int or tuple[int, ...]
            Target qudits.
        dagger : bool, default=False
            If True, apply the dagger version.

        Raises
        ------
        TypeError
            If position is not int.
        ValueError
            If qudit number mismatch occured or an attempt to
            place a 2-qudit gate on a single wire.
        IndexError
            If position or qudit number is out of range.

        NOTE: This operates on program order (the raw op list), not on ASAP columns.
        """
        if not isinstance(pos, int):
            raise TypeError("pos must be an int")
        if pos < 0 or pos > len(self._ops):
            raise IndexError(f"pos {pos} out of range 0..{len(self._ops)}")

        # normalize & validate targets
        if isinstance(qudits, int):
            qudits = (qudits,)
        if any(q < 0 or q >= self.n for q in qudits):
            raise IndexError("qudit index out of range")
        if len(qudits) != gate.n_qudits:
            raise ValueError(
                f"Gate {gate.name} expects {gate.n_qudits} qudits, got {len(qudits)}"
            )
        if gate.n_qudits == 2 and qudits[0] == qudits[1]:
            raise ValueError(
                f"Cannot apply 2-qudit gate {gate.name} to the same wire twice"
            )

        self._ops.insert(pos, (gate, tuple(qudits), bool(dagger)))

    def replace_at_position(
        self,
        pos: int,
        gate: Gate,
        qudits: Union[int, tuple[int, ...]],
        dagger: bool = False,
    ) -> None:
        """
        Replace the operation at a specific list index.

        Parameters
        ----------
        pos : int
            The list index to replace.
        gate : Gate
            The new gate.
        qudits : int or tuple[int, ...]
            Target qudits.
        dagger : bool, default=False
            If True, apply the dagger version.

        Raises
        ------
        TypeError
            If position is not int.
        ValueError
            If qudit number mismatch occured or an attempt to
            place a 2-qudit gate on a single wire.
        IndexError
            If position or qudit number is out of range.

        NOTE: This operates on program order (the raw op list), not on ASAP columns.
        """
        if not isinstance(pos, int):
            raise TypeError("pos must be an int")
        if pos < 0 or pos >= len(self._ops):
            raise IndexError(f"pos {pos} out of range 0..{len(self._ops)-1}")

        # normalize & validate targets
        if isinstance(qudits, int):
            qudits = (qudits,)
        if any(q < 0 or q >= self.n for q in qudits):
            raise IndexError("qudit index out of range")
        if len(qudits) != gate.n_qudits:
            raise ValueError(
                f"Gate {gate.name} expects {gate.n_qudits} qudits, got {len(qudits)}"
            )
        if gate.n_qudits == 2 and qudits[0] == qudits[1]:
            raise ValueError(
                f"Cannot apply 2-qudit gate {gate.name} to the same wire twice"
            )

        self._ops[pos] = (gate, tuple(qudits), bool(dagger))

    def delete_at_position(self, pos: int) -> None:
        """
        Delete the operation at a specific list index.

        Parameters
        ----------
        pos : int
            The list index to delete.

        Raises
        ------
        TypeError
            If position is not int.
        IndexError
            If position is out of range.

        NOTE: This operates on program order (the raw op list), not on ASAP columns.
        """
        if not isinstance(pos, int):
            raise TypeError("pos must be an int")
        if pos < 0 or pos >= len(self._ops):
            raise IndexError(f"pos {pos} out of range 0..{len(self._ops)-1}")
        del self._ops[pos]

    # -----------------------------------------------------------------------
    # Global Circuit Methods
    # -----------------------------------------------------------------------

    def compose(self, other: QuantumCircuit, in_place: bool = False) -> QuantumCircuit:
        """
        Append another circuit to this one.

        If the other circuit has more qudits, the circuit is extended.

        Parameters
        ----------
        other : QuantumCircuit
            The circuit to append.
        in_place : bool, default=False
            If True, modifies this circuit. If False, returns a new one.

        Returns
        -------
        QuantumCircuit
            The composed circuit.

        Raises
        ------
        ValueError
            If circuit dimensions (d) mismatch.
        """
        if self.d != other.d:
            raise ValueError(
                f"Cannot compose circuits with different dimensions: {self.d} vs {other.d}"
            )

        ## Select the target circuit
        if in_place:
            target_circuit = self
        else:
            target_circuit = self.copy()

        # Adjust qudit count
        target_circuit.n = max(self.n, other.n)

        # Append gates from the second circuit
        for gate, qidx, dagger in other._ops:
            if isinstance(gate, NoiseChannel):
                # Use add_noise_channel to merge the noise model
                # into the target circuit's registry.
                model = other.noise_registry[gate.model_id]
                model_reshaped = model.reshape((other.d, other.d))
                target_circuit.add_noise_channel(
                    qidx[0], model_reshaped, color=gate.color
                )
            else:
                target_circuit.append(gate, qidx, dagger)

        return target_circuit

    def compose_vertical(
        self, other: QuantumCircuit, in_place: bool = False
    ) -> QuantumCircuit:
        """
        Vertically compose two circuits by stacking `other`'s qudits below `self`.

        Parameters
        ----------
        other : QuantumCircuit
            The circuit to stack below.
        in_place : bool, default=False
            If True, modifies this circuit. If False, returns a new one.

        Returns
        -------
        QuantumCircuit
            The composed circuit.

        Raises
        ------
        ValueError
            If circuit dimensions (d) mismatch.
        """
        if self.d != other.d:
            raise ValueError(
                f"Cannot vertically compose circuits with different dimensions: {self.d} vs {other.d}"
            )

        ## Select the target circuit
        if in_place:
            target_circuit = self
        else:
            target_circuit = self.copy()

        offset = self.n

        # Update total qudit count
        target_circuit.n = self.n + other.n

        # Append gates from the second circuit with shifted qudit indices
        for gate, qidx, dagger in other._ops:
            shifted_qidx = tuple(q + offset for q in qidx)
            if isinstance(gate, NoiseChannel):
                model = other.noise_registry[gate.model_id]
                model_reshaped = model.reshape((other.d, other.d))
                target_circuit.add_noise_channel(
                    shifted_qidx[0], model_reshaped, color=gate.color
                )
            else:
                target_circuit.append(gate, shifted_qidx, dagger)

        return target_circuit

    def reverse(self, in_place: bool = False) -> QuantumCircuit:
        """
        Reverse the circuit (gates reversed in order and daggered).

        Parameters
        ----------
        in_place : bool, default=False
            If True, modifies this circuit. If False, returns a new one.

        Returns
        -------
        QuantumCircuit
            The reversed circuit.
        """
        new_ops = []
        temp_qc_for_registry = QuantumCircuit(self.n, self.d)

        # Rebuild the circuit in reverse, preserving noise models
        for gate, qidx, dagger in reversed(self._ops):
            if isinstance(gate, NoiseChannel):
                # Re-add the noise channel
                model = self.noise_registry[gate.model_id]
                model_reshaped = model.reshape((self.d, self.d))
                noise_instance = temp_qc_for_registry._create_noise_channel_instance(
                    model_reshaped, color=gate.color
                )
                new_ops.append((noise_instance, qidx, dagger))
            else:
                # Add the daggered version of the original gate
                new_ops.append((gate, qidx, not dagger))

        if in_place:
            target_circuit = self
        else:
            target_circuit = QuantumCircuit(self.n, self.d, name=self.name)

        target_circuit._ops = new_ops
        target_circuit.noise_registry = temp_qc_for_registry.noise_registry
        target_circuit._noise_model_cache = temp_qc_for_registry._noise_model_cache
        target_circuit._next_noise_model_id = temp_qc_for_registry._next_noise_model_id

        return target_circuit

    # -----------------------------------------------------------------------
    # Gate Addition Methods
    # -----------------------------------------------------------------------

    # --- 1-qudit Gates ---

    def I(self, q: int) -> None:
        """Add I on qudit q.

        Parameters
        ----------
        q : int
            Qudit index.
        """
        return self.append(I_gate, q)

    def Idag(self, q: int) -> None:
        """Add I† on qudit q.

        Parameters
        ----------
        q : int
            Qudit index.
        """
        return self.append(I_gate, q, dagger=True)

    def X(self, q: int) -> None:
        """Add X on q.

        Parameters
        ----------
        q : int
            Qudit index.
        """
        return self.append(X_gate, q)

    def Xdag(self, q: int) -> None:
        """Add X† on q.

        Parameters
        ----------
        q : int
            Qudit index.
        """
        return self.append(X_gate, q, dagger=True)

    def Y(self, q: int) -> None:
        """Add Y on q.

        Parameters
        ----------
        q : int
            Qudit index.
        """
        return self.append(Y_gate, q)

    def Ydag(self, q: int) -> None:
        """Add Y† on q.

        Parameters
        ----------
        q : int
            Qudit index.
        """
        return self.append(Y_gate, q, dagger=True)

    def Z(self, q: int) -> None:
        """Add Z on q.

        Parameters
        ----------
        q : int
            Qudit index.
        """
        return self.append(Z_gate, q)

    def Zdag(self, q: int) -> None:
        """Add Z† on q.

        Parameters
        ----------
        q : int
            Qudit index.
        """
        return self.append(Z_gate, q, dagger=True)

    def H(self, q: int) -> None:
        """Add H on q.

        Parameters
        ----------
        q : int
            Qudit index.
        """
        return self.append(H_gate, q)

    def Hdag(self, q: int) -> None:
        """Add H† on q.

        Parameters
        ----------
        q : int
            Qudit index.
        """
        return self.append(H_gate, q, dagger=True)

    def S(self, q: int) -> None:
        """Add S on q.

        Parameters
        ----------
        q : int
            Qudit index.
        """
        return self.append(S_gate, q)

    def Sdag(self, q: int) -> None:
        """Add S† on q.

        Parameters
        ----------
        q : int
            Qudit index.
        """
        return self.append(S_gate, q, dagger=True)

    def _W(self, a: int, b: int, color: str = "#FF0000") -> Gate:
        """
        Helper method for QuantumCircuit.W and Wdag.

        Parameters
        ----------
        a : int
            Parameter 'a' for Weyl gate.
        b : int
            Parameter 'b' for Weyl gate.
        color : str, optional
            Color for visualization, by default "#FF0000".

        Returns
        -------
        Gate
            Weyl operator Gate object.
        """
        a = a % self.d
        b = b % self.d

        weyl_gate = W_gate.copy(name=f"W_{a},{b}", color=color)

        weyl_gate.a = a
        weyl_gate.b = b

        weyl_gate.matrix_fn = lambda d_val: gates.W(d_val, a, b)
        weyl_gate.dagger_matrix_fn = lambda d_val: gates.W_dag(d_val, a, b)

        return weyl_gate

    def W(self, q: int, a: int, b: int, color: str = "#FF0000") -> None:
        """Add Weyl(a,b) on q.

        Parameters
        ----------
        q : int
            Qudit index.
        a : int
            Parameter 'a' for Weyl gate.
        b : int
            Parameter 'b' for Weyl gate.
        color : str, optional
            Color for visualization, by default "#FF0000".
        """
        weyl_gate = self._W(a, b, color=color)
        return self.append(weyl_gate, q, dagger=False)

    def Wdag(self, q: int, a: int, b: int, color: str = "#FF0000") -> None:
        """Add Weyl†(a,b) on q.

        Parameters
        ----------
        q : int
            Qudit index.
        a : int
            Parameter 'a' for Weyl gate.
        b : int
            Parameter 'b' for Weyl gate.
        color : str, optional
            Color for visualization, by default "#FF0000".
        """
        weyl_gate = self._W(a, b, color=color)
        return self.append(weyl_gate, q, dagger=True)

    def M(self, q: int) -> None:
        """Add measurement on q.

        Parameters
        ----------
        q : int
            Qudit index.
        """
        return self.append(M_gate, q)

    def P(self, q: int) -> None:
        """Add a placeholder gate on q.

        Parameters
        ----------
        q : int
            Qudit index.
        """
        return self.append(P_gate, q)

    def Pdag(self, q: int) -> None:
        """Add a daggered placeholder gate on q.

        Parameters
        ----------
        q : int
            Qudit index.
        """
        return self.append(P_gate, q, dagger=True)

    # --- 2-qudit Gates ---

    def CNOT(self, c: int, t: int) -> None:
        """Add CNOT(c,t).

        Parameters
        ----------
        c : int
            Control qudit index.
        t : int
            Target qudit index.
        """
        return self.append(CNOT_gate, (c, t))

    def CNOTdag(self, c: int, t: int) -> None:
        """Add CNOT†(c,t).

        Parameters
        ----------
        c : int
            Control qudit index.
        t : int
            Target qudit index.
        """
        return self.append(CNOT_gate, (c, t), dagger=True)

    def CX(self, c: int, t: int) -> None:
        """Alias of CNOT(c,t).

        Parameters
        ----------
        c : int
            Control qudit index.
        t : int
            Target qudit index.
        """
        return self.append(CNOT_gate, (c, t))

    def CXdag(self, c: int, t: int) -> None:
        """Alias of CNOT†(c,t).

        Parameters
        ----------
        c : int
            Control qudit index.
        t : int
            Target qudit index.
        """
        return self.append(CNOT_gate, (c, t), dagger=True)

    def CZ(self, c: int, t: int) -> None:
        """Add CZ(c,t).

        Parameters
        ----------
        c : int
            Control qudit index.
        t : int
            Target qudit index.
        """
        return self.append(CZ_gate, (c, t))

    def CZdag(self, c: int, t: int) -> None:
        """Add CZ†(c,t).

        Parameters
        ----------
        c : int
            Control qudit index.
        t : int
            Target qudit index.
        """
        return self.append(CZ_gate, (c, t), dagger=True)

    def SWAP(self, q1: int, q2: int) -> None:
        """Add SWAP(q1,q2).

        Parameters
        ----------
        q1 : int
            First qudit index.
        q2 : int
            Second qudit index.
        """
        return self.append(SWAP_gate, (q1, q2))

    def SWAPdag(self, q1: int, q2: int) -> None:
        """Add SWAP†(q1,q2).

        Parameters
        ----------
        q1 : int
            First qudit index.
        q2 : int
            Second qudit index.
        """
        return self.append(SWAP_gate, (q1, q2), dagger=True)

    def GROUP(self, q1: int, q2: int) -> None:
        """Add a placeholder group gate on qudits q1,q2.

        Parameters
        ----------
        q1 : int
            First qudit index of the group.
        q2 : int
            Last qudit index of the group.
        """
        return self.append(GROUP_gate, (q1, q2))

    # -----------------------------------------------------------------------
    # Simulation Methods
    # -----------------------------------------------------------------------

    # --- State Vector Simulation ---

    def measure(self, qudits: Union[int, tuple[int, ...]]) -> None:
        """
        Add a measurement operation to the circuit for specific qudit(s).

        This operation corresponds to a standard computational basis measurement.

        Parameters
        ----------
        qudits : int or tuple[int, ...]
            The integer index or a tuple of indices of the qudits to measure.

        Raises
        ------
        IndexError
            If any qudit index is out of the valid range [0, n-1].
        """
        if isinstance(qudits, int):
            qudits = (qudits,)
        for q in qudits:
            if not (0 <= q < self.n):
                raise IndexError(f"Qudit index {q} is out of range.")
            self.append(M_gate, (q,))

    def measure_all(self) -> None:
        """
        Add measurement operations for all qudits in the circuit.

        This is a convenience method that iterates through all qudits from 0 to n-1
        and appends a measurement gate for each.
        """
        for qudit in range(self.n):
            self.measure(qudit)

    def _perform_measurement(
        self, psi: np.ndarray, n: int, d: int, qudit: int
    ) -> tuple[np.ndarray, int]:
        """
        Perform a projective measurement on a single qudit for a statevector.

        This method calculates the probabilities of each outcome for the specified
        qudit, samples an outcome based on these probabilities, collapses the
        statevector to the subspace consistent with the outcome, and renormalizes it.

        Parameters
        ----------
        psi : np.ndarray
            The current statevector of size d^n.
        n : int
            The number of qudits in the system.
        d : int
            The dimension of each qudit.
        qudit : int
            The index of the qudit to measure.

        Returns
        -------
        tuple[np.ndarray, int]
            A tuple containing:
            - np.ndarray: The new, collapsed, and normalized statevector.
            - int: The measurement outcome (integer in [0, d-1]).
        """
        psi_nd = psi.reshape([d] * n)

        axes_to_sum = tuple(i for i in range(n) if i != qudit)
        probs = np.sum(np.abs(psi_nd) ** 2, axis=axes_to_sum)
        probs /= np.sum(probs)

        measured_val = np.random.choice(np.arange(d, dtype=np.int32), p=probs)

        new_psi_nd = np.zeros_like(psi_nd)
        sl = [slice(None)] * n
        sl[qudit] = measured_val
        new_psi_nd[tuple(sl)] = psi_nd[tuple(sl)]

        norm = np.linalg.norm(new_psi_nd)
        if norm > 1e-9:
            new_psi_nd /= norm

        non_zero_indices = np.where(np.abs(new_psi_nd.flatten()) > 1e-9)[0]
        if len(non_zero_indices) > 0:
            first_non_zero_idx = non_zero_indices[0]
            phase_value = new_psi_nd.flatten()[first_non_zero_idx]
            new_psi_nd /= phase_value

        return new_psi_nd.reshape(d**n), int(measured_val)

    def sample(
        self,
        shots: int,
        psi0: Optional[np.ndarray] = None,
        normalize: bool = True,
        seed: Optional[int] = None,
        return_digits: bool = True,
    ) -> Union[
        tuple[np.ndarray, np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]
    ]:
        """
        Execute the circuit once to obtain the final statevector, then sample measurement outcomes.

        This method performs a single statevector simulation and computes computational-basis
        probabilities p_k = |psi_k|^2 over all d**n joint outcomes. It then draws `shots`
        independent samples from this discrete distribution (with replacement).

        Parameters
        ----------
        shots : int
            Number of measurement samples to draw. Must be positive.
        psi0 : np.ndarray, optional
            Initial statevector of length d**n. If None, the zero state |0...0> is used.
        normalize : bool, default=True
            If True, normalizes the final statevector before computing probabilities.
        seed : int, optional
            Seed for the random number generator.
        return_digits : bool, default=True
            If True, also return per-qudit outcomes as base-d digits with shape (shots, n).

        Returns
        -------
        psi : np.ndarray
            Final complex statevector of shape (d**n,).
        digits : np.ndarray
            Integer outcomes per qudit with shape (shots, n), values in [0, d-1].
            Only returned if `return_digits` is True.
        flat : np.ndarray
            Flat outcome indices with shape (shots,), values in [0, d**n - 1].

        Raises
        ------
        ValueError
            If `shots <= 0`.
        """
        if shots <= 0:
            raise ValueError(f"`shots` must be positive, got {shots}.")

        psi, _ = self.execute(psi0=psi0, normalize=normalize)
        psi = np.asarray(psi)

        probs = np.abs(psi) ** 2
        s = probs.sum()
        if not np.isclose(s, 1.0):
            if s == 0:
                raise ValueError("Final statevector has zero norm; cannot sample.")
            probs = probs / s

        rng = np.random.default_rng(seed)
        flat = rng.choice(probs.size, size=shots, replace=True, p=probs).astype(
            np.int64, copy=False
        )

        if not return_digits:
            return psi, flat

        # Convert flat indices to base-d digits per qudit (q0,…,q_{n-1}); shape -> (shots, n)
        digits_tuple = np.unravel_index(flat, (self.d,) * self.n)
        digits = np.column_stack(digits_tuple).astype(np.int64, copy=False)

        return psi, digits, flat

    def execute(
        self,
        psi0: Optional[np.ndarray] = None,
        normalize: bool = True,
        print_state: bool = False,
        mid_circuit: bool = False
    ) -> tuple[np.ndarray, dict[int, int]]:
        """
        Execute the circuit using statevector simulation.

        The circuit is simulated once starting from ``psi0`` (or ``|0...0>`` if
        ``psi0`` is None). Gates are applied in order. Placeholder measurement
        operations are recognized by ``gate.name == "M"`` and trigger a computational
        basis measurement on the addressed qudit.

        Parameters
        ----------
        psi0 : np.ndarray, optional
            Initial statevector of shape ``(d**n,)``. If None, the simulation starts
            in the computational basis state ``|0...0>``.
        normalize : bool, default=True
            If True, renormalize the final statevector before returning it.
        print_state : bool, default=False
            If True, print the statevector at the start and after each operation.
        mid_circuit : bool, default=False
            If True, prohibits overwriting of already measured qubits.
            Enables mid-circuit-measurements.

        Returns
        -------
        psi : np.ndarray
            Final statevector of shape ``(d**n,)``.
        classical_results : dict[int, int]
            Dictionary mapping ``{measured_qudit_index: outcome}`` for each
            measurement encountered in the circuit. If a qudit is measured multiple
            times, later outcomes overwrite earlier ones. Executing with a set time_flag
            disables this, allowing for mid-circuit-measurements.

        Raises
        ------
        ValueError
            If ``normalize=True`` and the final statevector has zero norm.
        Exception
            Any exception raised by gate application (``gate.apply``) or measurement
            (``self._perform_measurement``) is propagated.
        """
        n, d = self.n, self.d
        size = d**n
        measure_count = 0

        if mid_circuit and self._stored_psi is not None:
            psi0 = self._stored_psi.copy()

        psi = (
            np.zeros(size, dtype=backend.COMPLEX_DTYPE)
            if psi0 is None
            else psi0.copy().astype(backend.COMPLEX_DTYPE)
        )

        classical_results = {}

        if psi0 is None:
            psi[0] = 1

        if print_state:
            print(f"Initial state:\n{psi}")

        for i in range(self._gate_idx, len(self._ops)):
            gate, qidx, dag = self._ops[i]
            if gate.matrix_fn is None and gate._apply_fn is None:
                if gate.name == "M":
                    measured_qudit = qidx[0]
                    psi, outcome = self._perform_measurement(psi, n, d, measured_qudit)
                    key = measured_qudit
                    classical_results[key] = outcome
                    if print_state:
                        print(f"Measured qudit   {measured_qudit}, outcome={outcome}")
                    if mid_circuit:
                        measure_count += 1
                        psi = self._normalize_state(psi, normalize)
                        self._gate_idx = i + 1
                        if measure_count >= sum(gate.name == "M" for gate, _, _ in self._ops):
                            self._stored_psi = psi.copy()
                            return psi, classical_results
                else:
                    if print_state:
                        print(
                            f"Skipping placeholder gate {gate.name} on qudit(s) {', '.join(map(str, qidx))}"
                        )
                    continue
            else:
                psi = gate.apply(psi, n, d, qidx, dag)
                if print_state:
                    print(
                        f"\nState after {gate.name}{'†' if dag else ''}"
                        f" on qudit{'s' if len(qidx) > 1 else ''}"
                        f" {', '.join(map(str, qidx))}:\n{psi}"
                    )
        psi = self._normalize_state(psi, normalize)
        self._gate_idx = 0
        self._stored_psi = None
        return psi, classical_results

    def _normalize_state(self, psi, normalize):
        if normalize:
            nrm = np.linalg.norm(psi)
            if nrm == 0:
                raise ValueError("Normalization error: final statevector is zero.")
            return (psi / nrm).astype(backend.COMPLEX_DTYPE, copy=False)
        else:
            return psi.astype(backend.COMPLEX_DTYPE, copy=False)

    # --- Density Matrices Simulation ---

    def execute_density_matrix(
        self, initial_state: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Execute the circuit using density-matrix simulation.

        This method evolves a density matrix ``rho`` through the circuit.
        Unitary gates are applied as::

            rho' = U @ rho @ U†

        Noise channels are handled by the gate's ``apply_density``
        implementation (e.g., via Kraus operators).

        Parameters
        ----------
        initial_state : np.ndarray, optional
            Initial state for the simulation. Accepted forms are:

            - None:
            Start in the pure state ``|0...0><0...0|``.
            - 1D array of shape ``(d**n,)``:
            Interpreted as a pure statevector ``|psi>``, initialized as
            ``rho = |psi><psi|``.
            - 2D array of shape ``(d**n, d**n)``:
            Interpreted as an explicit density matrix ``rho``.

        Returns
        -------
        rho : np.ndarray
            Final density matrix of shape ``(d**n, d**n)``.

        Raises
        ------
        ValueError
            If ``initial_state`` has an unsupported shape or rank.
        Exception
            Any exception raised by ``gate.apply_density`` is propagated.
        """
        n, d = self.n, self.d
        size = d**n

        # --- Initialize the density matrix ---
        if initial_state is None:
            rho = np.zeros((size, size), dtype=backend.COMPLEX_DTYPE)
            rho[0, 0] = 1
        elif initial_state.ndim == 1:  # It's a statevector
            psi = initial_state.astype(backend.COMPLEX_DTYPE)
            rho = np.outer(psi, np.conjugate(psi))
        elif initial_state.ndim == 2 and initial_state.shape == (size, size):
            rho = initial_state.astype(backend.COMPLEX_DTYPE)
        else:
            raise ValueError(
                f"Invalid initial_state. Must be a vector of size {size} or a matrix of size ({size}, {size})."
            )

        # Clear the cache at the beginning of each run to ensure freshness.
        self._kraus_operator_cache.clear()

        # --- Evolve the density matrix ---
        for gate, qidx, dag in self:
            rho = gate.apply_density(rho, self, qidx, dag)

        return rho

    # --- Full Circuit Unitary Matrix ---

    def get_unitary(self) -> np.ndarray:
        """
        Construct the full unitary matrix of the circuit.

        The result is a matrix of shape ``(d**n, d**n)``, constructed by
        left-multiplying each gate's full-system operator in circuit order.

        Placeholder operations (gates with both ``matrix_fn`` and ``_apply_fn`` set
        to None) are skipped.

        Returns
        -------
        U_total : np.ndarray
            Dense unitary matrix of shape ``(d**n, d**n)``.

        Raises
        ------
        Exception
            Any exception raised while constructing or applying a gate operator is
            propagated (e.g., from ``gate.apply`` inside ``_get_gate_unitary``).
        """
        n, d = self.n, self.d
        U_total = np.eye(d**n, dtype=backend.COMPLEX_DTYPE)

        for gate, qudits, dagger in self._ops:
            if gate.matrix_fn is None and gate._apply_fn is None:
                continue
            U_gate = self._get_gate_unitary(gate, qudits, dagger)
            U_total = U_gate @ U_total

        return U_total

    def _get_gate_unitary(
        self, gate: Gate, qudits: tuple[int, ...], dagger: bool
    ) -> np.ndarray:
        """
        Construct the full-system operator for a single gate.

        The operator is constructed by applying the gate to each computational basis
        vector ``|i>`` and stacking the resulting output states as columns.

        Parameters
        ----------
        gate : Gate
            The gate to convert into a full-system operator.
        qudits : tuple[int, ...]
            Target qudit indices the gate acts on.
        dagger : bool
            If True, apply the dagger of the gate.

        Returns
        -------
        U_gate : np.ndarray
            Dense operator matrix of shape ``(d**n, d**n)``.

        Raises
        ------
        Exception
            Any exception raised by ``gate.apply`` is propagated.
        """
        n, d = self.n, self.d
        size = d**n

        U_gate = np.zeros((size, size), dtype=backend.COMPLEX_DTYPE)

        for i in range(size):
            basis_vec = np.zeros(size, dtype=backend.COMPLEX_DTYPE)
            basis_vec[i] = 1

            U_gate[:, i] = gate.apply(basis_vec, n, d, qudits, dagger)

        return U_gate

    # -----------------------------------------------------------------------
    # Visualization
    # -----------------------------------------------------------------------

    def draw(
        self,
        scheduling: str = "asap",
        font_color: str = "black",
        show_edges: bool = False,
        gate_colors: tuple[str] = (
            "#EFBB04",  # Pauli
            "#0DB8B8",  # H/S
            "#4A90E2",  # CNOT
            "#9B59B6",  # CZ
            "#878CED",  # Other
        ),
        save_path: Optional[str] = None,
        dpi: int = 300,
        show_title: bool = False,
        show_info: bool = False,
        show_image: bool = True,
    ) -> None:
        """
        Draws a circuit diagram using matplotlib.

        Parameters
        ----------
        scheduling : str, default="asap"
            Layout strategy:
            - "asap": Packs gates as left as possible (visual layout).
            - "asap_packed": Creates maximally parallelized layers (logical depth).
            - "logical": Keeps the order gates were appended (one gate per column).
        font_color : str, default="black"
            Color for gate labels.
        show_edges : bool, default=False
            If True, draws borders around gate boxes.
        gate_colors : tuple[str]
            Custom colors (hex) for gate categories: [Pauli, H/S, CNOT, CZ, Other].
        save_path : str or None, optional
            Path to save the figure (e.g., "circuit.png").
        dpi : int, default=300
            Resolution for raster formats (PNG).
        show_title : bool, default=False
            Whether to show the circuit name as a title.
        show_info : bool, default=False
            Whether to show circuit stats (n, d, size, depth) as a title.
        show_image : bool, default=True
            Whether to display the image (plt.show()).
        """
        from .visualization import draw_circuit

        return draw_circuit(
            self,
            scheduling=scheduling,
            font_color=font_color,
            show_edges=show_edges,
            gate_colors=gate_colors,
            save_path=save_path,
            dpi=dpi,
            show_title=show_title,
            show_info=show_info,
            show_image=show_image,
        )

    # -----------------------------------------------------------------------
    # Noise Modeling
    # -----------------------------------------------------------------------

    def _prepare_weyl_probabilities(
        self, probs: Union[np.ndarray, float]
    ) -> np.ndarray:
        """
        Validate and normalize a Weyl error probability model.

        The circuit dimension ``d = self.d`` defines the Weyl operator set
        ``{W_{a,b}}`` with ``a,b in {0,...,d-1}``. This method accepts either:

        - A single scalar ``p`` for isotropic noise:
        ``P[0,0] = 1 - p`` and all other ``P[a,b] = p / (d^2 - 1)``.
        - A full ``(d, d)`` array ``P`` where ``P[a,b]`` is the probability of
        applying ``W_{a,b}``.

        The resulting model is normalized to sum to 1.

        Parameters
        ----------
        probs : float or np.ndarray
            Noise model specification. Either a scalar probability ``p`` or an
            array of shape ``(d, d)``.

        Returns
        -------
        np.ndarray
            Flattened probability vector of length ``d*d`` in row-major order.

        Raises
        ------
        ValueError
            If ``p`` is not in ``[0, 1]``, if the matrix has the wrong shape, if any
            probability is negative, or if the sum of probabilities is zero.
        """
        if isinstance(probs, (float, int)):
            p = float(probs)
            if not (0.0 <= p <= 1.0):
                raise ValueError("Error probability p must be between 0 and 1.")

            num_errors = self.d * self.d - 1
            if num_errors == 0:  # d=1 case
                return np.array([1.0])

            prob_per_error = p / num_errors
            p_matrix = np.full((self.d, self.d), prob_per_error, dtype=np.float64)
            p_matrix[0, 0] = 1.0 - p

        else:  # It's a matrix
            p_matrix = np.asarray(probs, dtype=np.float64)
            if p_matrix.shape != (self.d, self.d):
                raise ValueError(
                    f"Probability matrix must have shape ({self.d}, {self.d}), but got {p_matrix.shape}"
                )
            if np.any(p_matrix < 0):
                raise ValueError("Probabilities cannot be negative.")

        total_prob = np.sum(p_matrix)
        if not np.isclose(total_prob, 1.0):
            if total_prob == 0:
                raise ValueError("Sum of probabilities cannot be zero.")
            p_matrix /= total_prob  # Normalize

        return p_matrix.flatten()

    def _set_color(
        self, normalized_probs: np.ndarray, color: str
    ) -> tuple[float, float, float, float]:
        """
        Compute an RGBA color whose opacity encodes total error probability.

        Opacity is set to ``1 - P(no error)``, where ``P(no error)`` is the first
        probability (corresponding to ``W_{0,0}``) in the flattened model.

        Parameters
        ----------
        normalized_probs : np.ndarray
            Flattened Weyl probability vector of length ``d*d`` (assumed normalized).
        color : str
            Base color specification accepted by Matplotlib.

        Returns
        -------
        tuple[float, float, float, float]
            RGBA tuple with alpha equal to the total error probability.

        Raises
        ------
        ValueError
            If ``color`` is not a valid Matplotlib color specification.
        """
        from matplotlib import colors as mcolors

        try:
            base_rgba = mcolors.to_rgba(color)
        except ValueError:
            raise ValueError(f"Invalid color format: '{color}'")

        # Calculate opacity from the total error probability.
        # The probability of NO error (W_00) is the first element.
        prob_no_error = normalized_probs[0]
        opacity = 1.0 - prob_no_error

        # Create the RGBA color tuple for the gate.
        final_color = (*base_rgba[:3], opacity)

        return final_color

    def _create_noise_channel_instance(
        self, noise_model: Union[np.ndarray, float], color: str
    ) -> NoiseChannel:
        """
        Create a NoiseChannel gate instance for a given noise model.

        This normalizes the provided model, interns it into ``self.noise_registry``
        (deduplicated via a cache), and returns a ``NoiseChannel`` gate that stores
        only a model ID pointing into the registry.

        Parameters
        ----------
        noise_model : float or np.ndarray
            Noise model specification (scalar ``p`` or a ``(d, d)`` probability matrix).
        color : str
            Base color specification for visualization; opacity is derived from the
            total error probability.

        Returns
        -------
        NoiseChannel
            A NoiseChannel placeholder gate referencing the interned noise model.

        Raises
        ------
        ValueError
            If the noise model is invalid or the color specification is invalid.
        """
        normalized_probs = self._prepare_weyl_probabilities(noise_model)
        model_key = tuple(normalized_probs)
        model_id = self._noise_model_cache.get(model_key)

        if model_id is None:
            model_id = self._next_noise_model_id
            self.noise_registry[model_id] = normalized_probs
            self._noise_model_cache[model_key] = model_id
            self._next_noise_model_id += 1

        final_color = self._set_color(normalized_probs, color)
        return NoiseChannel(model_id=model_id, color=final_color)

    def add_noise_channel(
        self, qudit: int, noise_model: Union[np.ndarray, float], color: str = "#FF0000"
    ) -> None:
        """
        Append a noise channel acting on a single qudit.

        Parameters
        ----------
        qudit : int
            Target qudit index.
        noise_model : float or np.ndarray
            Noise model specification:

            - If a scalar ``p``: isotropic Weyl noise with total error probability ``p``.
            - If an array ``P`` of shape ``(d, d)``: ``P[a,b]`` is the probability of
            applying ``W_{a,b}`` and ``P[0,0]`` is the probability of no error.

        color : str, default="#FF0000"
            Base color for visualization; opacity is set from the total error probability.

        Raises
        ------
        ValueError
            If ``qudit`` is not an integer, or if the noise model / color is invalid.
        IndexError
            If ``qudit`` is out of range.
        """
        if not isinstance(qudit, int):
            raise ValueError("Qudit index must be an integer.")
        if qudit < 0 or qudit >= self.n:
            raise IndexError("Qudit index out of range.")

        noise_gate_instance = self._create_noise_channel_instance(noise_model, color)
        self.append(noise_gate_instance, (qudit,))

    def insert_noise_channel(
        self,
        pos: Union[int, tuple[int, int]],
        qudit_or_model: Union[int, np.ndarray, float],
        model_or_none: Optional[Union[np.ndarray, float]] = None,
        color: str = "#FF0000",
    ) -> None:
        """
        Insert a noise channel either by ASAP coordinates or by program order.

        Two calling modes are supported:

        - ASAP-scheduled insertion (visual coordinates):
        ``insert_noise_channel((qudit, column), noise_model, color="...")``

        - Program-order insertion (linear op list):
        ``insert_noise_channel(position_index, qudit, noise_model, color="...")``

        Parameters
        ----------
        pos : int or tuple[int, int]
            Insertion position. If an int, this is the index into the program-order
            operation list. If a tuple, it must be ``(qudit, column)`` for ASAP layout.
        qudit_or_model : int or float or np.ndarray
            If ``pos`` is an int: the target qudit index.
            If ``pos`` is a tuple: the noise model.
        model_or_none : float or np.ndarray, optional
            Only used in program-order mode. The noise model.
        color : str, default="#FF0000"
            Base color for visualization.

        Raises
        ------
        TypeError
            If arguments do not match the selected mode.
        ValueError
            If tuple coordinates are malformed or column is negative, or if the noise
            model / color is invalid.
        IndexError
            If qudit or insertion position is out of range.
        """
        # --- Dispatching logic based on the type of `pos` ---
        if isinstance(pos, tuple):
            # --- ASAP-based Mode ---
            # Signature: insert_noise_channel(pos=(qudit, column), noise_model=qudit_or_model)
            if model_or_none is not None:
                raise TypeError(
                    "Too many arguments for ASAP-scheduled insertion. Expected: (pos_tuple, noise_model, color=...)."
                )

            target_qudit, column = pos
            noise_model = qudit_or_model

            if not (isinstance(target_qudit, int) and isinstance(column, int)):
                raise ValueError(
                    "If pos is a tuple, it must be of the form (qudit, column)."
                )
            if not (0 <= target_qudit < self.n):
                raise IndexError(f"Qudit index {target_qudit} is out of range.")
            if column < 0:
                raise ValueError(f"Column index must be >= 0, got {column}")

            noise_gate_instance = self._create_noise_channel_instance(
                noise_model, color
            )
            self.insert(noise_gate_instance, target_qudit, column)

        elif isinstance(pos, int):
            # --- Program-based Mode ---
            # Signature: insert_noise_channel(pos=pos, qudit=qudit_or_model, noise_model=model_or_none)
            program_pos = pos
            target_qudit = qudit_or_model
            noise_model = model_or_none

            if noise_model is None:
                raise TypeError(
                    "Missing `noise_model` argument for program-order insertion. Expected: (pos_int, qudit, noise_model, color=...)."
                )
            if not isinstance(target_qudit, int):
                raise TypeError(
                    f"For program-order insertion, the second argument must be the target qudit (an int), but got {type(target_qudit)}."
                )

            if not (0 <= program_pos <= len(self._ops)):
                raise IndexError(
                    f"Position {program_pos} is out of range for insertion (0 to {len(self._ops)})."
                )
            if not (0 <= target_qudit < self.n):
                raise IndexError("Qudit index out of range.")

            noise_gate_instance = self._create_noise_channel_instance(
                noise_model, color
            )
            self.insert_at_position(program_pos, noise_gate_instance, (target_qudit,))

        else:
            raise TypeError(
                "`pos` must be an integer or a tuple of two integers (qudit, column)."
            )

    def replace_noise_channel(
        self,
        pos: Union[int, tuple[int, int]],
        qudit_or_model: Union[int, np.ndarray, float],
        model_or_none: Optional[Union[np.ndarray, float]] = None,
        color: str = "#FF0000",
    ) -> None:
        """
        Replace an operation with a noise channel, by ASAP coordinates or program order.

        Two calling modes are supported:

        - ASAP-scheduled replacement (visual coordinates):
        ``replace_noise_channel((qudit, column), noise_model, color="...")``

        - Program-order replacement (linear op list):
        ``replace_noise_channel(position_index, qudit, noise_model, color="...")``

        Parameters
        ----------
        pos : int or tuple[int, int]
            Replacement position. If an int, this is the index into the program-order
            operation list. If a tuple, it must be ``(qudit, column)`` for ASAP layout.
        qudit_or_model : int or float or np.ndarray
            If ``pos`` is an int: the target qudit index.
            If ``pos`` is a tuple: the noise model.
        model_or_none : float or np.ndarray, optional
            Only used in program-order mode. The noise model.
        color : str, default="#FF0000"
            Base color for visualization.

        Raises
        ------
        TypeError
            If arguments do not match the selected mode.
        ValueError
            If tuple coordinates are malformed or column is negative, or if the noise
            model / color is invalid.
        IndexError
            If qudit or replacement position is out of range.
        """
        if isinstance(pos, tuple):
            # --- ASAP-based Mode ---
            if model_or_none is not None:
                raise TypeError(
                    "Too many arguments for ASAP-scheduled replacement. Expected: (pos_tuple, noise_model, color=...)."
                )

            target_qudit, column = pos
            noise_model = qudit_or_model

            if not (isinstance(target_qudit, int) and isinstance(column, int)):
                raise ValueError(
                    "If pos is a tuple, it must be of the form (qudit, column)."
                )
            if not (0 <= target_qudit < self.n):
                raise IndexError(f"Qudit index {target_qudit} is out of range.")
            if column < 0:
                raise ValueError(f"Column index must be >= 0, got {column}")

            noise_gate_instance = self._create_noise_channel_instance(
                noise_model, color
            )
            self.replace(noise_gate_instance, target_qudit, column)

        elif isinstance(pos, int):
            # --- Program-based Mode ---
            program_pos = pos
            target_qudit = qudit_or_model
            noise_model = model_or_none

            if noise_model is None:
                raise TypeError(
                    "Missing `noise_model` argument for program-order replacement. Expected: (pos_int, qudit, noise_model, color=...)."
                )
            if not isinstance(target_qudit, int):
                raise TypeError(
                    f"For program-order replacement, the second argument must be the target qudit (an int), but got {type(target_qudit)}."
                )

            if not (0 <= program_pos < len(self._ops)):
                raise IndexError(
                    f"Position {program_pos} is out of range for replacement (0 to {len(self._ops) - 1})."
                )
            if not (0 <= target_qudit < self.n):
                raise IndexError("Qudit index out of range.")

            noise_gate_instance = self._create_noise_channel_instance(
                noise_model, color
            )
            self.replace_at_position(program_pos, noise_gate_instance, (target_qudit,))

        else:
            raise TypeError(
                "`pos` must be an integer or a tuple of two integers (qudit, column)."
            )

    # Note:
    # to delete noise gate use delete_at_position or delete

    def add_noise_global(
        self, noise_model: Union[np.ndarray, float], color: str = "#FF0000"
    ) -> None:
        """
        Insert a noise channel after every operation in the circuit.

        This implements "noise stacking": if a gate is already followed by one or more
        NoiseChannel placeholders, the newly inserted noise channels are placed after
        that contiguous block.

        Parameters
        ----------
        noise_model : float or np.ndarray
            Noise model to use for every inserted channel.
        color : str, default="#FF0000"
            Base color for inserted noise channels.

        Raises
        ------
        ValueError
            If ``noise_model`` is invalid or cannot be normalized.
        """
        try:
            self._prepare_weyl_probabilities(noise_model)
        except ValueError as e:
            raise ValueError(f"Invalid global noise model provided: {e}")

        offset = 0
        original_op_count = len(self._ops)

        for i in range(original_op_count):
            current_pos = i + offset
            gate, qudits, _ = self._ops[current_pos]

            if isinstance(gate, NoiseChannel) or gate.name == "NOISE_CHANNEL":
                continue

            base_insertion_pos = current_pos + 1

            while base_insertion_pos < len(self._ops):
                next_gate = self._ops[base_insertion_pos][0]
                if (
                    isinstance(next_gate, NoiseChannel)
                    or next_gate.name == "NOISE_CHANNEL"
                ):
                    base_insertion_pos += 1
                else:
                    break

            # Insert new noise channels at the final calculated position.
            num_inserted_this_step = 0
            for q in qudits:
                insertion_pos = base_insertion_pos + num_inserted_this_step
                self.insert_noise_channel(insertion_pos, q, noise_model, color)
                num_inserted_this_step += 1

            # Update the offset by the number of gates we just inserted
            offset += num_inserted_this_step

    def add_noise_layer(
        self,
        layer_index: int,
        noise_model: Union[
            float,
            np.ndarray,
            dict[int, Union[float, np.ndarray, tuple[Union[float, np.ndarray], str]]],
        ],
        mode: Literal["packed", "visual"] = "packed",
        where: Literal["before", "after"] = "after",
        include_idle: bool = True,
        color: str = "#FF0000",
        in_place: bool = False,
    ) -> Optional[QuantumCircuit]:
        """
        Add a layer of noise channels to a specific gate layer in the circuit.

        A "layer" is defined by an ASAP schedule (either packed/true-parallel or
        visual/layout-oriented). Noise channels can be inserted either before or
        after the specified layer. Noise may be uniform (one model for all qudits)
        or specified per qudit via a dictionary.

        Parameters
        ----------
        layer_index : int
            Zero-based index of the target layer.
        noise_model : float or np.ndarray or dict
            Noise specification:

            - float: isotropic noise with total error probability ``p``.
            - np.ndarray: matrix of shape ``(d, d)`` with Weyl probabilities.
            - dict: per-qudit specification, where values are either a model or
            ``(model, color)`` to override visualization color.

        mode : {"packed", "visual"}, default="packed"
            Which schedule to use to define layers.
        where : {"before", "after"}, default="after"
            Whether to insert noise before or after the target layer.
        include_idle : bool, default=True
            If True, insert noise for all qudits. If False, only for qudits active
            in the target layer.
        color : str, default="#FF0000"
            Default visualization color (used unless overridden per-qudit).
        in_place : bool, default=False
            If True, modify this circuit. If False, return a modified copy.

        Returns
        -------
        QuantumCircuit or None
            A modified circuit if ``in_place=False``; otherwise None.

        Raises
        ------
        ValueError
            If ``mode`` or ``where`` is invalid.
        IndexError
            If ``layer_index`` is negative or out of range for the chosen schedule.
        TypeError
            If ``noise_model`` has an unsupported type.
        """
        if mode not in ["packed", "visual"]:
            raise ValueError("mode must be either 'packed' or 'visual'.")
        if where not in ["before", "after"]:
            raise ValueError("where must be either 'before' or 'after'.")
        if layer_index < 0:
            raise IndexError("layer_index cannot be negative.")

        target_circuit = self if in_place else self.copy()

        # Get the schedule which defines the layers
        if mode == "packed":
            schedule = target_circuit._asap_schedule_packed()
        else:  # mode == "visual"
            schedule = target_circuit._asap_schedule()

        if layer_index >= len(schedule):
            raise IndexError(
                f"layer_index {layer_index} is out of range for the schedule with {len(schedule)} layers."
            )

        # This function generates the list of noise operations to be inserted
        def _get_noise_ops_for_layer(current_layer):
            if include_idle:
                qudits_to_noise = set(range(target_circuit.n))
            else:
                if not current_layer:
                    return []
                qudits_to_noise = {q for _, qudits, _ in current_layer for q in qudits}

            ops_to_add = []
            for q_idx in sorted(list(qudits_to_noise)):
                model, gate_color = None, color
                if isinstance(noise_model, (float, int, list, np.ndarray)):
                    model = noise_model
                elif isinstance(noise_model, dict):
                    if q_idx not in noise_model:
                        continue
                    spec = noise_model[q_idx]
                    model, gate_color = (
                        spec if isinstance(spec, tuple) else (spec, color)
                    )
                else:
                    raise TypeError(
                        f"Unsupported type for noise_model: {type(noise_model)}"
                    )

                if model is not None:
                    noise_gate = target_circuit._create_noise_channel_instance(
                        model, gate_color
                    )
                    ops_to_add.append((noise_gate, (q_idx,), False))
            return ops_to_add

        # Rebuild the entire _ops list from the schedule, inserting noise at the correct layer
        new_ops = []
        for i, current_layer in enumerate(schedule):
            # If this is the target layer, potentially add noise BEFORE
            if i == layer_index and where == "before":
                new_ops.extend(_get_noise_ops_for_layer(current_layer))

            # Add the actual gates from the current layer
            new_ops.extend(current_layer)

            # If this is the target layer, potentially add noise AFTER
            if i == layer_index and where == "after":
                new_ops.extend(_get_noise_ops_for_layer(current_layer))

        target_circuit._ops = new_ops

        if not in_place:
            return target_circuit
        return None

    def insert_noise_at_coords(
        self,
        coords: Optional[list[tuple[int, int]]] = None,
        noise_model: Optional[Union[int, float, list, np.ndarray]] = None,
        rand_coords: Optional[
            Union[list, tuple[tuple[int, int], tuple[int, int], float]]
        ] = None,
        color: str = "#FF0000",
        seed: Optional[int] = None,
        in_place: bool = False,
    ) -> Optional[QuantumCircuit]:
        """
        Insert noise channels at specified (qudit, column) coordinates in the visual layout.

        The circuit is scheduled using the ASAP visual schedule (``_asap_schedule``).
        For each unique column index in the requested coordinates, a new parallel layer
        is inserted into the schedule at that column, shifting subsequent layers right.

        Parameters
        ----------
        coords : list[tuple[int, int]], optional
            Explicit list of ``(qudit, column)`` coordinates.
        noise_model : int or float or list or np.ndarray, optional
            Noise model to apply at each inserted coordinate.
        rand_coords : list or tuple, optional
            Random coordinate specification of the form
            ``((q_start, q_end), (col_start, col_end), placement_prob)``.
            The index ranges are inclusive. ``placement_prob`` is in ``[0, 1]``.
        color : str, default="#FF0000"
            Base visualization color.
        seed : int, optional
            RNG seed used when generating random coordinates.
        in_place : bool, default=False
            If True, modify this circuit. If False, return a modified copy.

        Returns
        -------
        QuantumCircuit or None
            Modified circuit if ``in_place=False``; otherwise None.

        Raises
        ------
        ValueError
            If both/neither of ``coords`` and ``rand_coords`` are provided; if
            ``noise_model`` is missing; or if any parameter values are invalid.
        TypeError
            If ``coords`` format is invalid.
        """
        if (coords is not None and rand_coords is not None) or (
            coords is None and rand_coords is None
        ):
            raise ValueError("Provide either 'coords' or 'rand_coords', but not both.")

        if noise_model is None:
            raise ValueError("'noise_model' must be provided.")

        target_circuit = self if in_place else self.copy()

        try:
            noise_gate_instance = target_circuit._create_noise_channel_instance(
                noise_model, color
            )
        except ValueError as e:
            raise ValueError(f"Invalid noise_model or color provided: {e}") from e

        schedule = target_circuit._asap_schedule()
        final_coords = []

        # Coordinate Generation
        if rand_coords is not None:
            try:
                (q_range, col_range, prob) = rand_coords
                q_start, q_end = q_range
                col_start, col_end = col_range
            except (TypeError, ValueError):
                raise ValueError(
                    "`rand_coords` must be a list/tuple of the form "
                    "((q_start, q_end), (col_start, col_end), prob)."
                )

            if not (0 <= q_start <= q_end < target_circuit.n):
                raise ValueError(
                    f"Qudit range ({q_start}, {q_end}) is out of bounds for {target_circuit.n} qudits."
                )
            if not (0 <= col_start <= col_end):
                raise ValueError("Column range must be non-negative and ordered.")
            if not (0.0 <= prob <= 1.0):
                raise ValueError(f"placement_prob must be between 0 and 1, got {prob}.")

            rng = np.random.default_rng(seed)
            for q in range(q_start, q_end + 1):
                for c in range(col_start, col_end + 1):
                    if rng.random() < prob:
                        final_coords.append((q, c))
        else:  # coords is not None
            if not isinstance(coords, list) or not all(
                isinstance(c, tuple) and len(c) == 2 for c in coords
            ):
                raise TypeError("`coords` must be a list of (qudit, column) tuples.")
            for q, c in coords:
                if not isinstance(q, int) or not (0 <= q < target_circuit.n):
                    raise ValueError(f"Invalid qudit index {q} in coordinates.")
                if not isinstance(c, int) or c < 0:
                    raise ValueError(f"Invalid column index {c} in coordinates.")
            final_coords = coords

        if not final_coords:
            return None if in_place else target_circuit

        # Group insertions by column index
        insertions_by_col = {}
        for q, c in final_coords:
            if c not in insertions_by_col:
                insertions_by_col[c] = []
            # Append the full operation tuple for the noise gate
            insertions_by_col[c].append((noise_gate_instance, (q,), False))

        # Insert new columns into the schedule, iterating in reverse order
        # to prevent insertion from invalidating subsequent column indices.
        for col_idx in sorted(insertions_by_col.keys(), reverse=True):
            noise_ops_for_col = insertions_by_col[col_idx]

            # Pad the schedule if inserting beyond its current length
            while len(schedule) <= col_idx:
                schedule.append([])

            schedule.insert(col_idx, noise_ops_for_col)

        # Rebuild the main _ops list from the modified schedule
        new_ops = [op for col_ops in schedule for op in col_ops]
        target_circuit._ops = new_ops

        if not in_place:
            return target_circuit
        return None

    def push_noise(self, end: bool = False, in_place: bool = False) -> QuantumCircuit:
        """
        Move all NoiseChannel placeholders to the start or end of the circuit.

        Non-noise operations preserve their relative order. Noise operations preserve
        their relative order among themselves.

        Parameters
        ----------
        end : bool, default=False
            If True, move noise channels to the end. If False, move them to the start.
        in_place : bool, default=False
            If True, modify this circuit. If False, return a reordered copy.

        Returns
        -------
        QuantumCircuit
            The modified circuit (``self`` if in-place, otherwise a copy).
        """

        def _is_noise(g):
            return isinstance(g, NoiseChannel) or g.name == "NOISE_CHANNEL"

        ops = self._ops
        non_noise = [op for op in ops if not _is_noise(op[0])]
        noise = [op for op in ops if _is_noise(op[0])]

        target = self if in_place else self.copy()
        if end:
            target._ops = non_noise + noise
        else:
            target._ops = noise + non_noise
        return target

    def realize_noise(
        self,
        color: str = "#FF0000",
        show_identities: bool = False,
        seed: Optional[int] = None,
        in_place: bool = True,
    ) -> Union[
        list[tuple[Gate, int, int, tuple[int, int]]],
        tuple[QuantumCircuit, list[tuple[Gate, int, int, tuple[int, int]]]],
    ]:
        """
        Replace all NoiseChannel placeholders by sampled Weyl operators.

        For each NoiseChannel, a Weyl operator ``W_{a,b}`` is sampled from the stored
        probability model in ``noise_registry``.

        - If the sampled outcome is identity (a=b=0) and ``show_identities=False``,
        the placeholder is removed.
        - Otherwise, the placeholder is replaced by a concrete Weyl gate.

        After realization, the noise registry/cache is cleared.

        Parameters
        ----------
        color : str, default="#FF0000"
            Visualization color for realized Weyl gates.
        show_identities : bool, default=False
            If True, keep identity outcomes as explicit gates instead of deleting them.
        seed : int, optional
            RNG seed for reproducibility.
        in_place : bool, default=True
            If True, modify this circuit. If False, return a modified copy.

        Returns
        -------
        applied_errors : list[tuple[Gate, int, int, tuple[int, int]]]
            List of realized outcomes as tuples
            ``(realized_gate, qudit_index, new_position_in_ops, (a, b))``.
            If ``in_place=False``, returns ``(new_circuit, applied_errors)``.
        """
        if in_place:
            target_circuit = self
        else:
            target_circuit = self.copy()

        rng = np.random.default_rng(seed)
        applied_errors = []
        new_ops = []

        # Iterate through the original operations and build a new list
        for gate, qudits, dagger in target_circuit._ops:

            # Check for NoiseChannel instance
            if isinstance(gate, NoiseChannel):
                # Get the noise model from the registry using the gate's model_id
                normalized_probs = target_circuit.noise_registry.get(gate.model_id)
                if normalized_probs is None:
                    continue  # Should not happen if circuit is constructed correctly

                choice_index = rng.choice(
                    target_circuit.d * target_circuit.d, p=normalized_probs
                )
                a_b_pair = np.unravel_index(
                    choice_index, (target_circuit.d, target_circuit.d)
                )
                a, b = int(a_b_pair[0]), int(a_b_pair[1])

                is_identity = a == 0 and b == 0
                if is_identity and not show_identities:
                    pass  # Skip adding a gate entirely
                else:
                    realized_gate = self._W(a, b, color=color)
                    new_ops.append((realized_gate, qudits, False))
                    new_pos = len(new_ops) - 1
                    # Append all realized outcomes, including identities if shown
                    applied_errors.append((realized_gate, qudits[0], new_pos, (a, b)))
            else:
                # This is a regular gate, keep it.
                new_ops.append((gate, qudits, dagger))

        # The noise registry is cleared because there are no more NoiseChannel gates
        target_circuit._ops = new_ops
        target_circuit.noise_registry.clear()
        target_circuit._noise_model_cache.clear()
        target_circuit._next_noise_model_id = 0

        if in_place:
            return applied_errors
        else:
            return target_circuit, applied_errors

    def clear_noise(self) -> None:
        """
        Remove all NoiseChannel placeholders from the circuit in-place.

        Realized Weyl gates (concrete error gates) are not removed. The noise registry
        and associated caches are cleared.
        """
        # Build a new list containing only non-noise gates
        new_ops = []
        for gate, qudits, dagger in self._ops:
            if not isinstance(gate, NoiseChannel) and not gate.name == "NOISE_CHANNEL":
                new_ops.append((gate, qudits, dagger))

        # Replace the old operations list with the new, filtered list
        self._ops = new_ops

        # Clear all registry data as it's no longer relevant
        self.noise_registry.clear()
        self._noise_model_cache.clear()
        self._next_noise_model_id = 0
