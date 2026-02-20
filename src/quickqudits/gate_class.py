"""
Class definitions for quantum gates and noise channels.

This module defines the `Gate` class which holds the properties and matrix
generation logic for quantum operations. It also defines `NoiseChannel`,
a specialized gate subclass for handling noise models.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Optional

import numpy as np

from . import gates

if TYPE_CHECKING:
    from .circuit import QuantumCircuit


# ===========================================================================
# Gate Class
# ===========================================================================


class Gate:
    """
    Represents a quantum gate or channel.

    A Gate object stores the metadata (name, dimension) and the logic
    (matrix functions) required to simulate the operation.

    Parameters
    ----------
    name : str
        The name of the gate (e.g., "X", "CNOT").
    n_qudits : int
        The number of qudits this gate acts on.
    matrix_fn : Callable[[int], np.ndarray] or None
        A function that takes dimension `d` and returns the unitary matrix.
        Can be None for placeholder gates (like Measure).
    dagger_matrix_fn : Callable[[int], np.ndarray] or None, optional
        A function that returns the conjugate transpose matrix.
    apply_fn : Callable or None, optional
        An optional optimized function for applying the gate directly to a
        statevector without constructing the full matrix.
    color : str or None, optional
        Hex color code for visualization.
    a : float or None, optional
        Parameter 'a' for Weyl gates.
    b : float or None, optional
        Parameter 'b' for Weyl gates.
    """

    def __init__(
        self,
        name: str,
        n_qudits: int,
        matrix_fn: Optional[Callable[[int], np.ndarray]],
        dagger_matrix_fn: Optional[Callable[[int], np.ndarray]] = None,
        apply_fn: Optional[Callable] = None,
        color: Optional[str] = None,
        a: Optional[float] = None,
        b: Optional[float] = None,
    ):
        self.name = name
        self.n_qudits = n_qudits
        self.matrix_fn = matrix_fn
        self.dagger_matrix_fn = dagger_matrix_fn
        self._apply_fn = apply_fn
        self.color = color
        self.a = a
        self.b = b

    def __repr__(self) -> str:
        return f"Gate(name={self.name!r}, n_qudits={self.n_qudits})"

    def __str__(self) -> str:
        return f"<Gate {self.name}: {self.n_qudits}-qudit>"

    # -----------------------------------------------------------------------
    # Gate Utility Methods
    # -----------------------------------------------------------------------

    def copy(self, **overrides) -> Gate:
        """
        Create a shallow copy of this gate, optionally overriding attributes.

        Parameters
        ----------
        **overrides : dict
            Key-value pairs to override in the new instance (e.g., name="NewName").

        Returns
        -------
        Gate
            A new Gate instance.
        """
        kwargs = {
            "name": self.name,
            "n_qudits": self.n_qudits,
            "matrix_fn": self.matrix_fn,
            "dagger_matrix_fn": self.dagger_matrix_fn,
            "apply_fn": self._apply_fn,
            "color": self.color,
            "a": self.a,
            "b": self.b,
        }
        kwargs.update(overrides)
        return Gate(**kwargs)

    def __copy__(self) -> Gate:
        return self.copy()

    def matrix(self, d: int, dagger: bool = False) -> np.ndarray:
        """
        Return the numerical matrix for this gate at dimension d.

        Parameters
        ----------
        d : int
            Dimension of the qudit.
        dagger : bool, default=False
            If True, return the conjugate transpose matrix.

        Returns
        -------
        np.ndarray
            The unitary matrix.

        Raises
        ------
        ValueError
            If the gate does not define a matrix function.
        """
        if dagger:
            if not self.dagger_matrix_fn:
                raise ValueError(f"No dagger matrix available for gate {self.name}")
            return self.dagger_matrix_fn(d)
        if self.matrix_fn is None:
            raise ValueError(f"No matrix available for gate {self.name}")
        return self.matrix_fn(d)

    def apply(
        self,
        psi: np.ndarray,
        n: int,
        d: int,
        qudits: tuple[int, ...],
        dagger: bool = False,
    ) -> np.ndarray:
        """
        Apply this gate to a statevector.

        Uses the optimized `apply_fn` if available; otherwise falls back to
        tensor contraction using the matrix definition.

        Parameters
        ----------
        psi : np.ndarray
            Statevector of length d^n.
        n : int
            Total number of qudits.
        d : int
            Dimension of qudits.
        qudits : tuple[int, ...]
            Indices of the target qudits.
        dagger : bool, default=False
            If True, apply the inverse.

        Returns
        -------
        np.ndarray
            The updated statevector.
        """
        # Use optimized apply function if available
        if self._apply_fn is not None:
            if self.a is not None and self.b is not None:
                # Weyl gate special case
                return self._apply_fn(psi, n, d, qudits, dagger, a=self.a, b=self.b)
            else:
                # CNOT/CZ/SWAP
                return self._apply_fn(psi, n, d, qudits, dagger)

        # Optimized 1-qudit system case
        if n == 1:
            U = self.matrix(d, dagger)
            return U @ psi

        # General Tensor Contraction
        Umat = self.dagger_matrix_fn(d) if dagger else self.matrix_fn(d)
        k = self.n_qudits

        U = Umat.reshape((d,) * k + (d,) * k)
        psi_nd = psi.reshape((d,) * n)
        axes_U = list(range(k, 2 * k))
        axes_psi = list(qudits)
        contracted = np.tensordot(U, psi_nd, axes=(axes_U, axes_psi))
        rest = [i for i in range(n) if i not in qudits]

        perm = []
        for i in range(n):
            if i in qudits:
                perm.append(qudits.index(i))
            else:
                perm.append(k + rest.index(i))

        out = np.transpose(contracted, perm)
        return out.reshape(-1)

    # -----------------------------------------------------------------------
    # Density Matrix Application
    # -----------------------------------------------------------------------

    @staticmethod
    def _apply_operator_to_density_matrix(
        rho: np.ndarray, operator: np.ndarray, qudits: tuple[int, ...], n: int, d: int
    ) -> np.ndarray:
        """
        Applies rho' = U * rho * U_dagger by explicitly constructing the full d^n x d^n operator.

        Parameters
        ----------
        rho : np.ndarray
            Density matrix of shape (d^n, d^n).
        operator : np.ndarray
            The d^k x d^k matrix of the small gate.
        qudits : tuple[int, ...]
            Target qudits.
        n : int
            Total number of qudits.
        d : int
            Dimension of qudits.

        Returns
        -------
        np.ndarray
            The updated density matrix.
        """
        from .circuit import QuantumCircuit

        k = len(qudits)

        # Create a temporary gate object that holds the small operator.
        temp_gate = Gate("temp", k, lambda d_val: operator)

        # Create a temporary circuit to access the _get_gate_unitary helper.
        temp_qc = QuantumCircuit(n, d)

        # Use the helper to construct the full d^n x d^n operator.
        U = temp_qc._get_gate_unitary(temp_gate, qudits, False)

        # Get the conjugate transpose for the right-multiplication.
        U_dag = np.conjugate(U.T)

        # Perform the matrix multiplication.
        return U @ rho @ U_dag

    def apply_density(
        self,
        rho: np.ndarray,
        circuit: QuantumCircuit,
        qudits: tuple[int, ...],
        dagger: bool = False,
    ) -> np.ndarray:
        """
        Applies a gate to a density matrix: rho' = U rho U†.

        Parameters
        ----------
        rho : np.ndarray
            Density matrix of shape (d^n, d^n).
        circuit : QuantumCircuit
            Reference to the circuit (for accessing n, d).
        qudits : tuple[int, ...]
            Target qudits.
        dagger : bool, default=False
            If True, apply inverse.

        Returns
        -------
        np.ndarray
            Updated density matrix.
        """
        if self.matrix_fn is None:
            return rho

        U = self.matrix(circuit.d, dagger)

        return Gate._apply_operator_to_density_matrix(
            rho, U, qudits, circuit.n, circuit.d
        )


# ===========================================================================
# NoiseChannel Class
# ===========================================================================


class NoiseChannel(Gate):
    """
    Represents a noise channel.

    Unlike standard gates, this does not have a static matrix. Instead, it
    stores a `model_id` which points to a noise model in the circuit's
    registry.

    Parameters
    ----------
    model_id : int
        The ID of the noise model in the circuit registry.
    color : str or None, optional
        Visualization color. Opacity corresponds to error probability.
    """

    def __init__(self, model_id: int, color: Optional[tuple] = None):
        # Call the parent Gate constructor with fixed properties for a noise channel
        super().__init__(name="NOISE_CHANNEL", n_qudits=1, matrix_fn=None, color=color)
        self.model_id = model_id

    def apply_density(
        self,
        rho: np.ndarray,
        circuit: QuantumCircuit,
        qudits: tuple[int, ...],
        dagger: bool = False,
    ) -> np.ndarray:
        """
        Apply the noise channel to a density matrix using using the
        Kraus operator sum representation. Implements on-the-fly computation
        and caching of Kraus operators.

        rho' = sum(E_k * rho * E_k†)

        Parameters
        ----------
        rho : np.ndarray
            Density matrix of shape (d^n, d^n).
        circuit : QuantumCircuit
            Reference to the circuit.
        qudits : tuple[int, ...]
            Target qudits.
        dagger : bool
            Ignored for noise channels.

        Returns
        -------
        np.ndarray
            Updated density matrix.
        """
        # Check the circuit's cache for our Kraus operators.
        kraus_operators = circuit._kraus_operator_cache.get(self.model_id)

        # If not in the cache, compute and store them.
        if kraus_operators is None:
            # Retrieve the probability array from the permanent registry.
            prob_array = circuit.noise_registry.get(self.model_id)
            if prob_array is None:
                # This should not happen if the circuit is constructed correctly.
                return rho

            d = circuit.d
            prob_matrix = prob_array.reshape((d, d))

            # Compute the list of d x d Kraus operators.
            kraus_operators = []
            for a in range(d):
                for b in range(d):
                    prob = prob_matrix[a, b]
                    if (
                        prob > 1e-12
                    ):  # Only create operators for non-zero probabilities.
                        # E_k = sqrt(p_k) * W_k
                        kraus_op = np.sqrt(prob) * gates.W(d, a, b)
                        kraus_operators.append(kraus_op)

            # Store the computed list in the cache for future use in this run.
            circuit._kraus_operator_cache[self.model_id] = kraus_operators

        # Apply the Kraus sum: rho' = sum(E_k * rho * E_k_dagger)
        rho_new = np.zeros_like(rho)
        for E_k in kraus_operators:
            term = Gate._apply_operator_to_density_matrix(
                rho, E_k, qudits, circuit.n, circuit.d
            )
            rho_new += term

        return rho_new
