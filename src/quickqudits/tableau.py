"""
Stabilizer tableau for qudit Clifford simulation.

This module defines :class:`~Tableau`, a stabilizer tableau for ``n`` qudits of
dimension ``d``. The tableau stores:

- An ``X`` exponent block (mod ``d``)
- A ``Z`` exponent block (mod ``d``)
- A phase column encoded as powers of ``tau`` (mod ``2d``), where ``tau^2 = omega``

The tableau can be evolved by applying gates from :class:`~circuit.QuantumCircuit`.
Numba-accelerated kernels are used for faster application.
"""

from __future__ import annotations

import warnings
from functools import lru_cache
from typing import Optional, Union

import numpy as np

from . import backend, gates
from .circuit import QuantumCircuit
from .gate_class import Gate, NoiseChannel

try:
    # Numba kernels
    from .accelerators.numba_fast import (
        ID_TO_NAME,
        nb_apply_circuit,
        nb_apply_gate,
        pack_circuit_from_qc,
    )

    _HAS_NUMBA = True
    _NUMBA_IMPORT_ERROR: Optional[Exception] = None
except Exception as e:
    _HAS_NUMBA = False
    _NUMBA_IMPORT_ERROR = e

# ===========================================================================
# Helper Functions
# ===========================================================================


def choose_signed_dtype_for(max_dim: int) -> np.dtype:
    """
    Return the smallest *signed* integer dtype that can hold
    any intermediate up to ±2*(max_dim-1)^2 exactly.

    Parameters
    ----------
    max_dim : int
        Dimension ``d``.

    Returns
    -------
    numpy.dtype
        A signed integer dtype (e.g. ``np.int8``, ``np.int16``, ...).
    """
    max_need = 2 * (max_dim - 1) ** 2
    if max_need < 2**7:
        return np.int8
    elif max_need < 2**15:
        return np.int16
    elif max_need < 2**31:
        return np.int32
    else:
        return np.int64


def _scalar_to_int(x):
    """
    Convert a NumPy scalar (or Python number) into a Python ``int``.

    Parameters
    ----------
    x : object
        NumPy scalar.

    Returns
    -------
    int
        Python integer.
    """
    try:
        return int(x.item())
    except Exception:
        return int(x)


# ===========================================================================
# Tableau Class
# ===========================================================================


class Tableau:
    """
    Stabilizer tableau for ``n`` qudits of dimension ``d``.

    The tableau stores separate exponent blocks ``X`` and ``Z`` (entries in
    ``{0, ..., d-1}``) and a phase column ``tau_exp`` storing exponents of ``tau``
    in ``{0, ..., 2d-1}``, with ``tau^2 = omega``.

    Parameters
    ----------
    n : int
        Number of qudits.
    d : int
        Dimension.
    full : bool, optional
        If True, allocate a *full* tableau with ``2n`` rows (destabilizers + stabilizers).
        If False, allocate a stabilizer-only tableau with ``n`` rows. Default: True.
    X : numpy.ndarray, optional
        Initial X block with shape ``(rows, n)``, where ``rows = 2n`` if ``full`` else ``n``.
    Z : numpy.ndarray, optional
        Initial Z block with shape ``(rows, n)``.
    tau_exp : numpy.ndarray, optional
        Initial tau exponents with shape ``(rows,)``.

    Raises
    ------
    ValueError
        If provided arrays have incompatible shapes.
    """

    def __init__(
        self,
        n: int,
        d: int,
        *,
        full: bool = True,
        X: Optional[np.ndarray] = None,
        Z: Optional[np.ndarray] = None,
        tau_exp: Optional[np.ndarray] = None,
    ):
        self._n = n
        self._d = d
        self._full = full
        self._mod_d = d
        self._mod_2d = 2 * d

        # choose appropriate dtypes
        self._xz_dtype = choose_signed_dtype_for(self._d)
        self._tau_dtype = choose_signed_dtype_for(self._d)

        rows = 2 * n if full else n

        # Allocate defaults
        if X is not None:
            if X.shape != (rows, n):
                raise ValueError(
                    f"Provided X has shape {X.shape}, expected {(rows, n)}"
                )
            self._X = X.astype(self._xz_dtype) % self._mod_d
        else:
            self._X = np.zeros((rows, n), dtype=self._xz_dtype)

        if Z is not None:
            if Z.shape != (rows, n):
                raise ValueError(
                    f"Provided Z has shape {Z.shape}, expected {(rows, n)}"
                )
            self._Z = Z.astype(self._xz_dtype) % self._mod_d
        else:
            self._Z = np.zeros((rows, n), dtype=self._xz_dtype)

        if tau_exp is not None:
            if tau_exp.shape != (rows,):
                raise ValueError(
                    f"Provided tau_exp has shape {tau_exp.shape}, expected {(rows,)}"
                )
            self._tau_exp = tau_exp.astype(self._tau_dtype) % self._mod_2d
        else:
            self._tau_exp = np.zeros((rows,), dtype=self._tau_dtype)

        # default initializations if not overridden
        if X is None and Z is None and tau_exp is None:
            if full:
                # full input: first n rows have X_i, next n rows have Z_i
                eye = np.eye(n, dtype=self._xz_dtype)
                self._X[:n, :] = eye  # X stabilizers
                self._Z[n : 2 * n, :] = eye  # Z stabilizers
            else:
                # all-|0> state: Z_i stabilizers
                eye = np.eye(n, dtype=self._xz_dtype)
                self._Z[:n, :] = eye

    # -----------------------------------------------------------------------
    # Properties
    # -----------------------------------------------------------------------

    @property
    def n(self) -> int:
        """int: Number of qudits."""
        return self._n

    @property
    def d(self) -> int:
        """int: Dimension."""
        return self._d

    @property
    def full(self) -> bool:
        """bool: Whether this is a full (2n-row) tableau."""
        return self._full

    @property
    def mod_d(self) -> int:
        """int: Modulus for X/Z entries (equals ``d``)."""
        return self._mod_d

    @property
    def mod_2d(self) -> int:
        """int: Modulus for tau exponents (equals ``2d``)."""
        return self._mod_2d

    @property
    def X(self) -> np.ndarray:
        """numpy.ndarray: X exponent block (shape ``(rows, n)``)."""
        return self._X

    @property
    def Z(self) -> np.ndarray:
        """numpy.ndarray: Z exponent block (shape ``(rows, n)``)."""
        return self._Z

    @property
    def tau_exp(self) -> np.ndarray:
        """numpy.ndarray: Tau exponent column (shape ``(rows,)``)."""
        return self._tau_exp

    # -----------------------------------------------------------------------
    # Constructors / Basic Utilities
    # -----------------------------------------------------------------------

    @classmethod
    def zero_state(cls, n: int, d: int) -> Tableau:
        """
        Create a stabilizer-only tableau for the all-|0> state.

        Parameters
        ----------
        n : int
            Number of qudits.
        d : int
            Dimension.

        Returns
        -------
        Tableau
            A stabilizer-only tableau (``full=False``).
        """
        return cls(n, d, full=False)

    @classmethod
    def full_state(cls, n: int, d: int) -> Tableau:
        """
        Create a full (2n-row) input tableau.

        Parameters
        ----------
        n : int
            Number of qudits.
        d : int
            Dimension.

        Returns
        -------
        Tableau
            A full tableau (``full=True``).
        """
        return cls(n, d, full=True)

    @classmethod
    def from_rows(
        cls,
        n: int,
        d: int,
        X: np.ndarray,
        Z: np.ndarray,
        tau_exp: np.ndarray,
        full: bool = False,
    ) -> Tableau:
        """
        Construct a tableau from raw row blocks.

        Parameters
        ----------
        n : int
            Number of qudits.
        d : int
            Dimension.
        X : numpy.ndarray
            X block, shape ``(rows, n)``.
        Z : numpy.ndarray
            Z block, shape ``(rows, n)``.
        tau_exp : numpy.ndarray
            Tau exponents, shape ``(rows,)``.
        full : bool, optional
            Whether ``rows`` corresponds to a full tableau (2n) or stabilizer-only (n).
            Default: False.

        Returns
        -------
        Tableau
            New tableau instance.

        Raises
        ------
        ValueError
            If shapes are incompatible.
        """
        return cls(n, d, full=full, X=X, Z=Z, tau_exp=tau_exp)

    @classmethod
    def random_tableau(
        cls,
        n: int,
        d: int,
        *,
        full: bool = False,
        seed: Optional[int] = None,
    ) -> Tableau:
        """
        Construct a tableau with random entries.

        The returned tableau has:
        - X, Z entries sampled uniformly from ``{0, ..., d-1}``
        - tau exponents sampled uniformly from ``{0, ..., 2d-1}``

        Parameters
        ----------
        n : int
            Number of qudits.
        d : int
            Dimension.
        full : bool, optional
            If False, returns an ``n``-row tableau; if True, returns a ``2n``-row tableau.
            Default: False.
        seed : int, optional
            Seed for reproducibility.

        Returns
        -------
        Tableau
            Random tableau.
        """
        if seed is not None:
            np.random.seed(seed)
        rows = n if not full else 2 * n
        xz_dtype = choose_signed_dtype_for(d)
        tau_dtype = choose_signed_dtype_for(d)

        X = np.random.randint(0, d, size=(rows, n), dtype=xz_dtype)
        Z = np.random.randint(0, d, size=(rows, n), dtype=xz_dtype)
        tau_exp = np.random.randint(0, 2 * d, size=(rows,), dtype=tau_dtype)

        return cls(n, d, full=full, X=X, Z=Z, tau_exp=tau_exp)

    def copy(self) -> Tableau:
        """
        Return a copy of this tableau.

        Returns
        -------
        Tableau
            Independent copy (arrays are copied).
        """
        t = type(self)(self._n, self._d, full=self._full)
        t._X = self._X.copy()
        t._Z = self._Z.copy()
        t._tau_exp = self._tau_exp.copy()
        return t

    def _normalize(self) -> None:
        """Reduce all tableau entries modulo ``d`` / ``2d`` in-place."""
        self._X %= self._mod_d
        self._Z %= self._mod_d
        self._tau_exp %= self._mod_2d

    def normalize(self) -> None:
        """
        Normalize the tableau in-place by reducing entries modulo ``d`` / ``2d``.
        """
        self._normalize()

    def change_dim(self, new_d: int, in_place: bool = False) -> Tableau:
        """
        Change the dimension ``d -> new_d`` by reducing tableau entries.

        This performs:
        ``X := X mod new_d``, ``Z := Z mod new_d``, ``tau := tau mod (2*new_d)``.

        Parameters
        ----------
        new_d : int
            New dimension (>= 2).
        in_place : bool, optional
            If True, mutate this tableau and return it.
            If False, return a new tableau with remapped entries. Default: False.

        Returns
        -------
        Tableau
            Updated tableau (self if ``in_place=True``).

        Raises
        ------
        ValueError
            If ``new_d`` is invalid.
        """
        if not isinstance(new_d, int) or new_d < 2:
            raise ValueError(f"new_d must be an integer >= 2, got {new_d!r}")

        rows, n = self._X.shape
        if in_place:
            self._X %= new_d
            self._Z %= new_d
            self._tau_exp %= 2 * new_d
            self._d = int(new_d)
            return self

        # create same-shape tableau with the new dimension
        tgt = type(self)(n, int(new_d), full=self._full)
        tgt._X[:rows, :n] = self._X % new_d
        tgt._Z[:rows, :n] = self._Z % new_d
        tgt._tau_exp[:rows] = self._tau_exp % (2 * new_d)
        return tgt

    def _check_compat_with_circuit(self, qc: QuantumCircuit) -> None:
        """
        Check compatibility of tableau and circuit.
        The following should hold:
            tab.n = qc.n
            tab.d = qc.d

        Parameters
        ----------
        qc : QuantumCircuit
            Circuit to check against.

        Raises
        ------
        ValueError
            If either n or d mismatch between tableau and circuit occured.
        """
        if self.n != qc.n or self.d != qc.d:
            raise ValueError(
                f"Incompatible (n,d): tableau is (n={self.n}, d={self.d}), "
                f"circuit is (n={qc.n}, d={qc.d})."
            )

    # -----------------------------------------------------------------------
    # Tableau Visualization
    # -----------------------------------------------------------------------

    def print_stabilizers(self, show_omegas: bool = False) -> None:
        """
        Print a column-aligned table of stabilizer generators.

        Parameters
        ----------
        show_omegas : bool, optional
            If True, display the omega-power and tau-bit in the phase column.
            If False, show only tau (or 1). Default: False.
        """
        self._normalize()
        SUP = str.maketrans("0123456789-", "⁰¹²³⁴⁵⁶⁷⁸⁹⁻")
        sup = lambda n: str(n).translate(SUP)

        def single(a: int, b: int, d: int) -> str:
            a %= d
            b %= d
            if a == b == 0:
                return "I"
            out = []
            if a:
                out.append("X" if a == 1 else f"X{sup(a)}")
            if b:
                out.append("Z" if b == 1 else f"Z{sup(b)}")
            return "".join(out)

        d = self._d
        n = self._X.shape[1]
        rows = self._X.shape[0]
        labels = (
            ([f"d{i}" for i in range(n)] + [f"s{i}" for i in range(n)])
            if self._full
            else [str(i) for i in range(rows)]
        )

        entries = []
        for r in range(rows):
            tau_val = int(self._tau_exp[r] % (2 * d))
            omega_power = (tau_val // 2) % d
            has_tau = tau_val & 1
            if show_omegas:
                phase = (
                    (
                        "ω"
                        if omega_power == 1
                        else "ω" + sup(omega_power) if omega_power else ""
                    )
                    + (" τ" if has_tau else "")
                ).strip() or "1"
            else:
                phase = "τ" + sup(tau_val) if tau_val else "1"
            factors = [
                single(int(self._X[r, q]), int(self._Z[r, q]), d) for q in range(n)
            ]
            entries.append([labels[r], phase, *factors])

        widths = [
            max(len(col) for col in col_vals)
            for col_vals in zip(*entries, ["#", "phase", *map(str, range(n))])
        ]
        header = " | ".join(
            txt.ljust(widths[i])
            for i, txt in enumerate(["#", "phase", *map(str, range(n))])
        )
        print(header)
        top = "─" * len(header)
        mid = "-" * len(header)
        print(top)
        for r, row in enumerate(entries):
            print(" | ".join(cell.ljust(widths[i]) for i, cell in enumerate(row)))
            if self._full and r == n - 1:
                print(mid)

    def pretty(self) -> str:
        """
        Return a compact human-readable representation (one line per row).

        Returns
        -------
        str
            Multi-line string representation.
        """
        self._normalize()
        lines = []
        rows = self._X.shape[0]
        for i in range(rows):
            x_row = self._X[i]
            z_row = self._Z[i]
            x_str = ",".join(str(_scalar_to_int(v)) for v in x_row)
            z_str = ",".join(str(_scalar_to_int(v)) for v in z_row)
            tau = _scalar_to_int(self._tau_exp[i])
            lines.append(f"Row {i}: X=[{x_str}] Z=[{z_str}] τ^{tau}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        """str: Debug representation."""
        return (
            f"Tableau(n={self._n}, d={self._d}, full={self._full}, "
            f"rows={self._X.shape[0]})"
        )

    def __str__(self) -> str:
        """
        Return a column-aligned string representation of the full tableau.

        Returns
        -------
        str
            Pretty table string.

        Raises
        ------
        ValueError
            If self.n == 0.
        """
        if self.n == 0:
            raise ValueError(f"Empty Tableau: {self.__repr__()}")

        self._normalize()
        rows = self._X.shape[0]
        n = self._n

        x_strs = [[str(_scalar_to_int(v)) for v in self._X[i]] for i in range(rows)]
        z_strs = [[str(_scalar_to_int(v)) for v in self._Z[i]] for i in range(rows)]
        tau_strs = [str(_scalar_to_int(self._tau_exp[i])) for i in range(rows)]

        x_widths = [max(len(x_strs[i][j]) for i in range(rows)) for j in range(n)]
        z_widths = [max(len(z_strs[i][j]) for i in range(rows)) for j in range(n)]
        tau_width = max(len(s) for s in tau_strs + ["τ"])

        if self._full:
            labels = [f"d{i}" for i in range(n)] + [f"s{i}" for i in range(n)]
        else:
            labels = [str(i) for i in range(rows)]
        row_w = max(len("#"), max(len(s) for s in labels))

        def rpad(s, w):
            return s.ljust(w)

        def pad(s, w):
            return s.rjust(w)

        x_headers = [f"x{j}" for j in range(n)]
        z_headers = [f"z{j}" for j in range(n)]
        for j in range(n):
            x_widths[j] = max(x_widths[j], len(x_headers[j]))
            z_widths[j] = max(z_widths[j], len(z_headers[j]))
        row_label = rpad("#", row_w)
        x_block_hdr = " ".join(pad(x_headers[j], x_widths[j]) for j in range(n))
        z_block_hdr = " ".join(pad(z_headers[j], z_widths[j]) for j in range(n))
        tau_hdr = pad("τ", tau_width)

        header = f"{row_label} | {x_block_hdr} | {z_block_hdr} | {tau_hdr}"
        sep = "─" * len(header)
        sep_full = "-" * len(header)

        lines = [header, sep]
        for i in range(rows):
            rn = rpad(labels[i], row_w)
            x_block = " ".join(pad(x_strs[i][j], x_widths[j]) for j in range(n))
            z_block = " ".join(pad(z_strs[i][j], z_widths[j]) for j in range(n))
            tau_block = pad(tau_strs[i], tau_width)
            line = f"{rn} | {x_block} | {z_block} | {tau_block}"
            lines.append(line)
            if self._full and i == n - 1:
                lines.append(sep_full)
        return "\n".join(lines)

    # -----------------------------------------------------------------------
    # Equality Check & Indexing
    # -----------------------------------------------------------------------

    def __eq__(self, other: Tableau) -> bool:
        """
        Compare two tableaux for equality up to modular normalization.

        Parameters
        ----------
        other : Tableau
            Tableau to compare against.

        Returns
        -------
        bool
            True if equal (after normalization), else False.
        """
        if not isinstance(other, Tableau):
            return False
        if self._n != other._n or self._d != other._d or self._full != other._full:
            return False
        a = self.copy()
        b = other.copy()
        a._normalize()
        b._normalize()
        if not np.array_equal(a._X, b._X):
            return False
        if not np.array_equal(a._Z, b._Z):
            return False
        if not np.array_equal(a._tau_exp, b._tau_exp):
            return False
        return True

    def __getitem__(self, idx: int) -> np.ndarray:
        """
        Bracket indexing for tableau rows.

        Equivalent to :meth:`get_row`.

        Parameters
        ----------
        idx : int
            Row index (negative supported).

        Returns
        -------
        numpy.ndarray
            1D array of length ``2n+1``.
        """
        return self.get_row(idx)

    def get_row(self, idx: int) -> np.ndarray:
        """
        Return one tableau row as ``[x0..x_{n-1}, z0..z_{n-1}, tau]``.

        Parameters
        ----------
        idx : int
            Row index. Negative indices are supported.

        Returns
        -------
        numpy.ndarray
            1D array of length ``2n+1``.

        Raises
        ------
        TypeError
            If ``idx`` is not an int.
        IndexError
            If ``idx`` is out of range.
        """
        if not isinstance(idx, int):
            raise TypeError(f"Row index must be int, got {type(idx).__name__}")

        n_rows = self.X.shape[0]
        if idx < 0:
            idx += n_rows
        if idx < 0 or idx >= n_rows:
            raise IndexError(f"Row index {idx} out of range for {n_rows} rows")

        return np.concatenate([self.X[idx], self.Z[idx], [self.tau_exp[idx]]])

    # -----------------------------------------------------------------------
    # Circuit Application
    # -----------------------------------------------------------------------

    def apply_gate(
        self,
        gate: Gate,
        targets: Union[int, tuple[int, ...], list[int]],
        dagger: bool = False,
    ) -> None:
        """
        Apply a single gate to this tableau in-place.

        This dispatches to the tableau update rules for supported Clifford gates.
        For parametrized Weyl operators, it uses ``gate.a`` and ``gate.b``.

        Parameters
        ----------
        gate : Gate
            Gate to apply.
        targets : int or tuple[int, ...] or list[int]
            Target qudit index (1-qudit) or indices (2-qudit).
        dagger : bool, optional
            Whether to apply the daggered gate. Default: False.

        Raises
        ------
        ValueError
            If the gate name is None.
        """
        if isinstance(targets, int):
            targets = (targets,)

        name = getattr(gate, "name", None)

        if name is None:
            raise ValueError("Gate missing name attribute for dispatch")

        name_upper = name.upper()

        if name_upper == "I":
            return
        elif name_upper == "X":
            self._apply_X(targets[0], dagger)
        elif name_upper == "Y":
            self._apply_Y(targets[0], dagger)
        elif name_upper == "Z":
            self._apply_Z(targets[0], dagger)
        elif name_upper == "H":
            self._apply_H(targets[0], dagger=dagger)
        elif name_upper == "S":
            self._apply_S(targets[0], dagger=dagger)
        elif name_upper in ("CNOT", "CX"):
            self._apply_CNOT(targets[0], targets[1], dagger=dagger)
        elif name_upper == "CZ":
            self._apply_CZ(targets[0], targets[1], dagger=dagger)
        elif name_upper == "SWAP":
            self._apply_SWAP(targets[0], targets[1], dagger=dagger)
        elif gate.a is not None and gate.b is not None:
            self._apply_weyl(gate.a, gate.b, targets[0], dagger)
        else:
            pass

    def _apply_circuit_python(
        self, qc: QuantumCircuit, *, print_tableau: bool = False
    ) -> Tableau:
        """
        Apply a circuit using the pure-Python (reference) update-rule engine.

        Parameters
        ----------
        qc : QuantumCircuit
            Circuit to apply.
        print_tableau : bool, optional
            If True, print the tableau after each operation. Default: False.

        Returns
        -------
        Tableau
            Self (mutated).
        """
        if print_tableau:
            print("Initial tableau:")
            print(self)

        for gate, targets, dagger in qc:
            if print_tableau:
                name = getattr(gate, "name", "").upper()
                suffix = "†" if dagger else ""
                if gate.a == 0 and gate.b == 0:
                    suffix = ""
                suffix_q = "s" if len(targets) > 1 else ""
                print(
                    f"\n--- Applying {name}{suffix} on qudit{suffix_q} {', '.join(map(str, targets))} ---"
                )
            self.apply_gate(gate, targets, dagger=dagger)
            if print_tableau:
                print(self)
        return self

    def _apply_circuit_numba(
        self, qc: QuantumCircuit, *, print_tableau: bool = False
    ) -> Tableau:
        """
        Apply a circuit using Numba-accelerated kernels (if available).

        Parameters
        ----------
        qc : QuantumCircuit
            Circuit to apply.
        print_tableau : bool, optional
            If True, print the tableau after each operation.
            Default: False.

        Returns
        -------
        Tableau
            Self (mutated).
        """
        if not _HAS_NUMBA:
            warnings.warn(
                "Numba kernels are unavailable; falling back to the Python engine. "
                f"Original import error: {_NUMBA_IMPORT_ERROR!r}",
                RuntimeWarning,
                stacklevel=2,
            )
            return self._apply_circuit_python(qc, print_tableau=print_tableau)

        G = pack_circuit_from_qc(qc)
        if G.size == 0:
            return self

        self._X = np.ascontiguousarray(self._X)
        self._Z = np.ascontiguousarray(self._Z)
        self._tau_exp = np.ascontiguousarray(self._tau_exp)

        if print_tableau:
            print("Initial tableau:")
            print(self)

            for k in range(G.shape[0]):
                gid = int(G[k, 0])
                q1 = int(G[k, 1])
                q2 = int(G[k, 2])
                dag = int(G[k, 3])
                name = ID_TO_NAME.get(gid, "?")

                if name == "W_ab":
                    name = f"W_{abs(q2)},{abs(dag)}"
                    suffix = "†" if q2 < 0 and dag < 0 else ""
                    targets = f"{q1}"
                    suffix_q = ""
                else:
                    suffix = "†" if dag else ""
                    targets = f"{q1}" if q2 < 0 else f"{q1}, {q2}"
                    suffix_q = "s" if q2 >= 0 else ""

                print(f"\n--- Applying {name}{suffix} on qudit{suffix_q} {targets} ---")
                nb_apply_gate(
                    self._X,
                    self._Z,
                    self._tau_exp,
                    gid,
                    q1,
                    q2,
                    dag,
                    self._mod_d,
                    self._mod_2d,
                )
                print(self)

            return self

        nb_apply_circuit(self._X, self._Z, self._tau_exp, G, self._mod_d, self._mod_2d)
        return self

    def apply_circuit(
        self,
        qc: QuantumCircuit,
        *,
        engine: Optional[str] = None,
        print_tableau: bool = False,
    ) -> Tableau:
        """
        Apply a :class:`~qudit_sim.circuit.QuantumCircuit` to this tableau.

        Parameters
        ----------
        qc : QuantumCircuit
            Circuit to apply.
        engine : str or None, optional
            - None (default): use Numba if available, otherwise Python.
            - "numba": force Numba kernels.
            - "python": force Python update rules.
        print_tableau : bool, optional
            If True, print the tableau after each operation. Default: False.

        Returns
        -------
        Tableau
            Self (mutated).

        Raises
        ------
        ValueError
            If ``engine`` is not one of {None, "numba", "python"} or a mismatch of n/d between tab and qc occured.
        RuntimeError
            If ``engine="numba"`` is requested but kernels are unavailable.
        """
        self._check_compat_with_circuit(qc)

        if engine is None:
            return self._apply_circuit_numba(qc, print_tableau=print_tableau)

        if engine == "python":
            return self._apply_circuit_python(qc, print_tableau=print_tableau)

        if engine == "numba":
            if not _HAS_NUMBA:
                raise RuntimeError(
                    "Numba engine requested but kernels are unavailable. "
                    f"Original import error: {_NUMBA_IMPORT_ERROR!r}"
                )
            return self._apply_circuit_numba(qc, print_tableau=print_tableau)

        raise ValueError("engine must be None, 'numba', or 'python'.")

    # -----------------------------------------------------------------------
    # Update Rules
    # -----------------------------------------------------------------------

    def _apply_I(self, j: int, dagger: bool = False) -> None:
        """
        Identity on qudit j: no change.

        X / Z tableau entries unchanged.
        τ-exponent unchanged.

        Parameters
        ----------
        j : int
            Qudit index.
        dagger : bool, default=False
            Whether to apply the adjoint (†). Ignored for identity.
        """
        return

    def _apply_X(self, j: int, dagger: bool = False) -> None:
        """
        **X** (or **X†**) on qudit *j*.

        Update rules
        ------------
        ::

            τ  ←  τ  -  dir · 2 · Z_j (mod 2·d)

        No change to X / Z tableau entries.

        Parameters
        ----------
        j : int
            Qudit index.
        dagger : bool, default=False
            If False, apply **X**. If True, apply **X†**.
        """
        dir_ = 1 if not dagger else -1
        zj = self._Z[:, j]
        self._tau_exp = (self._tau_exp - dir_ * 2 * zj) % self._mod_2d

    def _apply_Y(self, j: int, dagger: bool = False) -> None:
        """
        Pauli **Y = τ^{-1}·X·Z** (and **Y†**) on qudit *j*.

        Conventions
        -----------
        dir = +1  →  Y          (conjugation by Y)

        dir = -1  →  Y†         (conjugation by Y†)

        Update rules
        ------------
        ::

            Let  x = X_j ,  z = Z_j .

            τ  ←  τ  +  dir · 2·(x - z) (mod 2·d)

        No change to X / Z tableau entries.

        Parameters
        ----------
        j : int
            Qudit index.
        dagger : bool, default=False
            If False, apply **Y**. If True, apply **Y†**.
        """
        dir_ = 1 if not dagger else -1
        xj = self._X[:, j]
        zj = self._Z[:, j]
        self._tau_exp = (self._tau_exp + dir_ * 2 * (xj - zj)) % self._mod_2d

    def _apply_Z(self, j: int, dagger: bool = False) -> None:
        """
        **Z** (or **Z†**) on qudit *j*.

        Update rules
        ------------
        ::

            τ  ←  τ  +  dir · 2 · X_j (mod 2·d)

        No change to X / Z tableau entries.

        Parameters
        ----------
        j : int
            Qudit index.
        dagger : bool, default=False
            If False, apply **Z**. If True, apply **Z†**.
        """
        dir_ = 1 if not dagger else -1
        xj = self._X[:, j]
        self._tau_exp = (self._tau_exp + dir_ * 2 * xj) % self._mod_2d

    def _apply_H(self, j: int, dagger: bool = False) -> None:
        """
        Hadamard on qudit *j* († if *dagger* is True).

        Conventions
        -----------
        • `dir = +1`  →  **H**

        • `dir = -1`  →  **H†**

        Update rules
        ------------
        Let *(x_old, z_old)* be the column (X, Z) of qudit *j* **before** the gate.
        ::

            X_j  ←  -dir · z_old (mod d)

            Z_j  ←   dir · x_old (mod d)

        Phase column (τ-exponents, modulus 2d):
        ::

            τ   ←   τ  +  2 · X_j · Z_j (mod 2·d)

        Parameters
        ----------
        j : int
            Qudit index.
        dagger : bool, default=False
            If False, apply **H**. If True, apply **H†**.
        """
        dir_ = 1 if not dagger else -1

        x_old = self._X[:, j].copy()
        z_old = self._Z[:, j].copy()

        self._X[:, j] = (-dir_ * z_old) % self._mod_d
        self._Z[:, j] = (dir_ * x_old) % self._mod_d

        x_new = self._X[:, j]
        z_new = self._Z[:, j]
        self._tau_exp = (self._tau_exp + 2 * x_new * z_new) % self._mod_2d

    def _apply_S(self, j: int, dagger: bool = False) -> None:
        """
        Phase gate **S** († if *dagger* is True) on qudit *j*.

        Conventions
        -----------
        • `dir = +1`  →  **S**

        • `dir = -1`  →  **S†**

        Update rules
        ------------
        Let *x = X_j* (before the gate).
        ::

            Z_j  ←  Z_j  +  dir · x (mod d)

            τ   ←  τ  +  dir · Δ (mod 2·d)

                with  Δ = x² (even d)

                Δ = x(x - 1) (odd  d)

        No change to any other tableau entry.

        Parameters
        ----------
        j : int
            Qudit index.
        dagger : bool, default=False
            If False, apply **S**. If True, apply **S†**.
        """
        dir_ = 1 if not dagger else -1

        xj = self._X[:, j]
        self._Z[:, j] = (self._Z[:, j] + dir_ * xj) % self._mod_d

        delta = xj * xj if self._mod_d % 2 == 0 else xj * (xj - 1)
        self._tau_exp = (self._tau_exp + dir_ * delta) % self._mod_2d

    def _apply_CNOT(self, c: int, t: int, dagger: bool = False) -> None:
        """
        CNOT/SUM between control *c* and target *t* († if *dagger* is True).

        Conventions
        -----------
        • `dir = +1`  →  **CNOT**

        • `dir = -1`  →  **CNOT†**

        Update rules
        ------------
        ::

            X_t  ←  X_t  +  dir · X_c (mod d)

            Z_c  ←  Z_c  -  dir · Z_t (mod d)

        Phase column **unchanged**.

        Parameters
        ----------
        c : int
            Control qudit index.
        t : int
            Target qudit index.
        dagger : bool, default=False
            If False, apply **CNOT**. If True, apply **CNOT†**.
        """
        if c == t:
            return

        dir_ = 1 if not dagger else -1

        self._X[:, t] = (self._X[:, t] + dir_ * self._X[:, c]) % self._mod_d
        self._Z[:, c] = (self._Z[:, c] - dir_ * self._Z[:, t]) % self._mod_d

    def _apply_CZ(self, c: int, t: int, dagger: bool = False) -> None:
        """
        CZ (controlled phase) between qudits *c* and *t* († if *dagger* is True).

        Basis action
        ------------
        ::

            |c, t⟩  →  ω^{dir · c t} |c, t⟩,  where  ω = exp(2·πi / d).

        Conventions
        -----------
        • `dir = +1`  →  **CZ**

        • `dir = -1`  →  **CZ†**

        Update rules
        ------------
        ::

            Z_{c}  ←  Z_{c}  +  dir · X_{t} (mod d)

            Z_{t}  ←  Z_{t}  +  dir · X_{c} (mod d)

            τ      ←  τ      +  2 · dir · X_{c} X_{t} (mod 2·d)

        `X` columns **unchanged** (only Z-columns and the phase exponent τ update).

        Parameters
        ----------
        c : int
            First qudit index.
        t : int
            Second qudit index.
        dagger : bool, default=False
            If False, apply **CZ**. If True, apply **CZ†**.
        """
        if c == t:
            return

        dir_ = -1 if dagger else 1

        self.Z[:, c] = (self.Z[:, c] + dir_ * self.X[:, t]) % self._mod_d
        self.Z[:, t] = (self.Z[:, t] + dir_ * self.X[:, c]) % self._mod_d
        self.tau_exp[:] = (
            self.tau_exp + 2 * dir_ * (self.X[:, c] * self.X[:, t])
        ) % self._mod_2d

    def _apply_SWAP(self, q1: int, q2: int, dagger: bool = False) -> None:
        """
        SWAP between qudits *q1* and *q2* († ignored — SWAP = SWAP†).

        Basis action
        ------------
        ::

            |q1, q2⟩  →  |q2, q1⟩

        Conventions
        -----------
        • SWAP is Hermitian and involutory:  SWAP = SWAP†  and  SWAP² = I

        • The ``dagger`` flag is ignored.

        Update rules
        ------------
        Exchange the X/Z columns of the two qudits in every stabilizer row:
        ::

            X_{⋅, q1} ↔ X_{⋅, q2}

            Z_{⋅, q1} ↔ Z_{⋅, q2}

        Phase column **unchanged** (τ is not modified).
        No other tableau entries change.

        Parameters
        ----------
        q1 : int
            First qudit index.
        q2 : int
            Second qudit index.
        dagger : bool, default=False
            Ignored (SWAP = SWAP†).
        """
        if q1 == q2:
            return

        self._X[:, [q1, q2]] = self._X[:, [q2, q1]]
        self._Z[:, [q1, q2]] = self._Z[:, [q2, q1]]

    def _apply_weyl(self, a: int, b: int, j: int, dagger: bool = False) -> None:
        """
        Applies the Weyl operator W_ab = τ^(-ab) X^a · Z^b to qudit j

        Update rule:
        ::

            τ  ←  τ  +  (2·b·X_j  -  2·a·Z_j)  (mod 2·d)

        The X and Z matrices are unchanged.

        Parameters
        ----------
        a : int
            Exponent of X in W_ab.
        b : int
            Exponent of Z in W_ab.
        j : int
            Qudit index.
        dagger : bool, default=False
            If False, apply W_ab. If True, apply its adjoint.
        """
        if a == 0 and b == 0:
            return

        dir_ = -1 if dagger else 1

        self._tau_exp = (
            self._tau_exp + 2 * dir_ * b * self._X[:, j] - 2 * dir_ * a * self._Z[:, j]
        ) % self._mod_2d

    # -----------------------------------------------------------------------
    # Statevector Extraction (expensive, intended for small n)
    # -----------------------------------------------------------------------

    def _statevector_from_blocks(
        self,
        X_block: np.ndarray,
        Z_block: np.ndarray,
        tau_exp: np.ndarray,
        d: int,
        *,
        normalize: bool = True,
    ) -> np.ndarray:
        """
        Construct the joint +1 eigenstate for a stabilizer set.

        This explicitly constructs stabilizer operators and diagonalizes their sum,
        which is only practical for small systems.

        Parameters
        ----------
        X_block : numpy.ndarray
            X exponents, shape ``(n_rows, n_qudits)``.
        Z_block : numpy.ndarray
            Z exponents, same shape as ``X_block``.
        tau_exp : numpy.ndarray
            Tau exponents, shape ``(n_rows,)``.
        d : int
            Dimension.
        normalize : bool, optional
            If True, normalize the resulting vector. Default: True.

        Returns
        -------
        numpy.ndarray
            Statevector of length ``d**n_qudits``.

        Raises
        ------
        ValueError
            If shapes are incompatible or no joint +1 eigenvector is found.
        """
        n_rows, n_qudits = X_block.shape
        if Z_block.shape != (n_rows, n_qudits):
            raise ValueError("Shape mismatch between X_block and Z_block.")
        if tau_exp.shape[0] != n_rows:
            raise ValueError("tau_exp length must match number of rows in X/Z blocks.")

        # Build single-qudit X and Z matrices
        X_mat = gates.X(d).astype(backend.COMPLEX_DTYPE)
        Z_mat = gates.Z(d).astype(backend.COMPLEX_DTYPE)

        # τ² = ω
        tau = np.exp(1j * np.pi * (d**2 + 1) / d)

        # Assemble each stabilizer operator
        mats = []
        for r in range(n_rows):
            factors = []
            for q in range(n_qudits):
                x = int(X_block[r, q])
                z = int(Z_block[r, q])

                # operator = X^x · Z^z
                op_q = np.linalg.matrix_power(X_mat, x) @ np.linalg.matrix_power(
                    Z_mat, z
                )
                factors.append(op_q)

            # Tensor them together to form the full d**n_op x d**n_op operator
            full_op = factors[0]
            for op_q in factors[1:]:
                full_op = np.kron(full_op, op_q)

            # Apply the global τ-phase from the tableau
            mats.append((tau ** int(tau_exp[r])) * full_op)

        # Find the joint +1 eigenvector of the sum of stabilizer operators
        M = sum(mats)
        w, v = np.linalg.eig(M)
        inds = np.where(np.isclose(w, len(mats)))[0]
        if inds.size == 0:
            raise ValueError("No +1 eigenvector found for this stabilizer set.")
        vec = v[:, int(inds[0])]
        if normalize:
            vec = vec / np.linalg.norm(vec)
            vec = vec.astype(backend.COMPLEX_DTYPE, copy=False)
        return vec

    def to_statevector(self, normalize: bool = True) -> np.ndarray:
        """
        Return the ``n``-qudit statevector corresponding to this tableau.

        For a full tableau, only the stabilizer half (rows ``n..2n-1``) is used.

        Parameters
        ----------
        normalize : bool, optional
            If True, normalize the output vector. Default: True.

        Returns
        -------
        numpy.ndarray
            Statevector of length ``d**n``.

        Raises
        ------
        ValueError
            If the tableau does not define a unique +1 eigenstate.
        """
        self._normalize()
        d, n = self._d, self._n

        if self._full:
            # take stabilizers only
            Xb = self._X[n:, :]
            Zb = self._Z[n:, :]
            ta = self._tau_exp[n:]
        else:
            Xb = self._X
            Zb = self._Z
            ta = self._tau_exp

        return self._statevector_from_blocks(Xb, Zb, ta, d, normalize=normalize)

    def to_statevector_universal(self, normalize: bool = True) -> np.ndarray:
        """
        Return the 2n-qudit universal (Choi-like) statevector, length d**(2n).

        Requires the internal X/Z to have shape (2n, n).

        Parameters
        ----------
        normalize : bool, optional
            If True, normalize the output vector. Default: True.

        Returns
        -------
        numpy.ndarray
            Statevector of length ``d**(2n)``.

        Raises
        ------
        ValueError
            If tableau shapes are incompatible or no joint +1 eigenstate is found.
        """
        self._normalize()
        d, n = self._d, self._n

        # Part A: The Evolved Half (from the 2n x n universal tableau)
        X_evolved = self._X
        Z_evolved = self._Z
        tau_exp = self._tau_exp

        # shape checks
        if (
            X_evolved.shape != (2 * n, n)
            or Z_evolved.shape != (2 * n, n)
            or tau_exp.shape[0] != 2 * n
        ):
            raise ValueError(
                "Tableau shape incompatible with universal/Choi construction: "
                f"X/Z must be (2n, n); got X={X_evolved.shape}, Z={Z_evolved.shape}."
            )

        # Part B: The Unchanged Half (initial basis operators on the second register)
        X_unchanged = np.vstack(
            [
                np.eye(n, dtype=self._xz_dtype),
                np.zeros((n, n), dtype=self._xz_dtype),
            ]
        )
        Z_unchanged = np.vstack(
            [
                np.zeros((n, n), dtype=self._xz_dtype),
                -np.eye(n, dtype=self._xz_dtype),  # Z† exponents
            ]
        )

        # Combine them to form the 2n x 2n tableau for the Choi state stabilizers
        X_block = np.hstack([X_evolved, X_unchanged])
        Z_block = np.hstack([Z_evolved, Z_unchanged])

        return self._statevector_from_blocks(
            X_block, Z_block, tau_exp, d, normalize=normalize
        )

    # -----------------------------------------------------------------------
    # Measurement (for prime d & full=True)
    # -----------------------------------------------------------------------

    def _is_prime(self, d: int) -> bool:
        """
        Check whether an integer is prime.

        Parameters
        ----------
        d : int
            Integer to test.

        Returns
        -------
        bool
            True if `d` is prime, else False.
        """
        if d < 2:
            return False
        if d in (2, 3):
            return True
        if d % 2 == 0 or d % 3 == 0:
            return False
        i = 5
        while i * i <= d:
            if d % i == 0 or d % (i + 2) == 0:
                return False
            i += 6
        return True

    @staticmethod
    def _coerce_rng(rng=None, seed=None) -> np.random.Generator:
        """
        Return a NumPy random Generator.

        Parameters
        ----------
        rng : numpy.random.Generator or None, optional
            If provided, return this RNG.
        seed : int or None, optional
            If `rng` is None and `seed` is provided, create a fresh Generator from `seed`.

        Returns
        -------
        numpy.random.Generator
            RNG to use for this call.
        """
        if rng is not None:
            return rng
        return np.random.default_rng(seed)

    @staticmethod
    @lru_cache(maxsize=None)
    def _mod_inverse(a: int, m: int) -> int:
        """
        Modular multiplicative inverse of `a` modulo `m`.

        Parameters
        ----------
        a : int
            Value to invert.
        m : int
            Modulus.

        Returns
        -------
        int
            `a^{-1} mod m`.
        """
        return pow(a, -1, m)

    def _product_logic(
        self,
        X1: np.ndarray,
        Z1: np.ndarray,
        tau1: int,
        X2: np.ndarray,
        Z2: np.ndarray,
        tau2: int,
    ) -> tuple[np.ndarray, np.ndarray, int]:
        """
        Multiply two stabilizer rows, g1 * g2, given raw row components.

        Parameters
        ----------
        X1, Z1 : np.ndarray
            X and Z components of the first row.
        tau1 : int
            τ-exponent of the first row (mod 2d).
        X2, Z2 : np.ndarray
            X and Z components of the second row.
        tau2 : int
            τ-exponent of the second row (mod 2d).

        Returns
        -------
        tuple[np.ndarray, np.ndarray, int]
            (new_X_row, new_Z_row, new_tau_exp).
        """
        # New X and Z parts are the element-wise sum mod d
        new_X = (X1 + X2) % self._mod_d
        new_Z = (Z1 + Z2) % self._mod_d

        # Phase update rule: phase_new = phase_1 + phase_2 + 2 * sum(Z1 * X2)
        phase_update = 2 * np.sum(Z1 * X2)
        new_tau_exp = (tau1 + tau2 + phase_update) % self._mod_2d

        return new_X, new_Z, new_tau_exp

    def _get_row_product(self, i: int, j: int) -> tuple[np.ndarray, np.ndarray, int]:
        """
        Multiply two stabilizer rows g_i * g_j by index.

        Parameters
        ----------
        i : int
            Row index of g_i.
        j : int
            Row index of g_j.

        Returns
        -------
        tuple[np.ndarray, np.ndarray, int]
            (new_X_row, new_Z_row, new_tau_exp).
        """
        return self._product_logic(
            self.X[i], self.Z[i], self.tau_exp[i], self.X[j], self.Z[j], self.tau_exp[j]
        )

    def _power(
        self,
        base_X: np.ndarray,
        base_Z: np.ndarray,
        base_tau: int,
        m: int,
    ) -> tuple[np.ndarray, np.ndarray, int]:
        """
        Compute (stabilizer row)^m using exponentiation by squaring.

        Parameters
        ----------
        base_X, base_Z : np.ndarray
            Base row X/Z.
        base_tau : int
            Base row τ-exponent (mod 2d).
        m : int
            Exponent.

        Returns
        -------
        tuple[np.ndarray, np.ndarray, int]
            (X, Z, tau) for the powered row.
        """
        # Exponentiation by squaring for stabilizer rows
        res_X, res_Z, res_tau = np.zeros_like(base_X), np.zeros_like(base_Z), 0

        # S^d = I, so we only need powers up to d.
        # S^0 = I, which is handled by the loop initialization.
        m %= self.d
        if m == 0:
            return res_X, res_Z, res_tau

        temp_X, temp_Z, temp_tau = base_X, base_Z, base_tau

        while m > 0:
            if m % 2 == 1:  # If exponent is odd, multiply by the current power of S
                res_X, res_Z, res_tau = self._product_logic(
                    res_X, res_Z, res_tau, temp_X, temp_Z, temp_tau
                )
            # Square the current power of S for the next iteration
            temp_X, temp_Z, temp_tau = self._product_logic(
                temp_X, temp_Z, temp_tau, temp_X, temp_Z, temp_tau
            )
            m //= 2

        return res_X, res_Z, res_tau

    def _row_add_mult(self, k: int, p: int, m: int) -> None:
        """
        Generalized row operation: row_k -> row_k + m * row_p.

        In stabilizer terms: g_k -> g_k * (g_p)^m.

        Parameters
        ----------
        k : int
            Destination row index.
        p : int
            Pivot/source row index.
        m : int
            Multiplier (mod d).
        """
        if m == 0:
            return

        # Calculate (g_p)^m
        g_p_m_X, g_p_m_Z, g_p_m_tau = self._power(
            self.X[p], self.Z[p], self.tau_exp[p], m
        )

        # Multiply g_k by the result
        self.X[k], self.Z[k], self.tau_exp[k] = self._product_logic(
            self.X[k], self.Z[k], self.tau_exp[k], g_p_m_X, g_p_m_Z, g_p_m_tau
        )

    def measure(
        self,
        qudits_to_measure: Union[int, list[int], tuple[int, ...]],
        desired: Optional[Union[int, list[int], tuple[int, ...]]] = None,
        rng: Optional[np.random.Generator] = None,
        seed: Optional[int] = None,
        in_place: bool = False,
        return_measure_info: bool = False,
    ) -> Union[tuple[Tableau, dict], tuple[Tableau, dict, dict]]:
        """
        Measures one or more qudits in the tableau using the CHP algorithm.
        Measurements happen in order of ascending qudit index, regardless of the input order.

        This method simulates a projective measurement in the Z-basis for each specified
        qudit. The tableau is updated to reflect the collapsed state after measurement.
        The algorithm is optimized to have a runtime complexity of O(q*n^2*log(d)), where
        q is the number of qudits being measured; O(n^3*log(d)) if measuring all qudits.

        Parameters
        ----------
        qudits_to_measure : int or list[int] or tuple[int, ...]
            The index or indices of the qudit(s) to measure.
        desired : int or list[int] or tuple[int, ...] or None
            If provided, forces the measurement outcome(s) to the specified value(s).
            If None (default), outcomes are random.
            If a list/tuple, its length must match the number and order of qudits being measured.
            If a single int, it can only be used when measuring a single qudit.
            If the outcome on the qudit is deterministic, the forced value is ignored;
            a tuple (det_value, "det") is stored in the outcomes dictionary.
            If the outcome on the qudit is random, a tuple (forced_value, "rand") is stored;
            tableau is updated accordingly to the forced value.
            If desired is out of range [0, d-1], a ValueError is raised.
        rng : np.random.Generator, optional
            RNG to use. If None, one is created from `seed`.
            If both provided, `rng` is used.
        seed : int, optional
            Seed to create a fresh RNG if `rng` is None.
        in_place : bool, optional
            If False (default), a copy of the tableau is measured and returned.
            If True, the measurement is performed directly on the current tableau.
        return_measure_info : bool, optional
            If True, returns pivot row's tau_exp (deterministic case) or pivot row's index (random)
            to use in the post_measure method.

        Returns
        -------
        (Tableau, dict) or (Tableau, dict, dict)
            If `return_measure_info=False`: (tableau_after, outcomes)
            If `return_measure_info=True`:  (tableau_after, outcomes, measurement_info)

        Raises
        ------
        ValueError
            If the tableau is not full.
        ValueError
            If d is not prime.
        ValueError
            If `qudits_to_measure` contains duplicates.
        ValueError
            If a qudit index is out of bounds.
        TypeError
            If `desired` is not int, list, tuple, or None.
        ValueError
            If `desired` is a list/tuple with wrong length.
        ValueError
            If `desired` is scalar while measuring multiple qudits.
        ValueError
            If any desired outcome is outside [0, d-1].
        """
        if not self._full:
            raise ValueError("Measurement is only supported for full tableaus.")
        if not self._is_prime(self._d):
            raise ValueError(
                f"Measurement is only supported for prime dimensions, but d={self._d}."
            )

        rng = self._coerce_rng(rng, seed)

        if isinstance(qudits_to_measure, (int, np.integer)):
            qudits_to_measure = [qudits_to_measure]
        else:
            qudits_to_measure = list(qudits_to_measure)

        if len(set(qudits_to_measure)) != len(qudits_to_measure):
            raise ValueError("Duplicate qudit indices in qudits_to_measure.")

        desired_map: dict[int, int] = {}
        if desired is not None:
            if isinstance(desired, (list, tuple)):
                if len(desired) != len(qudits_to_measure):
                    raise ValueError(
                        "If `desired` is a list/tuple, it must match the number of measured qudits."
                    )
                for q, v in zip(qudits_to_measure, desired):
                    v_int = int(v)
                    if not (0 <= v_int < self._d):
                        raise ValueError(
                            f"Desired outcome {v_int} for qudit {q} not in [0, {self._d - 1}]"
                        )
                    desired_map[q] = v_int
            elif isinstance(desired, int):
                if len(qudits_to_measure) != 1:
                    raise ValueError(
                        "Scalar `desired` can only be used when measuring a single qudit."
                    )
                v_int = int(desired)
                if not (0 <= v_int < self._d):
                    raise ValueError(
                        f"Desired outcome {v_int} not in [0, {self._d - 1}]"
                    )
                desired_map[qudits_to_measure[0]] = v_int
            else:
                raise TypeError("`desired` must be int, list, tuple, or None.")

        qudits_to_measure = sorted(qudits_to_measure)

        target_tableau = self if in_place else self.copy()
        n, d, mod_d = target_tableau.n, target_tableau.d, target_tableau.mod_d
        X, Z, tau_exp = target_tableau.X, target_tableau.Z, target_tableau.tau_exp

        outcomes = {}

        measurement_info = {} if return_measure_info else None

        for i in qudits_to_measure:  # Measure qudit i
            if not (0 <= i < n):
                raise ValueError(f"Qudit index {i} is out of bounds for n={n}.")

            stab_X = X[n:]
            # Find a stabilizer with an X component on qudit i
            pivot_row_relative_idx = np.where(stab_X[:, i] != 0)[0]

            if len(pivot_row_relative_idx) > 0:
                # --- Case 1: Random Measurement ---
                p = pivot_row_relative_idx[0]
                p_full_idx = p + n

                if return_measure_info:
                    measurement_info[i] = (p_full_idx, "rand")

                # Direct Row Reduction
                pivot_val_inv = target_tableau._mod_inverse(
                    _scalar_to_int(X[p_full_idx, i]), d
                )
                for k in range(2 * n):
                    if k != p_full_idx and X[k, i] != 0:
                        m = (mod_d - _scalar_to_int(X[k, i])) * pivot_val_inv % mod_d
                        target_tableau._row_add_mult(k, p_full_idx, m)

                # Normalization
                base_X, base_Z, base_tau = (
                    X[p_full_idx].copy(),
                    Z[p_full_idx].copy(),
                    tau_exp[p_full_idx],
                )

                m = target_tableau._mod_inverse(_scalar_to_int(base_X[i]), d)
                normalized_X, normalized_Z, normalized_tau = target_tableau._power(
                    base_X, base_Z, base_tau, m
                )

                # Assign the final result as the new destabilizer at row p
                X[p], Z[p], tau_exp[p] = normalized_X, normalized_Z, normalized_tau

                # Set the new stabilizer at row p_full_idx to be Z_i with a phase for the outcome
                X[p_full_idx, :] = 0
                Z[p_full_idx, :] = 0
                Z[p_full_idx, i] = 1

                if desired is None:
                    # Original behavior
                    outcome = int(rng.integers(0, d))
                    outcomes[i] = outcome
                    tau_exp[p_full_idx] = (-2 * outcome) % target_tableau.mod_2d
                else:
                    # Desired behavior
                    forced = desired_map[i]
                    outcomes[i] = (forced, "rand")
                    tau_exp[p_full_idx] = (-2 * forced) % target_tableau.mod_2d

            else:
                # --- Case 2: Deterministic Measurement ---
                destab_X = X[:n]
                temp_X, temp_Z, temp_tau_exp = (
                    np.zeros(n, dtype=target_tableau._xz_dtype),
                    np.zeros(n, dtype=target_tableau._xz_dtype),
                    0,
                )
                destab_indices_with_xi = np.where(destab_X[:, i] != 0)[0]

                for j in destab_indices_with_xi:
                    stab_to_add_idx = j + n
                    m = _scalar_to_int(X[j, i])

                    # Calculate (S_j)^m
                    powered_stab_X, powered_stab_Z, powered_stab_tau = (
                        target_tableau._power(
                            X[stab_to_add_idx],
                            Z[stab_to_add_idx],
                            tau_exp[stab_to_add_idx],
                            m,
                        )
                    )
                    # Add to the accumulator
                    temp_X, temp_Z, temp_tau_exp = target_tableau._product_logic(
                        temp_X,
                        temp_Z,
                        temp_tau_exp,
                        powered_stab_X,
                        powered_stab_Z,
                        powered_stab_tau,
                    )

                outcome = int((-temp_tau_exp // 2) % mod_d)

                if desired is None:
                    # Original behavior
                    outcomes[i] = outcome
                else:
                    # Desired behavior (ignored since deterministic)
                    outcomes[i] = (outcome, "det")

                if return_measure_info:
                    measurement_info[i] = (temp_tau_exp % target_tableau.mod_2d, "det")

        if return_measure_info:
            return target_tableau, outcomes, measurement_info

        return target_tableau, outcomes

    def measure_all(
        self,
        desired: Optional[Union[list[int], tuple[int, ...]]] = None,
        rng: Optional[np.random.Generator] = None,
        seed: Optional[int] = None,
        in_place: bool = False,
    ) -> tuple[Tableau, np.ndarray]:
        """
        Measures all qudits in the tableau by calling the measure method.

        If `desired` is None (default):
        Returns (tableau, outcomes) where `outcomes` is a 1D array of ints of length n,
        corresponding to the measurement outcomes on each qudit in ascending order.

        If `desired` is provided (list/tuple of length n, in ascending qudit order),
        it forces the measurement outcome(s) to the specified value(s).
        Returns (tableau, outcomes) where:
        where `outcomes` is a 1D array of tuples (int, str) of length n,
        where each tuple is (forced_value, "rand") or (det_value, "det"),
        depending on whether the outcome was random or deterministic.
        If the measurement outcome on a qudit for which there is a value in desired is deterministic,
        the forced value is ignored, and you get (det_value, "det") in the outcomes array.

        If you pass a seed, a single Generator is created
        for the whole shot so each qudit consumes the next random number, i.e. you
        get reproducible but *different* values across qudits.

        Parameters
        ----------
        desired : list[int] or tuple[int, ...] or None, optional
            Desired outcomes (ascending qudit order).
        rng : np.random.Generator, optional
            RNG to use. If None, one is created from `seed`.
        seed : int, optional
            Seed to create a fresh RNG if `rng` is None.
        in_place : bool, optional
            If False (default), a copy is measured. If True, measure in-place.

        Returns
        -------
        (Tableau, np.ndarray)
            Tableau after measurement and a 1D array of outcomes (ascending qudit order).

        Raises
        ------
        ValueError
            Propagated from `measure()` (e.g. non-full tableau, non-prime d, etc.).
        TypeError
            Propagated from `measure()` (e.g. invalid `desired` type).
        """
        rng = self._coerce_rng(rng, seed)
        target_tableau, outcomes_dict = self.measure(
            qudits_to_measure=range(self.n), desired=desired, rng=rng, in_place=in_place
        )

        outcomes_array = np.array([outcomes_dict[i] for i in range(self.n)])

        return target_tableau, outcomes_array

    def post_measure(
        self,
        qudits_to_measure: Union[int, list[int], tuple[int, ...]],
        desired: Optional[Union[int, list[int], tuple[int, ...]]] = None,
        rng: Optional[np.random.Generator] = None,
        seed: Optional[int] = None,
        in_place: bool = False,
    ) -> tuple[Tableau, dict[int, object]]:
        """
        Performs projective measurement(s) on the specified qudits and returns a reduced tableau.

        This method simulates the measurement of one or more qudits in the computational (Z) basis,
        updates the state, and removes the measured qudits from the system (partial trace).

        Algorithm Steps:
        1. Determine the measurement outcome (random or deterministic) for the target qudit.
        2. If Random:
           - Identify the pivot row (the destabilizer of the measured qudit).
           - Eliminate the Z-component of the measured qudit from all other stabilizers.
           - Mark the pivot row for removal.
           - Destabilizers are not updated in this step.
        3. If Deterministic:
           - Construct a virtual measurement operator from the outcome.
           - Eliminate the Z-component of the measured qudit from all stabilizers.
           - Perform Gaussian Elimination (RREF) on the remaining stabilizers to identify
             the linearly dependent (redundant) row, while updating destabilizers to
             maintain the symplectic structure.
           - Mark the redundant row for removal.
        4. Remove the measured columns and the identified rows (stabilizer + paired destabilizer).

        If `qudits_to_measure` is a list, measurements are performed sequentially
        in descending order of their indices to preserve index validity for subsequent steps.

        Parameters
        ----------
        qudits_to_measure : int or list[int] or tuple[int]
            Index or indices of qudits to measure.
        desired : int or list[int] or tuple[int] or None
            Outcomes to force (if possible).
        rng :
            A numpy.random.Generator to draw randomness from, or
        seed :
            An int to seed a local Generator for this call only.
        in_place : bool
            If True, modifies this tableau instance. If False, returns a new one.

        Returns
        -------
        (Tableau, dict[int, object])
            The reduced tableau and a dictionary of outcomes {qudit_index: result}.

        Raises
        ------
        ValueError
            Propagated from `measure()` (e.g. non-full tableau, non-prime d, etc.).
        TypeError
            If measuring multiple qudits but `desired` is a scalar.
        ValueError
            If measuring multiple qudits and `desired` length does not match.
        """

        # ==============================================================================
        # MULTI-QUDIT SEQUENTIAL HANDLER
        # ==============================================================================

        # check if input is an iterable (list, tuple, ndarray)
        is_scalar = isinstance(qudits_to_measure, (int, np.integer))
        if not is_scalar:
            target_tab = self if in_place else self.copy()

            qudits_list = list(qudits_to_measure)

            if desired is None:
                desired_list = [None] * len(qudits_list)
            elif isinstance(desired, (int, np.integer)):
                raise TypeError(
                    "If measuring multiple qudits, 'desired' must be a list/tuple."
                )
            else:
                desired_list = list(desired)
                if len(desired_list) != len(qudits_list):
                    raise ValueError(
                        "Length of 'desired' must match 'qudits_to_measure'."
                    )

            combined = sorted(
                zip(qudits_list, desired_list), key=lambda x: x[0], reverse=True
            )

            all_outcomes = {}

            rng = self._coerce_rng(rng, seed)

            # Apply post_measure sequentially
            for q, des in combined:
                # We recurse with in_place=True because we are working on 'target_tab'
                _, out = target_tab.post_measure(
                    q, desired=des, rng=rng, seed=seed, in_place=True
                )
                all_outcomes.update(out)

            all_outcomes = dict(sorted(all_outcomes.items()))

            return target_tab, all_outcomes

        # ==============================================================================
        # SINGLE-QUDIT LOGIC
        # ==============================================================================

        q = int(qudits_to_measure)

        target_tab = self if in_place else self.copy()

        # perform the standard measurement
        target_tab, outcomes, measure_info = target_tab.measure(
            qudits_to_measure=q,
            desired=desired,
            rng=rng,
            seed=seed,
            in_place=True,
            return_measure_info=True,
        )

        n, d, mod_d = target_tab.n, target_tab.d, target_tab.mod_d
        rows_to_remove = set()

        val, type_tag = measure_info[q]

        # Construct the measurement operator S_op
        if type_tag == "rand":
            # in the random case "val" is the pivot row index
            pivot_idx = val
            rows_to_remove.add(pivot_idx)

            op_X = target_tab.X[pivot_idx]
            op_Z = target_tab.Z[pivot_idx]
            op_tau = target_tab.tau_exp[pivot_idx]

        else:
            # in the deterministic case "val" is temp_tau_exp
            pivot_idx = -1
            op_X = np.zeros(n, dtype=target_tab._xz_dtype)
            op_Z = np.zeros(n, dtype=target_tab._xz_dtype)
            op_Z[q] = 1
            op_tau = val

        # Eliminate Z_q from all other stabilizers
        for r in range(n, 2 * n):
            if r == pivot_idx:
                continue
            if r in rows_to_remove:
                continue

            val_Z = int(target_tab.Z[r, q])
            if val_Z != 0:
                m = (mod_d - val_Z) % mod_d

                # Update Stabilizer (Phase and Matrix)
                op_m_X, op_m_Z, op_m_tau = target_tab._power(op_X, op_Z, op_tau, m)
                target_tab.X[r], target_tab.Z[r], target_tab.tau_exp[r] = (
                    target_tab._product_logic(
                        target_tab.X[r],
                        target_tab.Z[r],
                        target_tab.tau_exp[r],
                        op_m_X,
                        op_m_Z,
                        op_m_tau,
                    )
                )

                # IMPORTANT NOTE: we should not update the paired destabilizers here,
                # because it'd break the anticommutation relations between S_r and D_r

                # regarding the pivot's destabilizer, we don't need to update it,
                # because we discard it in the end anyway

        # Identify Redundant Row (Deterministic Only)
        if type_tag == "det":
            # Get indices of stabilizers we are keeping so far
            candidates = [r for r in range(n, 2 * n) if r not in rows_to_remove]

            valid_cols = [c for c in range(n) if c != q]

            # We create a list of (qudit, is_z) tuples to iterate over 2*m columns.
            # 0 = X dimension, 1 = Z dimension
            logical_cols = []
            for c in valid_cols:
                logical_cols.append((c, 0))  # X column
                logical_cols.append((c, 1))  # Z column

            pivot_idx_rref = 0

            for col, c_type in logical_cols:
                if pivot_idx_rref >= len(candidates):
                    break

                # Find a candidate row with a non-zero value in this logical column
                found_pivot = False
                for i in range(pivot_idx_rref, len(candidates)):
                    r = candidates[i]

                    # Select value based on whether we are looking at X or Z dim
                    val = target_tab.Z[r, col] if c_type == 1 else target_tab.X[r, col]

                    if val != 0:
                        # Swap candidate rows
                        candidates[pivot_idx_rref], candidates[i] = (
                            candidates[i],
                            candidates[pivot_idx_rref],
                        )
                        found_pivot = True
                        break

                if not found_pivot:
                    continue

                # Eliminate this column from other rows
                curr_r = candidates[pivot_idx_rref]

                # Value of the pivot element
                pivot_val = (
                    target_tab.Z[curr_r, col]
                    if c_type == 1
                    else target_tab.X[curr_r, col]
                )
                inv = target_tab._mod_inverse(int(pivot_val), d)

                for i in range(pivot_idx_rref + 1, len(candidates)):
                    target_r = candidates[i]

                    # Value of the target element in the same logical column
                    target_val = (
                        target_tab.Z[target_r, col]
                        if c_type == 1
                        else target_tab.X[target_r, col]
                    )

                    if target_val != 0:
                        m = (int(target_val) * inv) % mod_d

                        # Update Stabilizer: S_target -= m * S_curr
                        m_neg = (mod_d - m) % mod_d
                        target_tab._row_add_mult(target_r, curr_r, m_neg)

                        # Update Destabilizer: D_curr += m * D_target
                        target_tab._row_add_mult(curr_r - n, target_r - n, m)

                pivot_idx_rref += 1

            # The redundant row is the one pushed to the bottom
            redundant_r = candidates[-1]
            rows_to_remove.add(redundant_r)

        # Final Reduction
        stab_rows_to_remove = rows_to_remove
        destab_rows_to_remove = {r - n for r in stab_rows_to_remove}

        all_rows_to_remove = stab_rows_to_remove.union(destab_rows_to_remove)
        keep_rows = [r for r in range(2 * n) if r not in all_rows_to_remove]
        keep_qudits = [k for k in range(n) if k != q]

        new_X = target_tab.X[np.ix_(keep_rows, keep_qudits)]
        new_Z = target_tab.Z[np.ix_(keep_rows, keep_qudits)]
        new_tau = target_tab.tau_exp[keep_rows]

        target_tab._X = new_X
        target_tab._Z = new_Z
        target_tab._tau_exp = new_tau
        target_tab._n = len(keep_qudits)

        return target_tab, outcomes

    def measurement_kinds(
        self,
        qudits: Optional[Union[int, list[int], tuple[int, ...]]] = None,
    ) -> dict[int, str]:
        """
        Classify whether measuring given qudits in the Z basis is random or deterministic.

        A qudit i is:
        - "rand" if some stabilizer generator has X[:, i] ≠ 0
        - "det"  otherwise

        Parameters
        ----------
        qudits : int or list[int] or tuple[int, ...] or None, optional
            Qudit indices to check. If None (default), checks all qudits 0..n-1.

        Returns
        -------
        dict[int, str]
            Mapping {qudit_index: "rand" or "det"}.

        Raises
        ------
        ValueError
            If this is not a full tableau.
        ValueError
            If any qudit index is out of bounds.
        """
        if not self._full:
            raise ValueError("Classification requires a full tableau.")

        n = self.n
        if qudits is None:
            qs = list(range(n))
        elif isinstance(qudits, int):
            qs = [qudits]
        else:
            qs = list(qudits)

        bad = [q for q in qs if q < 0 or q >= n]
        if bad:
            raise ValueError(f"Qudit index/indices out of bounds for n={n}: {bad}")

        stab_X = self.X[n:]
        col_has_nonzero = np.any(stab_X != 0, axis=0)

        return {q: ("rand" if bool(col_has_nonzero[q]) else "det") for q in qs}

    # -----------------------------------------------------------------------
    # Fast Sampling
    # -----------------------------------------------------------------------

    @staticmethod
    def _rref_mod_d(matrix: np.ndarray, d: int) -> np.ndarray:
        """
        Computes the Row-Reduced Echelon Form of a matrix over the finite field Z_d (prime d).

        Parameters
        ----------
        matrix : np.ndarray
            Input matrix.
        d : int
            Prime modulus.

        Returns
        -------
        np.ndarray
            Row-reduced echelon form of `matrix` modulo d.

        Raises
        ------
        ValueError
            If a modular inverse is requested for a non-invertible pivot
            (should not occur for prime d).
        """
        A = matrix.copy()
        n_rows, n_cols = A.shape
        pivot_row = 0
        for j in range(n_cols):  # Iterate through columns
            if pivot_row >= n_rows:
                break

            # Find a pivot in the current column
            i = pivot_row
            while i < n_rows and A[i, j] == 0:
                i += 1

            if i < n_rows:  # Pivot found at (i, j)
                # Swap the pivot row into place
                A[[pivot_row, i]] = A[[i, pivot_row]]

                # Normalize the pivot row
                inv = Tableau._mod_inverse(int(A[pivot_row, j]), d)
                A[pivot_row, :] = (A[pivot_row, :] * inv) % d

                # Eliminate other entries in the column
                for k in range(n_rows):
                    if k != pivot_row:
                        factor = A[k, j]
                        A[k, :] = (A[k, :] - factor * A[pivot_row, :]) % d

                pivot_row += 1

        return A

    def sample_measurements(
        self,
        shots: int,
        rng: Optional[np.random.Generator] = None,
        seed: Optional[int] = None,
    ) -> np.ndarray:
        """
        Efficiently samples measurement outcomes using the affine subspace method.

        This method is significantly faster than calling measure_all() repeatedly.
        It first computes the affine subspace of all possible measurement outcomes
        and then generates random samples from this space.

        The complexity is O(n^3 * log(d)) for a one-time setup, followed by
        a per-shot cost of O(k*n) where k is the rank of the stabilizer X-matrix.

        Uses a single RNG stream (seeded if provided) across all shots.

        Parameters
        ----------
        shots : int
            The number of measurement samples to generate.
        rng : np.random.Generator, optional
            RNG to use. If None, one is created from `seed`.
        seed : int, optional
            Seed to create a fresh RNG if `rng` is None.

        Returns
        -------
        np.ndarray
            A 2D array of shape (shots, n) where each row is a single measurement outcome.

        Raises
        ------
        ValueError
            If this is not a full tableau.
        ValueError
            If `d` is not prime.
        ValueError
            If `shots <= 0`.
        """
        if shots <= 0:
            raise ValueError("shots must be a positive integer.")

        if not self._full:
            raise ValueError(
                "This sampling method is only supported for full tableaus."
            )
        if not self._is_prime(self._d):
            raise ValueError(
                f"This sampling method is only supported for prime dimensions, but d={self._d}."
            )

        rng = self._coerce_rng(rng, seed)

        # --- Setup Phase (One-time cost) ---

        # Get a single measurement outcome to serve as the "offset vector" v0
        v0 = self.measure_all(rng=rng, in_place=False)[1]

        # Get the basis vectors for the linear subspace from the stabilizer X-matrix
        stabilizer_X = self.X[self.n :, :]
        rref_matrix = Tableau._rref_mod_d(stabilizer_X, self.d)

        # The basis vectors are the non-zero rows of the RREF matrix
        basis_vectors = rref_matrix[np.any(rref_matrix, axis=1)]
        k = basis_vectors.shape[0]

        # --- Sampling Phase (Per-shot cost) ---

        # If k=0, the subspace is trivial; all outcomes are just v0
        if k == 0:
            return np.tile(v0, (shots, 1))

        # Generate all random coefficients at once for efficiency
        coeffs = rng.integers(0, self.d, size=(shots, k))

        # Use broadcasting to compute all linear combinations
        # This computes s = c_1*b_1 + c_2*b_2 + ... for each shot
        linear_combinations = np.dot(coeffs, basis_vectors) % self.d

        # Add the offset vector v0 to each linear combination to get the final samples
        samples = (v0 + linear_combinations) % self.d

        return samples

    def to_probvector(
        self,
        shots: int,
        rng: Optional[np.random.Generator] = None,
        seed: Optional[int] = None,
    ) -> np.ndarray:
        """
        Estimates the probability vector by efficiently sampling measurement outcomes.

        This is the fast estimator. It reconstructs the full
        probability vector from samples produced by `sample_measurements()`.

        Parameters
        ----------
        shots : int
            The number of samples to use for the estimation. Must be positive.
        rng : np.random.Generator, optional
            RNG to use. If None, one is created from `seed`.
        seed : int, optional
            Seed to create a fresh RNG if `rng` is None.

        Returns
        -------
        np.ndarray
            A 1D array of size d**n representing the estimated probability of measuring
            each basis state.

        Raises
        ------
        ValueError
            If `shots <= 0`.
        ValueError
            Propagated from `sample_measurements()` (e.g., non-full tableau, non-prime d).
        """
        if shots <= 0:
            raise ValueError("Number of shots must be a positive integer.")

        rng = self._coerce_rng(rng, seed)

        shots = int(shots)

        n, d = self.n, self.d

        # Get all samples using the fast method
        samples = self.sample_measurements(shots, rng=rng)

        # Find unique outcomes and their counts
        unique_samples, counts = np.unique(samples, return_counts=True, axis=0)

        # Convert outcomes to flat indices for the probability vector
        powers = np.power(d, np.arange(n - 1, -1, -1, dtype=np.int64))
        indices = np.dot(unique_samples, powers)

        # Create the probability vector
        prob_vector = np.zeros((d**n,), dtype=float)

        # Populate it with the calculated probabilities
        probabilities = counts / shots
        prob_vector[indices] = probabilities

        return prob_vector

    # -----------------------------------------------------------------------
    # Exact Probvector (use when d^n is small)
    # -----------------------------------------------------------------------

    def enumerate_outcomes(self) -> np.ndarray:
        """
        Systematically generates all possible measurement outcomes for the state.

        This method determines the affine subspace of outcomes and iterates through
        every element, returning the complete set. The complexity is
        O(n^3) + d^k * O(n), where k is the rank of the stabilizer X-matrix.

        Returns
        -------
        np.ndarray
            A 2D array of shape (d^k, n) containing all possible measurement outcomes.

        Raises
        ------
        ValueError
            If this is not a full tableau.
        ValueError
            If `d` is not prime.
        """
        if not self._full:
            raise ValueError("This method is only supported for full tableaus.")
        if not self._is_prime(self._d):
            raise ValueError(
                f"This method is only supported for prime dimensions, but d={self._d}."
            )

        n, d = self.n, self.d
        v0 = self.measure_all(in_place=False)[1]

        stabilizer_X = self.X[n:, :]
        rref_matrix = Tableau._rref_mod_d(stabilizer_X, d)
        basis_vectors = rref_matrix[np.any(rref_matrix, axis=1)]
        k = basis_vectors.shape[0]

        if k == 0:
            return v0.reshape(1, n)

        num_outcomes = d**k
        outcomes = np.zeros((num_outcomes, n), dtype=v0.dtype)
        outcomes[0] = v0
        current_v = v0.copy()

        if num_outcomes > 1:
            coeffs = np.zeros(k, dtype=int)
            for i in range(1, num_outcomes):
                j = 0
                while True:
                    # Subtract the old contribution of basis_vectors[j]
                    current_v = (current_v - coeffs[j] * basis_vectors[j]) % d
                    coeffs[j] = (coeffs[j] + 1) % d
                    # Add the new contribution
                    current_v = (current_v + coeffs[j] * basis_vectors[j]) % d

                    if coeffs[j] != 0:
                        break
                    j += 1
                outcomes[i] = current_v

        return outcomes

    def to_probvector_exact(self) -> np.ndarray:
        """
        Computes the exact probability vector of the stabilizer state.

        This method systematically enumerates all possible measurement outcomes
        and calculates their exact, uniform probability. It is deterministic
        and avoids any sampling error.

        Returns
        -------
        np.ndarray
            A 1D array of size d**n representing the exact probability of measuring
            each basis state.

        Raises
        ------
        ValueError
            Propagated from `enumerate_outcomes()` (e.g., non-full tableau, non-prime d).
        """
        n, d = self.n, self.d

        # Get all possible outcomes
        all_outcomes = self.enumerate_outcomes()
        num_outcomes = all_outcomes.shape[0]

        if num_outcomes == 0:
            return np.zeros((d**n,), dtype=float)

        # The probability is uniform over the subspace
        probability = 1.0 / num_outcomes

        # Create the probability vector
        prob_vector = np.zeros((d**n,), dtype=float)

        # Convert outcomes to indices and place the probability
        powers = np.power(d, np.arange(n - 1, -1, -1, dtype=np.int64))
        indices = np.dot(all_outcomes, powers)

        prob_vector[indices] = probability

        return prob_vector

    # -----------------------------------------------------------------------
    # Sample from Noisy Circuits (Monte-Carlo)
    # -----------------------------------------------------------------------

    def sample_measurements_noisy_naive(
        self,
        qc: QuantumCircuit,
        shots: int,
        color: str = "#FF0000",
        show_identities: bool = False,
        rng: Optional[np.random.Generator] = None,
        seed: Optional[int] = None,
        return_circuits: bool = False,
    ) -> tuple[np.ndarray, Optional[list[QuantumCircuit]]]:
        """
        Naive Monte-Carlo sampling of measurement outcomes from a noisy circuit.

        This method is a Monte-Carlo estimator: each shot realizes a full noisy circuit
        instance (via `qc.realize_noise(...)`), simulates it, and measures. It is
        much slower than `sample_measurements_noisy_pf()`.

        This method does not mutate `self`. A fresh tableau is used per shot.

        Parameters
        ----------
        qc : QuantumCircuit
            A circuit that may contain noise-gate placeholders. Each shot will
            realize a concrete noisy instance via `qc.realize_noise(...)`.
        shots : int
            Number of Monte-Carlo shots.
        color : str, optional
            Color to use for Weyl Operators in the drawn circuits.
        show_identities : bool, optional
            If True, identity gates are added to the generated circuits.
        rng : np.random.Generator, optional
            Random number generator to use. If None, one is created from `seed`.
            If both `rng` and `seed` are provided, `rng` is used.
        seed : int, optional
            Seed to create a fresh RNG when `rng` is None.
        return_circuits : bool, optional
            If True, also return the list of realized per-shot noisy circuits.

        Returns
        -------
        outcomes : np.ndarray
            Array of shape (shots, n) with integer outcomes in base-d for each qudit.
        circuits : list[QuantumCircuit] or None
            If `return_circuits=True`, a list of the realized noisy circuits (length = shots).
            Otherwise, None.

        Raises
        ------
        ValueError
            If `shots <= 0`  or a mismatch of n/d between tab and qc occured.
        """
        self._check_compat_with_circuit(qc)

        if shots <= 0:
            raise ValueError("shots must be a positive integer.")

        rng = self._coerce_rng(rng, seed)

        per_shot_noise_seeds = rng.integers(
            0, np.iinfo(np.int64).max, size=shots, dtype=np.int64
        )
        per_shot_meas_seeds = rng.integers(
            0, np.iinfo(np.int64).max, size=shots, dtype=np.int64
        )

        n, d = self.n, self.d
        outcomes = np.empty((shots, n), dtype=np.int64)
        circuits_list = [] if return_circuits else None

        for i in range(shots):
            noisy_qc, _ = qc.realize_noise(
                color=color,
                show_identities=show_identities,
                seed=int(per_shot_noise_seeds[i]),
                in_place=False,
            )

            if circuits_list is not None:
                circuits_list.append(noisy_qc)

            tab = type(self)(n, d, full=True)
            tab.apply_circuit(noisy_qc)

            _, shot_outcome = tab.measure_all(
                in_place=False, seed=int(per_shot_meas_seeds[i])
            )

            outcomes[i] = np.asarray(shot_outcome, dtype=np.int64)

        return outcomes, circuits_list

    def to_probvector_noisy_naive(
        self,
        qc: QuantumCircuit,
        shots: int,
        rng: Optional[np.random.Generator] = None,
        seed: Optional[int] = None,
    ) -> np.ndarray:
        """
        Naive Monte-Carlo estimate of the output probability vector for a noisy circuit.

        This method estimates the output distribution by realizing a full noisy circuit
        per shot (via `qc.realize_noise(...)`), simulating, and measuring. It is
        much slower than `to_probvector_noisy_pf()`.

        The mapping from an n-digit base-d string x_{n-1}...x_0 to index is
        sum_{k=0}^{n-1} x_k * d^{n-1-k}.

        Parameters
        ----------
        qc : QuantumCircuit
            A circuit that may contain noise-gate placeholders. Each shot will
            realize a concrete noisy instance via `qc.realize_noise(...)`.
        shots : int
            Number of Monte-Carlo shots used for the estimate.
        rng : np.random.Generator, optional
            Random number generator to use. If None, one is created from `seed`.
            If both `rng` and `seed` are provided, `rng` is used.
        seed : int, optional
            Seed to create a fresh RNG when `rng` is None.

        Returns
        -------
        np.ndarray
            A length d**n vector of probabilities.

        Raises
        ------
        ValueError
            Propagated from `sample_measurements_noisy_naive()` if `shots <= 0` or a mismatch of n/d between tab and qc occured.
        """
        self._check_compat_with_circuit(qc)

        outcomes, _ = self.sample_measurements_noisy_naive(
            qc=qc, shots=shots, rng=rng, seed=seed, return_circuits=False
        )

        n, d = self.n, self.d
        unique_samples, counts = np.unique(outcomes, return_counts=True, axis=0)
        powers = np.power(d, np.arange(n - 1, -1, -1, dtype=np.int64))
        indices = np.dot(unique_samples, powers)
        prob_vector = np.zeros((d**n,), dtype=float)
        prob_vector[indices] = counts.astype(float) / float(shots)
        return prob_vector

    # -----------------------------------------------------------------------
    # Push NoiseChannels to the End of the Circuit
    # -----------------------------------------------------------------------

    @classmethod
    def clean_run(
        cls, qc: QuantumCircuit
    ) -> tuple[Tableau, list[tuple[int, np.ndarray, np.ndarray, object]]]:
        """
        Run qc once, skipping NoiseChannel ops but caching
        (q, X[:,q], Z[:,q], model_id) for each noise insertion point.

        Parameters
        ----------
        qc : QuantumCircuit
            Circuit to run.

        Returns
        -------
        (Tableau, list[tuple[int, np.ndarray, np.ndarray, object]])
            (T_clean, noise_frames).
        """
        T = cls(qc.n, qc.d, full=True)
        noise_frames = []

        for gate, qudits, dagger in qc._ops:
            if isinstance(gate, NoiseChannel):
                q = qudits[0]
                Xcol = T.X[:, q].copy()
                Zcol = T.Z[:, q].copy()
                noise_frames.append((q, Xcol, Zcol, gate.model_id))
            else:
                T.apply_gate(gate, qudits, dagger)

        return T, noise_frames

    @staticmethod
    def _sample_delta_taus(
        qc: QuantumCircuit,
        T_clean: Tableau,
        noise_frames: list[tuple[int, np.ndarray, np.ndarray, object]],
        shots: int,
        rng: Optional[np.random.Generator] = None,
        seed: Optional[int] = None,
    ) -> np.ndarray:
        """
        Sample Δτ for all noise frames & all shots.
        Output shape: (shots, 2n)

        Parameters
        ----------
        qc : QuantumCircuit
            Circuit whose noise models are sampled.
        T_clean : Tableau
            Tableau after running the circuit without NoiseChannels.
        noise_frames : list[tuple[int, np.ndarray, np.ndarray, object]]
            Cached noise insertion frames (q, Xcol, Zcol, model_id).
        shots : int
            Number of shots.
        rng : np.random.Generator, optional
            RNG to use. If None, one is created from `seed`.
        seed : int, optional
            Seed to create a fresh RNG if `rng` is None.

        Returns
        -------
        np.ndarray
            Δτ array of shape (shots, 2n), mod 2d.
        """
        if rng is None:
            rng = np.random.default_rng(seed)

        d = qc.d
        mod_2d = np.int64(2 * d)
        n = qc.n
        rows = 2 * n

        delta_taus = np.zeros((shots, rows), dtype=T_clean._tau_dtype)

        for q, Xcol, Zcol, model_id in noise_frames:
            probs = qc.noise_registry[model_id].reshape(-1)

            # sample S indices for this noise location
            choice = rng.choice(d * d, size=shots, p=probs)

            # convert to a_j, b_j
            a_j, b_j = np.unravel_index(choice, (d, d))

            contrib = (
                2 * b_j[:, None] * Xcol[None, :] - 2 * a_j[:, None] * Zcol[None, :]
            )

            delta_taus = (delta_taus + contrib) % mod_2d

        return delta_taus

    def sample_measurements_noise_push(
        self,
        qc: QuantumCircuit,
        shots: int,
        rng: Optional[np.random.Generator] = None,
        seed: Optional[int] = None,
    ) -> np.ndarray:
        """
        Monte-Carlo sampling of measurement outcomes from a noisy circuit via noise pushing.

        Algorithm:
        1. clean run: T_clean, noise_frames
        2. sample Δτ vectors
        3. apply phase + measure using CHP

        For the naive per-shot circuit-realization approach, use
        `sample_measurements_noisy_naive()`.

        Parameters
        ----------
        qc : QuantumCircuit
            Circuit (with NoiseChannels).
        shots : int
            Number of shots.
        rng : np.random.Generator, optional
            RNG to use. If None, one is created from `seed`.
        seed : int, optional
            Seed to create a fresh RNG if `rng` is None.

        Returns
        -------
        np.ndarray
            Array of shape (shots, n) with measurement outcomes.

        Raises
        ------
        ValueError
            If `shots <= 0` or a mismatch of n/d between tab and qc occured.
        """
        self._check_compat_with_circuit(qc)

        if shots <= 0:
            raise ValueError("shots must be a positive integer.")
        if rng is None:
            rng = np.random.default_rng(seed)

        per_shot_meas_seeds = rng.integers(
            0, np.iinfo(np.int64).max, size=shots, dtype=np.int64
        )

        T_clean, noise_frames = type(self).clean_run(qc)
        delta_taus = type(self)._sample_delta_taus(
            qc, T_clean, noise_frames, shots, rng
        )

        mod_2d = T_clean.mod_2d
        base_tau = T_clean._tau_exp.copy()

        results = np.zeros((shots, qc.n), dtype=np.int64)

        for s in range(shots):
            Ts = T_clean.copy()
            Ts._tau_exp = (base_tau + delta_taus[s]) % mod_2d
            m = Ts.measure_all(in_place=False, seed=int(per_shot_meas_seeds[s]))
            res = m[1] if isinstance(m, tuple) else m
            results[s] = res

        return results

    def to_probvector_noise_push(
        self,
        qc: QuantumCircuit,
        shots: int,
        rng: Optional[np.random.Generator] = None,
        seed: Optional[int] = None,
    ) -> np.ndarray:
        """
        Fast Monte-Carlo estimate of the output probability vector for a noisy circuit.

        This method estimates the output distribution by sampling noisy realizations and
        converting the sampled outcomes into a probability vector. It uses `sample_measurements_noise_push()`.

        For the naive per-shot circuit-realization estimator, use
        `to_probvector_noisy_naive()`.

        Parameters
        ----------
        qc : QuantumCircuit
            Circuit (with NoiseChannels).
        shots : int
            Number of samples to use for the estimate.
        rng : np.random.Generator, optional
            RNG to use. If None, one is created from `seed`.
        seed : int, optional
            Seed to create a fresh RNG if `rng` is None.

        Returns
        -------
        np.ndarray
            A length d**n vector of probabilities.

        Raises
        ------
        ValueError
            If a mismatch of n/d between tab and qc occured.
        """
        self._check_compat_with_circuit(qc)

        outcomes = self.sample_measurements_noise_push(
            qc=qc, shots=shots, rng=rng, seed=seed
        )

        n, d = self.n, self.d
        unique_samples, counts = np.unique(outcomes, return_counts=True, axis=0)
        powers = np.power(d, np.arange(n - 1, -1, -1, dtype=np.int64))
        indices = np.dot(unique_samples, powers)
        prob_vector = np.zeros((d**n,), dtype=float)
        prob_vector[indices] = counts.astype(float) / float(shots)
        return prob_vector

    # -----------------------------------------------------------------------
    # Pauli Frames Sampler
    # -----------------------------------------------------------------------

    @staticmethod
    def _pf_gate_key(g: Gate) -> str:
        name = getattr(g, "name", None)
        return str(name).upper()

    @staticmethod
    def _pf_init_frames(
        shots: int, n: int, d: int, rng: np.random.Generator
    ) -> tuple[np.ndarray, np.ndarray]:
        x = np.zeros((shots, n), dtype=np.int64)
        z = rng.integers(0, d, size=(shots, n), dtype=np.int64)
        return x, z

    @staticmethod
    def _pf_apply_noise_channel(
        x: np.ndarray,
        z: np.ndarray,
        *,
        q: int,
        probs: np.ndarray,
        d: int,
        rng: np.random.Generator,
    ) -> None:
        idx = rng.choice(d * d, size=x.shape[0], p=probs).astype(np.int64, copy=False)
        a = idx // d
        b = idx % d
        x[:, q] = (x[:, q] + a) % d
        z[:, q] = (z[:, q] + b) % d

    @staticmethod
    def _pf_apply_gate(
        x: np.ndarray,
        z: np.ndarray,
        *,
        key: str,
        qudits: tuple[int, ...],
        dagger: bool,
        d: int,
    ) -> None:
        dir_ = -1 if dagger else 1

        if key == "H":
            q = qudits[0]
            xq = x[:, q].copy()
            zq = z[:, q]
            x[:, q] = (-dir_ * zq) % d
            z[:, q] = (dir_ * xq) % d
            return
        elif key == "S":
            q = qudits[0]
            z[:, q] = (z[:, q] + dir_ * x[:, q]) % d
            return
        elif key == "CNOT":
            c, t = qudits
            x[:, t] = (x[:, t] + dir_ * x[:, c]) % d
            z[:, c] = (z[:, c] - dir_ * z[:, t]) % d
            return
        elif key == "CZ":
            c, t = qudits
            z[:, c] = (z[:, c] + dir_ * x[:, t]) % d
            z[:, t] = (z[:, t] + dir_ * x[:, c]) % d
            return
        elif key == "SWAP":
            a, b = qudits
            x_a = x[:, a].copy()
            z_a = z[:, a].copy()
            x[:, a] = x[:, b]
            z[:, a] = z[:, b]
            x[:, b] = x_a
            z[:, b] = z_a
            return
        else:
            return

    def sample_measurements_noisy_pf(
        self,
        qc: QuantumCircuit,
        shots: int,
        rng: Optional[np.random.Generator] = None,
        seed: Optional[int] = None,
    ) -> np.ndarray:
        """
        Sample Z-basis measurement outcomes from a noisy Clifford circuit using a Pauli-frame method.

        This method:
        1. Runs the circuit once without NoiseChannels to get a clean final tableau.
        2. Measures it once to get a reference outcome vector r.
        3. Propagates `shots` Pauli frames (x,z) through the circuit, vectorized over shots:
           - Initialize x=0 and z uniform in Z_d^n.
           - Conjugate through Clifford gates.
           - At each NoiseChannel, sample a Weyl error W(a,b)=X^a Z^b and add (a,b) into the frame.
        4. Returns outcomes m = (r + x_final) mod d.

        Parameters
        ----------
        qc : QuantumCircuit
            Circuit (with NoiseChannels).
        shots : int
            Number of samples to use for the estimate.
        rng : np.random.Generator, optional
            RNG to use. If None, one is created from `seed`.
        seed : int, optional
            Seed to create a fresh RNG if `rng` is None.

        Returns
        -------
        np.ndarray
            An integer array of shape (shots, qc.n), where each row is one sampled outcome
            vector in {0, 1, ..., d-1}^n (mod d).

        Raises
        ------
        ValueError
            If `shots <= 0`.
        """
        self._check_compat_with_circuit(qc)

        if shots <= 0:
            raise ValueError("shots must be a positive integer.")

        rng = self._coerce_rng(rng, seed)
        n, d = qc.n, qc.d

        T_clean, _ = type(self).clean_run(qc)
        ref_rng = np.random.default_rng(rng.integers(0, 2**63 - 1, dtype=np.int64))
        _, r = T_clean.measure_all(in_place=False, rng=ref_rng)
        r = np.asarray(r, dtype=np.int64) % d

        x, z = type(self)._pf_init_frames(shots, n, d, rng)

        for gate, qudits, dagger in qc._ops:
            if isinstance(gate, NoiseChannel):
                q = int(qudits[0])
                probs = qc.noise_registry.get(gate.model_id)
                type(self)._pf_apply_noise_channel(x, z, q=q, probs=probs, d=d, rng=rng)
                continue

            key = type(self)._pf_gate_key(gate)
            type(self)._pf_apply_gate(x, z, key=key, qudits=qudits, dagger=dagger, d=d)

        return (r[None, :] + x) % d

    def to_probvector_noisy_pf(
        self,
        qc: QuantumCircuit,
        shots: int,
        rng: Optional[np.random.Generator] = None,
        seed: Optional[int] = None,
    ) -> np.ndarray:
        """
        Monte-Carlo estimate of the output probability vector for a noisy circuit with Pauli frames.

        Uses `self.sample_measurements_noisy_pf()`.

        Parameters
        ----------
        qc : QuantumCircuit
            Circuit (with NoiseChannels).
        shots : int
            Number of samples to use for the estimate.
        rng : np.random.Generator, optional
            RNG to use. If None, one is created from `seed`.
        seed : int, optional
            Seed to create a fresh RNG if `rng` is None.

        Returns
        -------
        np.ndarray
            A length d**n vector of probabilities.

        Raises
        ------
        ValueError
            If a mismatch of n/d between tab and qc occured.
        """
        self._check_compat_with_circuit(qc)

        outcomes = self.sample_measurements_noisy_pf(
            qc=qc, shots=shots, rng=rng, seed=seed
        )

        n, d = self.n, self.d
        unique_samples, counts = np.unique(outcomes, return_counts=True, axis=0)
        powers = np.power(d, np.arange(n - 1, -1, -1, dtype=np.int64))
        indices = np.dot(unique_samples, powers)
        prob_vector = np.zeros((d**n,), dtype=float)
        prob_vector[indices] = counts.astype(float) / float(shots)
        return prob_vector
