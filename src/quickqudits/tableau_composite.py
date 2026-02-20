"""
Composite-dimension tableau measurement via diagonal-subgroup extraction and SNF.

This module provides the `CompositeTableau` class, which
inherits from `tableau.Tableau` but overrides measurement for
arbitrary dimensions ``d``.

Computational-basis measurement outcomes x ∈ Z_d^n are characterized by the diagonal
subgroup of the stabilizer group. A product of generators specified by coefficients
u ∈ Z_d^n is diagonal iff its X-part cancels:
    u^T X_s ≡ 0 (mod d),
where X_s are the stabilizer-generator X rows.

For each diagonal stabilizer with Z-exponents b and even phase exponent t, valid
outcomes satisfy the linear congruence
    b · x ≡ -t/2 (mod d).

Stacking all such constraints yields
    B x ≡ c (mod d),
which is solved and sampled exactly using Smith normal decomposition (SNF)
provided by SymPy:
    D = U B V,
reducing the system to independent 1D congruences in the transformed variables.
"""

from __future__ import annotations

from typing import List, Optional, Tuple, Union

import numpy as np
from sympy import Matrix
from sympy.matrices.normalforms import smith_normal_decomp

from .gate_class import NoiseChannel
from .tableau import Tableau


def _np_to_sympy(A: np.ndarray) -> Matrix:
    """
    Convert a NumPy integer array to a SymPy Matrix (exact integers).

    Parameters
    ----------
    A : numpy.ndarray
        Integer array.

    Returns
    -------
    sympy.Matrix
        Matrix with Python ``int`` entries.
    """
    return Matrix([[int(v) for v in row] for row in A.tolist()])


def _sympy_to_np(A: Matrix, mod=None) -> np.ndarray:
    """
    Convert a SymPy Matrix to a NumPy integer array.

    Parameters
    ----------
    A : sympy.Matrix
        Integer matrix.

    Returns
    -------
    numpy.ndarray
        Array with dtype ``int``.
    """
    arr = np.array(A.tolist(), dtype=object)
    if mod is not None:
        arr %= int(mod)
        arr = arr.astype(int)
    return arr


def _snf_int(
    A: np.ndarray, mod: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Smith normal decomposition over the integers using SymPy.

    Parameters
    ----------
    A : numpy.ndarray
        Integer matrix.

    Returns
    -------
    D : numpy.ndarray
        Diagonal Smith form.
    U : numpy.ndarray
        Left unimodular transform.
    V : numpy.ndarray
        Right unimodular transform.

    Notes
    -----
    Uses SymPy's ``smith_normal_decomp`` which returns matrices satisfying
        D = U * A * V .
    """
    M = _np_to_sympy(A)
    D, U, V = smith_normal_decomp(M)
    if mod is None:
        return _sympy_to_np(D), _sympy_to_np(U), _sympy_to_np(V)

    return _sympy_to_np(D, mod=mod), _sympy_to_np(U, mod=mod), _sympy_to_np(V, mod=mod)


def _kernel_generators_mod_d_snf(A: np.ndarray, d: int) -> np.ndarray:
    """
    Compute a generating set for the solution module of:

        A x ≡ 0 (mod d)

    using integer SNF.

    Parameters
    ----------
    A : numpy.ndarray
        Integer matrix of shape (m, n).
    d : int
        Modulus.

    Returns
    -------
    numpy.ndarray
        Array of shape (k, n) whose rows generate the kernel over Z_d.
    """
    A = np.array(A, dtype=int)
    m, n = A.shape

    if n == 0:
        return np.zeros((0, 0), dtype=int)

    D, U, V = _snf_int(A, mod=d)

    gens_y: List[np.ndarray] = []

    r = min(m, n)
    for i in range(r):
        s = abs(int(D[i, i]))
        if s == 0:
            e = np.zeros(n, dtype=int)
            e[i] = 1
            gens_y.append(e)
        else:
            g = int(np.gcd(s, d))
            step = d // g
            e = np.zeros(n, dtype=int)
            e[i] = step % d
            gens_y.append(e)

    for i in range(r, n):
        e = np.zeros(n, dtype=int)
        e[i] = 1
        gens_y.append(e)

    if len(gens_y) == 0:
        return np.zeros((0, n), dtype=int)

    GY = np.vstack(gens_y) % d
    X = (V @ GY.T) % d
    return X.T.astype(int, copy=False)


def _solve_congruences_mod_d_snf(
    B: np.ndarray, c: np.ndarray, d: int
) -> Tuple[np.ndarray, np.ndarray, List[Tuple[int, int, int]], List[int]]:
    """
    Solve B x ≡ c (mod d) using integer SNF.

    This returns a sampler description for the solution set, sufficient to
    sample uniformly without enumerating all solutions.

    Parameters
    ----------
    B : numpy.ndarray
        Constraint matrix of shape (m, n).
    c : numpy.ndarray
        RHS vector of shape (m,).
    d : int
        Modulus.

    Returns
    -------
    V : numpy.ndarray
        Unimodular matrix from SNF, used to map y -> x via x = V y (mod d).
    y0 : numpy.ndarray
        One particular solution y0 in Z_d^n for the transformed system D y ≡ c' (mod d).
    diag_info : list[tuple[int, int, int]]
        For each i < r with diagonal s_i != 0, stores (i, modulus_red, g),
        where:
            - modulus_red = d / gcd(s_i, d)
            - g = gcd(s_i, d)
        Sampling rule: y_i = y0_i + modulus_red * k, k uniform in {0,...,g-1}
    free_indices : list[int]
        Indices of y that are free uniform in {0,...,d-1}.
    """
    B = np.array(B, dtype=int)
    c = np.array(c, dtype=int).reshape(-1)

    m, n = B.shape

    D, U, V = _snf_int(B, mod=d)

    cp = (U @ c.reshape(-1, 1)).reshape(-1)
    cp = np.mod(cp, d).astype(int)

    y0 = np.zeros(n, dtype=int)
    diag_info: List[Tuple[int, int, int]] = []
    free_indices: List[int] = []

    r = min(m, n)

    for i in range(r):
        s = int(D[i, i])
        rhs = int(cp[i] % d)

        if s == 0:
            free_indices.append(i)
            continue

        s_abs = abs(s)
        g = int(np.gcd(s_abs, d))

        s_red = (s // g) % (d // g)
        rhs_red = (rhs // g) % (d // g)
        mod_red = d // g

        inv = pow(int(s_red), -1, int(mod_red))
        yi0 = (inv * rhs_red) % mod_red

        y0[i] = int(yi0)
        diag_info.append((i, mod_red, g))

    for i in range(r, n):
        free_indices.append(i)

    return V % d, y0 % d, diag_info, free_indices


class CompositeTableau(Tableau):
    """
    Composite-dimension stabilizer tableau with exact measurement via SNF.

    Notes
    -----
    This class implements computational-basis measurement for arbitrary dimensions ``d`` by:

    1. Extracting the diagonal subgroup of the stabilizer group:
       find exponent vectors ``u`` such that combining stabilizer generators with
       exponents ``u`` cancels the X-part, producing a diagonal stabilizer.

    2. Converting diagonal stabilizers into linear congruence constraints on outcomes:
       for diagonal element with Z-exponents ``b`` and phase ``tau_exp`` (even),
       valid outcomes ``x`` satisfy ``b·x ≡ -tau_exp/2 (mod d)``.

    3. Solving and sampling from the congruence system using Smith normal form (SNF).
    """

    def _diagonal_constraints(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        Construct diagonal stabilizer constraints B x ≡ c (mod d).

        Returns
        -------
        (B, c) or None
            If diagonal constraints exist, returns:
            - B : ndarray, shape (m, n)
            - c : ndarray, shape (m,)
            such that valid outcomes x satisfy B x ≡ c (mod d).
            If no diagonal stabilizers exist, returns None (uniform distribution).
        """
        if not self.full:
            raise ValueError("Composite measurement requires full tableau.")

        n = self.n
        d = self.d

        Xs = (self.X[n:, :] % d).astype(int)

        U = _kernel_generators_mod_d_snf(Xs.T, d)
        if U.shape[0] == 0:
            return None

        diag_Z: List[np.ndarray] = []
        diag_tau: List[int] = []

        for u in U:
            acc_X = np.zeros(n, dtype=self._xz_dtype)
            acc_Z = np.zeros(n, dtype=self._xz_dtype)
            acc_tau = 0

            for j in range(n):
                mj = int(u[j] % d)
                if mj == 0:
                    continue

                Xp, Zp, taup = self._power(
                    self.X[n + j], self.Z[n + j], int(self.tau_exp[n + j]), mj
                )
                acc_X, acc_Z, acc_tau = self._product_logic(
                    acc_X, acc_Z, acc_tau, Xp, Zp, taup
                )

            if np.any((acc_X % d) != 0):
                continue

            t = int(acc_tau % (2 * d))

            diag_Z.append((acc_Z % d).astype(int))
            diag_tau.append(t)

        if len(diag_Z) == 0:
            return None

        B = np.vstack(diag_Z) % d
        c = (-np.array(diag_tau, dtype=int) // 2) % d
        return B, c

    def sample_measurements(
        self,
        shots: int,
        rng: Optional[np.random.Generator] = None,
        seed: Optional[int] = None,
    ) -> np.ndarray:
        """
        Sample computational-basis measurement outcomes for all qudits.

        The sampling is non-destructive: it does not update the tableau to a collapsed state.
        Outcomes are sampled exactly from the stabilizer state's computational-basis
        distribution for arbitrary dimensions ``d``.

        Parameters
        ----------
        shots : int
            Number of samples to draw.
        rng : numpy.random.Generator, optional
            Random number generator to use. If ``None``, a new generator is created.
        seed : int, optional
            Seed used to initialize the generator when ``rng`` is ``None``.

        Returns
        -------
        numpy.ndarray
            Integer array of shape ``(shots, n)`` where each row is an outcome
            in ``{0, ..., d-1}^n``.
        """
        if shots < 1:
            raise ValueError(f"shots must be >= 1, got {shots}")

        rng = np.random.default_rng(seed) if rng is None else rng

        n = self.n
        d = self.d

        cons = self._diagonal_constraints()
        if cons is None:
            return rng.integers(0, d, size=(shots, n), endpoint=False)

        B, c = cons

        V, y0, diag_info, free_indices = _solve_congruences_mod_d_snf(B, c, d)

        samples = np.empty((shots, n), dtype=int)

        for s in range(shots):
            y = np.zeros(n, dtype=int)

            y[:] = y0

            for i, mod_red, g in diag_info:
                if g == 1:
                    continue
                k = int(rng.integers(0, g))
                y[i] = (y0[i] + (mod_red % d) * k) % d

            for i in free_indices:
                y[i] = int(rng.integers(0, d))

            x = (V @ y) % d
            samples[s] = x.astype(int, copy=False)

        return samples

    def sample_measurements_noisy_pf(
        self,
        qc,
        shots: int,
        rng: Optional[np.random.Generator] = None,
        seed: Optional[int] = None,
    ) -> np.ndarray:
        """
        This method overrides the corresponding method of tableau.Tableau,
        so that it works for composite d.

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
        r = T_clean.measure_all(rng=ref_rng)
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

    def measure_all(
        self,
        rng: Optional[np.random.Generator] = None,
        seed: Optional[int] = None,
        **kwargs,
    ) -> np.ndarray:
        """
        Measure all qudits in the computational basis (single shot).

        The tableau is not modified.

        Parameters
        ----------
        rng : numpy.random.Generator, optional
            Random number generator to use. If ``None``, a new generator is created.
        seed : int, optional
            Seed used to initialize the generator when ``rng`` is ``None``.

        Returns
        -------
        numpy.ndarray
            Integer array of shape ``(n,)`` containing the measurement outcome.
        """
        return self.sample_measurements(1, rng=rng, seed=seed)[0]

    def measure(
        self,
        qudits_to_measure: Union[int, list[int], tuple[int, ...]],
        rng: Optional[np.random.Generator] = None,
        seed: Optional[int] = None,
        **kwargs,
    ) -> dict[int, int]:
        """
        Measure a subset of qudits in the computational basis (single shot).

        The tableau is not modified.

        Parameters
        ----------
        qudits_to_measure : int or sequence of int
            Index (or indices) of the qudit(s) to measure.
        rng : numpy.random.Generator, optional
            Random number generator to use. If ``None``, a new generator is created.
        seed : int, optional
            Seed used to initialize the generator when ``rng`` is ``None``.

        Returns
        -------
        dict[int, int]
            Mapping from qudit index to measured outcome in ``{0, ..., d-1}``.
        """
        if isinstance(qudits_to_measure, (int, np.integer)):
            qudits = [int(qudits_to_measure)]
        else:
            qudits = [int(q) for q in qudits_to_measure]

        full = self.sample_measurements(1, rng=rng, seed=seed)[0]
        return {q: int(full[q]) for q in qudits}
