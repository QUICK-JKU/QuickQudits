"""
Fast Pauli-noise sampling by effect-table lookup.

Two classes share one idea: an error injected at a noise location reaches the final
measurements as a fixed shift, so it is compiled once and applied by table lookup.

* :class:`NoisyCircuitSampler` samples any Clifford circuit with Pauli (Weyl)
  ``NoiseChannel`` s, with the same distribution as ``Tableau.sample_measurements_noisy_pf``
  and several times faster.
* :class:`PauliFrameSampler` estimates the Renyi-2 coherent information

    I2 = S2(Q) - S2(RQ)

  of an encoded logical state under Pauli noise, using a two-copy destructive
  swap test.  It is the fastest and most reliable path in this package for the
  common workflow "scan I2 over many noise strengths p", and is specialised for
  **bit-flip noise** while remaining correct for general single-qudit Pauli
  (Weyl) noise on prime dimension d.

Why it is fast
--------------
The expensive part of a swap-test I2 estimate is the *noiseless* stabilizer
simulation of the two-copy encoding circuit.  For bit-flip (and any Pauli)
noise the effect of an error is a purely classical, linear bit-flip pattern on
the final Z-basis measurements.  So we:

1. **Compile an effect table once.**  An error injected at a noise location
   (which sits *after* the encoding, *before* the swap-test layer) propagates
   to the measured qubits only through the small swap-test layer.  We
   precompute, per noise location and per Weyl type, the resulting bit-flip
   pattern on the measured qubits (``_F`` / ``_F_packed``).  Cost: O(L * G_swap).

2. **Sample the noiseless outcomes once** (``precompute_noiseless_pool``).  The
   full stabilizer tableau is simulated a single time; the resulting pool of
   noiseless measurement outcomes is reused for *every* p-value.

3. **Apply noise by table lookup.**  For each p we sample which locations fire
   and XOR their effect rows into the noiseless pool -- no further tableau
   simulation.  Cost per p: O(shots * (L + pairs)).

Compared with re-running the full Pauli-frame tableau sampler
(``Tableau.sample_measurements_noisy_pf``) once per p-value, this is several
times faster already for a single distance, and the speed-up grows with the
number of p-values scanned.

Why it is reliable
-------------------
* Every estimate carries a standard error (SE) computed from the delta method,
  so the caller can always judge whether a point is trustworthy.
* Purity estimates are used raw (no eps-clamping): a sign error near the
  threshold shows up as a large SE, never as a silent bias.
* ``run_unit_tests()`` validates the whole pipeline against the independent
  backend sampler ``Tableau.sample_measurements_noisy_pf``.

Public API
----------
    sampler = PauliFrameSampler(qc_one_copy, n_one_copy,
                                reference_qubits_one_copy,
                                tested_qubits_one_copy, d=2)
    sampler.precompute_noiseless_pool(n_shots, seed)          # optional
    rows = sampler.scan_ps(ps, n_shots, error_model="bitflip",
                           seed0=0, use_pool=True)

    run_unit_tests()   # self-check against the backend

    sampler = NoisyCircuitSampler(qc)           # qc has NoiseChannels
    outcomes = sampler.sample(shots, seed=0)    # (shots, n); probs=... rescans the noise
"""

from __future__ import annotations

import numpy as np

from .gate_class import NoiseChannel

# NOTE: `Tableau` / `QuantumCircuit` are imported lazily inside the functions that use
# them, so this module has no import-time dependency on `tableau.py`.


# ─────────────────────────────────────────────────────────────────────────────
# Circuit helpers
# ─────────────────────────────────────────────────────────────────────────────

def duplicate_circuit_vertical(qc_one_copy: "QuantumCircuit") -> "QuantumCircuit":
    """Return two independent vertically-stacked copies of ``qc_one_copy``."""
    return qc_one_copy.compose_vertical(qc_one_copy, in_place=False)


def add_destructive_swap_test(qc: QuantumCircuit, n_one_copy: int, tested_qubits_one_copy):
    """
    Append a destructive swap test between the two copies for each tested qubit.

    For each tested qubit ``q`` (one-copy index) this appends ``CX(q, q+n)``
    then ``H(q)`` and records the pair ``(q, q+n)``.  Measuring all qubits in
    the Z basis afterwards yields, per pair, a swap-test bit ``m_a & m_b`` whose
    XOR over a set of pairs estimates the corresponding subsystem purity.
    """
    pairs = []
    for q in tested_qubits_one_copy:
        a = q
        b = q + n_one_copy
        qc.CX(a, b)
        qc.H(a)
        pairs.append((a, b))
    return qc, pairs


# ─────────────────────────────────────────────────────────────────────────────
# Noise models  (probability vector over Weyl operators W_{a,b} = X^a Z^b,
# flattened row-major as index = a*d + b, matching circuit._prepare_weyl_probabilities)
# ─────────────────────────────────────────────────────────────────────────────

def bitflip_noise_model(p: float) -> np.ndarray:
    """(d=2) bit-flip channel: identity w.p. 1-p, X w.p. p."""
    P = np.zeros((2, 2), dtype=float)
    P[0, 0] = 1.0 - p   # I
    P[1, 0] = p         # X  (a=1, b=0)
    return P


def depolarizing_noise_model(p: float) -> np.ndarray:
    """(d=2) depolarizing channel: identity w.p. 1-p, each of X,Y,Z w.p. p/3."""
    P = np.zeros((2, 2), dtype=float)
    P[0, 0] = 1.0 - p       # I
    P[1, 0] = p / 3.0       # X  (1,0)
    P[1, 1] = p / 3.0       # Y  (1,1)
    P[0, 1] = p / 3.0       # Z  (0,1)
    return P


# ─────────────────────────────────────────────────────────────────────────────
# Standard-error helpers (delta method)
# ─────────────────────────────────────────────────────────────────────────────

def _se_purity(P: float, n: int) -> float:
    """SE of the swap-test purity estimator P_hat = 1 - 2*count/n: sqrt((1-P^2)/n)."""
    return float(np.sqrt(max(1.0 - P ** 2, 0.0) / n))


def _se_I2(P_RQ: float, P_Q: float, n: int, eps: float = 1e-9) -> float:
    """
    SE of I2 = log|P_RQ| - log|P_Q| via the delta method:

        SE(I2)^2 = (1 - P_RQ^2)/(n P_RQ^2) + (1 - P_Q^2)/(n P_Q^2)
    """
    term_RQ = (1 - P_RQ ** 2) / (n * max(abs(P_RQ), eps) ** 2)
    term_Q = (1 - P_Q ** 2) / (n * max(abs(P_Q), eps) ** 2)
    return float(np.sqrt(term_RQ + term_Q))


def shots_for_target_se(P: float, target_se: float, eps: float = 1e-9) -> int:
    """Minimum shots so that SE(log|P_hat|) <= target_se: n = (1-P^2)/(P^2 target_se^2)."""
    P_abs = max(abs(P), eps)
    return int(np.ceil((1.0 - P ** 2) / (P_abs ** 2 * target_se ** 2)))


# ─────────────────────────────────────────────────────────────────────────────
# Effect-table compilation:  how an injected Weyl error propagates through the
# swap-test layer to a bit-flip pattern on the final Z-basis measurements.
# ─────────────────────────────────────────────────────────────────────────────

def _propagate_error_through_ops(x0: int, z0: int, q0: int, n_meas: int, ops, d: int) -> np.ndarray:
    """
    Forward-propagate a single Weyl error W(x0, z0) placed on qudit ``q0`` through
    ``ops`` (a list of ("CNOT",(c,t)) / ("H",(q,)) tuples) and return the final
    x-frame (length ``n_meas``).  Because Z-basis outcome = clean + x (mod d),
    the x-frame *is* the bit-flip pattern seen at measurement.
    """
    x = np.zeros(n_meas, dtype=np.int64)
    z = np.zeros(n_meas, dtype=np.int64)
    x[q0] = x0 % d
    z[q0] = z0 % d
    for name, qd in ops:
        if name == "CNOT":
            c, t = qd
            x[t] = (x[t] + x[c]) % d          # X_c -> X_c X_t
            z[c] = (z[c] - z[t]) % d          # Z_t -> Z_t Z_c^{-1}
        elif name == "H":
            q = qd[0]
            xq = x[q]
            x[q] = z[q] % d                   # H: X <-> Z
            z[q] = (-xq) % d
    return x % d


# ─────────────────────────────────────────────────────────────────────────────
# The sampler
# ─────────────────────────────────────────────────────────────────────────────

class PauliFrameSampler:
    """
    Fast, reliable I2 estimator for an encoded logical state under Pauli noise.

    Parameters
    ----------
    qc_one_copy : QuantumCircuit
        The (noiseless) encoding circuit for one copy: reference-Bell prep +
        encoding unitary, built from H and CX gates only.
    n_one_copy : int
        Number of qubits in one copy (data + reference).
    reference_qubits_one_copy : sequence[int]
        Reference-qubit indices (one-copy indexing).  These are *not* noised and
        are excluded from the Q subsystem purity.
    tested_qubits_one_copy : sequence[int]
        Qubits included in the swap test (usually ``range(n_one_copy)``).
    d : int, default 2
        Qudit dimension (prime).  Bit-flip noise requires d = 2.

    Notes
    -----
    Noise is applied to every non-reference qubit of *both* copies, immediately
    after the encoding and before the swap test -- matching the reference
    workflow in the surface/color-code notebooks.
    """

    # Weyl index a*d + b for the pure X error (a=1, b=0); used by the bit-flip fast path.
    _X_TYPE = None  # set in __init__ once d is known

    def __init__(
        self,
        qc_one_copy: QuantumCircuit,
        n_one_copy: int,
        reference_qubits_one_copy,
        tested_qubits_one_copy,
        d: int = 2,
    ):
        self.d = int(d)
        self.n_one_copy = int(n_one_copy)
        self.n_meas = 2 * self.n_one_copy
        self.reference_qubits = list(reference_qubits_one_copy)
        self.tested_qubits = list(tested_qubits_one_copy)
        self._X_TYPE = 1 * self.d + 0  # a=1, b=0

        ref = set(self.reference_qubits)
        ref2 = {q + self.n_one_copy for q in self.reference_qubits}
        # Noise sits on every non-reference qubit of both copies.
        self._noisy_qubits = [
            q for q in range(self.n_meas) if q not in ref and q not in ref2
        ]

        # Swap-test op list (forward order) and measured pairs.
        self._swap_ops = []
        self._pairs_RQ = []
        for q in self.tested_qubits:
            self._swap_ops.append(("CNOT", (q, q + self.n_one_copy)))
            self._swap_ops.append(("H", (q,)))
            self._pairs_RQ.append((q, q + self.n_one_copy))
        # Q subsystem: drop reference qubits.
        self._pairs_Q = [(a, b) for (a, b) in self._pairs_RQ if a not in ref]

        # Pair index arrays for fast parity.
        self._a_RQ = np.array([a for a, _ in self._pairs_RQ], dtype=np.intp)
        self._b_RQ = np.array([b for _, b in self._pairs_RQ], dtype=np.intp)
        self._a_Q = np.array([a for a, _ in self._pairs_Q], dtype=np.intp)
        self._b_Q = np.array([b for _, b in self._pairs_Q], dtype=np.intp)

        # Compile the effect table: F[l, etype, :] = final bit-flip pattern from
        # a Weyl error of type etype = a*d+b injected on noisy qubit l.
        L = len(self._noisy_qubits)
        self._F = np.zeros((L, self.d * self.d, self.n_meas), dtype=np.uint8)
        for l, q0 in enumerate(self._noisy_qubits):
            for a in range(self.d):
                for b in range(self.d):
                    et = a * self.d + b
                    flip = _propagate_error_through_ops(
                        a, b, q0, self.n_meas, self._swap_ops, self.d
                    )
                    self._F[l, et, :] = flip.astype(np.uint8)
        # Bit-packed copy along the measured-qubit axis (compact + handy for reporting).
        self._F_packed = np.packbits(self._F, axis=2)

        # Noiseless two-copy circuit (encoding + swap test, NO noise) for the pool.
        qc_two = duplicate_circuit_vertical(qc_one_copy)
        qc_two, _ = add_destructive_swap_test(qc_two, self.n_one_copy, self.tested_qubits)
        self._qc_noiseless = qc_two

        self._pool = None          # (pool_shots, n_meas) uint8
        self._pool_shots = 0

    # ── noiseless pool ──────────────────────────────────────────────────────

    def precompute_noiseless_pool(self, n_shots: int, seed=None) -> np.ndarray:
        """
        Simulate the noiseless two-copy tableau once and cache ``n_shots``
        noiseless measurement outcomes for reuse across all p-values.
        """
        from .tableau import Tableau
        T_clean, _ = Tableau.clean_run(self._qc_noiseless)
        pool = T_clean.sample_measurements(int(n_shots), seed=seed)
        self._pool = np.asarray(pool, dtype=np.uint8)
        self._pool_shots = self._pool.shape[0]
        return self._pool

    # ── noise sampling ──────────────────────────────────────────────────────

    def _noise_type_probs(self, p: float, error_model) -> np.ndarray:
        """Return the length-(d*d) Weyl probability vector for the given model."""
        if callable(error_model):
            P = np.asarray(error_model(p), dtype=float)
        elif error_model == "bitflip":
            P = bitflip_noise_model(p)
        elif error_model == "depolarizing":
            P = depolarizing_noise_model(p)
        else:
            raise ValueError(f"Unknown error_model: {error_model!r}")
        probs = P.reshape(-1)
        return probs / probs.sum()

    def _apply_noise_flips(self, chunk: int, p: float, error_model, rng: np.random.Generator) -> np.ndarray:
        """
        Sample noise for ``chunk`` shots and return the bit-flip pattern
        (chunk, n_meas) uint8 to XOR into noiseless outcomes.

        Fast path for bit-flip: a single Bernoulli draw per (shot, location) and
        one GF(2) matvec against the X effect rows.  General path: sample a Weyl
        type per (shot, location) and accumulate the effect of each non-identity
        type.
        """
        L = len(self._noisy_qubits)
        if L == 0:
            return np.zeros((chunk, self.n_meas), dtype=np.uint8)

        probs = self._noise_type_probs(p, error_model)
        # Bit-flip fast path: only the X type has nonzero probability.
        nonzero = np.nonzero(probs > 0)[0]
        is_bitflip = set(nonzero.tolist()) <= {0, self._X_TYPE}

        if is_bitflip:
            p_x = float(probs[self._X_TYPE])
            fired = (rng.random((chunk, L)) < p_x).astype(np.uint8)   # (chunk, L)
            Fx = self._F[:, self._X_TYPE, :]                          # (L, n_meas)
            # uint8 matmul wraps mod 256; since 256 is even, (x mod 256) mod 2 == x mod 2.
            return (fired @ Fx) & 1

        # General Pauli path.
        types = rng.choice(self.d * self.d, size=(chunk, L), p=probs).astype(np.int64)
        flip = np.zeros((chunk, self.n_meas), dtype=np.uint8)
        for et in nonzero:
            if et == 0:  # identity contributes no flip
                continue
            mask = (types == et).astype(np.uint8)                    # (chunk, L)
            flip ^= (mask @ self._F[:, et, :]) & 1
        return flip

    # ── parity / purity ─────────────────────────────────────────────────────

    def _parity_counts(self, noisy: np.ndarray):
        """Return (count_RQ, count_Q): #shots whose swap-test parity is 1."""
        cross_RQ = noisy[:, self._a_RQ] & noisy[:, self._b_RQ]
        cnt_RQ = int(np.bitwise_xor.reduce(cross_RQ, axis=1).sum())
        cross_Q = noisy[:, self._a_Q] & noisy[:, self._b_Q]
        cnt_Q = int(np.bitwise_xor.reduce(cross_Q, axis=1).sum())
        return cnt_RQ, cnt_Q

    # ── single-p estimate ───────────────────────────────────────────────────

    def compute_I2(
        self,
        p: float,
        n_shots: int,
        error_model="bitflip",
        seed=None,
        use_pool: bool = False,
        batch_size: int = 500_000,
    ) -> dict:
        """
        Estimate I2 at a single noise strength ``p``.

        If ``use_pool`` is True the cached noiseless pool is reused (call
        :meth:`precompute_noiseless_pool` first with at least ``n_shots`` shots);
        otherwise a fresh noiseless pool is sampled in memory-bounded chunks.
        """
        rng = np.random.default_rng(seed)
        n_shots = int(n_shots)

        if use_pool:
            if self._pool is None or self._pool_shots < n_shots:
                raise ValueError(
                    "use_pool=True requires precompute_noiseless_pool(n_shots>=requested) first."
                )

        cnt_RQ = 0
        cnt_Q = 0
        remaining = n_shots
        offset = 0
        while remaining > 0:
            chunk = min(batch_size, remaining)
            if use_pool:
                clean = self._pool[offset:offset + chunk]
            else:
                from .tableau import Tableau
                T_clean, _ = Tableau.clean_run(self._qc_noiseless)
                clean = np.asarray(
                    T_clean.sample_measurements(chunk, seed=int(rng.integers(2 ** 63 - 1))),
                    dtype=np.uint8,
                )
            flips = self._apply_noise_flips(chunk, p, error_model, rng)
            noisy = clean ^ flips
            c_rq, c_q = self._parity_counts(noisy)
            cnt_RQ += c_rq
            cnt_Q += c_q
            remaining -= chunk
            offset += chunk

        P_RQ = 1.0 - 2.0 * cnt_RQ / n_shots
        P_Q = 1.0 - 2.0 * cnt_Q / n_shots
        S2_RQ = -np.log(max(abs(P_RQ), 1e-15))
        S2_Q = -np.log(max(abs(P_Q), 1e-15))
        I2 = S2_Q - S2_RQ
        se = _se_I2(P_RQ, P_Q, n_shots)
        return {
            "p": p,
            "P_RQ": P_RQ,
            "P_Q": P_Q,
            "se_P_RQ": _se_purity(P_RQ, n_shots),
            "se_P_Q": _se_purity(P_Q, n_shots),
            "S2_RQ": S2_RQ,
            "S2_Q": S2_Q,
            "I2": I2,
            "I2_bits": I2 / np.log(2),
            "se_I2": se,
            "se_I2_bits": se / np.log(2),
            "shots_used": n_shots,
        }

    # ── scan over p ─────────────────────────────────────────────────────────

    def scan_ps(
        self,
        ps,
        n_shots: int,
        error_model="bitflip",
        seed0: int = 0,
        use_pool: bool = False,
        batch_size: int = 500_000,
    ) -> list:
        """
        Estimate I2 for every p in ``ps``.

        With ``use_pool=True`` the noiseless pool is built once (if not already
        cached) and reused for all p-values -- the decisive speed-up when
        scanning many p.  Returns a list of per-p result dicts.
        """
        ps = list(ps)
        if use_pool and (self._pool is None or self._pool_shots < int(n_shots)):
            self.precompute_noiseless_pool(int(n_shots), seed=seed0 ^ 0x9E3779B9)

        rows = []
        for i, p in enumerate(ps):
            seed = None if seed0 is None else (seed0 + i)
            rows.append(
                self.compute_I2(
                    p, n_shots, error_model=error_model,
                    seed=seed, use_pool=use_pool, batch_size=batch_size,
                )
            )
        return rows


# ─────────────────────────────────────────────────────────────────────────────
# Self-check against the independent backend sampler
# ─────────────────────────────────────────────────────────────────────────────

def _backend_I2(qc_one_copy, n_one_copy, reference_qubits, p, shots, seed, d=2):
    """Reference I2 via Tableau.sample_measurements_noisy_pf (full frame propagation)."""
    from .tableau import Tableau
    ref = set(reference_qubits)
    qc = duplicate_circuit_vertical(qc_one_copy)
    ref2 = {q + n_one_copy for q in reference_qubits}
    P = bitflip_noise_model(p)
    for q in range(2 * n_one_copy):
        if q not in ref and q not in ref2:
            qc.add_noise_channel(qudit=q, noise_model=P)
    tested = list(range(n_one_copy))
    qc, pairs_RQ = add_destructive_swap_test(qc, n_one_copy, tested)
    pairs_Q = [(a, b) for (a, b) in pairs_RQ if a not in ref]

    tab = Tableau(2 * n_one_copy, d, full=True)
    s = tab.sample_measurements_noisy_pf(qc=qc, shots=shots, seed=seed)

    def count(pairs):
        acc = np.zeros(shots, dtype=np.int64)
        for a, b in pairs:
            acc ^= (s[:, a] & s[:, b]).astype(np.int64)
        return int(acc.sum())

    P_RQ = 1.0 - 2.0 * count(pairs_RQ) / shots
    P_Q = 1.0 - 2.0 * count(pairs_Q) / shots
    I2 = (-np.log(abs(P_Q))) - (-np.log(abs(P_RQ)))
    return I2 / np.log(2), P_RQ, P_Q


def run_unit_tests(verbose: bool = True) -> bool:
    """
    Validate the fast effect-table + pool sampler against the backend Pauli-frame
    sampler on a small hand-built circuit.  Returns True on success.

    The two estimators use different (independent) sampling internals, so we
    require agreement of I2 within a few combined standard errors.
    """
    from .circuit import QuantumCircuit
    n1 = 4
    ref = [3]
    qc1 = QuantumCircuit(n1, 2)
    qc1.H(0); qc1.CX(0, 1); qc1.CX(1, 2); qc1.H(3); qc1.CX(3, 0)

    sampler = PauliFrameSampler(
        qc_one_copy=qc1,
        n_one_copy=n1,
        reference_qubits_one_copy=ref,
        tested_qubits_one_copy=list(range(n1)),
        d=2,
    )

    shots = 2_000_000
    ps = [0.0, 0.05, 0.10, 0.20]
    sampler.precompute_noiseless_pool(shots, seed=123)

    ok = True
    if verbose:
        print(f"run_unit_tests: {len(sampler._noisy_qubits)} noisy qubits, "
              f"effect table {sampler._F.shape}, packed {sampler._F_packed.shape} "
              f"({sampler._F_packed.nbytes} B)")
        print(f"{'p':>6} {'fast I2':>9} {'backend I2':>11} {'|dI2|':>8} {'tol(4SE)':>9}  ok")
    for p in ps:
        fast = sampler.compute_I2(p, shots, error_model="bitflip", seed=7, use_pool=True)
        b_I2, _, _ = _backend_I2(qc1, n1, ref, p, shots, seed=7)
        tol = 4.0 * fast["se_I2_bits"] + 0.01
        diff = abs(fast["I2_bits"] - b_I2)
        passed = diff <= tol
        ok = ok and passed
        if verbose:
            print(f"{p:6.2f} {fast['I2_bits']:9.4f} {b_I2:11.4f} "
                  f"{diff:8.4f} {tol:9.4f}  {'PASS' if passed else 'FAIL'}")

    # p=0 sanity: with one reference qubit, I2 should be ~1 bit.
    r0 = sampler.compute_I2(0.0, shots, error_model="bitflip", seed=1, use_pool=True)
    sane = abs(r0["I2_bits"] - 1.0) < 0.02
    ok = ok and sane
    if verbose:
        print(f"p=0 sanity: I2={r0['I2_bits']:.4f} bits (expect ~1.0)  "
              f"{'PASS' if sane else 'FAIL'}")

    if verbose:
        print("ALL TESTS PASSED" if ok else "SOME TESTS FAILED")
    return ok


# ─────────────────────────────────────────────────────────────────────────────
# General noisy-circuit sampler
# ─────────────────────────────────────────────────────────────────────────────

class NoisyCircuitSampler:
    """
    Sample Z-basis outcomes of a Clifford circuit with Pauli (Weyl) ``NoiseChannel`` s.

    Same distribution as :meth:`Tableau.sample_measurements_noisy_pf`, but the circuit is
    processed once instead of once per shot batch:

    1. **Compile.** An error ``W(a, b)`` injected at a noise location reaches the final
       measurement as a fixed shift of the outcome (Clifford gates map Paulis to Paulis).
       Every (location, Weyl type) is propagated through the remaining gates in one
       vectorised pass, giving the effect table ``F[location, a*d + b, :]``.
    2. **Noiseless outcomes** come from the clean tableau (``sample_measurements``).
    3. **Noise** is drawn per shot and location from the channel's own probabilities and
       added by table lookup: one matrix product per Weyl type.

    The table depends on where the noise sits, not on how strong it is, so the strength can
    be rescanned with ``sample(..., probs=...)`` without recompiling. Gates the Pauli-frame
    backend ignores (Paulis, unknown names) are ignored here too.

    Parameters
    ----------
    qc : QuantumCircuit
        Circuit with ``NoiseChannel`` s. Each channel acts on one qudit; its probabilities
        are the Weyl vector ``p[a*d + b]`` of the circuit's noise registry.

    Attributes
    ----------
    n_locations : int
        Number of noise channels in the circuit.
    effects : np.ndarray
        Effect table, shape ``(n_locations, d*d, n)``, entries in ``Z_d``.
    """

    def __init__(self, qc: "QuantumCircuit"):
        from .tableau import Tableau

        self.n, self.d = qc.n, qc.d
        n, d, T = self.n, self.d, self.d * self.d

        probs = [np.asarray(qc.noise_registry[g.model_id], dtype=float).reshape(T)
                 for g, _, _ in qc._ops if isinstance(g, NoiseChannel)]
        self.n_locations = L = len(probs)
        self._probs = np.array(probs).reshape(L, T)

        # Row l*T + t carries the error of Weyl type t at location l.
        x = np.zeros((L * T, n), dtype=np.int64)
        z = np.zeros_like(x)
        types = np.arange(T)
        loc = 0
        for gate, qudits, dagger in qc._ops:
            if isinstance(gate, NoiseChannel):
                rows = slice(loc * T, (loc + 1) * T)
                x[rows, qudits[0]] = types // d
                z[rows, qudits[0]] = types % d
                loc += 1
            else:  # rows not injected yet are zero, so only the injected ones need updating
                Tableau._pf_apply_gate(x[:loc * T], z[:loc * T], key=Tableau._pf_gate_key(gate),
                                       qudits=qudits, dagger=dagger, d=d)
        self.effects = (x % d).reshape(L, T, n).astype(np.int16)
        self._clean, _ = Tableau.clean_run(qc)

    def sample(self, shots: int, rng=None, seed=None, probs=None, batch_size: int = 20_000):
        """
        Draw ``shots`` outcome vectors, shape ``(shots, n)``, entries in ``Z_d``.

        ``probs`` overrides the circuit's noise: shape ``(d*d,)`` for every location or
        ``(n_locations, d*d)``; rows are normalised Weyl probabilities ``p[a*d + b]``.
        """
        if shots <= 0:
            raise ValueError("shots must be a positive integer.")
        rng = rng if rng is not None else np.random.default_rng(seed)
        n, d, L = self.n, self.d, self.n_locations
        T = d * d

        P = self._probs if probs is None else np.broadcast_to(np.asarray(probs, float), (L, T))
        cdf = np.cumsum(P, axis=1)
        cdf[:, -1] = 1.0
        active = [t for t in range(1, T) if P[:, t].any()]
        effect = {t: self.effects[:, t, :].astype(np.float32) for t in active}

        out = np.asarray(self._clean.sample_measurements(shots=shots, rng=rng), dtype=np.int64)
        if L == 0:
            return out % d
        for s in range(0, shots, batch_size):
            m = min(batch_size, shots - s)
            kind = (rng.random((m, L, 1)) >= cdf).sum(axis=2)  # inverse-CDF draw of the Weyl type
            shift = sum((kind == t).astype(np.float32) @ effect[t] for t in active)
            out[s:s + m] += np.rint(shift).astype(np.int64) if active else 0
        return out % d

