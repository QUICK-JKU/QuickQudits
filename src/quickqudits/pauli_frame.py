"""
Fast Pauli-noise sampling by effect-table lookup.

An error injected at a noise location reaches the final Z-basis measurements as a fixed
shift of the outcome, because Clifford gates map Paulis to Paulis. :class:`NoisyCircuitSampler`
compiles those shifts once and then samples a noisy Clifford circuit by table lookup, with the
same distribution as ``Tableau.sample_measurements_noisy_pf`` and several times faster.

    sampler = NoisyCircuitSampler(qc)                  # qc has NoiseChannels
    sampler.precompute_noiseless_pool(100_000, seed=1) # optional: reuse the clean outcomes
    outcomes = sampler.sample(50_000, seed=2, use_pool=True)   # (shots, n); pool_offset=k skips k
    outcomes = sampler.sample(50_000, seed=3, probs=p_weyl)    # rescan the noise strength
"""

from __future__ import annotations

import numpy as np

from .gate_class import NoiseChannel


class NoisyCircuitSampler:
    """
    Sample Z-basis outcomes of a Clifford circuit with Pauli (Weyl) ``NoiseChannel`` s.

    1. **Compile.** Every (location, Weyl type) error is propagated through the remaining gates
       in one vectorised pass, giving the effect table ``F[location, a*d + b, :]``.
    2. **Noiseless outcomes** come from the clean tableau (``sample_measurements``), either
       drawn per call or cached once with :meth:`precompute_noiseless_pool`.
    3. **Noise** is drawn per shot and location from the channel's probabilities and added by
       table lookup: one matrix product per Weyl type.

    The table depends on where the noise sits, not on how strong it is, so the strength can be
    rescanned with ``sample(..., probs=...)`` without recompiling. Gates the Pauli-frame backend
    ignores (Paulis, unknown names) are ignored here too.

    Parameters
    ----------
    qc : QuantumCircuit
        Circuit with ``NoiseChannel`` s. Each channel acts on one qudit; its probabilities are
        the Weyl vector ``p[a*d + b]`` of the circuit's noise registry.

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
        self._pool = None

    def precompute_noiseless_pool(self, shots: int, seed=None) -> np.ndarray:
        """Draw ``shots`` noiseless outcomes once; reuse them with ``sample(..., use_pool=True)``."""
        self._pool = np.asarray(self._clean.sample_measurements(shots=int(shots), seed=seed),
                                dtype=np.int64)
        return self._pool

    def sample(self, shots: int, rng=None, seed=None, probs=None, batch_size: int = 20_000,
               use_pool: bool = False, pool_offset: int = 0):
        """
        Draw ``shots`` outcome vectors, shape ``(shots, n)``, entries in ``Z_d``.

        ``probs`` overrides the circuit's noise: shape ``(d*d,)`` for every location or
        ``(n_locations, d*d)``; rows are normalised Weyl probabilities ``p[a*d + b]``.
        ``use_pool`` takes the cached noiseless outcomes ``pool[pool_offset : pool_offset + shots]``
        instead of sampling them.
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

        if use_pool:
            if self._pool is None or len(self._pool) < pool_offset + shots:
                raise ValueError("use_pool=True needs a pool of at least pool_offset + shots outcomes.")
            out = self._pool[pool_offset:pool_offset + shots].copy()
        else:
            out = np.asarray(self._clean.sample_measurements(shots=shots, rng=rng), dtype=np.int64)
        if L == 0:
            return out % d
        for s in range(0, shots, batch_size):
            m = min(batch_size, shots - s)
            kind = (rng.random((m, L, 1)) >= cdf).sum(axis=2)  # inverse-CDF draw of the Weyl type
            shift = sum((kind == t).astype(np.float32) @ effect[t] for t in active)
            out[s:s + m] += np.rint(shift).astype(np.int64) if active else 0
        return out % d
