import numpy as np
import pytest

from quickqudits import NoisyCircuitSampler, QuantumCircuit, Tableau


def bitflip_noise_model(p):
    return np.array([[1 - p, 0.0], [p, 0.0]])


def depolarizing_noise_model(p):
    return np.array([[1 - p, p / 3], [p / 3, p / 3]])

SHOTS = 300_000


def _probvector(outcomes, d):
    n = outcomes.shape[1]
    idx = outcomes @ (d ** np.arange(n - 1, -1, -1))
    return np.bincount(idx, minlength=d**n) / len(outcomes)


def _circuit_d2(model):
    qc = QuantumCircuit(3, 2)
    qc.H(0)
    qc.CX(0, 1)
    qc.add_noise_channel(1, model)
    qc.CX(1, 2)
    qc.S(2)
    qc.H(2)
    qc.add_noise_channel(0, model)
    qc.add_noise_channel(2, model)
    return qc


def _circuit_d3():
    qc = QuantumCircuit(3, 3)
    qc.H(0)
    qc.CX(0, 1)
    qc.add_noise_global(0.15)
    qc.CX(1, 2)
    qc.H(1)
    qc.add_noise_global(0.1)
    return qc


@pytest.mark.parametrize("model", [bitflip_noise_model(0.1), depolarizing_noise_model(0.15),
                                   np.array([[0.7, 0.1], [0.05, 0.15]])])
def test_matches_frame_propagation_qubits(model):
    qc = _circuit_d2(model)
    ours = _probvector(NoisyCircuitSampler(qc).sample(SHOTS, seed=1), 2)
    theirs = Tableau(3, 2).to_probvector_noisy_pf(qc, SHOTS, seed=2)
    assert np.abs(ours - theirs).max() < 0.008


def test_matches_frame_propagation_qutrits():
    qc = _circuit_d3()
    ours = _probvector(NoisyCircuitSampler(qc).sample(SHOTS, seed=3), 3)
    theirs = Tableau(3, 3).to_probvector_noisy_pf(qc, SHOTS, seed=4)
    assert np.abs(ours - theirs).max() < 0.008


def test_probs_override_equals_rebuilt_circuit():
    s = NoisyCircuitSampler(_circuit_d2(bitflip_noise_model(0.3)))
    a = _probvector(s.sample(SHOTS, seed=5, probs=bitflip_noise_model(0.05).reshape(-1)), 2)
    b = _probvector(NoisyCircuitSampler(_circuit_d2(bitflip_noise_model(0.05))).sample(SHOTS, seed=6), 2)
    assert np.abs(a - b).max() < 0.008


def test_noiseless_circuit_and_shapes():
    qc = QuantumCircuit(2, 2)
    qc.H(0)
    qc.CX(0, 1)
    s = NoisyCircuitSampler(qc)
    out = s.sample(1000, seed=0)
    assert s.n_locations == 0 and out.shape == (1000, 2)
    assert (out[:, 0] == out[:, 1]).all()  # Bell pair


def test_pool_reuse_matches_fresh_sampling_and_checks_size():
    qc = _circuit_d2(depolarizing_noise_model(0.15))
    s = NoisyCircuitSampler(qc)
    s.precompute_noiseless_pool(SHOTS, seed=7)
    pooled = _probvector(s.sample(SHOTS, seed=8, use_pool=True), 2)
    theirs = Tableau(3, 2).to_probvector_noisy_pf(qc, SHOTS, seed=9)
    assert np.abs(pooled - theirs).max() < 0.008
    with pytest.raises(ValueError):
        s.sample(SHOTS + 1, use_pool=True)


def test_pool_offset_gives_disjoint_slices():
    s = NoisyCircuitSampler(_circuit_d2(bitflip_noise_model(0.0)))
    pool = s.precompute_noiseless_pool(1000, seed=1)
    a = s.sample(400, seed=2, use_pool=True)
    b = s.sample(400, seed=2, use_pool=True, pool_offset=400)
    assert (a == pool[:400]).all() and (b == pool[400:800]).all()
