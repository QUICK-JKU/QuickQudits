import numpy as np
import pytest

from quickqudits import (
    PauliFrameSampler,
    QuantumCircuit,
    bitflip_noise_model,
    depolarizing_noise_model,
    shots_for_target_se,
)
from quickqudits.pauli_frame import _backend_I2, _propagate_error_through_ops

N_ONE_COPY = 4
REF = [3]


def _encoder():
    qc = QuantumCircuit(N_ONE_COPY, 2)
    qc.H(0)
    qc.CX(0, 1)
    qc.CX(1, 2)
    qc.H(3)
    qc.CX(3, 0)
    return qc


def _sampler():
    return PauliFrameSampler(_encoder(), N_ONE_COPY, REF, list(range(N_ONE_COPY)), d=2)


@pytest.mark.parametrize("model", [bitflip_noise_model, depolarizing_noise_model])
def test_noise_models_are_distributions(model):
    P = model(0.1)
    assert P.shape == (2, 2)
    assert P.sum() == pytest.approx(1.0)
    assert (P >= 0).all()


def test_shots_for_target_se():
    assert shots_for_target_se(0.5, 0.1) == 300  # ceil((1 - 0.25) / (0.25 * 0.01))


def test_error_propagation_through_swap_layer():
    ops = [("CNOT", (0, 1)), ("H", (0,))]
    assert _propagate_error_through_ops(1, 0, 0, 2, ops, 2).tolist() == [0, 1]  # X0
    assert _propagate_error_through_ops(0, 1, 0, 2, ops, 2).tolist() == [1, 0]  # Z0


def test_effect_table_shape_and_noisy_qubits():
    s = _sampler()
    assert s._noisy_qubits == [0, 1, 2, 4, 5, 6]  # reference of both copies stays clean
    assert s._F.shape == (6, 4, 2 * N_ONE_COPY)


def test_noiseless_point_is_one_bit():
    s = _sampler()
    s.precompute_noiseless_pool(20_000, seed=1)
    r = s.compute_I2(0.0, 20_000, error_model="bitflip", seed=1, use_pool=True)
    assert r["I2_bits"] == pytest.approx(1.0, abs=0.02)


def test_pool_is_reproducible():
    s = _sampler()
    s.precompute_noiseless_pool(5_000, seed=3)
    a = s.compute_I2(0.1, 5_000, error_model="bitflip", seed=9, use_pool=True)
    b = s.compute_I2(0.1, 5_000, error_model="bitflip", seed=9, use_pool=True)
    assert a["I2_bits"] == b["I2_bits"]


@pytest.mark.parametrize("p", [0.05, 0.2])
def test_sampler_matches_frame_propagation(p):
    shots = 200_000
    s = _sampler()
    s.precompute_noiseless_pool(shots, seed=5)
    fast = s.compute_I2(p, shots, error_model="bitflip", seed=7, use_pool=True)
    ref, _, _ = _backend_I2(_encoder(), N_ONE_COPY, REF, p, shots, seed=7)
    assert abs(fast["I2_bits"] - ref) <= 4 * fast["se_I2_bits"] + 0.02
