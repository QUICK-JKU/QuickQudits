"""Analytic gate-noise checks shared by the three noisy tableau samplers."""

import numpy as np
import pytest

import quickqudits as qq
from quickqudits.gate_class import NoiseChannel


SAMPLERS = (
    "sample_measurements_noisy_naive",
    "sample_measurements_noise_push",
    "sample_measurements_noisy_pf",
)


def _certain_x(d):
    probabilities = np.zeros((d, d))
    probabilities[1, 0] = 1.0
    return probabilities


def _outcomes(tableau, sampler, circuit, shots=8, seed=13):
    result = getattr(tableau, sampler)(circuit, shots=shots, seed=seed)
    return result[0] if isinstance(result, tuple) else result


def _density_probabilities(circuit):
    # Circuit qudit 0 is the leftmost/base-d most significant digit.
    return np.diag(circuit.execute_density_matrix()).real.reshape((circuit.d,) * circuit.n)


@pytest.mark.parametrize("d", [2, 3])
@pytest.mark.parametrize("side", ["before", "after"])
@pytest.mark.parametrize("sampler", SAMPLERS)
def test_h_x_order_is_distinguishable_in_computational_basis(d, side, sampler):
    circuit = qq.QuantumCircuit(1, d)
    circuit.H(0)
    config = qq.GateNoiseConfig()
    config.set("H", **{side: _certain_x(d)})
    noisy = config.apply(circuit)
    # Added after materialization so only the first H receives gate noise.
    noisy.H(0)

    # Fourier H squared maps |j> to |-j>. Thus X before the first H sends
    # |0> to |-1>; X between the two H gates is conjugated to a Z error and
    # leaves the measured |0> unchanged (up to phase).
    expected = d - 1 if side == "before" else 0
    expected_probs = np.eye(d)[expected]
    np.testing.assert_allclose(_density_probabilities(noisy), expected_probs, atol=1e-12)
    np.testing.assert_array_equal(
        _outcomes(qq.Tableau(1, d), sampler, noisy),
        np.full((8, 1), expected),
    )


@pytest.mark.parametrize("d", [2, 3])
@pytest.mark.parametrize("side", ["before", "after"])
@pytest.mark.parametrize("sampler", SAMPLERS)
def test_cnot_noise_is_local_to_both_operands(d, side, sampler):
    circuit = qq.QuantumCircuit(2, d)
    circuit.CNOT(0, 1)
    config = qq.GateNoiseConfig()
    config.set("CNOT", **{side: _certain_x(d)})
    noisy = config.apply(circuit)

    ops = [(type(gate), qudits) for gate, qudits, _ in noisy.ops]
    local_channels = [(NoiseChannel, (0,)), (NoiseChannel, (1,))]
    if side == "before":
        assert ops[:2] == local_channels
        assert ops[2][1] == (0, 1)
    else:
        assert ops[0][1] == (0, 1)
        assert ops[1:] == local_channels

    # Independent X on both inputs gives |1,1>, then CNOT maps the target
    # to 1+1 mod d. Applied afterwards, both X errors give |1,1> instead.
    expected = (1, 2 % d) if side == "before" else (1, 1)
    expected_probs = np.zeros((d, d))
    expected_probs[expected] = 1.0
    np.testing.assert_allclose(_density_probabilities(noisy), expected_probs, atol=1e-12)
    np.testing.assert_array_equal(
        _outcomes(qq.Tableau(2, d), sampler, noisy),
        np.tile(expected, (8, 1)),
    )


@pytest.mark.parametrize("sampler", SAMPLERS)
def test_stochastic_gate_noise_matches_density_probabilities(sampler):
    circuit = qq.QuantumCircuit(1, 2)
    circuit.H(0)
    config = qq.GateNoiseConfig()
    model = np.array([[0.7, 0.3], [0.0, 0.0]])
    config.set("H", after=model)
    noisy = config.apply(circuit)
    noisy.H(0)

    # I yields |0> and Z between the two H gates yields |1>.
    # For 400 shots, a 0.12 absolute tolerance exceeds five binomial standard
    # deviations at p=.3 while still detecting a substantial wrong rate.
    reference = _density_probabilities(noisy)
    np.testing.assert_allclose(reference, [0.7, 0.3], atol=1e-12)
    outcomes = _outcomes(qq.Tableau(1, 2), sampler, noisy, shots=400, seed=107)
    observed = np.bincount(outcomes[:, 0], minlength=2) / 400
    np.testing.assert_allclose(observed, reference, atol=0.12)


def test_reapplying_config_and_global_noise_stack_channels():
    circuit = qq.QuantumCircuit(1, 2)
    circuit.H(0)
    config = qq.GateNoiseConfig()
    config.set("H", after=_certain_x(2))

    once = config.apply(circuit)
    twice = config.apply(once)
    twice.add_noise_global(_certain_x(2))

    assert [gate.name for gate, _, _ in circuit.ops] == ["H"]
    assert [gate.name for gate, _, _ in once.ops] == ["H", "NOISE_CHANNEL"]
    assert [gate.name for gate, _, _ in twice.ops] == [
        "H", "NOISE_CHANNEL", "NOISE_CHANNEL", "NOISE_CHANNEL"
    ]


def test_plain_apply_circuit_ignores_noise_placeholders():
    circuit = qq.QuantumCircuit(1, 2)
    circuit.add_noise_channel(0, _certain_x(2))
    tableau = qq.Tableau(1, 2)

    tableau.apply_circuit(circuit, engine="python")
    np.testing.assert_array_equal(tableau.measure_all(seed=13)[1], [0])
    np.testing.assert_array_equal(
        tableau.sample_measurements_noisy_pf(circuit, shots=8, seed=13),
        np.ones((8, 1), dtype=int),
    )


@pytest.mark.parametrize("sampler", SAMPLERS)
def test_composite_dimension_four_uses_its_measurement_api(sampler):
    d = 4
    circuit = qq.QuantumCircuit(1, d)
    circuit.H(0)
    config = qq.GateNoiseConfig()
    config.set("H", before=_certain_x(d))
    noisy = config.apply(circuit)
    noisy.H(0)

    np.testing.assert_allclose(_density_probabilities(noisy), np.eye(d)[3], atol=1e-12)
    np.testing.assert_array_equal(
        _outcomes(qq.CompositeTableau(1, d), sampler, noisy),
        np.full((8, 1), 3),
    )
