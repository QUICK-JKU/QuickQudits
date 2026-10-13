import numpy as np
import pytest

import quickqudits as qq
from quickqudits.gate_class import Gate, NoiseChannel


def _certain_x_error():
    model = np.zeros((2, 2))
    model[1, 0] = 1.0
    return model


@pytest.mark.parametrize("method", ["execute", "sample"])
def test_explicit_noise_is_rejected_by_statevector_paths(method):
    circuit = qq.QuantumCircuit(1, 2)
    circuit.add_noise_channel(0, _certain_x_error())

    with pytest.raises(NotImplementedError, match="does not support NoiseChannel") as exc:
        if method == "sample":
            circuit.sample(shots=2, seed=42)
        else:
            circuit.execute()

    assert "execute_density_matrix()" in str(exc.value)
    assert "mid-circuit measurements" in str(exc.value)
    assert "realize_noise(seed=" in str(exc.value)


@pytest.mark.parametrize("method", ["execute", "sample"])
def test_gate_noise_config_circuit_is_rejected_by_statevector_paths(method):
    circuit = qq.QuantumCircuit(1, 2)
    circuit.H(0)
    config = qq.GateNoiseConfig()
    config.set("H", after=0.1)
    noisy = config.apply(circuit)

    with pytest.raises(NotImplementedError, match="NoiseChannel"):
        if method == "sample":
            noisy.sample(shots=2, seed=42)
        else:
            noisy.execute()


def test_ideal_and_realized_circuits_still_execute():
    ideal = qq.QuantumCircuit(1, 2)
    ideal.X(0)
    state, results = ideal.execute()
    np.testing.assert_allclose(state, [0, 1])
    assert results == {}
    _, digits, flat = ideal.sample(shots=3, seed=42)
    np.testing.assert_array_equal(digits, [[1], [1], [1]])
    np.testing.assert_array_equal(flat, [1, 1, 1])

    noisy = qq.QuantumCircuit(1, 2)
    noisy.add_noise_channel(0, _certain_x_error())
    realized, _ = noisy.realize_noise(seed=42, in_place=False)
    assert any(isinstance(gate, NoiseChannel) for gate, _, _ in noisy.ops)
    assert not any(isinstance(gate, NoiseChannel) for gate, _, _ in realized.ops)
    np.testing.assert_allclose(realized.execute()[0], [0, 1])
    np.testing.assert_allclose(noisy.execute_density_matrix(), [[0, 0], [0, 1]])


@pytest.mark.parametrize("method", ["execute", "sample", "mid_circuit"])
def test_noise_is_rejected_before_gates_measurements_or_rng(monkeypatch, method):
    circuit = qq.QuantumCircuit(1, 2)
    circuit.X(0)
    circuit.measure(0)
    circuit.add_noise_channel(0, _certain_x_error())

    def unexpected(*args, **kwargs):
        pytest.fail("execution started before NoiseChannel validation")

    monkeypatch.setattr(circuit.ops[0][0], "apply", unexpected)
    monkeypatch.setattr(circuit, "_perform_measurement", unexpected)
    monkeypatch.setattr(np.random, "choice", unexpected)
    monkeypatch.setattr(np.random, "default_rng", unexpected)

    before = (
        circuit._gate_idx,
        circuit._stored_psi,
        circuit._measurement_results.copy(),
        circuit._qudit_timestamps.copy(),
        circuit._latest_measurements.copy(),
    )
    with pytest.raises(NotImplementedError, match="NoiseChannel"):
        if method == "sample":
            circuit.sample(shots=2, seed=42)
        else:
            circuit.execute(mid_circuit=(method == "mid_circuit"))
    after = (
        circuit._gate_idx,
        circuit._stored_psi,
        circuit._measurement_results.copy(),
        circuit._qudit_timestamps.copy(),
        circuit._latest_measurements.copy(),
    )
    assert after == before


def test_check_uses_noise_channel_type_not_placeholder_name():
    circuit = qq.QuantumCircuit(1, 2)
    circuit.append(Gate("NOISE_CHANNEL", 1, None), 0)
    np.testing.assert_allclose(circuit.execute()[0], [1, 0])
