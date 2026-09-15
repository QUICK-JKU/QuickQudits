import numpy as np
import pytest

import quickqudits as qq
from quickqudits.gate_class import NoiseChannel


def _op_names_and_qudits(circuit):
    return [(gate.name, qudits) for gate, qudits, _ in circuit.ops]


def test_materializes_before_and_after_each_matching_one_qudit_gate():
    circuit = qq.QuantumCircuit(2, 2)
    circuit.H(0)
    circuit.X(1)
    circuit.Hdag(1)

    config = qq.GateNoiseConfig()
    config.set("H", before=0.1, after=0.2)

    noisy = config.apply(circuit)

    assert _op_names_and_qudits(noisy) == [
        ("NOISE_CHANNEL", (0,)),
        ("H", (0,)),
        ("NOISE_CHANNEL", (0,)),
        ("X", (1,)),
        ("NOISE_CHANNEL", (1,)),
        ("H", (1,)),
        ("NOISE_CHANNEL", (1,)),
    ]
    assert _op_names_and_qudits(circuit) == [
        ("H", (0,)),
        ("X", (1,)),
        ("H", (1,)),
    ]

    noise_gates = [gate for gate, _, _ in noisy if isinstance(gate, NoiseChannel)]
    before_probs = noisy.noise_registry[noise_gates[0].model_id]
    after_probs = noisy.noise_registry[noise_gates[1].model_id]
    np.testing.assert_allclose(before_probs, [0.9, 0.1 / 3, 0.1 / 3, 0.1 / 3])
    np.testing.assert_allclose(after_probs, [0.8, 0.2 / 3, 0.2 / 3, 0.2 / 3])


def test_materialized_noise_executes_like_explicit_channels():
    before = np.zeros((2, 2))
    before[1, 0] = 1.0
    after = np.zeros((2, 2))
    after[0, 1] = 1.0

    circuit = qq.QuantumCircuit(1, 2)
    circuit.H(0)

    config = qq.GateNoiseConfig()
    config.set("H", before=before, after=after)
    actual = config.apply(circuit).execute_density_matrix()

    explicit = qq.QuantumCircuit(1, 2)
    explicit.add_noise_channel(0, before)
    explicit.H(0)
    explicit.add_noise_channel(0, after)
    expected = explicit.execute_density_matrix()

    np.testing.assert_allclose(actual, expected)


def test_in_place_application_returns_and_updates_original_circuit():
    circuit = qq.QuantumCircuit(1, 3)
    circuit.X(0)

    config = qq.GateNoiseConfig()
    config.set("X", after=0.15)

    result = config.apply(circuit, in_place=True)

    assert result is circuit
    assert _op_names_and_qudits(circuit) == [
        ("X", (0,)),
        ("NOISE_CHANNEL", (0,)),
    ]


def test_configuration_copies_probability_arrays():
    model = np.array([[0.75, 0.1], [0.1, 0.05]])
    config = qq.GateNoiseConfig()
    config.set("X", after=model)
    model[:] = 0.25

    circuit = qq.QuantumCircuit(1, 2)
    circuit.X(0)
    noisy = config.apply(circuit)

    noise_gate = noisy.ops[1][0]
    np.testing.assert_allclose(
        noisy.noise_registry[noise_gate.model_id], [0.75, 0.1, 0.1, 0.05]
    )


def test_materializes_noise_on_both_operands_of_two_qudit_gate():
    circuit = qq.QuantumCircuit(2, 2)
    circuit.CNOT(0, 1)

    config = qq.GateNoiseConfig()
    config.set("CNOT", before=0.1, after=0.2)

    noisy = config.apply(circuit)

    assert _op_names_and_qudits(noisy) == [
        ("NOISE_CHANNEL", (0,)),
        ("NOISE_CHANNEL", (1,)),
        ("CNOT", (0, 1)),
        ("NOISE_CHANNEL", (0,)),
        ("NOISE_CHANNEL", (1,)),
    ]
    assert _op_names_and_qudits(circuit) == [("CNOT", (0, 1))]


def test_two_qudit_gate_noise_executes_like_explicit_local_channels():
    before = np.zeros((2, 2))
    before[1, 0] = 1.0
    after = np.zeros((2, 2))
    after[0, 1] = 1.0

    circuit = qq.QuantumCircuit(2, 2)
    circuit.CNOT(1, 0)

    config = qq.GateNoiseConfig()
    config.set("CNOT", before=before, after=after)
    actual = config.apply(circuit).execute_density_matrix()

    explicit = qq.QuantumCircuit(2, 2)
    explicit.add_noise_channel(1, before)
    explicit.add_noise_channel(0, before)
    explicit.CNOT(1, 0)
    explicit.add_noise_channel(1, after)
    explicit.add_noise_channel(0, after)
    expected = explicit.execute_density_matrix()

    np.testing.assert_allclose(actual, expected)


def test_invalid_noise_is_atomic_for_in_place_application():
    circuit = qq.QuantumCircuit(1, 2)
    circuit.H(0)
    circuit.X(0)

    config = qq.GateNoiseConfig()
    config.set("H", after=0.1)
    config.set("X", after=2.0)

    with pytest.raises(ValueError, match="between 0 and 1"):
        config.apply(circuit, in_place=True)

    assert _op_names_and_qudits(circuit) == [("H", (0,)), ("X", (0,))]
    assert circuit.noise_registry == {}


def test_rules_can_be_replaced_removed_and_cleared():
    config = qq.GateNoiseConfig()

    with pytest.raises(ValueError, match="At least one"):
        config.set("H")

    config.set("H", before=0.1)
    config.set("H", after=0.2)
    assert len(config) == 1

    circuit = qq.QuantumCircuit(1, 2)
    circuit.H(0)
    assert _op_names_and_qudits(config.apply(circuit)) == [
        ("H", (0,)),
        ("NOISE_CHANNEL", (0,)),
    ]

    config.remove("H")
    assert len(config) == 0
    config.set("X", after=0.1)
    config.clear()
    assert len(config) == 0
