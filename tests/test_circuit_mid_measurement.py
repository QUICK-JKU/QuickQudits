import numpy as np

from quickqudits import QuantumCircuit


ZERO = np.array([1.0, 0.0], dtype=complex)
ONE = np.array([0.0, 1.0], dtype=complex)


def test_normal_execute_starts_fresh_after_mid_circuit_pause():
    circuit = QuantumCircuit(1, 2)
    circuit.X(0)
    circuit.measure(0)
    circuit.X(0)

    paused_state, paused_results = circuit.execute(mid_circuit=True)

    np.testing.assert_allclose(paused_state, ONE)
    assert paused_results == {0: 1}

    # Disabling mid-circuit execution must run the complete circuit from its
    # documented initial state, independently of an earlier paused execution.
    final_state, results = circuit.execute(mid_circuit=False)

    np.testing.assert_allclose(final_state, ZERO)
    assert results == {0: 1}


def test_each_appended_measurement_stage_pauses_before_following_gates():
    circuit = QuantumCircuit(1, 2)
    circuit.measure(0)

    first_state, first_results = circuit.execute(mid_circuit=True)

    np.testing.assert_allclose(first_state, ZERO)
    assert first_results == {0: 0}

    circuit.X(0)
    circuit.measure(0)
    circuit.X(0)

    # The second invocation must resume after the first measurement and stop
    # at the newly appended measurement, leaving the final X for the next call.
    second_state, second_results = circuit.execute(mid_circuit=True)

    np.testing.assert_allclose(second_state, ONE)
    assert second_results == {0: 1}

    final_state, final_results = circuit.execute(mid_circuit=True)

    np.testing.assert_allclose(final_state, ZERO)
    assert final_results == {0: 1}


def test_completed_measurement_stages_are_not_replayed(monkeypatch):
    circuit = QuantumCircuit(1, 2)
    measurement_count = 0
    perform_measurement = circuit._perform_measurement

    def counting_measurement(psi, n, d, qudit):
        nonlocal measurement_count
        measurement_count += 1
        return perform_measurement(psi, n, d, qudit)

    monkeypatch.setattr(circuit, "_perform_measurement", counting_measurement)

    circuit.measure(0)
    _, first_results = circuit.execute(mid_circuit=True)

    circuit.X(0)
    circuit.measure(0)
    _, second_results = circuit.execute(mid_circuit=True)

    circuit.X(0)
    final_state, final_results = circuit.execute(mid_circuit=True)

    assert first_results == {0: 0}
    assert second_results == {0: 1}
    assert measurement_count == 2
    np.testing.assert_allclose(final_state, ZERO)
    assert final_results == {0: 1}


def test_repeated_measurements_are_timestamped_and_latest_result_is_returned():
    circuit = QuantumCircuit(1, 2)
    circuit.measure(0)
    circuit.X(0)
    circuit.measure(0)

    final_state, results = circuit.execute(mid_circuit=True)

    # Both outcomes are deterministic: history retains the first result while
    # the final classical result contains the latest value for the qudit.
    np.testing.assert_allclose(final_state, ONE)
    assert circuit.get_measurement() == [(0, 0, 0), (0, 2, 1)]
    assert circuit.get_measurement(0) == [(0, 0, 0)]
    assert circuit.get_measurement(2) == [(0, 2, 1)]
    assert circuit.get_measurement({0, 2}) == [(0, 0, 0), (0, 2, 1)]
    assert results == {0: 1}


def test_latest_results_and_statevector_span_resumed_measurement_stages():
    circuit = QuantumCircuit(2, 2)
    circuit.measure(0)

    first_state, first_results = circuit.execute(mid_circuit=True)

    np.testing.assert_allclose(first_state, np.array([1.0, 0.0, 0.0, 0.0]))
    assert first_results == {0: 0}

    circuit.X(1)
    circuit.measure(1)

    second_state, second_results = circuit.execute(mid_circuit=True)

    # The second stage resumes from the state collapsed by the first. Its final
    # results retain the latest outcome for every qudit measured in either stage.
    np.testing.assert_allclose(second_state, np.array([0.0, 1.0, 0.0, 0.0]))
    assert circuit.get_measurement(0) == [(0, 0, 0)]
    assert circuit.get_measurement(1) == [(1, 1, 1)]
    assert second_results == {0: 0, 1: 1}


def test_timestamps_follow_each_qudits_interactions():
    circuit = QuantumCircuit(2, 2)
    circuit.X(0)
    circuit.CNOT(0, 1)
    circuit.measure_all()

    state, results = circuit.execute(mid_circuit=True)

    np.testing.assert_allclose(state, np.array([0.0, 0.0, 0.0, 1.0]))
    assert results == {0: 1, 1: 1}
    assert circuit.get_measurement() == [(0, 2, 1), (1, 1, 1)]
    assert circuit.get_measurement(1) == [(1, 1, 1)]
    assert circuit.get_measurement({1, 2}) == [(0, 2, 1), (1, 1, 1)]
    assert circuit.get_measurement(qudits=0) == [(0, 2, 1)]
    assert circuit.get_measurement(qudits={0, 1}) == [
        (0, 2, 1),
        (1, 1, 1),
    ]
    assert circuit.get_measurement(timestamp={1, 2}, qudits=1) == [(1, 1, 1)]


def test_first_gate_is_timestamp_zero_and_following_measurement_is_one():
    circuit = QuantumCircuit(1, 2)
    circuit.X(0)
    circuit.measure(0)

    _, results = circuit.execute(mid_circuit=True)

    assert circuit.get_measurement() == [(0, 1, 1)]
    assert results == {0: 1}
