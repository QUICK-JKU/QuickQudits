import numpy as np
import pytest

from quickqudits import QuantumCircuit, backend, gates
from quickqudits.gate_class import Gate


@pytest.fixture(autouse=True)
def restore_default_precision():
    yield
    backend.set_precision("double")


@pytest.mark.parametrize("precision,dtype", [("single", np.complex64), ("double", np.complex128)])
@pytest.mark.parametrize("d", [2, 3, 4])
def test_weyl_work_arrays_and_matrix_cache_follow_precision(monkeypatch, precision, dtype, d):
    backend.set_precision(precision)
    rng = np.random.default_rng(210 + d)
    psi = (rng.normal(size=d**3) + 1j * rng.normal(size=d**3)).astype(dtype)
    original = psi.copy()
    multiply = np.multiply
    work_dtypes = []

    def record_multiply(left, right, *args, **kwargs):
        work_dtypes.append((left.dtype, np.asarray(right).dtype, kwargs["out"].dtype))
        return multiply(left, right, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(gates.np, "multiply", record_multiply)
        result = gates.apply_W_reshape_idx(psi, 3, d, (1,), a=1, b=1)
    assert result.dtype == dtype
    assert work_dtypes == [(np.dtype(dtype),) * 3] * d
    assert not np.shares_memory(result, psi)
    np.testing.assert_array_equal(psi, original)

    for matrix in (gates.H(d), gates.S(d), gates.Z(d), gates.CZ(d), gates.W(d, 1, 1)):
        assert matrix.dtype == dtype


def _independent_weyl_matrix(n, d, target, a, b, dagger):
    """Map basis columns directly, including the τ^{-ab} global factor."""
    a = (-a if dagger else a) % d
    b = (-b if dagger else b) % d
    omega = np.exp(2j * np.pi / d)
    tau = np.exp(1j * np.pi * (d**2 + 1) / d)
    stride = d ** (n - 1 - target)
    size = d**n
    matrix = np.zeros((size, size), dtype=np.complex128)
    for column in range(size):
        digit = (column // stride) % d
        row = column + (((digit + a) % d) - digit) * stride
        matrix[row, column] = tau ** (-a * b) * omega ** (b * digit)
    return matrix


@pytest.mark.parametrize(
    "precision,dtype", [("single", np.complex64), ("double", np.complex128)]
)
@pytest.mark.parametrize("d", [2, 3, 4])
@pytest.mark.parametrize("target", [0, 1, 2])
def test_weyl_matches_independent_global_phase_reference(precision, dtype, d, target):
    backend.set_precision(precision)
    rng = np.random.default_rng(680 + d + target)
    psi = (rng.normal(size=d**3) + 1j * rng.normal(size=d**3)).astype(dtype)
    psi /= np.linalg.norm(psi)
    original = psi.copy()
    for a in range(-1, d):
        for b in range(-1, d):
            for dagger in (False, True):
                actual = gates.apply_W_reshape_idx(psi, 3, d, (target,), dagger, a, b)
                expected = _independent_weyl_matrix(3, d, target, a, b, dagger) @ original
                assert actual.dtype == dtype
                assert not np.shares_memory(actual, psi)
                tolerance = 3e-6 if precision == "single" else 1e-13
                np.testing.assert_allclose(actual, expected, rtol=tolerance, atol=tolerance)
                actual_norm = np.linalg.norm(actual.astype(np.complex128))
                initial_norm = np.linalg.norm(original.astype(np.complex128))
                assert abs(actual_norm - initial_norm) < 2e-6
    np.testing.assert_array_equal(psi, original)


def test_qubit_combined_weyl_keeps_its_global_phase():
    backend.set_precision("single")
    zero = np.array([1, 0], dtype=np.complex64)
    one = np.array([0, 1], dtype=np.complex64)
    np.testing.assert_allclose(
        gates.apply_W_reshape_idx(zero, 1, 2, (0,), a=1, b=1),
        [0, -1j],
        atol=1e-7,
    )
    np.testing.assert_allclose(
        gates.apply_W_reshape_idx(one, 1, 2, (0,), a=1, b=1),
        [1j, 0],
        atol=1e-7,
    )
    np.testing.assert_allclose(
        gates.apply_W_reshape_idx(zero, 1, 2, (0,), True, 1, 1),
        [0, -1j],
        atol=1e-7,
    )


@pytest.mark.parametrize("precision,dtype", [("single", np.complex64), ("double", np.complex128)])
@pytest.mark.parametrize("d", [2, 3, 4])
def test_one_qudit_weyl_matches_local_matrix(precision, dtype, d):
    backend.set_precision(precision)
    rng = np.random.default_rng(301 + d)
    psi = (rng.normal(size=d) + 1j * rng.normal(size=d)).astype(dtype)
    for a in range(d):
        for b in range(d):
            for dagger in (False, True):
                actual = gates.apply_W_reshape_idx(psi, 1, d, (0,), dagger, a, b)
                local = gates.W_dag(d, a, b) if dagger else gates.W(d, a, b)
                expected = local @ psi
                assert actual.dtype == dtype
                np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=2e-6)


@pytest.mark.parametrize("d", [2, 3, 4])
def test_precision_switch_rebuilds_local_matrices(d):
    backend.set_precision("single")
    single_h = gates.H(d)
    single_cz = gates.CZ(d)
    backend.set_precision("double")
    double_h = gates.H(d)
    double_cz = gates.CZ(d)
    assert single_h.dtype == single_cz.dtype == np.complex64
    assert double_h.dtype == double_cz.dtype == np.complex128
    assert single_h is not double_h
    assert single_cz is not double_cz
    backend.set_precision("single")
    assert gates.H(d) is not single_h
    assert gates.CZ(d) is not single_cz


@pytest.mark.parametrize("precision,dtype", [("single", np.complex64), ("double", np.complex128)])
@pytest.mark.parametrize("d", [2, 3, 4])
def test_predefined_local_gate_application_preserves_state_dtype(precision, dtype, d):
    backend.set_precision(precision)
    rng = np.random.default_rng(900 + d)
    psi = (rng.normal(size=d**3) + 1j * rng.normal(size=d**3)).astype(dtype)
    for name, targets in (
        ("I", (0,)), ("X", (1,)), ("Y", (2,)), ("Z", (0,)),
        ("H", (1,)), ("S", (2,)), ("CNOT", (2, 0)),
        ("CZ", (0, 2)), ("SWAP", (2, 0)), ("W", (1,)),
    ):
        circuit = QuantumCircuit(3, d)
        if name == "W":
            circuit.W(targets[0], 1, 1)
        else:
            getattr(circuit, name)(*targets)
        gate, qudits, dagger = circuit.ops[0]
        psi = gate.apply(psi, 3, d, qudits, dagger)
        assert psi.dtype == dtype, name


@pytest.mark.parametrize("precision,dtype", [("single", np.complex64), ("double", np.complex128)])
def test_final_normalization_writes_directly_to_selected_complex_dtype(
    monkeypatch, precision, dtype
):
    backend.set_precision(precision)
    circuit = QuantumCircuit(2, 3)
    circuit.H(0)
    divide = np.divide
    einsum = np.einsum
    observed = []
    norm_inputs = []

    def record_divide(left, right, *args, **kwargs):
        observed.append((left.dtype, kwargs["out"].dtype))
        return divide(left, right, *args, **kwargs)

    def record_einsum(subscripts, *operands, **kwargs):
        norm_inputs.append((operands[0].dtype, kwargs["dtype"]))
        return einsum(subscripts, *operands, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(np, "divide", record_divide)
        patch.setattr(np, "einsum", record_einsum)
        state, _ = circuit.execute()
    assert observed == [(np.dtype(dtype), np.dtype(dtype))]
    real_dtype = np.float32 if precision == "single" else np.float64
    assert norm_inputs == [(np.dtype(real_dtype), np.float64)] * 2
    assert state.dtype == dtype


@pytest.mark.parametrize("precision,dtype", [("single", np.complex64), ("double", np.complex128)])
@pytest.mark.parametrize("n", [1, 3])
def test_local_matrix_is_cast_before_statevector_contraction(monkeypatch, precision, dtype, n):
    backend.set_precision(precision)
    gate = Gate("custom_phase", 1, lambda d: np.diag([1, 1j]).astype(np.complex128))
    psi = np.ones(2**n, dtype=dtype)
    recorded = []
    tensordot = np.tensordot

    def record_tensordot(left, right, *args, **kwargs):
        recorded.append((left.dtype, right.dtype))
        return tensordot(left, right, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(np, "tensordot", record_tensordot)
        actual = gate.apply(psi, n, 2, (0,))
    assert actual.dtype == dtype
    if n > 1:
        assert recorded == [(np.dtype(dtype), np.dtype(dtype))]


@pytest.mark.parametrize("precision,dtype", [("single", np.complex64), ("double", np.complex128)])
@pytest.mark.parametrize("d", [2, 3, 4])
def test_initialization_measurement_and_resume_keep_state_dtype(
    monkeypatch, precision, dtype, d
):
    backend.set_precision(precision)
    circuit = QuantumCircuit(2, d)
    initial = np.zeros(d**2, dtype=np.complex128)
    initial[0] = 1
    circuit.H(0)
    circuit.M(0)

    observed_probs = []

    def choose_maximum(values, p):
        observed_probs.append(p.copy())
        assert p.dtype == np.float64
        return int(np.argmax(p))

    monkeypatch.setattr(np.random, "choice", choose_maximum)
    first, _ = circuit.execute(psi0=initial, mid_circuit=True)
    assert first.dtype == circuit._stored_psi.dtype == dtype
    np.testing.assert_array_equal(initial, np.eye(1, d**2, dtype=np.complex128)[0])

    circuit.W(1, 1, 1)
    circuit.H(1)
    circuit.M(1)
    second, _ = circuit.execute(mid_circuit=True)
    assert second.dtype == circuit._stored_psi.dtype == dtype

    circuit.CNOT(0, 1)
    final, _ = circuit.execute(mid_circuit=True)
    assert final.dtype == dtype
    assert circuit._stored_psi is None
    assert len(observed_probs) == 2
    for probabilities in observed_probs:
        assert np.isclose(probabilities.sum(dtype=np.float64), 1.0)


@pytest.mark.parametrize("precision", ["single", "double"])
def test_sample_uses_float64_probabilities(monkeypatch, precision):
    backend.set_precision(precision)
    circuit = QuantumCircuit(2, 3)
    circuit.H(0)
    circuit.W(1, 1, 1)

    class RecordingRng:
        def choice(self, population, *, p, **kwargs):
            assert p.dtype == np.float64
            assert np.isclose(p.sum(), 1.0)
            assert population == 3**2
            return np.zeros(kwargs["size"], dtype=np.int64)

    monkeypatch.setattr(np.random, "default_rng", lambda seed: RecordingRng())
    state, digits, flat = circuit.sample(5, seed=31)
    assert state.dtype == backend.COMPLEX_DTYPE
    assert digits.shape == (5, 2)
    assert flat.shape == (5,)


@pytest.mark.parametrize("d", [2, 3, 4])
@pytest.mark.parametrize("depth", [10, 100, 300])
def test_single_precision_interference_accuracy_increases_with_depth(d, depth):
    rng = np.random.default_rng(90210 + d)
    circuit = QuantumCircuit(4, d)
    for _ in range(depth):
        q = int(rng.integers(0, 4))
        q2 = (q + 1 + int(rng.integers(0, 3))) % 4
        circuit.H(q)
        circuit.S(q2)
        circuit.W(q, int(rng.integers(d)), int(rng.integers(d)))
        circuit.CNOT(q, q2)
        control = (q + 2) % 4
        circuit.CZ(control, q2 if q2 != control else q)

    backend.set_precision("double")
    double_state, _ = circuit.execute(normalize=False)
    backend.set_precision("single")
    single_state, _ = circuit.execute(normalize=False)
    assert single_state.dtype == np.complex64

    # Cast only in this small test oracle; production never needs a complex128
    # copy of the single-precision state for norm or measurement probabilities.
    high_precision_single = single_state.astype(np.complex128)
    double_norm = np.linalg.norm(double_state)
    single_norm = np.linalg.norm(high_precision_single)
    double_unit = double_state / double_norm
    single_unit = high_precision_single / single_norm
    fidelity = abs(np.vdot(double_unit, single_unit)) ** 2
    probability_error = np.max(np.abs(np.abs(double_unit) ** 2 - np.abs(single_unit) ** 2))

    # 300 rounds contain 1500 gates. These bounds allow accumulated float32
    # rounding while still detecting a wrong phase, ordering, or dtype path.
    assert abs(double_norm - 1) < 1e-10
    assert abs(single_norm - 1) < 2e-5
    assert fidelity > 1 - 1e-6
    assert probability_error < 2e-6
