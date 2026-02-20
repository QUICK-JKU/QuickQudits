import numpy as np
import pytest

from quickqudits import QuantumCircuit
from quickqudits.gate_class import NoiseChannel
from quickqudits.io import load_circuit, save_circuit


def test_circuit_io_roundtrip_basic(tmp_path):
    c = QuantumCircuit(2, 3, name="demo")
    c.H(0)
    c.CNOT(0, 1)
    c.W(1, 2, 1)
    c.add_noise_channel(0, 0.2, color="#00FF00")
    c.Sdag(1)

    path = tmp_path / "circuit"
    save_circuit(c, str(path))
    c2 = load_circuit(str(path))

    assert c2.n == c.n
    assert c2.d == c.d
    assert c2.name == c.name
    assert len(c2.ops) == len(c.ops)

    for (g1, q1, dag1), (g2, q2, dag2) in zip(c.ops, c2.ops):
        assert type(g2) is type(g1)
        assert q2 == q1
        assert dag2 == dag1
        assert g2.name == g1.name

        if g1.name.startswith("W_"):
            assert g2.a == g1.a
            assert g2.b == g1.b

        if isinstance(g1, NoiseChannel):
            assert g2.model_id == g1.model_id
            assert g2.color == g1.color

    assert set(c2.noise_registry.keys()) == set(c.noise_registry.keys())
    for mid in c.noise_registry:
        np.testing.assert_allclose(
            c2.noise_registry[mid], c.noise_registry[mid], rtol=0, atol=0
        )


def test_circuit_io_roundtrip_custom_noise_models_and_many_gates(tmp_path):
    d = 5
    qc = QuantumCircuit(3, d, name="noisy-demo")

    P1 = np.zeros((d, d), dtype=float)
    P1[0, 0] = 0.70
    P1[1, 0] = 0.10
    P1[0, 2] = 0.05
    P1[3, 4] = 0.15
    assert np.isclose(P1.sum(), 1.0)

    P2 = np.zeros((d, d), dtype=float)
    P2[0, 0] = 0.40
    P2[2, 1] = 0.25
    P2[4, 3] = 0.20
    P2[1, 1] = 0.15
    assert np.isclose(P2.sum(), 1.0)

    qc.H(0)
    qc.CNOT(0, 1)

    qc.add_noise_channel(0, P1, color="#00FF00")
    mid1 = qc.ops[-1][0].model_id

    qc.W(2, 3, 4)
    qc.Sdag(1)
    qc.CZ(1, 2)

    qc.add_noise_channel(2, P2, color="#FF00FF")
    mid2 = qc.ops[-1][0].model_id

    qc.SWAP(0, 2)
    qc.P(2)
    qc.M(0)

    assert mid1 != mid2
    assert mid1 in qc.noise_registry
    assert mid2 in qc.noise_registry

    np.testing.assert_allclose(qc.noise_registry[mid1], P1.reshape(-1), rtol=0, atol=0)
    np.testing.assert_allclose(qc.noise_registry[mid2], P2.reshape(-1), rtol=0, atol=0)

    path = tmp_path / "c"
    save_circuit(qc, str(path))
    qc2 = load_circuit(str(path))

    assert qc2.n == qc.n
    assert qc2.d == qc.d
    assert qc2.name == qc.name
    assert len(qc2.ops) == len(qc.ops)

    assert set(qc2.noise_registry.keys()) == set(qc.noise_registry.keys())
    for mid in qc.noise_registry:
        np.testing.assert_allclose(
            qc2.noise_registry[mid], qc.noise_registry[mid], rtol=0, atol=0
        )

    for (g1, q1, dag1), (g2, q2, dag2) in zip(qc.ops, qc2.ops):
        assert type(g2) is type(g1)
        assert q2 == q1
        assert dag2 == dag1
        assert g2.name == g1.name

        if g1.name.startswith("W_"):
            assert g2.a == g1.a
            assert g2.b == g1.b

        if isinstance(g1, NoiseChannel):
            assert g2.model_id == g1.model_id
            assert g2.color == g1.color


def test_circuit_io_roundtrip_scalar_and_array_noise_models(tmp_path):
    d = 3
    qc = QuantumCircuit(2, d)

    qc.H(0)
    qc.add_noise_channel(0, 0.2, color="#112233")
    mid_scalar = qc.ops[-1][0].model_id

    P = np.zeros((d, d), dtype=float)
    P[0, 0] = 0.60
    P[1, 2] = 0.25
    P[2, 1] = 0.15
    assert np.isclose(P.sum(), 1.0)

    qc.CNOT(0, 1)
    qc.add_noise_channel(1, P, color="#AABBCC")
    mid_array = qc.ops[-1][0].model_id

    assert mid_scalar != mid_array
    assert mid_scalar in qc.noise_registry
    assert mid_array in qc.noise_registry

    v_scalar = qc.noise_registry[mid_scalar]
    v_array = qc.noise_registry[mid_array]

    assert v_scalar.shape == (d * d,)
    assert v_array.shape == (d * d,)
    assert np.isfinite(v_scalar).all()
    assert np.isfinite(v_array).all()
    assert np.isclose(v_scalar.sum(), 1.0)
    assert np.isclose(v_array.sum(), 1.0)

    np.testing.assert_allclose(v_array, P.reshape(-1), rtol=0, atol=0)

    path = tmp_path / "c"
    save_circuit(qc, str(path))
    qc2 = load_circuit(str(path))

    assert len(qc2.ops) == len(qc.ops)

    noise_ops_1 = [(g, q) for (g, q, _) in qc.ops if isinstance(g, NoiseChannel)]
    noise_ops_2 = [(g, q) for (g, q, _) in qc2.ops if isinstance(g, NoiseChannel)]
    assert len(noise_ops_2) == len(noise_ops_1)

    for (g1, q1), (g2, q2) in zip(noise_ops_1, noise_ops_2):
        assert q2 == q1
        assert g2.model_id == g1.model_id
        assert g2.color == g1.color

    for mid in qc.noise_registry:
        np.testing.assert_allclose(
            qc2.noise_registry[mid], qc.noise_registry[mid], rtol=0, atol=0
        )


def test_circuit_io_schema_mismatch_raises(tmp_path):
    c = QuantumCircuit(1, 3)
    c.H(0)

    path = tmp_path / "c"
    save_circuit(c, str(path))

    p = str(path) + ".npz"
    with np.load(p, allow_pickle=False) as f:
        data = dict(f)

    data["schema"] = np.int64(999)
    np.savez_compressed(p, **data)

    with pytest.raises(ValueError, match="Unsupported circuit schema"):
        load_circuit(str(path))


def test_circuit_io_unknown_gate_raises(tmp_path):
    from quickqudits.gate_class import Gate

    c = QuantumCircuit(1, 3)
    c.append(Gate("NONEXISTENT", 1, None, None), 0)

    with pytest.raises(ValueError, match="Unknown gate name"):
        save_circuit(c, str(tmp_path / "c"))


def test_circuit_io_inconsistent_noise_color_raises(tmp_path):
    c = QuantumCircuit(1, 3)
    c.add_noise_channel(0, 0.2, color=(0.0, 1.0, 0.0, 0.2))
    mid = c.ops[-1][0].model_id

    c.ops.append((NoiseChannel(model_id=mid, color=(1.0, 0.0, 1.0, 0.2)), (0,), False))

    with pytest.raises(ValueError, match="Inconsistent colors"):
        save_circuit(c, str(tmp_path / "c"))
