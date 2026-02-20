import numpy as np
import pytest

from quickqudits import (
    CNOT_gate,
    CZ_gate,
    H_gate,
    I_gate,
    S_gate,
    SWAP_gate,
    Tableau,
    W_gate,
    X_gate,
    Y_gate,
    Z_gate,
)
from quickqudits.utils import check_symplectic_structure, random_circuit

DS = [2, 3, 4, 5, 6, 7, 8, 9, 10]
NS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]


# ---------------------------------------------------------------------------
# Single-qudit gates
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("d", DS)
@pytest.mark.parametrize("n", NS)
def test_identity_gate_is_noop(n, d):
    tab0 = Tableau(n, d, full=True)

    tab = Tableau(n, d, full=True)
    for q in range(n):
        tab.apply_gate(I_gate, q)

    assert check_symplectic_structure(tab)

    np.testing.assert_array_equal(tab.X, tab0.X)
    np.testing.assert_array_equal(tab.Z, tab0.Z)
    np.testing.assert_array_equal(tab.tau_exp, tab0.tau_exp)


@pytest.mark.parametrize("gate", [X_gate, Y_gate, Z_gate, H_gate, S_gate])
@pytest.mark.parametrize("d", DS)
@pytest.mark.parametrize("n", NS)
def test_single_qudit_gates_preserve_symplectic_structure(gate, n, d):
    tab = Tableau(n, d, full=True)

    for q in range(n):
        t = tab.copy()
        t.apply_gate(gate, q)
        assert check_symplectic_structure(t)

        t.apply_gate(gate, q, dagger=True)
        assert check_symplectic_structure(t)


# ---------------------------------------------------------------------------
# Two-qudit gates
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("gate", [CNOT_gate, CZ_gate, SWAP_gate])
@pytest.mark.parametrize("d", DS)
@pytest.mark.parametrize("n", [2, 3, 4, 5, 6, 7, 8, 9, 10])
def test_two_qudit_gates_preserve_symplectic_structure(gate, n, d):
    base = Tableau(n, d, full=True)

    for a in range(n):
        for b in range(n):
            if a == b:
                continue

            tab = base.copy()
            tab.apply_gate(gate, (a, b))
            assert check_symplectic_structure(tab)

            tab.apply_gate(gate, (a, b), dagger=True)
            assert check_symplectic_structure(tab)


# ---------------------------------------------------------------------------
# Weyl gate W(a,b)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("a,b", [(1, 0), (0, 1), (1, 1), (2, 3)])
@pytest.mark.parametrize("d", [3, 5, 6, 10])
@pytest.mark.parametrize("n", NS)
def test_weyl_gate_preserves_symplectic_structure(a, b, d, n):
    a %= d
    b %= d
    if a == 0 and b == 0:
        pytest.skip("W(0,0) is identity")

    tab = Tableau(n, d, full=True)
    gate = W_gate.copy(a=a, b=b)

    for q in range(n):
        t = tab.copy()
        t.apply_gate(gate, q)
        assert check_symplectic_structure(t)

        t.apply_gate(gate, q, dagger=True)
        assert check_symplectic_structure(t)


# ---------------------------------------------------------------------------
# Random circuits
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("d", DS)
@pytest.mark.parametrize("n", NS)
def test_random_circuit_application_preserves_symplectic_structure(n, d):
    seeds = [0, 1, 2, 3, 4]

    for seed in seeds:
        circuit = random_circuit(
            num_qudits=n,
            dim=d,
            max_size=100,
            max_depth=100,
            seed=seed,
        )

        tab = Tableau(n, d, full=True)
        assert check_symplectic_structure(tab)

        tab.apply_circuit(circuit)

        assert check_symplectic_structure(tab)
