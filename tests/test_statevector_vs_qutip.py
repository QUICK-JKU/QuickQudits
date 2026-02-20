import numpy as np
import pytest

qutip = pytest.importorskip("qutip")
import qutip as qt

from quickqudits import QuantumCircuit, gates
from quickqudits.utils import compare_statevectors, random_circuit

DS = [2, 3, 4, 5, 6, 7, 8, 9]
NS = [1, 2, 3, 4, 5]
SEEDS = [0, 1, 2, 3]

DS_W = [2, 3, 4, 5, 6, 7, 8, 9]
NS_W = [1, 2, 3, 4, 5]
SEEDS_W = [0, 1, 2, 3]


def _zero_state_qutip(n: int, d: int) -> qt.Qobj:
    return qt.tensor([qt.basis(d, 0) for _ in range(n)])


def _qutip_single_qudit_gate(gate, d: int) -> qt.Qobj:
    mat = gate.matrix_fn(d)
    return qt.Qobj(
        mat,
        dims=[[d], [d]],
    )


def _qutip_two_qudit_gate(gate, d: int) -> qt.Qobj:
    mat = gate.matrix_fn(d)
    return qt.Qobj(
        mat,
        dims=[[d, d], [d, d]],
    )


def _embed_gate(U: qt.Qobj, n: int, targets: tuple[int, ...], d: int) -> qt.Qobj:
    return qt.expand_operator(
        U,
        dims=[d] * n,
        targets=list(targets),
    )


@pytest.mark.parametrize("d", DS)
@pytest.mark.parametrize("n", NS)
def test_statevector_vs_qutip_random_circuits(n: int, d: int):
    for seed in SEEDS:
        qc = random_circuit(
            num_qudits=n,
            dim=d,
            max_size=50,
            max_depth=50,
            seed=seed,
        )

        # QuickQudits
        psi_my, _ = qc.execute()

        # QuTiP
        psi_qt = _zero_state_qutip(n, d)

        for gate, qidx, dagger in qc.ops:
            if gate.matrix_fn is None:
                continue

            if gate.n_qudits == 1:
                U = _qutip_single_qudit_gate(gate, d)
            else:
                U = _qutip_two_qudit_gate(gate, d)

            if dagger:
                U = U.dag()

            U_full = _embed_gate(U, n, qidx, d)
            psi_qt = U_full @ psi_qt

        psi_qutip = psi_qt.full().ravel()

        assert compare_statevectors(
            psi_my,
            psi_qutip,
            atol=1e-8,
        )


@pytest.mark.parametrize("d", DS_W)
@pytest.mark.parametrize("n", NS_W)
def test_statevector_vs_qutip_weyl_gate_sequences(n: int, d: int):
    for seed in SEEDS_W:
        rng_seed = np.random.default_rng(seed)

        qc = QuantumCircuit(num_qudits=n, dim=d)

        ops = []
        depth = 40
        for _ in range(depth):
            q = int(rng_seed.integers(0, n))
            a = int(rng_seed.integers(0, d))
            b = int(rng_seed.integers(0, d))
            dagger = bool(rng_seed.integers(0, 2))

            if dagger:
                qc.Wdag(q, a=a, b=b)
            else:
                qc.W(q, a=a, b=b)

            ops.append((q, a, b, dagger))

        psi_my, _ = qc.execute()

        psi_qt = _zero_state_qutip(n, d)
        for q, a, b, dagger in ops:
            mat = gates.W(d, a, b)
            U = qt.Qobj(mat, dims=[[d], [d]])
            if dagger:
                U = U.dag()
            U_full = _embed_gate(U, n, (q,), d)
            psi_qt = U_full @ psi_qt

        psi_qutip = psi_qt.full().ravel()

        assert compare_statevectors(psi_my, psi_qutip, atol=1e-8)
