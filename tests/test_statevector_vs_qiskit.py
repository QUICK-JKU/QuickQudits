import pytest

qiskit = pytest.importorskip("qiskit")
from qiskit import QuantumCircuit as QiskitCircuit
from qiskit.quantum_info import Statevector

from quickqudits.utils import compare_statevectors, random_circuit

NS = range(1, 16)
SEEDS = [0, 1, 2, 3]


def _map_to_qiskit_index(q: int, n: int) -> int:
    return n - 1 - q


def _apply_gate_qiskit(
    qc: QiskitCircuit, name: str, qidx: tuple[int, ...], dagger: bool, n: int
) -> None:
    if len(qidx) == 1:
        q = _map_to_qiskit_index(qidx[0], n)

        if name == "I":
            return

        if name == "X":
            qc.x(q)
            return

        if name == "Y":
            qc.y(q)
            return

        if name == "Z":
            qc.z(q)
            return

        if name == "H":
            qc.h(q)
            return

        if name == "S":
            if dagger:
                qc.sdg(q)
            else:
                qc.s(q)
            return

    if len(qidx) == 2:
        a = _map_to_qiskit_index(qidx[0], n)
        b = _map_to_qiskit_index(qidx[1], n)

        if name == "CNOT":
            qc.cx(a, b)
            return

        if name == "CZ":
            qc.cz(a, b)
            return

        if name == "SWAP":
            qc.swap(a, b)
            return


@pytest.mark.parametrize("n", NS)
def test_statevector_vs_qiskit_random_qubit_circuits(n: int):
    d = 2

    for seed in SEEDS:
        qq = random_circuit(
            num_qudits=n,
            dim=d,
            max_size=50,
            max_depth=50,
            seed=seed,
        )

        psi_my, _ = qq.execute()

        qc = QiskitCircuit(n)

        for gate, qidx, dagger in qq.ops:
            if gate.matrix_fn is None:
                continue

            name = gate.name

            _apply_gate_qiskit(qc, name, qidx, dagger, n)

        psi_qiskit = Statevector.from_int(0, 2**n).evolve(qc).data

        assert compare_statevectors(psi_my, psi_qiskit, atol=1e-8)
