import numpy as np
import pytest

from quickqudits.utils import density_matrix_to_probvector, random_circuit

DS = [2, 3, 4]
NS = [1, 2, 3, 4]
SEEDS = [0, 1, 2, 3]


@pytest.mark.parametrize("d", DS)
@pytest.mark.parametrize("n", NS)
def test_density_matrix_probabilities_match_statevector(n: int, d: int):
    for seed in SEEDS:
        qc = random_circuit(
            num_qudits=n,
            dim=d,
            max_size=50,
            max_depth=50,
            seed=seed,
        )

        psi, _ = qc.execute()
        p_sv = np.abs(psi) ** 2

        rho = qc.execute_density_matrix()
        p_dm = density_matrix_to_probvector(rho)

        assert p_sv.shape == p_dm.shape
        assert np.isclose(np.sum(p_dm), 1.0, atol=1e-10)
        assert np.all(p_dm >= -1e-12)

        assert np.allclose(p_sv, p_dm, atol=1e-8)


@pytest.mark.parametrize("d", DS)
@pytest.mark.parametrize("n", NS)
def test_density_matrix_trace_is_one(n: int, d: int):
    for seed in SEEDS:
        qc = random_circuit(
            num_qudits=n,
            dim=d,
            max_size=30,
            max_depth=30,
            seed=seed,
        )

        rho = qc.execute_density_matrix()
        rho = np.asarray(rho, dtype=np.complex128)
        tr = np.trace(rho)
        assert np.isclose(tr.real, 1.0, atol=1e-10)
        assert np.isclose(tr.imag, 0.0, atol=1e-10)


@pytest.mark.parametrize("d", DS)
@pytest.mark.parametrize("n", NS)
def test_density_matrix_is_hermitian(n: int, d: int):
    for seed in SEEDS:
        qc = random_circuit(
            num_qudits=n,
            dim=d,
            max_size=30,
            max_depth=30,
            seed=seed,
        )

        rho = np.asarray(qc.execute_density_matrix(), dtype=np.complex128)
        assert np.allclose(rho, rho.conj().T, atol=1e-10)
