import numpy as np
import pytest

import quickqudits as qq
from quickqudits.utils import density_matrix_to_probvector, random_circuit

PRIMES = [2, 3, 5]
NON_PRIMES = [4, 6]
DS = PRIMES + NON_PRIMES
NS = [1, 2, 3]
SEEDS = [0, 1, 2]

MAX_SIZE = 20
MAX_DEPTH = 20


SHOTS_DM_REF = 100_000
TOL_MAXABS = 0.02
TOL_L2 = 0.06


def _tab_class_for_d(d: int):
    return qq.Tableau if d in PRIMES else qq.CompositeTableau


def _l2(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def _random_weyl_channel(
    rng: np.random.Generator, d: int, p_total: float
) -> np.ndarray:
    P = rng.random((d, d))
    P /= P.sum()
    P *= p_total
    P[0, 0] = 1.0 - p_total
    P /= P.sum()
    return P


@pytest.mark.parametrize("d", DS)
@pytest.mark.parametrize("n", NS)
@pytest.mark.parametrize("p_noise", [0.0, 0.05, 0.15, 0.35])
def test_pf_global_scalar_matches_density_matrix(n: int, d: int, p_noise: float):
    Tab = _tab_class_for_d(d)

    for seed in SEEDS:
        qc = random_circuit(
            num_qudits=n,
            dim=d,
            max_size=MAX_SIZE,
            max_depth=MAX_DEPTH,
            seed=seed,
        )

        qc.add_noise_global(noise_model=float(p_noise))

        rho = qc.execute_density_matrix()
        p_dm = density_matrix_to_probvector(rho)

        tab = Tab(n=n, d=d, full=True)

        p_pf = tab.to_probvector_noisy_pf(
            qc=qc,
            shots=SHOTS_DM_REF,
            seed=10_000_000 + 1000 * seed + 17 * n + d,
        )

        assert p_pf.shape == p_dm.shape
        assert np.isclose(p_pf.sum(), 1.0, atol=1e-10)
        assert np.all(p_pf >= -1e-12)

        assert np.max(np.abs(p_pf - p_dm)) <= TOL_MAXABS
        assert _l2(p_pf, p_dm) <= TOL_L2


@pytest.mark.parametrize("d", DS)
@pytest.mark.parametrize("n", NS)
def test_pf_global_full_matrix_matches_density_matrix(n: int, d: int):
    Tab = _tab_class_for_d(d)

    for seed in SEEDS:
        rng = np.random.default_rng(12345 + 1000 * seed + 19 * n + d)

        qc = random_circuit(
            num_qudits=n,
            dim=d,
            max_size=MAX_SIZE,
            max_depth=MAX_DEPTH,
            seed=seed,
        )

        p_total = float(rng.uniform(0.05, 0.35))
        P = _random_weyl_channel(rng, d=d, p_total=p_total)
        qc.add_noise_global(noise_model=P)

        rho = qc.execute_density_matrix()
        p_dm = density_matrix_to_probvector(rho)

        tab = Tab(n=n, d=d, full=True)

        p_pf = tab.to_probvector_noisy_pf(
            qc=qc,
            shots=SHOTS_DM_REF,
            seed=20_000_000 + 1000 * seed + 23 * n + d,
        )

        assert p_pf.shape == p_dm.shape
        assert np.max(np.abs(p_pf - p_dm)) <= TOL_MAXABS
        assert _l2(p_pf, p_dm) <= TOL_L2


@pytest.mark.parametrize("d", DS)
@pytest.mark.parametrize("n", NS)
def test_pf_multiple_custom_channels_matches_density_matrix(n: int, d: int):
    Tab = _tab_class_for_d(d)

    for seed in SEEDS:
        rng = np.random.default_rng(54321 + 1000 * seed + 29 * n + d)

        qc = random_circuit(
            num_qudits=n,
            dim=d,
            max_size=MAX_SIZE,
            max_depth=MAX_DEPTH,
            seed=seed,
        )

        for q in range(n):
            for _ in range(2):
                p_total = float(rng.uniform(0.05, 0.30))
                Pq = _random_weyl_channel(rng, d=d, p_total=p_total)
                qc.add_noise_channel(q, noise_model=Pq)

        rho = qc.execute_density_matrix()
        p_dm = density_matrix_to_probvector(rho)

        tab = Tab(n=n, d=d, full=True)

        p_pf = tab.to_probvector_noisy_pf(
            qc=qc,
            shots=SHOTS_DM_REF,
            seed=30_000_000 + 1000 * seed + 31 * n + d,
        )

        assert p_pf.shape == p_dm.shape
        assert np.max(np.abs(p_pf - p_dm)) <= TOL_MAXABS
        assert _l2(p_pf, p_dm) <= TOL_L2
