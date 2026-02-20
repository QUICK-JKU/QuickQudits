import numpy as np
import pytest

from quickqudits.utils import density_matrix_to_probvector, measureall, random_circuit

DS = [2, 3, 5]
NS = [2, 3]
SEEDS = [0]

GLOBAL_NOISE_PS = [0.0, 0.25, 1]

SHOTS = 5000
TOL_L2 = 0.03
MAX_SIZE = 25
MAX_DEPTH = 25


def _probvector_from_mc_outcomes(outcomes: list[int], dim: int) -> np.ndarray:
    counts = np.bincount(outcomes, minlength=dim)
    p = counts.astype(np.float64) / float(len(outcomes))
    return p


def _mc_probvector_from_noisy_statevector(
    qc, n: int, d: int, shots: int, seed: int
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    outcomes = []

    for _ in range(shots):
        shot_seed = int(rng.integers(0, 2**31 - 1))

        noisy_qc, _ = qc.realize_noise(seed=shot_seed, in_place=False)

        psi_final, _ = noisy_qc.execute()

        _, outcome_idx = measureall(psi_final, n, d, seed=shot_seed)
        outcomes.append(int(outcome_idx))

    return _probvector_from_mc_outcomes(outcomes, d**n)


@pytest.mark.parametrize("p_noise", GLOBAL_NOISE_PS)
@pytest.mark.parametrize("d", DS)
@pytest.mark.parametrize("n", NS)
def test_noisy_global_dm_matches_statevector_mc(n: int, d: int, p_noise: float):
    for seed in SEEDS:
        qc = random_circuit(
            num_qudits=n,
            dim=d,
            max_size=MAX_SIZE,
            max_depth=MAX_DEPTH,
            seed=seed,
        )

        qc.add_noise_global(noise_model=p_noise)

        rho = qc.execute_density_matrix()
        p_dm = density_matrix_to_probvector(rho)

        p_mc = _mc_probvector_from_noisy_statevector(
            qc,
            n=n,
            d=d,
            shots=SHOTS,
            seed=10_000_000 + 1000 * seed + int(100 * p_noise) + d + 17 * n,
        )

        diff = np.linalg.norm(p_dm - p_mc)
        assert diff <= TOL_L2


@pytest.mark.parametrize("d", DS)
@pytest.mark.parametrize("n", NS)
def test_custom_weyl_noisechannels_dm_matches_statevector_mc(n: int, d: int):
    for seed in SEEDS:
        rng = np.random.default_rng(12345 + 1000 * seed + 10 * d + n)

        qc = random_circuit(
            num_qudits=n,
            dim=d,
            max_size=MAX_SIZE,
            max_depth=MAX_DEPTH,
            seed=seed,
        )

        for q in range(n):
            P = rng.random((d, d))
            P = P / P.sum()

            p_total_error = float(rng.uniform(0.05, 0.25))
            P *= p_total_error
            P[0, 0] = 1.0 - p_total_error

            qc.add_noise_channel(q, noise_model=P)

        qc.H(0)
        if n >= 2:
            qc.CNOT(0, 1)
        qc.S(0)

        rho = qc.execute_density_matrix()
        p_dm = density_matrix_to_probvector(rho)

        p_mc = _mc_probvector_from_noisy_statevector(
            qc,
            n=n,
            d=d,
            shots=SHOTS,
            seed=20_000_000 + 1000 * seed + 13 * n + d,
        )

        l2 = float(np.linalg.norm(p_dm - p_mc))

        assert l2 <= TOL_L2
