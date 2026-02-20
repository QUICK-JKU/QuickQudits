import numpy as np
import pytest

import quickqudits as qq
from quickqudits.utils import (
    density_matrix_to_probvector,
    fidelity_np,
    fidelity_pf,
    random_circuit,
    univ_input_state_circuit,
)

SEEDS = [0, 1, 2]
NOISE_PROBS = [0.0, 0.1, 0.5, 1]

SHOTS = 10_000

EPS_ABS = 2e-3
SIGMA_MULT = 5.0


def _within_mc_error(p_true: float, p_est: float, shots: int) -> bool:
    p = float(np.clip(p_true, 0.0, 1.0))
    se = float(np.sqrt(p * (1.0 - p) / shots))
    tol = max(SIGMA_MULT * se, EPS_ABS)
    return abs(p_true - p_est) <= tol


# Entanglement fidelity
ENT_PAIRS = [(1, 6), (2, 2), (2, 4), (3, 2)]  # (n, d)


# State fidelity
STATE_PAIRS = [(1, 2), (1, 4), (2, 3)]


@pytest.mark.parametrize("n,d", ENT_PAIRS)
@pytest.mark.parametrize("prob", NOISE_PROBS)
def test_entanglement_fidelity_matches_density_matrix(n: int, d: int, prob: float):
    circuit_size = max(10, n**2)
    circuit_depth = circuit_size

    for i, seed in enumerate(SEEDS):
        qc_ideal = random_circuit(
            num_qudits=n,
            dim=d,
            max_size=circuit_size,
            max_depth=circuit_depth,
            seed=seed,
        )

        qc_noisy = qc_ideal.copy()
        qc_noisy.add_noise_global(prob)

        qc_univ = univ_input_state_circuit(n, d)
        qc_univ_rev = qc_univ.reverse()
        qc_ideal_rev = qc_ideal.reverse()

        qc_sandwich = qc_univ.copy()
        qc_sandwich = qc_sandwich.compose(qc_noisy, in_place=False)
        qc_sandwich = qc_sandwich.compose(qc_ideal_rev, in_place=False)
        qc_sandwich = qc_sandwich.compose(qc_univ_rev, in_place=False)

        rho_final = qc_sandwich.execute_density_matrix()
        fid_dm = float(density_matrix_to_probvector(rho_final)[0])

        fid_tab = float(
            fidelity_np(
                qc_noisy,
                shots=SHOTS,
                entanglement_fidelity=True,
                seed=10_000 + 100 * i + seed,
            )
        )

        fid_pf = float(
            fidelity_pf(
                qc_noisy,
                shots=SHOTS,
                entanglement_fidelity=True,
                seed=30_000 + 100 * i + seed,
            )
        )

        assert _within_mc_error(fid_dm, fid_tab, SHOTS), (n, d, prob, fid_dm, fid_tab)
        assert _within_mc_error(fid_dm, fid_pf, SHOTS), (n, d, prob, fid_dm, fid_pf)


@pytest.mark.parametrize("n,d", STATE_PAIRS)
@pytest.mark.parametrize("prob", NOISE_PROBS)
def test_state_fidelity_matches_density_matrix(n: int, d: int, prob: float):
    circuit_size = max(10, n**2)
    circuit_depth = circuit_size

    for i, seed in enumerate(SEEDS):
        qc_ideal = random_circuit(
            num_qudits=n,
            dim=d,
            max_size=circuit_size,
            max_depth=circuit_depth,
            seed=seed,
        )

        qc_noisy = qc_ideal.copy()
        qc_noisy.add_noise_global(prob)

        qc_ideal_rev = qc_ideal.reverse()

        qc_sandwich = qc_noisy.compose(qc_ideal_rev, in_place=False)

        rho_final = qc_sandwich.execute_density_matrix()
        fid_dm = float(density_matrix_to_probvector(rho_final)[0])

        fid_tab = float(
            fidelity_np(
                qc_noisy,
                shots=SHOTS,
                entanglement_fidelity=False,
                seed=20_000 + 100 * i + seed,
            )
        )

        fid_pf = float(
            fidelity_pf(
                qc_noisy,
                shots=SHOTS,
                entanglement_fidelity=False,
                seed=40_000 + 100 * i + seed,
            )
        )

        assert _within_mc_error(fid_dm, fid_tab, SHOTS), (n, d, prob, fid_dm, fid_tab)
        assert _within_mc_error(fid_dm, fid_pf, SHOTS), (n, d, prob, fid_dm, fid_pf)
