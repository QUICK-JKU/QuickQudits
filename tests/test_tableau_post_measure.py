import numpy as np
import pytest

from quickqudits import Tableau
from quickqudits.utils import (
    check_symplectic_structure,
    compare_statevectors,
    get_projected_state,
    get_projected_state_multi,
    random_circuit,
)

SEEDS = [0, 1, 2, 3]
PRIMES = [2, 3, 5, 7]

MAX_HILBERT = 2_000
MAX_SIZE = 25
MAX_DEPTH = 25


def _dim_n_pairs_for_limit(ds, ns, max_hilbert: int):
    out = []
    for d in ds:
        for n in ns:
            if d**n <= max_hilbert:
                out.append((n, d))
    return out


PAIRS_REDUCED = _dim_n_pairs_for_limit(PRIMES, [2, 3, 4], MAX_HILBERT)


@pytest.mark.parametrize("n,d", PAIRS_REDUCED)
def test_post_measure_single_qudit_matches_statevector_projection(n: int, d: int):
    for seed in SEEDS:
        rng = np.random.default_rng(100_000 + 997 * seed + 31 * n + d)

        qc = random_circuit(
            num_qudits=n,
            dim=d,
            max_size=MAX_SIZE,
            max_depth=MAX_DEPTH,
            seed=seed,
        )

        tab = Tableau(n=n, d=d, full=True)
        tab.apply_circuit(qc)

        psi_original = tab.to_statevector()

        q_target = int(rng.integers(0, n))

        reduced_tab, outcome_dict = tab.post_measure(
            qudits_to_measure=q_target,
            in_place=False,
            seed=int(rng.integers(0, 2**63 - 1)),
        )

        outcome = int(outcome_dict[q_target])

        psi_from_tableau = reduced_tab.to_statevector()
        psi_expected = get_projected_state(psi_original, n, d, q_target, outcome)

        assert psi_expected is not None
        assert compare_statevectors(psi_from_tableau, psi_expected, atol=1e-8)
        assert check_symplectic_structure(reduced_tab)


@pytest.mark.parametrize("n,d", PAIRS_REDUCED)
def test_post_measure_multi_qudit_matches_statevector_projection(n: int, d: int):
    for seed in SEEDS:
        rng = np.random.default_rng(200_000 + 991 * seed + 37 * n + d)

        qc = random_circuit(
            num_qudits=n,
            dim=d,
            max_size=MAX_SIZE,
            max_depth=MAX_DEPTH,
            seed=seed,
        )

        tab = Tableau(n=n, d=d, full=True)
        tab.apply_circuit(qc)

        psi_original = tab.to_statevector()

        num_to_measure = int(rng.integers(1, n))
        qudits_to_measure = (
            rng.choice(np.arange(n), size=num_to_measure, replace=False)
            .astype(int)
            .tolist()
        )

        reduced_tab, outcome_dict = tab.post_measure(
            qudits_to_measure=qudits_to_measure,
            in_place=False,
            seed=int(rng.integers(0, 2**63 - 1)),
        )

        outcomes = [int(outcome_dict[q]) for q in qudits_to_measure]

        psi_from_tableau = reduced_tab.to_statevector()
        psi_expected = get_projected_state_multi(
            psi_original, n, d, qudits_to_measure, outcomes
        )

        assert psi_expected is not None
        assert compare_statevectors(psi_from_tableau, psi_expected, atol=1e-6)
        assert check_symplectic_structure(reduced_tab)
