import numpy as np
import pytest

from quickqudits import CompositeTableau, Tableau, io
from quickqudits.utils import compare_statevectors, random_circuit

SEEDS = [0, 1, 2, 3]

PRIMES = [2, 3, 7]
COMPOSITE = [4, 6]

MAX_HILBERT = 2_000

MAX_SIZE = 25
MAX_DEPTH = 25


def _dim_n_pairs_for_limit(ds, ns, max_hilbert: int):
    out = []
    for d in ds:
        for n in ns:
            if (d**n) <= max_hilbert:
                out.append((n, d))
    return out


PAIRS_TO_STATEVECTOR = _dim_n_pairs_for_limit(
    PRIMES + COMPOSITE, [1, 2, 3, 4, 5], MAX_HILBERT
)


@pytest.mark.parametrize("n,d", PAIRS_TO_STATEVECTOR)
@pytest.mark.parametrize("full", [False, True])
def test_tableau_to_statevector_matches_statevector_sim(n: int, d: int, full: bool):
    for seed in SEEDS:
        qc = random_circuit(
            num_qudits=n,
            dim=d,
            max_size=MAX_SIZE,
            max_depth=MAX_DEPTH,
            seed=seed,
        )

        psi_sv, _ = qc.execute()

        tab = Tableau(n=n, d=d, full=full)
        tab.apply_circuit(qc)

        psi_tab = tab.to_statevector()

        assert compare_statevectors(psi_sv, psi_tab, atol=1e-8)


PAIRS_PROBVECTOR = _dim_n_pairs_for_limit(PRIMES + COMPOSITE, [1, 2, 3, 4], MAX_HILBERT)


@pytest.mark.parametrize("n,d", PAIRS_PROBVECTOR)
def test_tableau_to_probvector_matches_statevector_probabilities(n: int, d: int):
    for seed in SEEDS:
        qc = random_circuit(
            num_qudits=n,
            dim=d,
            max_size=MAX_SIZE,
            max_depth=MAX_DEPTH,
            seed=seed,
        )

        psi_sv, _ = qc.execute()
        p_sv = np.abs(psi_sv) ** 2

        if d in PRIMES:
            tab = Tableau(n=n, d=d, full=True)
        else:
            tab = CompositeTableau(n=n, d=d, full=True)
        tab.apply_circuit(qc)

        p_tab = tab.to_probvector(shots=3_000_000)

        if not np.allclose(p_tab, p_sv, atol=1e-3, rtol=0):
            io.save_circuit("faulty_circuit", qc)
        assert p_tab.shape == p_sv.shape
        assert np.isclose(np.sum(p_tab), 1.0, atol=1e-10)
        assert np.all(p_tab >= -1e-12)

        assert np.allclose(p_tab, p_sv, atol=1e-3, rtol=0)


PAIRS_PROBVECTOR_EXACT = _dim_n_pairs_for_limit(PRIMES, [1, 2, 3, 4, 5], MAX_HILBERT)


@pytest.mark.parametrize("n,d", PAIRS_PROBVECTOR_EXACT)
def test_tableau_to_probvector_exact_matches_statevector_probabilities(n: int, d: int):
    for seed in SEEDS:
        qc = random_circuit(
            num_qudits=n,
            dim=d,
            max_size=MAX_SIZE,
            max_depth=MAX_DEPTH,
            seed=seed,
        )

        psi_sv, _ = qc.execute()
        p_sv = np.abs(psi_sv) ** 2

        tab = Tableau(n=n, d=d, full=True)
        tab.apply_circuit(qc)

        p_tab = tab.to_probvector_exact()

        assert p_tab.shape == p_sv.shape
        assert np.isclose(np.sum(p_tab), 1.0, atol=1e-10)
        assert np.all(p_tab >= -1e-12)

        assert np.allclose(p_tab, p_sv, atol=1e-8, rtol=0)
