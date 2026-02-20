import pytest

from quickqudits.utils import compare_statevectors, get_zero_vector, random_circuit

DS = [2, 3, 4, 5, 6, 7, 8, 9]
NS = [1, 2, 3, 4, 5]
SEEDS = [0, 1, 2, 3]


@pytest.mark.parametrize("d", DS)
@pytest.mark.parametrize("n", NS)
def test_forward_then_reverse_returns_to_initial_state(n: int, d: int):
    for seed in SEEDS:
        qc = random_circuit(
            num_qudits=n,
            dim=d,
            max_size=50,
            max_depth=50,
            seed=seed,
        )

        psi0 = get_zero_vector(n, d)

        qc_rev = qc.copy()
        qc_rev.reverse(in_place=True)

        qc.compose(qc_rev, in_place=True)

        psi_final, _ = qc.execute()

        assert compare_statevectors(psi_final, psi0, atol=1e-8)
