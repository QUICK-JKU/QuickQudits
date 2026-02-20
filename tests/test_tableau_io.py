import numpy as np

from quickqudits import Tableau
from quickqudits.io import load_tableau, save_tableau


def test_tableau_io_roundtrip(tmp_path):
    t = Tableau(2, 3, full=True)
    t.normalize()

    path = tmp_path / "tab"
    save_tableau(t, str(path))
    t2 = load_tableau(str(path))

    assert t2.n == t.n
    assert t2.d == t.d
    assert t2.full == t.full
    np.testing.assert_array_equal(t2.X, t.X)
    np.testing.assert_array_equal(t2.Z, t.Z)
    np.testing.assert_array_equal(t2.tau_exp, t.tau_exp)
