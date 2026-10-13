import subprocess
import sys
import textwrap


def test_base_import_and_cpu_simulation_do_not_require_optional_packages():
    script = textwrap.dedent(
        """
        import importlib.abc
        import sys
        import warnings

        import numpy as np


        class BlockOptionalImports(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname.split(".", 1)[0] in {"numba", "matplotlib"}:
                    raise ModuleNotFoundError(
                        "optional packages are unavailable in a base installation"
                    )
                return None


        sys.meta_path.insert(0, BlockOptionalImports())

        import quickqudits as qq

        circuit = qq.QuantumCircuit(1, 2)
        circuit.X(0)
        np.testing.assert_allclose(circuit.execute()[0], [0, 1])
        np.testing.assert_allclose(circuit.execute_density_matrix(), [[0, 0], [0, 1]])

        tableau = qq.Tableau(1, 2)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            tableau.apply_circuit(circuit)
        np.testing.assert_array_equal(tableau.measure_all(seed=1)[1], [1])

        noisy = qq.QuantumCircuit(1, 2)
        error = np.array([[0.0, 0.0], [1.0, 0.0]])
        noisy.add_noise_channel(0, error, color="#0F0")
        np.testing.assert_allclose(noisy.ops[0][0].color[:3], [0, 1, 0])
        np.testing.assert_allclose(noisy.execute_density_matrix(), [[0, 0], [0, 1]])
        np.testing.assert_array_equal(
            qq.Tableau(1, 2).sample_measurements_noisy_pf(noisy, shots=3, seed=1),
            [[1], [1], [1]],
        )

        config = qq.GateNoiseConfig()
        config.set("X", after=0.0)
        assert len(config.apply(circuit).noise_registry) == 1

        try:
            qq.QuantumCircuit(1, 2).add_noise_channel(0, 0.1, color="red")
        except ValueError as exc:
            assert "quickqudits[plot]" in str(exc)
        else:
            raise AssertionError("Named colors require the optional plot extra")

        try:
            qq.Tableau(1, 2).apply_circuit(circuit, engine="numba")
        except RuntimeError as exc:
            assert "Numba engine requested but kernels are unavailable" in str(exc)
        else:
            raise AssertionError("An unavailable requested accelerator must fail")
        """
    )

    completed = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr


def test_public_import_and_cpu_execution_do_not_load_cuda_modules():
    script = textwrap.dedent(
        """
        import sys
        import quickqudits as qq

        circuit = qq.QuantumCircuit(1, 2)
        circuit.H(0)
        circuit.execute()
        assert not any(name.startswith("numba.cuda") for name in sys.modules)
        """
    )
    completed = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
