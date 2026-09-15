import subprocess
import sys
import textwrap


def test_package_import_does_not_require_optional_numba():
    script = textwrap.dedent(
        """
        import importlib.abc
        import sys


        class BlockNumbaImports(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname == "numba" or fullname.startswith("numba."):
                    raise ModuleNotFoundError(
                        "numba is unavailable in a base quickqudits installation"
                    )
                return None


        sys.meta_path.insert(0, BlockNumbaImports())

        import quickqudits
        """
    )

    completed = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
