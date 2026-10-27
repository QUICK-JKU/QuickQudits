"""Reproduce the H statevector audit with the current and original H paths.

Run from the repository root with one BLAS thread set before NumPy imports::

    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 .venv/bin/python benchmarks/h_statevector.py

The original path is selected by temporarily clearing H_gate._apply_fn.
Each output line is a JSON record. RSS records come from fresh processes.
"""

import argparse
import json
import os
import resource
import subprocess
import sys
import time

import numpy as np

from quickqudits import QuantumCircuit, backend, gates
from quickqudits.predefined_gates import H_gate


THREAD_ENV = (
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)
SIZES = ((20, 2), (13, 3))
KINDS = ("product", "ghz", "dense", "weyl")


def _set_precision(dtype):
    backend.set_precision("single" if dtype == np.complex64 else "double")


def _run_h(psi, n, d, q, mode):
    original = H_gate._apply_fn
    try:
        H_gate._apply_fn = None if mode == "old" else gates.apply_H_matmul
        return H_gate.apply(psi, n, d, (q,))
    finally:
        H_gate._apply_fn = original


def _run_circuit(circuit, mode):
    original = H_gate._apply_fn
    try:
        H_gate._apply_fn = None if mode == "old" else gates.apply_H_matmul
        return circuit.execute()
    finally:
        H_gate._apply_fn = original


def _summary(values):
    return [round(float(value), 3) for value in np.quantile(values, (0.25, 0.5, 0.75))]


def _paired(run, repeats):
    times = {"old": [], "new": []}
    for repetition in range(repeats):
        order = ("old", "new") if repetition % 2 == 0 else ("new", "old")
        for mode in order:
            start = time.perf_counter_ns()
            run(mode)
            times[mode].append((time.perf_counter_ns() - start) / 1e6)
    return {mode: _summary(values) for mode, values in times.items()}


def isolated(repeats):
    for n, d in (*SIZES, (10, 4)):
        real = np.random.default_rng(400 + n).normal(size=d**n)
        imag = np.random.default_rng(500 + d).normal(size=d**n)
        raw = real + 1j * imag
        for dtype in (np.complex64, np.complex128):
            _set_precision(dtype)
            psi = raw.astype(dtype)
            for q in sorted({0, n // 2, n - 1}):
                yield {
                    "kind": "isolated",
                    "n": n,
                    "d": d,
                    "dtype": np.dtype(dtype).name,
                    "target": q,
                    "repeats_per_mode": repeats,
                    "ms_q25_median_q75": _paired(
                        lambda mode: _run_h(psi, n, d, q, mode), repeats
                    ),
                }


def _build(kind, n, d):
    circuit = QuantumCircuit(n, d)
    if kind == "product":
        for q in range(n):
            circuit.H(q)
    elif kind == "ghz":
        circuit.H(0)
        for q in range(n - 1):
            circuit.CNOT(q, q + 1)
    elif kind == "dense":
        for q in range(n):
            circuit.H(q)
        for q in range(n - 1):
            if q < n - 1 - q:
                circuit.CZ(q, n - 1 - q)
        for q in range(n - 2):
            circuit.CNOT(q, q + 2)
        for q in range(n):
            circuit.S(q)
    elif kind == "weyl":
        rng = np.random.default_rng(20261013)
        for q in range(n):
            circuit.H(q)
        for layer in range(3):
            for q in range(n):
                circuit.W(q, int(rng.integers(d)), int(rng.integers(d)))
            for q in range(layer % 2, n - 1, 2):
                circuit.CZ(q, q + 1)
            for q in range(n):
                circuit.H(q)
    return circuit


def circuits(repeats):
    for n, d in SIZES:
        for dtype in (np.complex64, np.complex128):
            _set_precision(dtype)
            for kind in KINDS:
                start = time.perf_counter_ns()
                circuit = _build(kind, n, d)
                build_ms = (time.perf_counter_ns() - start) / 1e6
                yield {
                    "kind": kind,
                    "n": n,
                    "d": d,
                    "dtype": np.dtype(dtype).name,
                    "build_ms": round(build_ms, 3),
                    "repeats_per_mode": repeats,
                    "execute_ms_q25_median_q75": _paired(
                        lambda mode: _run_circuit(circuit, mode), repeats
                    ),
                }


def rss_child(mode, kind, n, d, dtype_name):
    dtype = np.dtype(dtype_name).type
    _set_precision(dtype)
    import_kib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    start = time.perf_counter_ns()
    circuit = _build(kind, n, d)
    build_ms = (time.perf_counter_ns() - start) / 1e6
    _run_circuit(circuit, mode)
    return {
        "mode": mode,
        "kind": kind,
        "n": n,
        "d": d,
        "dtype": dtype_name,
        "import_mib": round(import_kib / 1024, 2),
        "peak_mib": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 2),
        "build_ms": round(build_ms, 3),
    }


def rss():
    for n, d in SIZES:
        for dtype in (np.complex64, np.complex128):
            for kind in KINDS:
                modes = {}
                for mode in ("old", "new"):
                    command = [
                        sys.executable,
                        __file__,
                        "--rss-child",
                        mode,
                        kind,
                        str(n),
                        str(d),
                        np.dtype(dtype).name,
                    ]
                    result = subprocess.run(command, check=True, capture_output=True, text=True)
                    modes[mode] = json.loads(result.stdout)
                yield {
                    "kind": kind,
                    "n": n,
                    "d": d,
                    "dtype": np.dtype(dtype).name,
                    "import_mib": {mode: value["import_mib"] for mode, value in modes.items()},
                    "peak_mib": {mode: value["peak_mib"] for mode, value in modes.items()},
                }


def components(repeats):
    for n, d in (*SIZES, (10, 4)):
        for dtype in (np.complex64, np.complex128):
            _set_precision(dtype)
            psi = np.ones(d**n, dtype=dtype)
            for q in sorted({0, n // 2, n - 1}):
                def measure(fn):
                    values = []
                    for _ in range(repeats):
                        start = time.perf_counter_ns()
                        result = fn()
                        values.append((time.perf_counter_ns() - start) / 1e6)
                    return round(float(np.median(values)), 4), result

                timings = {}
                timings["matrix_lookup"], matrix = measure(lambda: gates.H(d))
                timings["dtype_conversion"], matrix = measure(
                    lambda: np.asarray(matrix, dtype=psi.dtype)
                )
                timings["tensordot"], contracted = measure(
                    lambda: np.tensordot(matrix, psi.reshape((d,) * n), axes=([1], [q]))
                )
                rest = [index for index in range(n) if index != q]
                permutation = [0 if index == q else 1 + rest.index(index) for index in range(n)]
                timings["transpose"], transposed = measure(
                    lambda: np.transpose(contracted, permutation)
                )
                timings["reshape"], flattened = measure(lambda: transposed.reshape(-1))
                outer, inner = d**q, d ** (n - q - 1)
                if q == 0:
                    view = psi.reshape(d, -1)
                    multiply = lambda: matrix @ view
                elif inner == 1:
                    view = psi.reshape(-1, d)
                    multiply = lambda: view @ matrix.T
                else:
                    view = psi.reshape(outer, d, inner)
                    multiply = lambda: np.matmul(matrix, view)
                timings["matmul"], multiplied = measure(multiply)
                timings["new_reshape"], new_flattened = measure(
                    lambda: multiplied.reshape(-1)
                )
                yield {
                    "kind": "components",
                    "n": n,
                    "d": d,
                    "dtype": np.dtype(dtype).name,
                    "target": q,
                    "median_ms": timings,
                    "input_shape": view.shape,
                    "input_strides": view.strides,
                    "old_output_strides": transposed.strides,
                    "old_reshape_copies": not np.shares_memory(flattened, transposed),
                    "new_reshape_copies": not np.shares_memory(new_flattened, multiplied),
                }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", choices=("isolated", "circuits", "rss", "components"))
    parser.add_argument("--repeats", type=int, default=8)
    parser.add_argument("--rss-child", nargs=5, metavar=("MODE", "KIND", "N", "D", "DTYPE"))
    args = parser.parse_args()
    if args.rss_child:
        mode, kind, n, d, dtype_name = args.rss_child
        print(json.dumps(rss_child(mode, kind, int(n), int(d), dtype_name)))
        return
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    print(json.dumps({"kind": "environment", "numpy": np.__version__,
                      "threads": {key: os.environ.get(key) for key in THREAD_ENV}}), flush=True)
    if args.only is None or args.only == "isolated":
        for record in isolated(args.repeats):
            print(json.dumps(record), flush=True)
    if args.only is None or args.only == "circuits":
        for record in circuits(args.repeats):
            print(json.dumps(record), flush=True)
    if args.only is None or args.only == "components":
        for record in components(args.repeats):
            print(json.dumps(record), flush=True)
    if args.only is None or args.only == "rss":
        for record in rss():
            print(json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
