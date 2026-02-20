"""
Acceleration backends for `quickqudits`.

This subpackage provides optimized kernels for tableau updates.
The accelerators are optional: if the required dependencies are
not available, the core library still works using pure NumPy implementations.

## Submodules

### `accelerators.numba_fast`
Numba-accelerated kernels used by the tableau simulator.

Contents:
- JIT-compiled update routines for applying Clifford gates to tableau `X/Z` blocks

## Notes

- To reduce first call JIT latency, you can trigger compilation via
  `quickqudits.utils.warmup_numba(...)`.
- If you run into Numba import/JIT issues, simply avoid the accelerator path; the
  simulator will fall back to non-JIT code.
"""
