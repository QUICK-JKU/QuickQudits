"""
Backend configuration.

This module handles the numerical precision settings (single vs double)
used throughout the package. It maintains global references to data types.
"""

from typing import Literal

import numpy as np

# --- Global Backend Definitions ---

# Default precision settings (double).
COMPLEX_DTYPE = np.complex128
FLOAT_DTYPE = np.float64

# Default integer types
INT_DTYPE = np.int64
FAST_INT_DTYPE = np.int64


def set_precision(precision: Literal["single", "double"] = "double") -> None:
    """
    Set the floating-point and complex precision.

    This updates global DTYPE variables and clears the gate cache.

    Parameters
    ----------
    precision : Literal["single", "double"], default="double"
        The desired precision level.
        - 'single': Uses `np.float32`, `np.complex64`, and `np.int64`.
        - 'double': Uses `np.float64`, `np.complex128`, and `np.int64`.

    Raises
    ------
    ValueError
        If the provided precision string is not 'single' or 'double'.
    """
    global COMPLEX_DTYPE, FLOAT_DTYPE, INT_DTYPE

    if precision == "single":
        COMPLEX_DTYPE = np.complex64
        FLOAT_DTYPE = np.float32
        INT_DTYPE = np.int64
    elif precision == "double":
        COMPLEX_DTYPE = np.complex128
        FLOAT_DTYPE = np.float64
        INT_DTYPE = np.int64
    else:
        raise ValueError(f"Precision must be 'single' or 'double', got '{precision}'.")

    # Clear gate cache to rebuild matrices with the new dtypes.
    try:
        from . import gates

        gates.clear_cache()
    except ImportError:
        pass


def show_backend_info() -> None:
    """
    Print current backend and precision configuration.
    """
    print("Array backend: NumPy (CPU)")
    print(f"Precision:     {COMPLEX_DTYPE.__name__} / {FLOAT_DTYPE.__name__}")
    print(f"Integer width: {INT_DTYPE.__name__}")


# Initialize backend with default settings on import
set_precision("double")
