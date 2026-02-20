"""
I/O subpackage.

This subpackage contains serialization utilities for circuits and tableaux.

## Submodules

### `tableau_io`
Save/load `tableau.Tableau` objects in ``.npz`` format.

---

### `circuit_io`
Save/load `circuit.QuantumCircuit` objects in ``.npz`` format.
"""

from .circuit_io import *
from .tableau_io import *
