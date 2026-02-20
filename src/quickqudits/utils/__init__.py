"""
Core utilities for `quickqudits`.

This subpackage contains high-level tools built on top of the core
simulation layer (circuits, tableaus, gates). It includes routines for random circuit
generation, Monte-Carlo fidelity estimation, entanglement/Schmidt-rank calculations,
and statevector/density-matrix analysis.

You can import utilities like:

```python
from quickqudits.utils import random_circuit, fidelity_pf, schmidt_rank, check_symplectic_structure
```

## Submodules

### `utils.circuit_utils`
Small circuit-construction helpers.

Functions:
- `univ_input_state_circuit(...)` — build a “universal input state” preparation circuit (used for 2n-qudit style constructions)
- `encoding_circuit(...)` — build a specific encoding / preparation circuit used in examples/tests

---

### `utils.random_circuits`
Random circuit generation.

Functions:
- `random_circuit(...)` — generate a random `QuantumCircuit` (bounded depth/size)

---

### `utils.statevector`
Statevector helpers: measurement, projections, comparisons, and simple linear-algebra utilities.

Functions:
- `measureall(...)` — sample a full computational-basis measurement from a statevector
- `flip_endianness_statevector(...)` — reverse qudit-axis order of a length `d**n` statevector
- `nonzero_dary_indices(...)` — list nonzero indices in base-`d` (d-ary) digit form
- `get_zero_vector(...)` — construct the `|0...0⟩` statevector
- `entangle_input(...)` — construct a maximally entangled input on `2n` qudits
- `collapse_universal_to_basis(...)` — extract a basis-conditioned output from a “universal” (2n-qudit) run
- `get_projected_state(...)` — project onto a single-qudit measurement outcome (and normalize)
- `get_projected_state_multi(...)` — project onto multiple qudit outcomes (and trace them out)
- `compare_statevectors(...)` — compare two statevectors up to global phase
- `norm_diff(...)` — compute a norm difference between two statevectors
- `apply_unitary(...)` — apply a full `(d**n x d**n)` unitary to a statevector

---

### `utils.density`
Density-matrix helpers.

Functions:
- `get_zero_density_matrix(...)` — construct `|0...0⟩⟨0...0|`
- `is_valid_density_matrix(...)` — validate (Hermitian/trace/PSD) density matrices
- `purity(...)` — compute purity `Tr(ρ²)`
- `density_matrix_to_probvector(...)` — computational-basis probability vector from a density matrix

---

### `utils.fidelity`
Monte-Carlo fidelity estimation for noisy circuits.

Functions:
- `fidelity_pf(...)` & `fidelity_np(...)` — estimate computational-basis fidelity of a noisy circuit via sampling

---

### `utils.entanglement`
Entanglement utilities for stabilizer states.

Functions:
- `schmidt_rank(...)` — Schmidt exponent / Schmidt rank across a bipartition
- `schmidt_rank_vector(...)` — compute a Schmidt-rank vector across multiple cuts/partitions

---

### `utils.tableau_utils`
Tableau-focused helpers.

Functions:
- `warmup_numba(...)` — trigger JIT compilation of Numba kernels (to avoid first-call overhead)
- `check_symplectic_structure(...)` — validate full-tableau commutation structure
"""

from .circuit_utils import *
from .density import *
from .entanglement import *
from .fidelity import *
from .random_circuits import *
from .statevector import *
from .tableau_utils import *
