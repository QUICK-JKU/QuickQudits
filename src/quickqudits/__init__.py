"""
`quickqudits` is a library for the efficient classical simulation of Clifford circuits acting on d-dimensional qudits,
including Pauli/Weyl noise.

All dimensions are supported, with prime dimensions favoured.

Core features:

- Qudit stabilizer tableau simulator (`Tableau`) for `n` qudits of dimension `d`
- Tableau phases tracked as powers of `τ mod 2d` (with `τ² = ω`)
- A circuit container (`QuantumCircuit`) with Clifford/Weyl gates (allows statevector and density matrix simulation)
- Noise channels (e.g., depolarizing-style Weyl noise) and fast Monte-Carlo sampling utilities
- Optional Numba acceleration for fast tableau updates
- High-level tools and helpers for tableaus/statevectors/density matrices, fidelity estimation, random circuits, etc.

---

## Example

```python
import quickqudits as qq

# initialize circuit
n, d = 5, 3 # 5 qudits of dimension 3
qc = qq.QuantumCircuit(n, d)

# add gates
qc.CXdag(2, 0)
qc.CX(1, 0)
qc.H(0)
qc.H(1)
qc.CXdag(0, 2)
qc.CX(0, 4)
qc.X(2)
qc.S(4)
qc.Hdag(4)
qc.CXdag(2, 0)
qc.Sdag(3)
qc.CX(2, 0)
qc.CZdag(4, 0)

# add global depolarizing noise
qc.add_noise_global(0.5)

# visualize circuit
qc.draw(show_info=True)

# initialize tableau
tab = qq.Tableau(5, 3)

# draw samples from the noisy circuit
samples = tab.sample_measurements_noisy_pf(qc=qc, shots=5, seed=42)
print(samples, end='\\n\\n')

# execute the circuit and obtain tableau (ignoring noise channels)
tab.apply_circuit(qc)

# visualize tableau
print(tab)
```

Output:
```plaintext
[[2 0 0 0 1]
 [0 2 2 0 1]
 [2 2 1 1 1]
 [0 1 0 0 2]
 [0 1 1 0 0]]

#  | x0 x1 x2 x3 x4 | z0 z1 z2 z3 z4 | τ
────────────────────────────────────────
d0 |  0  0  0  0  0 |  1  0  0  0  0 | 0
d1 |  0  0  0  0  0 |  1  1  0  0  0 | 0
d2 |  0  0  1  0  0 |  2  0  0  0  0 | 0
d3 |  0  0  0  1  0 |  0  0  0  2  0 | 0
d4 |  0  0  0  0  1 |  2  0  0  0  2 | 4
----------------------------------------
s0 |  2  1  1  0  2 |  2  0  1  0  2 | 2
s1 |  0  2  0  0  0 |  0  0  0  0  0 | 0
s2 |  0  0  0  0  0 |  1  0  1  0  0 | 4
s3 |  0  0  0  0  0 |  0  0  0  1  0 | 0
s4 |  0  0  0  0  1 |  1  0  0  0  0 | 0
```

## Modules

- `circuit` — `QuantumCircuit`: build circuits, run statevector/density simulations, sampling helpers
- `tableau` — `Tableau`: stabilizer tableau evolution, measurement, conversions, noisy sampling
- `tableau_composite` — `CompositeTableau`: composite d
- `io.tableau_io` — save/load tableaus via compressed NumPy `.npz`
- `io.circuit_io` — save/load circuits via compressed NumPy `.npz`
- `accelerators.numba_fast` — Numba JIT kernels for fast tableau updates (optional)
- `backend` — global dtype/precision configuration (`set_precision`)
- `gate_class` — `Gate` and `NoiseChannel` definitions
- `gates` — numerical matrix builders for standard qudit gates
- `predefined_gates` — pre-instantiated gate objects (e.g. `H_gate`, `CNOT_gate`, …)
- `visualization` — Matplotlib circuit drawing backend
- `utils` — utilities (see below)

`utils` submodules:

- `utils.circuit_utils` — small circuit constructors (e.g. universal input-state prep)
- `utils.random_circuits` — random circuit generator (`random_circuit`)
- `utils.statevector` — statevector helpers (projection, sampling, comparisons, etc.)
- `utils.density` — density-matrix helpers (validity checks, purity, probvectors)
- `utils.fidelity` — Monte-Carlo fidelity estimators for noisy circuits
- `utils.entanglement` — Schmidt-rank utilities for stabilizer states
- `utils.tableau_utils` — Numba warmup + tableau symplectic checks

---

## Main classes and utilities

Classes:

- `QuantumCircuit` — build / store circuits; provides simulation + sampling routines
- `Tableau` — stabilizer tableau representation for qudit Clifford simulation
- `CompositeTableau` — stabilizer tableau for composite d
- `Gate` — gate metadata + matrix generation logic
- `NoiseChannel` — gate subclass representing noise models

Selected utilities:

- `utils.random_circuit`
- `utils.fidelity_pf`
- `utils.fidelity_np`
- `utils.check_symplectic_structure`
- `utils.get_projected_state`, `utils.compare_statevectors`, `utils.apply_unitary`, `utils.measureall`
- `utils.get_zero_density_matrix`, `utils.is_valid_density_matrix`, `utils.purity`
- ...

See utils documentation for the rest.
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("quickqudits")
except PackageNotFoundError:
    __version__ = "0.0.0"

from . import backend, io, utils
from .circuit import QuantumCircuit
from .predefined_gates import *
from .tableau import Tableau
from .tableau_composite import CompositeTableau
