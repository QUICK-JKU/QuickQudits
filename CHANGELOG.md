# Changelog

All notable changes to this project will be documented in this file.

The format follows [Semantic Versioning](https://semver.org/).

---

## [Unreleased]

### Added
- `quickqudits.pauli_frame.NoisyCircuitSampler`: samples any Clifford circuit with Pauli (Weyl)
  noise channels from a compiled effect table. Same distribution as
  `Tableau.sample_measurements_noisy_pf`, about 4-5x faster on surface-code encoders
  (d = 3 ... 9). `precompute_noiseless_pool` caches the clean outcomes; `sample(..., probs=...)`
  rescans the noise strength without recompiling.
- `tests/test_noisy_circuit_sampler.py`.

---

## [1.0.1] - 2026-03-07

### Added
- GitHub Pages documentation.
- ORCID ids in CITATION.
- Circuit drawing example in the README.

### Changed
- Updated project URLs to point to the published documentation.

## [1.0.0] - 2026-02-20

### Added
- Initial release of QuickQudits