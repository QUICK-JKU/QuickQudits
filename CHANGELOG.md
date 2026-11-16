# Changelog

All notable changes to this project will be documented in this file.

The format follows [Semantic Versioning](https://semver.org/).

---

## [Unreleased]

### Added
- `quickqudits.pauli_frame`: `PauliFrameSampler`, a fast swap-test estimator of the Renyi-2
  coherent information `I2 = S2(Q) - S2(RQ)` under Pauli noise. It compiles an effect table
  once, samples the noiseless outcomes once, and applies noise by table lookup for every `p`.
  Standard errors by the delta method; `shots_for_target_se` sizes the shot budget.
- Helpers: `duplicate_circuit_vertical`, `add_destructive_swap_test`, `bitflip_noise_model`,
  `depolarizing_noise_model`, `run_unit_tests` (self-check against `sample_measurements_noisy_pf`).
- `tests/test_pauli_frame.py`.

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