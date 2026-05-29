# Changelog

All notable changes to `slicedcommittor` will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- LICENSE file (MIT) at the repository root.
- `py.typed` marker so downstream type-checkers honour the library's type hints.
- Sphinx + Read the Docs scaffold under `docs/`.
- `examples/` directory with three self-contained scripts (1D OU analytical,
  2D double-well with optional inlined PDE baseline, higher-dim synthetic) and
  a per-slice diagnostics notebook.
- New test suite covering validation against the 1D OU closed form, validation
  against an inlined 2D Jacobi PDE solver, RMSE-vs-`n_directions` convergence,
  Gram internals (`_compute_derivative_matrix`, `_assemble_gram_matrix`), the
  three epsilon estimators and the dispatcher, plain-BMC KKT residuals, full-Gram
  weight numerics, diagonal RD weights, affine + ABC calibration, recalibration
  monotonicity, solver edge cases, evaluator boundary enforcement, and the input
  validation surface from Phase 6.
- Public diagnostics surface on `SlicedCommittorResult`: `summary()`,
  `__repr__`, plus module-level helpers `summarize_gram_diagnostics(result_dict)`
  and `why_masked(result, slice_index)`.
- Input validation at the public API boundary:
  - `compute_sliced_committor` rejects empty basins, overlapping basins, NaN
    samples, mismatched mask lengths, and non-2D sample arrays; warns when
    `dim > n_samples`.
  - The EBMC / PESB / BMC solvers raise a single actionable `ValueError` when
    `jax_enable_x64=False` (instead of cryptic JAX traces).
  - The epsilon estimator dispatcher returns a helpful "valid choices" message
    on unknown names.
- GitHub Actions workflows (`test.yml`, `lint.yml`, `docs.yml`) and a
  `release.yml` stub commented out until a public remote exists.
- Ruff configuration in `pyproject.toml`, pre-commit hooks including a local
  em-dash linter.
- `CONTRIBUTING.md`, expanded `README.md` with theory primer + badges.
- PyPI metadata: `keywords`, `classifiers`, `project.urls`, `maintainers`,
  granular extras (`test`, `docs`, `lint`, `examples`, `dev`).

### Changed
- `pyproject.toml` migrated to SPDX `license = "MIT"` plus `license-files`.
- `boundary_quantile` / `absorption_quantile` and `store_projected_samples`
  defaults documented at the API boundary; behaviour unchanged.

### Fixed
- Naming consistency pass across `solver.py`: standardised on `s_grid` for bin
  grids and `s_projected` for projected samples.

## [0.4.0] - 2025-05-28

### Added
- PESB-EBMC higher-order softmix basis (`compute_enriched_basin_moment_weights_power`)
  with the `Ψ_n(v) = v^n / (v^n + (1-v)^n)` family.
- `compute_epsilon_flux1d` flux-importance-reweighted epsilon estimator and
  the dispatcher `compute_epsilon(name, ctx)`.
- gcPCA direction sampler (`directions_gcpca`, `gcpca_basis`) v4.1.
- Calibration module: `calibrate_weights_affine`, ABC v2 affine calibration,
  per-slice and global recalibration curves.

### Changed
- EBMC promoted to recommended default in the README quickstart.
- Diagonal RD and full-Gram simplex solvers retained for compatibility.

## [0.3.0] - 2025-04

### Added
- Initial extracted library version with `compute_sliced_committor`,
  `evaluate_committor`, diagonal RD weights (`corrected_dirichlet_inv_rd`),
  full-Gram solver (`full_gram_weights`), plain BMC (`basin_moment_weights`),
  EBMC (`enriched_basin_moment_weights`).
- Direction sampling: uniform on the sphere, LDA single-axis bias, TICA-EMA
  (single-trajectory + bias-aware), PCA basis.
- Three equilibrium / RMS epsilon estimators.
