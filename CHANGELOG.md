# Changelog

All notable changes to `sliced-committor` will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **`sliced-committor-us --reference NAME=VALUE[:UNITS]`** (repeatable): plot custom
  reference rate(s) on the comparison chart, e.g. `--reference expt=0.0105:1/us`
  (UNITS default `1/s`). Overrides the registry rates; when omitted, known systems
  use their registry reference and an unregistered folder gets no reference band.
  Threaded through `run_pipeline(references=...)` -> `run_sweep`.
- **Verbose discovery logging** (`sliced-committor-us -v`): `load_us_dataset` now
  logs each discovery decision (config-source match, window count, resolved topology
  + atom selection, and the representative window's COLVAR / CV-columns / restraint
  center+kappa / trajectory).
- **LDA is the fast-mode informed direction default** (fast sweep is now
  `{uniform, lda}` instead of `{uniform, tica_ema_decomposed}`): LDA is supervised
  (uses the A/B labels) and dynamics-free, so it is cheaper than the TICA modes,
  which need per-window trajectories. `lda` is also added to the exhaustive mode set.
- **Comprehensive US-workflow documentation** (`docs/umbrella_sampling.md`): the CLI
  reference, auto-discovery, the `SystemConfig` registry, the sidecar schema (with
  `Region` box/ball examples), featurizations, direction modes, the rate estimators,
  reference rates, and outputs.

### Fixed
- **`--no-plot` is now honored.** It was wired into argparse but never threaded to
  `write_report`, so plots were always generated; `run_pipeline` now takes `plot=`.
- **Clearer sidecar-on-failure message.** When no system config matches an
  unregistered folder, the error now names the required sidecar keys (region_A /
  region_B, cv_columns, kappa, beta, topology) and their shapes, and the `--help`
  epilog documents them.
- **`sliced-committor-us .` / `./` from inside a US folder now works.** The registry
  match used `Path(folder).name`, which is empty for `.`/`./`; the folder is now
  resolved to a canonical path first.
- **Informed direction modes on a 1D feature** (e.g. `cv_only`) no longer error
  (`dim must be >= 2`): in 1D there is a single axis up to sign, so `lda`/`pca`/etc.
  fall back to the trivial uniform single-axis path.

### Changed
- **Workflows kinetics: replaced the feature-space `TPT_cv` estimator with the
  coordinate-invariant `TPT_cvmap`.** The old `TPT_cv` (a frozen scalar CV-space
  diffusion times the FEATURE-space `|grad q|^2`) was not coordinate-invariant: it
  scales as `1/alpha^2` under a feature rescaling `f -> alpha*f` (the committor and
  `<|grad q|^2>` transform, the scalar `D_cv` does not). `TPT_cvmap` reads the TPT
  flux PLATEAU on the CV-mapped committor diffusion
  `D_q(q) = D_s * <|grad q|^2>_q / <|grad s|^2>` (the same diffusion as
  `BS_mfpt_cvmap`, with the plateau reduction instead of the harmonic MFPT); the
  configurational scale `D0 = D_s/<|grad s|^2>` transforms as `alpha^2` and cancels
  the gradient, so the estimate is feature-scaling invariant. Affects
  `rates_by_method` keys, the `profiles.png` flux panel, and `report.md`.

## [0.5.0] - 2026-06-11

### Added
- **Dirichlet energy + settings sweep.** `committor_dirichlet_energy(result,
  weights, *, mode="auto")` returns the variational objective 𝓓[q̂] = wᵀG w of
  the recombined committor (the physical gradient energy, no Tikhonov ridge), a
  label-free relative quality ranker. `fit_committor(..., return_details=True)`
  now populates the new `CommittorFit.dirichlet_energy` field with it. The
  dedicated `sweep_committor(samples, *, in_A, in_B, grid={...}, select_by=
  "dirichlet", on_error="skip", **fixed)` fits the full Cartesian product of a
  settings grid and returns a `SweepResult` (all fits + their Dirichlet energies
  + boundary errors + configs, plus `best_idx` / `.best_committor`). Purely
  additive: existing fit / weight-solver numerics are unchanged.
- **Callable committor API.** The committor is now a callable object built via
  `build_committor(result, weights, *, clip=True, enforce_boundary_conditions=True,
  rescale_transition=False)` (returns `q`, called as `q(points, *, in_A=None,
  in_B=None)`) or the one-shot `fit_committor(samples, *, in_A, in_B,
  weights="ebmc", n_directions=256, seed=42, ...)`. Committor gradients are
  available via `committor_gradient(q, points) -> (P, dim)`.
- New `sliced_committor.rates` subpackage for transport-coefficient and rate
  estimation: the quantity primitives `diffusion_coefficient`, `density`,
  `reactive_flux`; the rate formulas `dirichlet_rate`, `tpt_rate`,
  `berezhkovskii_szabo_rate`, `kramers_rate`, and the coordinate-invariant
  `committor_rate` (unified `{D_q, π}` reductions); the calibrated-`D` bridge
  `saddle_bridge_D` / `BridgeD`; and the automatic flux-flatness plateau
  `find_plateau` / `PlateauWindow`.
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
  weight numerics, diagonal RD weights, solver edge cases, evaluator boundary
  enforcement, and the input validation surface from Phase 6.
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
- The seven algorithm modules (`solver`, `weights`, `gram`, `directions`,
  `_internal`, `_bmc`, `_bmc_enriched`) moved under the `sliced_committor.core`
  subpackage. The top-level `import sliced_committor as sc` public API is
  unchanged; only direct submodule imports gain the `.core` prefix
  (e.g. `from sliced_committor.core.weights import compute_epsilon_rms`).
- `pyproject.toml` migrated to SPDX `license = "MIT"` plus `license-files`.
- `boundary_quantile` / `absorption_quantile` and `store_projected_samples`
  defaults documented at the API boundary; behaviour unchanged.
- **`rates` API consistency pass (behaviour-preserving).**
  `berezhkovskii_szabo_rate` is now a thin alias for `committor_rate`
  (`mode="local"` -> `reduction="local"`, `mode="mfpt"` -> `reduction="harmonic"`);
  the iso-committor bin-count argument of `reactive_flux` / `dirichlet_rate` /
  `tpt_rate` is renamed `n_strata` -> `n_bins` to match the other primitives;
  `dirichlet_rate` reports the flat `plateau` / `plateau_flatness` / `plateau_ok`
  keys (shared with `tpt_rate` / `committor_rate`) instead of a nested `section`
  dict; `BridgeD` is a `NamedTuple`. Rate values are unchanged.
- **`rates` naming / settings tidy-up (follow-up).** Dropped the unused
  `lag_selection` / `lag_tolerance` knobs from `diffusion_coefficient` and the
  rates that forward them (the lag scan always uses the robust diffusive-peak
  pick); renamed `BridgeD.D_cart` -> `BridgeD.D` (it is a general calibrated
  diffusion, not cartesian) and `PlateauWindow.n_bins` -> `PlateauWindow.n_valid`
  (it counts valid bins, distinct from the histogram `n_bins`); unified the
  default `n_bins` of `reactive_flux` / `dirichlet_rate` / `tpt_rate` to 200 (was
  50) to match the other primitives. Recommended-method rate values are unchanged.
- **Library-wide hygiene / dedup pass (behaviour-preserving, no numeric
  change).** Removed the 36 MB stale `docs/_build/` artifact (it referenced the
  removed `calibration` subpackage and pre-`.core` module paths) and added a
  `.gitignore`. Factored the duplicated per-basin epsilon weighting in
  `core/weights.py` into `_basin_weight_sums`, and filled in missing `Args`
  entries (`clamp_epsilon` / `return_overlap` / `gram_dtype`) on the weight-solver
  wrappers in `core/solver.py`. Added `tests/_helpers.py` with a single
  parametrizable `two_basin_samples` sampler (collapsing seven near-duplicate
  per-file samplers) and the reused `TOL_SOLVER` tolerance. Added `__all__ = []`
  to `core/__init__.py`.
- **`directions_tica_ema` ridge is now trace-relative** (`η = ridge·tr(C0)/d`,
  matching `compute_lda_axis` / `gcpca_basis` so a ridge value ports across them);
  the 1e-6 default is a near-no-op vs the prior absolute ridge.
  `directions_tica_ema_decomposed` keeps its tuned *absolute* whitening ridge,
  now documented as such (not interchangeable with the relative ridges).
- **Degenerate-input guards.** A zero-width plateau range (`at=(c, c)`) now
  raises; `directions_tica_ema_decomposed` warns when between-window variance
  collapses; a noise-dominated TICA solve (all eigenvalues ≤ 0) warns and falls
  back to uniform directions. `committor_dirichlet_energy` masks invalid slices
  for the centered / PESB ansätze so it always matches the built committor.
- **Further DRY consolidation (behaviour-preserving).** A shared
  Gram mask+Tikhonov helper (`_mask_and_regularize_gram`) across the constrained
  / BMC / EBMC KKT solvers; a shared boundary-condition helper across
  `build_committor` and the internal evaluator; a shared epsilon basin-moment
  base; `dirichlet_rate` builds the co-area profile once; `committor.py` and
  `sweep.py` share one `_energy_basis` classifier.

### Removed
- **`evaluate_committor` removed from the public API.** Use the callable
  committor instead: `build_committor(result, weights)(points, in_A=..., in_B=...)`
  (or `fit_committor(...)` for the one-shot path). The `in_A`/`in_B` basin masks
  now move to the call site; `clip`, `enforce_boundary_conditions`, and
  `rescale_transition` are set on `build_committor`.
- **Post-hoc calibration subpackage** (`sliced_committor.calibration`): the
  affine label-mean calibration (`calibrate_weights_affine`), the ABC_v2 global
  affine calibration (`AffineCalibration`, `compute_affine_calibration`,
  `apply_affine`, `evaluate_committor_calibrated`), and the 1D-RD recalibration
  (`compute_recalibration_curve`, `apply_recalibration`,
  `evaluate_committor_with_recal`). These evaluation-level corrections did not
  reliably improve the estimate.
- `evaluate_committor` no longer accepts the calibration knobs
  `adaptive_boundary_correction`, `min_transition_fraction`, `affine_correction`,
  `affine_calibration`, or `affine_clip` (and the deprecated ABC v1 code path).
  The evaluation-level post-processing it retains is `clip` (the `[0, 1]` bound),
  `enforce_boundary_conditions` (q=0 on A, q=1 on B), and `rescale_transition`.
- **Public weight-solver surface trimmed.** The bare `basin_moment_weights`,
  `enriched_basin_moment_weights`, and `enriched_basin_moment_weights_power`
  names are no longer top-level exports; use the `compute_*` wrappers (the
  documented, `result`-based interface). `EnrichedBMCRepresentationError` stays
  public.
- **Dropped the `center` argument** from `pca_basis` / `gcpca_basis` (they always
  centre, the standard for unsupervised bases).
- **Removed unreachable internal code:** the thin-shell BCM solver and its
  multi-constraint KKT, `precompute_gram` / `precompute_gram_and_overlap` /
  `solve_gram_weights` / `compute_gram_diagnostics` (a redundant `full_gram_weights`
  alias), the residual-decomposition helpers, and the dead `needs_inversion`
  plumbing in the 1-D RD solver. None were exported, tested, or reachable. The
  Gram-quality diagnostics remain available on every solver's result dict via the
  public `summarize_gram_diagnostics`.

### Fixed
- Naming consistency pass across `solver.py`: standardised on `s_grid` for bin
  grids and `s_projected` for projected samples.
- `committor_rate(reduction="plateau")` now reports `plateau_flatness` for an
  explicit range too (previously only the `at="auto"` branch), and the auto
  branch of `dirichlet_rate` / `tpt_rate` reports the plateau's own flatness
  rather than recomputing it on a differently-filtered subset.
- The settings-sweep mixed-energy-family warning now also fires for a callable
  `select_by` that ranks on the Dirichlet energy, not only the string default.

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
