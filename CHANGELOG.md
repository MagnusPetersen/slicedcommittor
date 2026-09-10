# Changelog

All notable changes to `sliced-committor` are documented in this file. The
format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and
the project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2026-09-09

The first public release. The public surface is `sliced_committor.__all__`
(36 names) and `sliced_committor.umbrella.__all__`, and it is the stability
promise from here on. The paper's figures and rate table are reproduced by
the Zenodo package against this version. Versions 0.3 to 0.6 were never
published; their history is kept below and their code at tag `v0.6.0`.

### The committor

- One weight solver, `solve_weights(result, *, tikhonov="halfset_eigen",
  heldout_cap=False, raise_on_degenerate=True, counts=None)`, returning a
  typed `Weights`
  (`w`, `c`, `dirichlet_energy`, `moment_gap`, `cond`, `ridge`, `tikhonov`,
  `heldout_cap`, `diagnostics`). `build_committor(result, weights, clip=True)`
  returns the callable `q(x, in_A=None, in_B=None)`; `fit_committor(samples,
  in_A=, in_B=, n_directions=256, seed=42, tikhonov=, heldout_cap=,
  return_details=, **solver_kwargs)` is the one-call path and returns
  `(q, CommittorFit)` with details.
- The half-set spectral filter is the default ridge. `tikhonov="auto"` is the
  closed-form scalar ridge and a float is an absolute ridge in the units of
  `G`. `heldout_cap=True` needs the filter and returns the cap, its standard
  error, the per-fold values and the train-versus-held-out `gap`. The
  regularisation lives in `core/_halfset.py`, absorbed from the RECOVAR
  research package together with the contiguous folds and the block
  bootstrap.
- `bootstrap_weights(result, weights, *, n_boot=40, block_len, run_ids=None,
  seed=0, points=None, min_basin_frames=5)` block-bootstraps the weights with
  the basis fixed, skips a replicate that leaves fewer than
  `min_basin_frames` frames in a basin, and returns a `Bootstrap` (`n_ok`
  counts the replicates kept).
- `rescale_transition(q_values, in_A, in_B)` is a function of an evaluated
  batch; the committor callable is a pure function of `x`.
- `compute_sliced_committor` takes `n_directions`, `n_bins`, `seed`,
  `binning_method`, `density_floor`, `n_min`, `rd_kappa`,
  `boundary_quantile`, `direction_batch_size`, `sample_weights`,
  `directions`, `direction_sampling` and `feature_metric`;
  `store_projected_samples`, `quantile_subsample` and `absorption_quantile`
  are gone, `binning_method` is validated, `sample_weights` must sum to one
  (the solve uses them verbatim, so the Dirichlet energies of fits are
  comparable), `rd_kappa` is `1e12` in one place, and `valid_mask` now means
  that the 1D solve is finite.
- Directions: `DirectionSamplingConfig(mode="uniform" | "lda", lda_shrinkage,
  mu, alpha, axis)`; `compute_lda_axis`, `sample_power_spherical_mixture(key,
  axis, n_directions, dim, mu=, alpha=)` (one axis) and `directions_uniform`
  are public so that the axis can be inspected or replaced and the draw
  reproduced outside the solve.
- The feature-space metric `sincos_pullback_metric` and `AngleSign` are
  exported; `AngleSign` is required.
- `scipy>=1.10` is a hard dependency (the half-set path used it undeclared);
  `jax>=0.5.3` is the oldest version tested, and CI runs it.
- Uniform sample weights are exactly `ones(N) / N` (a one-ulp renormalisation
  was amplified to `4e-4` in the half-set weights) and the `(M, M)` solve stays
  on scipy's Cholesky (the JAX one differs by `1e-4` at condition `1e12`).
  Golden references under `tests/golden/` pin both: bit for bit where the
  frozen slice basis reproduces bit for bit (JAX 0.5.3 on the machine that
  froze them), and at the measured drift between JAX builds and machines
  anywhere else (`docs/reproducibility.md`).

### Rates

- The state is the pair `{D_q, pi}`. `density`, `committor_grad_sq` and
  `basin_populations` read the static ensemble; `diffusion_profile` (at an
  explicit `lag`), `lag_scan`, `hummer_diffusion` and `pooled_acf_diffusion`
  (the paper's estimator, promoted from the reproduction package with
  `run_ids` and `window_band`) measure the diffusion on a trajectory (no
  estimator reads across a run join, `diffusion_profile` included);
  `committor_diffusion_from_cv` is the Jacobian map from a collective
  variable, with `linear_response_grad_sq` for the CV's gradient and
  `committor_diffusion_from_cv_reparam` (gated by the squared Spearman rank
  correlation) as the gradient-free alternative. Quantities take per-sample
  values, not the committor, except `committor_grad_sq`, which differentiates it.
- `rate_from_profiles(pi, D_q, rho_A, rho_B, *, reduction="plateau", band=,
  q_star=, window=)` unifies the reductions (plateau, arithmetic, harmonic,
  local); each accepts only its own parameter. Every result carries the
  arithmetic and harmonic values, the profiles `nu` and `D_q` on the
  density's grid, and `flatness`; `flux_flatness` is the paper's `flux_cv`
  on a fixed band.
- `committor_rate(q, samples, *, D_q= | trajectory=, dt=, lag=, window_ids=,
  run_ids=, ...)`: the two routes are mutually exclusive.
- The MFPT quadrature uses midpoint cumulative masses, so the discrete
  identity `int M_A dq = rho_A` is exact and every reduction returns the same
  rate to rounding on an exact committor.
- `rates.baselines.pmf_kramers_rate(pi, D)` takes profiles along a physical
  coordinate; `rates.units` holds the unit conversions.

### Umbrella sampling

- `sliced_committor.workflows` is `sliced_committor.umbrella`: `USDataset`
  (without `system_name` and `cv_names`), `reweight` / `mbar_weights` /
  `wham_weights`, the reading helpers `read_colvar`, `parse_restraint`,
  `load_trajectory` and `align_colvar_traj` in `io`, `mdtraj_metric`, and
  `fit_and_rate(dataset, features, sample_weights, *, n_directions=, n_bins=,
  seed=, tikhonov=, bridge_metric=, D_s=, run_ids=, diffusion=("cvmap",),
  reductions=, lag=, n_diff_bins=, strict=True, **solver_kwargs)` (the solver
  settings, `direction_sampling`, `directions` and `feature_metric` among
  them, ride in `solver_kwargs`) returning one bundle (`committor`,
  `q_samples`, `D_s`, `cv_grad_sq`, `profiles`, `rates`, `flux_cv`,
  `kramers`, `errors`), with `kramers_baseline(dataset, weights)` as the
  baseline on its own. `fit_and_rate` no longer switches `jax_enable_x64` on
  in the caller's process (the solve raises when it is off). The extra is
  `[umbrella]` (mdtraj, pymbar).

### Removed

Everything below is at tag `v0.6.0`; `docs/design_decisions.md` gives the
evidence for each item.

- Weight solvers: plain BMC, PESB-EBMC, the full-Gram simplex and its three
  boundary-error estimators, the diagonal RD weights, the Nitsche boundary
  conditions, `compute_weights_multi` and both registries, `WeightingContext`,
  `committor_dirichlet_energy`, `summarize_gram_diagnostics`, `why_masked`.
- Ridge rules `cv`, `cv_refit`, `auto_lambda`, `auto_m`, and the tuple and
  relative-float spellings.
- Direction samplers: PCA, generalised-contrastive PCA, both TICA-EMA
  factories, LDA by mean difference; `sweep_committor` and `SweepResult`.
- The metric's `M0Kind`, `m0_atom`, `shrink_metric`, `torsion_g_matrix`,
  `array_sha256`.
- Rates: `dirichlet_rate`, `tpt_rate`, `berezhkovskii_szabo_rate`,
  `kramers_rate`, `reactive_flux`, `diffusion_coefficient`,
  `mapped_committor_diffusion`, `saddle_bridge_D`, `BridgeD`, the whole
  bridge-v2 module (`flux_reductions`, `conditional_mean`,
  `mapped_committor_diffusion_field`, `smooth_Ds_profile`,
  `constancy_reconstruction`, `bootstrap_barrier_Ds`,
  `bayesian_smoluchowski_diffusion`, `hummer_Ds_profile`,
  `committor_grad_profile`, `committor_populations`), the `at=` selector.
- The workflow orchestrator: the system registry (`config`), `loaders`,
  `featurize`, `sweep`, `pipeline`, `report`, `plotting`, the
  `sliced-committor-us` command, and the `pyyaml` and `matplotlib` extras.
- The `recovar` research package.

### Migration from 0.6.0

| was | now |
|---|---|
| `compute_enriched_basin_moment_weights(res, X, sw, tik, **kw)` | `solve_weights(res, tikhonov=tik, **kw)` |
| `weights["w"]`, `["c"]`, `["M_gap"]`, `["heldout_cap"]` | `weights.w`, `.c`, `.moment_gap`, `.heldout_cap` |
| `fit_committor(weights="ebmc", weight_kwargs={"tikhonov": t})` | `fit_committor(tikhonov=t)` |
| `tikhonov=<float>` (relative), `("ridge_abs", r)`, `"cv"`, `"auto_lambda"` | `tikhonov="halfset_eigen"`, `"auto"`, or an absolute float |
| `build_committor(..., enforce_boundary_conditions=True, rescale_transition=True)` | `q = build_committor(res, w)`; `rescale_transition(q(X, in_A=, in_B=), in_A, in_B)` |
| `evaluate_committor(res, pts, w, ...)` | `build_committor(res, w)(pts, in_A=, in_B=)` |
| `DirectionSamplingConfig(mode="lda", lda_method="fisher")` | `DirectionSamplingConfig(mode="lda")` |
| `sweep_committor(grid=...)` | `itertools.product` + `fit_committor(..., heldout_cap=True, return_details=True)` |
| `density(q, X, coordinate=cv)` | `density(q(X))`, `density(cv, span=(lo, hi))` |
| `diffusion_coefficient(q, traj, dt=, lag=None, method="hummer", per_window=)` | `diffusion_profile(q(traj), dt=, lag=)`, `hummer_diffusion(s, window_ids, dt=)`, `lag_scan` |
| `mapped_committor_diffusion(q, X, D_s=, cv_grad_sq=)` | `committor_diffusion_from_cv(committor_grad_sq(q, X), D_s=, cv_grad_sq=)` |
| `dirichlet_rate(q, X, D=D0)` | `committor_rate(q, X, D_q=committor_diffusion_from_cv(g_q, D_s=D0, cv_grad_sq=1), reduction="arithmetic")` |
| `tpt_rate(q, X, D=D0, at=(lo, hi))` | the same with `reduction="plateau", band=(lo, hi)` |
| `berezhkovskii_szabo_rate(mode="mfpt" / "local")` | `reduction="harmonic"` / `"local"` |
| `committor_rate(q, X, traj, dt=, D_profile=D_q, at=...)` | `committor_rate(q, X, D_q=D_q, band= / q_star= / window=)` |
| `reactive_flux(q, X, D=D)` | `committor_rate(...)["nu"]` |
| `_cv_feature_grad_sq`, `hummer_pooled_acf` (reproduction package) | `linear_response_grad_sq`, `pooled_acf_diffusion` |
| `workflows.{_containers, reweight, colvar, bias, trajectory, metric_inputs, committor_rates, pmf_kramers, _units}` | `umbrella.{dataset, reweight, io, io, io, mdtraj_metric, rates, rates}`, `rates.units` |
| `fit_and_rate(direction_mode=, weight_solver=, diffusion_mode=, has_dynamics=, committor_sweep_grid=, solver_kwargs=, weight_kwargs=)` | `fit_and_rate(tikhonov=, diffusion=, D_s=, direction_sampling=, **solver_kwargs)` |

## Pre-release history (never published)

## [0.6.0] - 2026-08-03

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
