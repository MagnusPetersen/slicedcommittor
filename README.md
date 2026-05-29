# slicedcommittor

<!--
The repository is private during initial collaboration; the badges below
will resolve once the remote is public.
-->
[![CI](https://github.com/magnuspetersen/slicedcommittor/actions/workflows/test.yml/badge.svg)](https://github.com/magnuspetersen/slicedcommittor/actions/workflows/test.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

Sample-based committor function approximation via 1D projections (analogous to
sliced optimal transport). Given samples and basin labels, the library returns
the committor q(x): the probability of reaching state B before state A
starting from configuration x.

## Method

Instead of solving the full committor PDE in d dimensions, the algorithm:

1. Projects samples onto many random 1D directions (uniform on the sphere by default).
2. Solves a 1D reaction-diffusion committor problem along each slice.
3. Combines the slice committors with principled weights. The recommended
   default is the **enriched basin-moment-constrained solver (EBMC)**, which
   adds a free bias to the linear aggregator and yields a closed-form
   solution that strictly improves on the vanilla BMC simplex in Dirichlet
   energy. A higher-order variant (**PESB-EBMC**) augments each per-slice
   basis with a smoothstep family `Ψ_n(v) = v^n / (v^n + (1-v)^n)` for
   additional resolving power on heterogeneous transitions. Diagonal RD
   and full-Gram simplex solvers remain available for compatibility.
   Optional post-hoc affine / monotone calibration further refines the
   estimate near the basins.

## Theory primer

The committor `q(x)` is the probability that a stochastic trajectory starting
at `x` reaches basin B before basin A. Solving its PDE
`∇ · (e^{-βV} ∇q) = 0` in `d` dimensions becomes infeasible for `d ≳ 5`.

The sliced approach replaces the d-dimensional solve with `M` independent
1D solves:

1. **Project.** For each random direction `θ_j` on the unit sphere, project
   every sample `x_i` to a scalar `s_{ij} = θ_j · x_i`.
2. **Solve along the slice.** Fit a histogram free-energy `F_j(s)`, build the
   1D Smoluchowski operator with absorbing boundaries on the projected basins
   `θ_j(A)`, `θ_j(B)`, and solve the resulting tridiagonal linear system for
   the per-slice committor `q_{θ_j}(s)`.
3. **Aggregate.** Recombine the slice committors into the d-dimensional
   estimator
   `q̂(x) = c + Σ_j w_j · q_{θ_j}(θ_j · x)`,
   choosing weights `w_j` and an optional bias `c` to satisfy basin-mean
   constraints (`E_A[q̂] = 0`, `E_B[q̂] = 1`) while minimising aggregate
   Dirichlet energy.

The recommended weight solver (**EBMC**) computes `w, c` in closed form via
a single regularised Gram solve; the higher-order **PESB-EBMC** variant
enriches the per-slice basis with smoothstep functions `Ψ_n(v)`. Both fall
back gracefully on a diagonal-RD or full-Gram simplex aggregator.

A boundary-error term `ε_j` measures how far the 1D solve fails to hit
`q = 0` / `q = 1` at the projected basin edges; weights are corrected by
`(1 - ε_j)_+` to suppress unreliable slices. Three estimators ship
(equilibrium, RMS, 1D flux), differing in which moments of `q` they
penalise.

The accompanying paper derives the constraint algebra and the
Galerkin-monotonicity property in full.

## Install

```bash
# from a clone of the repo:
pip install -e .

# directly from GitHub (requires read access to the private repo):
pip install git+ssh://git@github.com/magnuspetersen/slicedcommittor.git
```

Hard dependencies: `jax >= 0.4.20`, `numpy >= 1.24`. Users pick the JAX
accelerator flavour (`jax[cpu]`, `jax[cuda12]`, …) themselves; the library
does not pin one.

## Quickstart (recommended defaults)

```python
import jax
jax.config.update("jax_enable_x64", True)   # required for EBMC

import jax.numpy as jnp
from sliced_committor import (
    compute_sliced_committor,
    compute_enriched_basin_moment_weights,
    evaluate_committor,
)

# samples: (N, dim) array of configurations
# in_A, in_B: (N,) bool arrays, basin labels
result = compute_sliced_committor(
    samples, in_A=in_A, in_B=in_B, n_directions=256,
)
ebmc = compute_enriched_basin_moment_weights(result, samples)
q = evaluate_committor(
    result, points, ebmc,
    in_A=in_A_at_points, in_B=in_B_at_points,    # optional, for q=0/q=1 enforcement
)
```

`evaluate_committor` autodetects the centered-basis (EBMC) result dict and
routes to the affine-aware aggregator: `q̂(x) = c + Σ_j w_j q_{θ_j}(θ_j·x)`.

## Recommended settings (cross-system consensus)

The library's defaults were tuned against the combined paper figures on
AIB9, villin HP-35, and chignolin. They are optimal-but-general; you should
not need to touch them for a typical run.

| Setting | Default | Where it varies |
|---|---|---|
| `n_directions` | 256 | 512 (AIB9 / mid-d), 4096 (villin / large) |
| `n_bins` | 200 | 2000 (AIB9 paper polish), 300 (villin) |
| `seed` | 42 | (any) |
| `binning_method` | `'quantile'` | unchanged across systems |
| `n_min` | 10 | unchanged across systems |
| `boundary_quantile` | 1.0 | 0.95-0.98 if d ≫ k and halo artefacts appear |
| `rd_kappa` | 1e12 | hard absorption, unchanged across systems |
| `direction_sampling` | `None` (uniform) | `DirectionSamplingConfig(mode=...)` for `'lda'`, `'pca'`, `'gcpca'`; or `directions_tica_ema(...)` for time-lagged trajectories |
| weight solver | **EBMC** (this README's quickstart) | PESB-EBMC P=2 for heterogeneous transitions |
| `tikhonov` | `'auto'` | unchanged across systems |
| `gram_dtype` | `'float64'` for EBMC/PESB/BMC (required); `'float32'` for full-Gram | matches paper runs |
| ε estimator | equilibrium (cached) | `compute_epsilon_rms` for fat-tailed states |

For very high-dimensional Cartesian features (d ≳ 200) the paper runs lower
`boundary_quantile` to ~0.95 to suppress halo inflation in the projected
state shadows. For low-d torsion features, the default 1.0 is fine.

## Direction-sampling alternatives

The default sampler is uniform on the sphere. For directed bias, the
library ships:

```python
from sliced_committor import (
    DirectionSamplingConfig,         # config NamedTuple
    directions_uniform,              # baseline
    directions_pca, pca_basis,       # top-/bottom-K PCA axes (label-free)
    directions_gcpca, gcpca_basis,   # generalised contrastive PCA v4.1
                                     #   (target = A ∪ B vs background)
    directions_tica_ema,             # EMA-integrated slow modes (single trajectory)
    directions_tica_ema_decomposed,  # bias-aware TICA-EMA (umbrella sampling)
    compute_lda_axis,                # single-axis LDA bias
    sample_power_spherical_mixture,  # the underlying mixture sampler
)
```

Pass `direction_sampling=DirectionSamplingConfig(mode='pca', n_bias_axes=4)`
(or similar) to `compute_sliced_committor` and the dispatcher computes the
bias axes from the samples, then feeds them through a power-spherical +
uniform mixture (per-axis weights default to the `|λ|`-simplex). The
`(μ, α)` knobs are dimension-invariant: `α=1` is pure uniform, `α=0` is
pure informed (the dispatcher warns at `α < 0.1`).

Kernel-based axes (diffusion maps, RBF kernel PCA via random Fourier
features) are deliberately not shipped in the library, they are unstable
in our hands.

## Higher-order softmix basis (PESB-EBMC)

For heterogeneous transition regions, augment each per-slice basis with the
smoothstep family `Ψ_n(v) = v^n / (v^n + (1−v)^n)` for `n ∈ n_values`:

```python
from sliced_committor import compute_enriched_basin_moment_weights_power

pesb = compute_enriched_basin_moment_weights_power(
    result, samples, P=2,                # n_values defaults to [1.0, 2.0]
)
q = evaluate_committor(result, points, pesb)
print("Dirichlet improvement over plain EBMC:", pesb["improvement_over_ebmc"])
```

Only the smoothstep ("softmix") basis is shipped: the monomial alternative
was disconfirmed (asymmetric around `v = 0.5`, Hilbert-like within-block
conditioning at large P). `P = 2` is a robust default; `P = 3` is the
practical ceiling.

## All weight solvers

```python
from sliced_committor import (
    # Recommended default (closed-form, strict Dirichlet improvement):
    compute_enriched_basin_moment_weights,

    # Higher-order softmix basis on top of EBMC:
    compute_enriched_basin_moment_weights_power,

    # Vanilla BMC (two basin-moment constraints, no bias):
    compute_basin_moment_weights,

    # Full-Gram simplex (Σw = 1):
    compute_full_gram_weights,

    # Diagonal RD-corrected, dispatched through compute_weights_multi:
    corrected_dirichlet_inv_rd,
    compute_weights_multi,
    get_default_weight_functions,
)
```

The Gram-based solvers (EBMC, PESB-EBMC, BMC, full-Gram) do not go through
`compute_weights_multi`: they take richer kwargs (Tikhonov, ε-estimator,
Gram dtype) and return a dict carrying weights + diagnostics rather than a
bare vector.

**Float64 requirement.** EBMC, PESB-EBMC, and plain BMC all require
`jax.config.update("jax_enable_x64", True)` before the result is built.

## Three ε estimators

```python
from sliced_committor import (
    compute_epsilon_equilibrium,   # equilibrium-weighted, default
    compute_epsilon_rms,           # RMS variant (≥ equilibrium pointwise)
    compute_epsilon_flux1d,        # flux-importance-reweighted (1D)
    compute_epsilon,               # dispatcher: name -> estimator
)
```

Pass any of these as the `epsilon_fn` argument to
`compute_full_gram_weights` to override the cached equilibrium ε
(`compute_epsilon_rms` is the most common alternative).

### Utility

```python
from sliced_committor import make_weighting_context, WeightingContext
```

`make_weighting_context(result)` builds the `WeightingContext` bundle that
weight functions and ε estimators consume. Call this if you want to invoke
a weight function directly (`corrected_dirichlet_inv_rd(ctx)`) rather than
via `compute_weights_multi`.

## Calibration

```python
from sliced_committor.calibration import (
    calibrate_weights_affine,        # affine label-mean (centered basis)
    compute_affine_calibration,      # ABC_v2 global affine
    apply_affine,
    evaluate_committor_calibrated,
    compute_recalibration_curve,     # 1D-RD recalibration along q̄
    apply_recalibration,
    evaluate_committor_with_recal,
)
```

## Why no `beta` argument?

The committor is β-invariant given fixed samples: the algorithm computes
`F = -(1/β) log ρ` from histogram density, then uses `ρ = exp(-β(F - F_min))`
internally; the two β's cancel. Verified numerically across every downstream
output (committors, all weight solvers, all ε estimators, calibration,
recalibration). Including β in the API would be misleading; the library uses
β = 1 internally so `result.free_energies` stores `-log ρ` directly. Multiply
by `1/β_physical` to recover physical-units free energy.

## Why bool arrays instead of state functions?

The method only needs to know *which* training samples are in A vs B. State
functions (geometric expressions, RMSD thresholds, clustering output …) are
how *you* compute those labels; the library doesn't need to inspect that
logic. Boundary enforcement at evaluation points uses the same mechanism:
pass `in_A` / `in_B` arrays for the points being evaluated.

This means:
- a user with cluster labels has nothing to wrap;
- a user with a geometric state writes one line:
  `in_A = jnp.linalg.norm(samples - centre, axis=-1) < radius`;
- no `FunctionalState` class to learn or maintain.

## Memory note

`compute_sliced_committor(..., store_projected_samples=True)` (the default)
materialises an `(M, N)` array of projected samples. For `N ≈ 10⁶` and `M = 256`
that's ~1 GB at float32. Pass `store_projected_samples=False` if you don't
need the full Gram or BMC solvers afterwards; the diagonal weight solver
re-projects on demand.

## Citation

If you use this library in published work, please cite the accompanying paper.
The DOI and full BibTeX block will be filled in on the first tagged release.

```bibtex
@article{petersen_sliced_committor,
  title   = {Sliced committor functions via random one-dimensional projections},
  author  = {Petersen, Magnus and others},
  year    = {2026},
  journal = {TODO},
  doi     = {TODO},
  note    = {Fill in once published},
}
```
