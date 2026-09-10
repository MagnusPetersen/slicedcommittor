# Settings, diagnostics and uncertainty

The defaults are the paper's. This page goes through what each setting
does, when to change it, and how to tell without a reference committor
whether a change helped.

## What to look at

| number | where | what it says |
|---|---|---|
| boundary error per slice | `fit.result.boundary_errors` | how well each direction separates the states: the raw material of the fit |
| representation gate | `fit.weights.cond` | whether any combination of the slices separates the states; the solve raises below `1e-6` |
| Dirichlet energy | `fit.weights.dirichlet_energy` | the objective; ranks fits of the same data within one trial space |
| held-out cap and gap | `fit.weights.heldout_cap` with `heldout_cap=True` | the objective read out of sample; ranks trial spaces; the gap is the overfitting guard |
| flux flatness | `rate["flatness"]`, `flux_flatness` | how constant the reactive flux is along `q`; the end-to-end quality score, after a rate |
| bootstrap spread | `bootstrap_weights` | statistical error bars on all of the above (variance only) |

The energy and the cap are read on the same scale: the sample weights are
used verbatim, so two fits are comparable only on the same samples with
the same weights.

## Features and the metric

The Dirichlet form asserts a diffusion tensor in the space the samples live
in, the identity by default. Torsion features `(sin phi, cos phi)` are not
Cartesian coordinates, and the identity is wrong for them: a torsion peaked
near `-60` degrees has `<cos^2> ~ 0.25` against `<sin^2> ~ 0.75`, so its
two features carry diffusion in a ratio near 3 to 1. The pull-back metric
`Mbar = <J M0 J^T>` from the Cartesian Jacobian of the features corrects
this. It enters the Gram matrix through `theta_j^T Mbar theta_k` only, so
the slices are untouched and only the optimal combination moves. Build it
once from the trajectory's coordinates and the torsions' atom quadruples
and pass it as `feature_metric=`:

```python skip
from sliced_committor import AngleSign, sincos_pullback_metric
from sliced_committor.umbrella.mdtraj_metric import dihedral_index_arrays

quads, counts = dihedral_index_arrays(traj, kinds=("phi", "psi", "omega", "chi1", "chi2"))
metric = sincos_pullback_metric(xyz_chunks, quads, angle_sign=AngleSign.IUPAC)
result = compute_sliced_committor(samples, in_A=in_A, in_B=in_B, feature_metric=metric.M)
```

`angle_sign` is required: mdtraj's dihedral sign convention and the IUPAC
one differ by a sign that flips the Jacobian, and no norm-based check can
catch a mix-up. The same metric must go into the rate's change of
variables (`committor_grad_sq(..., metric=)` and
`linear_response_grad_sq(..., metric=)`). Only the shape of the metric
matters; the functions return it normalised to `trace / d = 1`, and the
solve is invariant under its scale.

## States and the halo: `boundary_quantile`

The states enter the fit twice: as the absorbing regions of every 1D solve,
and as the frames over which the basin moments are averaged. With many
uninformative dimensions the projection of a state along a random
direction acquires a halo, a spread of random projections that inflates
its shadow and pushes the absorbing region into the transition region.
`boundary_quantile` moves the inner edge of each shadow from its extreme
sample toward its centroid, and truncates the absorption past the same
quantile:

```python
from sliced_committor import compute_sliced_committor

result = compute_sliced_committor(
    samples, in_A=in_A, in_B=in_B, n_directions=128, boundary_quantile=0.98, direction_batch_size=64
)
```

The paper uses 0.99 on AIB9 (52 features) and 0.98 on villin (350
features), and 1.0 on chignolin and in 2D. Too low a value moves the
boundary into the state, which shows up as a growing boundary error, so
scan it against `boundary_errors` and the held-out cap.

## Directions: `n_directions` and the cone

`n_directions` is the size of the trial space. The cost of the fit is linear
in it, and the half-set filter is what keeps a large `M` from over-fitting
the Gram matrix; the paper uses 256 to 2048.

Uniform directions are the default. When the states are known, a cone
around the Fisher discriminant axis between them, mixed with a uniform
floor, puts the directions where the transition is:

```python
from sliced_committor import DirectionSamplingConfig, compute_sliced_committor

cone = DirectionSamplingConfig(mode="lda", mu=0.5, alpha=0.2, lda_shrinkage=1e-2)
result = compute_sliced_committor(samples, in_A=in_A, in_B=in_B, n_directions=128, direction_sampling=cone)
```

`mu` is the mean cosine to the axis (0 is uniform, the paper uses 0.5 to
0.8), `alpha` the fraction of uniform directions (0.2 in the paper; it is
a coverage floor, so do not set it to zero), and `lda_shrinkage` the ridge
on the within-state covariance. Select `(mu, alpha)` per system by the
held-out cap, as the paper does; the energy alone cannot compare cones,
because they are different trial spaces. The pieces are public, so the
axis can be inspected (`info["mean_mahalanobis"]` is the states'
separation in the whitened metric), replaced by a coordinate of your own,
and the draw reproduced outside the solver and passed back as
`directions=`:

```python
import jax

from sliced_committor import compute_lda_axis, sample_power_spherical_mixture

axis, info = compute_lda_axis(samples, in_A, in_B, shrinkage=1e-2)  # (unit vector, diagnostics)
directions = sample_power_spherical_mixture(
    jax.random.PRNGKey(0), axis, 128, samples.shape[1], mu=0.5, alpha=0.2
)
result = compute_sliced_committor(samples, in_A=in_A, in_B=in_B, directions=directions)
```

## Bins: `n_bins`, `binning_method`, `n_min`, `density_floor`

Each slice is a histogram of the projections. `n_bins` sets its resolution
(200 by default; the paper uses 100 to 2000, more for more samples), and
`binning_method` its placement: `"quantile"` bins hold equal counts, so
they resolve the dense regions and are the default; `"equal_width"` bins
resolve the sparsely sampled barrier of biased data, which is why the
umbrella package and the 2D benchmark use them (quantile bins under-resolve
the barrier there and inflate the rate). `n_min` and `density_floor` floor
the 1D density so that empty bins do not produce infinite free energies.

## Regularisation: `tikhonov`

`tikhonov="halfset_eigen"` is the default and the paper's universal
setting; it has no constant and reads the samples ([theory](theory.md)).
`tikhonov="auto"` is the closed-form scalar ridge
`max(1e-12, 1/sqrt(N_eff)) median(diag G)`, and a float is an absolute
ridge in the units of `G`; both are there for controlled comparisons and
for problems with too few valid directions for a half-set split (below
two the solve falls back to `"auto"` on its own).

```python
from sliced_committor import solve_weights

w_auto = solve_weights(result, tikhonov="auto")
w_ridge = solve_weights(result, tikhonov=1e-6)
```

`solve_weights` raises `RepresentationError` when the moment gap collapses,
which means no combination of the slices separates the states; pass
`raise_on_degenerate=False` to get the degenerate weights back with
`weights.cond` as the flag.

## Sample weights

A biased ensemble enters through `sample_weights`, one weight per frame,
summing to one (the solve refuses anything else). The weights are used
verbatim in the Gram matrix and the density, so the Dirichlet energies of
fits are comparable only for the same weights. The basin moments are
unweighted averages over the labelled frames.

## Choosing settings without a reference

Two label-free numbers compare fits of the same data. The Dirichlet energy
ranks fits within one trial space. The held-out Dirichlet cap ranks trial
spaces: it is the same objective read on folds the solve did not see, and
its gap against the in-sample value is the overfitting guard.

```python
import itertools

from sliced_committor import fit_committor

rows = []
for M, bins in itertools.product((64, 128), (100, 200)):
    _, fit = fit_committor(
        samples, in_A=in_A, in_B=in_B, n_directions=M, n_bins=bins, heldout_cap=True, return_details=True
    )
    cap = fit.weights.heldout_cap
    rows.append((M, bins, fit.weights.dirichlet_energy, cap["cap"], cap["gap"]))
best = min(rows, key=lambda r: r[3])  # the trial space with the lowest held-out cap
```

The cap costs the per-fold moments and ten small solves on top of the fit.
The paper picks the LDA cone's `(mu, alpha)` this way, and the label-free
score of a finished fit is the flatness of its reactive flux
([rates](rates.md)).

## Error bars: the block bootstrap

`bootstrap_weights` resamples contiguous blocks of frames with the 1D basis
held fixed and re-solves the weights under the fit's own `tikhonov` rule.
Blocks never cross `run_ids`, so independent trajectories are never
spliced. It returns the replicate weights, moment gaps, Dirichlet energies
and, with `points=`, committor values:

```python
from sliced_committor import bootstrap_weights, fit_committor

_, fit = fit_committor(samples, in_A=in_A, in_B=in_B, n_directions=64, return_details=True)
boot = bootstrap_weights(fit.result, fit.weights, n_boot=20, block_len=50, points=points)
se_energy = boot.dirichlet_energy.std()
se_q = boot.q.std(axis=0)  # (P,)
```

`block_len` must reach the integrated autocorrelation time of the frames or
the spread is optimistic, and the spread is variance only: when the barrier
is undersampled the bias dominates and the bootstrap cannot see it. A
replicate that leaves a state with fewer than `min_basin_frames` frames is
skipped; `boot.n_ok` counts the rest.

## Memory and scale

The solver holds the `(M, N)` array of projected samples and a few working
arrays of that size. `direction_batch_size` bounds the transient memory of
the slice construction, and quantile binning subsamples the projections
above 20,000 frames to find its bin edges. The committor callable and the
rate quantities evaluate in batches of points on their own, so a large
evaluation costs no more memory than the fit. The same code runs on a GPU;
[reproducibility](reproducibility.md) says what moves between devices.

## Determinism

`seed` fixes the direction draw, and nothing else is random: the same
samples, labels and settings reproduce the committor to the bit on one
machine with one JAX build. Across JAX builds, machines and devices the
committor moves by about `1e-5` and individual weights by up to `1e-2`
relative, from rounding that the Gram solve amplifies;
[reproducibility](reproducibility.md) has the contract and the mechanism.
