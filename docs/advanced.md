# Advanced settings

## The direction cone

Uniform directions on the sphere are the default. When the basins are known
the Fisher discriminant axis between them is a better centre for the
directions, and the paper draws a power-spherical cone around it mixed with
a uniform floor:

```python
from sliced_committor import DirectionSamplingConfig, compute_sliced_committor

cone = DirectionSamplingConfig(mode="lda", mu=0.5, alpha=0.2, lda_shrinkage=1e-2)
result = compute_sliced_committor(samples, in_A=in_A, in_B=in_B, n_directions=128, direction_sampling=cone)
```

`mu` is the mean cosine to the axis (0 is uniform), `alpha` the fraction of
uniform directions, and the paper selects `(mu, alpha)` per system by the
held-out cap. At large `M` build the cone outside the solver and pass it as
`directions=`: folding the sampler into the jitted solve blows JAX's
constant budget.

```python
import jax

from sliced_committor import compute_lda_axis, sample_power_spherical_mixture

axis, _ = compute_lda_axis(samples, in_A, in_B, shrinkage=1e-2)  # (unit vector, diagnostics)
directions = sample_power_spherical_mixture(
    jax.random.PRNGKey(0), axis[None, :], 128, samples.shape[1], mu=0.5, alpha=0.2
)
result = compute_sliced_committor(samples, in_A=in_A, in_B=in_B, n_directions=128, directions=directions)
```

## The feature-space metric

Torsion features `(sin phi, cos phi)` are not Cartesian coordinates, and
the Dirichlet form in that space asserts a diffusion tensor that is the
identity there. The pull-back metric `Mbar = <J M0 J^T>` from the Cartesian
Jacobian of the features corrects this; it enters the Gram matrix through
`theta_j^T Mbar theta_k` only, so the slices are untouched and only the
optimal combination moves. Build it once from the trajectory's coordinates
and the torsions' atom quadruples and pass it as `feature_metric=`:

```python skip
from sliced_committor import AngleSign, sincos_pullback_metric
from sliced_committor.umbrella.mdtraj_metric import dihedral_index_arrays

quads, counts = dihedral_index_arrays(traj, kinds=("phi", "psi", "omega", "chi1", "chi2"))
metric = sincos_pullback_metric(xyz_chunks, quads, angle_sign=AngleSign.IUPAC)
result = compute_sliced_committor(samples, in_A=in_A, in_B=in_B, feature_metric=metric.M)
```

`angle_sign` is required: mdtraj's dihedral sign convention and the IUPAC
one differ by a sign that flips the Jacobian. The same metric must go into
the rate's change of variables (`committor_grad_sq(..., metric=)` and
`linear_response_grad_sq(..., metric=)`).

## Regularisation

`tikhonov="halfset_eigen"` is the default and the paper's universal setting.
`tikhonov="auto"` is the closed-form scalar ridge
`max(1e-12, 1/sqrt(N_eff)) median(diag G)`, and a float is an absolute
ridge in the units of `G`; both are there for controlled comparisons and
for problems with too few valid directions for a half-set split (below two
the solve falls back to `"auto"` on its own).

```python
from sliced_committor import solve_weights

w_auto = solve_weights(result, tikhonov="auto")
w_ridge = solve_weights(result, tikhonov=1e-6)
```

`solve_weights` raises `RepresentationError` when the moment gap collapses,
which means no combination of the slices separates the basins; pass
`raise_on_degenerate=False` to get the degenerate weights back with
`weights.cond` as the flag.

## Halo mitigation and memory

`boundary_quantile` moves the absorbing boundaries of the 1D solves from
the extreme projected basin sample toward the centroid, for high-dimensional
features whose basin shadows inflate ([theory.md](theory.md)). The solver
holds an `(M, N)` array of projected samples: `direction_batch_size` bounds
the transient memory of the slice construction, and quantile binning
subsamples the projections above 20,000 frames to find its bin edges.

```python
result = compute_sliced_committor(
    samples, in_A=in_A, in_B=in_B, n_directions=128, boundary_quantile=0.98, direction_batch_size=64
)
```

## Uncertainty

`bootstrap_weights` resamples contiguous blocks of frames with the 1D basis
held fixed and re-solves the weights under the fit's own `tikhonov` rule.
Blocks never cross `run_ids`, so independent trajectories are never spliced.
It returns the replicate weights, moment gaps, Dirichlet energies and,
with `points=`, committor values:

```python
from sliced_committor import bootstrap_weights, solve_weights

weights = solve_weights(result)
boot = bootstrap_weights(result, weights, n_boot=20, block_len=50, points=points)
se_q = boot.q.std(axis=0)
```

The spread is variance only. When the barrier is undersampled the bias
dominates and the bootstrap cannot see it; `block_len` must reach the
integrated autocorrelation time of the frames.
