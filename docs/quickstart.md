# Quickstart

## Install

```bash
pip install sliced-committor              # jax, numpy, scipy
pip install sliced-committor[umbrella]    # + mdtraj, pymbar (umbrella-sampling data)
pip install sliced-committor[examples]    # + matplotlib (the example scripts)
pip install sliced-committor[dev]         # everything, plus ruff, pytest, sphinx
```

The JAX build (`jax[cpu]`, `jax[cuda12]`, ...) is yours to choose. Float64
is required: the weight solve inverts a Gram matrix whose condition number
reaches `1e12`, and the library raises if `jax_enable_x64` is off.

## What the library needs from you

**Samples: an `(N, dim)` float64 array from the equilibrium ensemble.**
The committor is the minimiser of the Dirichlet form over the equilibrium
density, so the density is what the fit reads, and the samples must
represent it: a long unbiased trajectory, or a biased ensemble with its
reweighting weights. The feature space is yours: Cartesian coordinates,
torsion sines and cosines, distances, anything that resolves the
transition. The cost is linear in `dim`, so a full torsion representation
is affordable; the price of many uninformative dimensions is a halo around
the projected states, which `boundary_quantile` treats
([settings](settings.md)).

**States: two boolean masks, `in_A` and `in_B`.** The frames inside the
two metastable states. The masks must be disjoint and both populated: the
solve constrains the committor to average 0 over A and 1 over B, and those
averages need frames behind them. Define each state as the core of its
basin, a ball or an ellipsoid in a few coordinates that identify it, and
leave everything else unlabelled; the committor between the states is what
the fit produces.

**Weights, when the ensemble is biased.** `sample_weights`, one per frame,
summing to one. MBAR or WHAM weights of umbrella-sampling data come from
`sliced_committor.umbrella.reweight`.

**Time order.** The default regularisation deals the frames into
contiguous, basin-stratified blocks and compares Gram matrices built from
alternate blocks; the block bootstrap resamples contiguous blocks. Both
rely on consecutive frames being consecutive in time. Concatenate
independent trajectories end to end, mark the joins with `run_ids` where an
estimator takes them, and never shuffle.

**Float64.** Enable `jax_enable_x64` before building any array.

## The committor in one call

```python
import jax

jax.config.update("jax_enable_x64", True)

from sliced_committor import committor_gradient, fit_committor, rescale_transition

q, fit = fit_committor(
    samples, in_A=in_A, in_B=in_B, n_directions=256, n_bins=200, seed=0, return_details=True
)
q_vals = q(points)  # (P,) committor values, clipped to [0, 1]
q_snapped = q(samples, in_A=in_A, in_B=in_B)  # exactly 0 on A and 1 on B
grad_q = committor_gradient(q, points)  # (P, dim)
q_figure = rescale_transition(q_snapped, in_A, in_B)  # the paper's affine post-processor for figures
```

The callable is a pure function of `x`: `jax.grad(q)` differentiates it,
and it holds only the slice basis and the weights, not the samples it was
fitted on. Passing the state masks with the points snaps the values to the
boundary conditions. `rescale_transition` stretches the transition region
of an evaluated batch so that its extremes are exactly 0 and 1; it is a
property of the batch, which is why it is a function and not an option of
the callable.

## What came back

`fit` is a `CommittorFit` with three fields: the callable, the slice basis
and the weights.

`fit.result`, a `SlicedCommittorResult`, is the slice basis: the
`directions (M, dim)`, the 1D grids `slice_coords`, the 1D free energies
`free_energies` (as `-log rho`), the 1D committors `committors_1d`, the
projected samples `projected_samples (M, N)`, and per slice the
`boundary_errors` and the `valid_mask` (True where the 1D solve is finite).
`fit.result.summary()` prints the essentials.

`fit.weights`, a `Weights`, is the solved combination and what the solve
knows about it: the weights `w (M,)` and the bias `c`, the
`dirichlet_energy`, the `moment_gap` and the representation gate `cond`,
the `heldout_cap` when requested, and a `diagnostics` dict with the basin
moments and the constraint checks.

```python
print(fit.result.summary())
print(fit.weights.dirichlet_energy)  # w^T G w, the variational objective: lower is better
print(fit.weights.moment_gap, fit.weights.cond)  # the basin-moment gap and the representation gate
print(fit.weights.diagnostics["mu_A_check"], fit.weights.diagnostics["mu_B_check"])  # ~0 and ~1
```

## Reading the fit

Look at three things, in this order.

1. **The boundary errors**, `fit.result.boundary_errors`. Each slice's 1D
   committor misses its boundary values by `eps_j = <q_j>_A + <1 - q_j>_B`.
   A slice with a small error is one whose direction separates the states;
   the aggregate constraint repairs the rest. When every slice is bad, the
   features do not resolve the states or the states overlap.
2. **The representation gate**, `fit.weights.cond`. It is the moment gap
   relative to the basin moments' own scale, and it collapses when no
   combination of the slices separates the states; the solve then raises
   `RepresentationError` (pass `raise_on_degenerate=False` to get the
   degenerate weights back and inspect them).
3. **The Dirichlet energy**, `fit.weights.dirichlet_energy`. The
   variational objective of the committor that was built: lower is closer
   to the true committor. It compares fits of the same data within one
   trial space (the same directions, bins and metric).
   [Settings](settings.md) explains how the held-out cap compares trial
   spaces and how the block bootstrap puts an error bar on any of these.

## Step by step

`fit_committor` is three calls, and taking them one at a time lets you
reuse the slice basis: solve the weights under another rule, bootstrap
them, or inspect the slices.

```python
from sliced_committor import build_committor, compute_sliced_committor, solve_weights

result = compute_sliced_committor(samples, in_A=in_A, in_B=in_B, n_directions=256, n_bins=200, seed=0)
weights = solve_weights(result)  # tikhonov="halfset_eigen"
q = build_committor(result, weights)
```

## Next

[Theory](theory.md) explains what each of these objects is and why the fit
needs no dynamics; [settings](settings.md) goes through every setting and
the label-free ways to choose it; [rates](rates.md) turns `q` into a rate.
