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

## The committor in one call

```python
import jax

jax.config.update("jax_enable_x64", True)

from sliced_committor import fit_committor

# samples: (N, dim) equilibrium configurations
# in_A, in_B: (N,) bool basin labels, defined however you like
q = fit_committor(samples, in_A=in_A, in_B=in_B, n_directions=256, n_bins=200, seed=0)
q_vals = q(points)  # (P,) committor values, clipped to [0, 1]
q_snapped = q(samples, in_A=in_A, in_B=in_B)  # exactly 0 on A and 1 on B
```

The callable is a pure function of `x`: `jax.grad(q)` differentiates it, and
`committor_gradient(q, points)` returns the gradients at many points.
`rescale_transition(q_vals, in_A, in_B)` is the paper's affine post-processor
for figures: it stretches the transition region so that the boundary values
are exactly 0 and 1.

## Step by step

`fit_committor` is three calls:

```python
from sliced_committor import build_committor, compute_sliced_committor, solve_weights

result = compute_sliced_committor(samples, in_A=in_A, in_B=in_B, n_directions=256, n_bins=200, seed=0)
weights = solve_weights(result)  # tikhonov="halfset_eigen"
q = build_committor(result, weights)
```

`result` (a `SlicedCommittorResult`) is the slice basis: the directions,
the 1D grids, the 1D free energies and committors, the projected samples,
and a `valid_mask` marking the slices whose 1D solve is finite.
`result.summary()` prints it. `weights` (a `Weights`) is the solved
combination and what the solve knows about it:

```python
print(weights.dirichlet_energy)  # w^T G w, the variational objective: lower is better
print(weights.moment_gap, weights.cond)  # the basin-moment gap and the representation gate
```

## Choosing settings without a reference

Two label-free numbers compare fits of the same data. The Dirichlet energy
ranks fits within one trial space. The held-out Dirichlet cap ranks trial
spaces: it is the same objective read on folds the solve did not see, and
its gap against the in-sample value is the overfitting guard.

```python
import itertools

rows = []
for M, bins in itertools.product((64, 128), (100, 200)):
    _, fit = fit_committor(
        samples, in_A=in_A, in_B=in_B, n_directions=M, n_bins=bins, heldout_cap=True, return_details=True
    )
    cap = fit.weights.heldout_cap
    rows.append((M, bins, fit.dirichlet_energy, cap["cap"], cap["gap"]))
best = min(rows, key=lambda r: r[3])  # the trial space with the lowest held-out cap
```

The paper picks the LDA cone's `(mu, alpha)` this way. Error bars come from
a block bootstrap over frames with the basis held fixed:

```python
from sliced_committor import bootstrap_weights

_, fit = fit_committor(samples, in_A=in_A, in_B=in_B, n_directions=64, return_details=True)
boot = bootstrap_weights(fit.result, fit.weights, n_boot=20, block_len=50)
se_energy = boot.dirichlet_energy.std()
```

`block_len` must reach the integrated autocorrelation time of the frames, and
the spread is variance only: it cannot see the bias of an undersampled
barrier. See [advanced.md](advanced.md) for the direction cone, the
feature-space metric and the memory knobs, and [rates.md](rates.md) for the
rate.
