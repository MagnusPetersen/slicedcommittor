# Quickstart

## Install

```bash
pip install sliced-committor
# or, from the monorepo:
pip install -e ./lib
```

Optional extras:

```bash
pip install sliced-committor[examples]   # matplotlib + jupyter
pip install sliced-committor[docs]       # sphinx + furo + myst-parser
pip install sliced-committor[dev]        # all of the above + ruff + pytest
```

Hard dependencies are `jax >= 0.4.20` and `numpy >= 1.24`. JAX flavour
(`jax[cpu]`, `jax[cuda12]`, …) is up to you; the library does not pin one.

## Minimal example

```python
import jax
jax.config.update("jax_enable_x64", True)   # required for EBMC / PESB / BMC

import jax.numpy as jnp
from sliced_committor import fit_committor

# `samples` is your (N, dim) feature array.
# `in_A`, `in_B` are (N,) bool masks indicating basin membership.
q = fit_committor(
    samples, in_A=in_A, in_B=in_B, weights="ebmc",
    n_directions=256, n_bins=200, seed=0,
)
q_vals = q(points)   # q is a callable
```

For more control, fit the `SlicedCommittorResult` once and `build_committor`
explicitly:

```python
from sliced_committor import (
    compute_sliced_committor,
    compute_enriched_basin_moment_weights,
    build_committor,
)

result = compute_sliced_committor(
    samples, in_A=in_A, in_B=in_B, n_directions=256, n_bins=200, seed=0,
)
ebmc = compute_enriched_basin_moment_weights(result, samples)
q = build_committor(result, ebmc)
q_vals = q(points)
```

The `result` carries everything downstream solvers need; the EBMC dict
auto-routes through `build_committor`'s centered-basis path. See
[theory.md](theory.md) for what each knob means and [recipes.md](recipes.md) for
common variations (custom direction samplers, alternative epsilon
estimators).

## Inspecting the result

```python
print(result.summary())
```

prints a multi-line block with the number of directions, valid slices,
mean boundary error, mean Dirichlet energy per slice, and whether
projected samples are stored.

For per-slice debugging, use `why_masked`:

```python
from sliced_committor import why_masked
for j, ok in enumerate(result.valid_mask):
    if not ok:
        print(j, why_masked(result, j))
```

For Gram-solver diagnostics:

```python
from sliced_committor import summarize_gram_diagnostics
print(summarize_gram_diagnostics(ebmc))
```

## Choosing settings (label-free)

`fit_committor(..., return_details=True)` returns `(q, fit)`, where
`fit.dirichlet_energy` is the variational objective `𝓓[q̂]` of the fit — a
label-free, ground-truth-free quality score (lower is closer to the true
committor). Sweep a grid and keep the best by that score with `sweep_committor`:

```python
from sliced_committor import sweep_committor

res = sweep_committor(
    samples, in_A=in_A, in_B=in_B,
    grid={"n_directions": [128, 256], "weights": ["ebmc", "full_gram"]},
    n_bins=200,                       # held fixed across the sweep
)
print(res.summary())                 # one row per combination; best marked *
q = res.best_committor
```

The energy is comparable only within the Gram-family solvers (`ebmc` / `pesb` /
`bmc` / `full_gram`). See [recipes.md](recipes.md) for more.
