# Quickstart

## Install

```bash
# from a clone of the repo:
pip install -e .

# or directly from GitHub (requires read access):
pip install git+ssh://git@github.com/magnuspetersen/slicedcommittor.git
```

Optional extras:

```bash
pip install -e .[examples]   # matplotlib + jupyter
pip install -e .[docs]       # sphinx + furo + myst-parser
pip install -e .[dev]        # all of the above + ruff + pytest
```

Hard dependencies are `jax >= 0.4.20` and `numpy >= 1.24`. JAX flavour
(`jax[cpu]`, `jax[cuda12]`, …) is up to you; the library does not pin one.

## Minimal example

```python
import jax
jax.config.update("jax_enable_x64", True)   # required for EBMC / PESB / BMC

import jax.numpy as jnp
from sliced_committor import (
    compute_sliced_committor,
    compute_enriched_basin_moment_weights,
    evaluate_committor,
)

# `samples` is your (N, dim) feature array.
# `in_A`, `in_B` are (N,) bool masks indicating basin membership.
result = compute_sliced_committor(
    samples, in_A=in_A, in_B=in_B, n_directions=256, n_bins=200, seed=0,
)
ebmc = compute_enriched_basin_moment_weights(result, samples)
q = evaluate_committor(result, points, ebmc)
```

The `result` carries everything downstream solvers need; the EBMC dict
auto-routes through `evaluate_committor`'s centered-basis path. See
[theory.md](theory.md) for what each knob means and [recipes.md](recipes.md) for
common variations (custom direction samplers, alternative epsilon
estimators, calibration).

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
