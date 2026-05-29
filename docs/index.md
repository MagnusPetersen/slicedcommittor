# slicedcommittor

Sample-based committor function approximation via 1D projections. Given
samples and basin labels, the library returns the committor `q(x)`: the
probability that a stochastic trajectory starting at `x` reaches state B
before state A.

```{toctree}
:maxdepth: 2
:caption: Contents

quickstart
theory
recipes
reproducibility
api/index
changelog
```

## At a glance

```python
import jax
jax.config.update("jax_enable_x64", True)

from sliced_committor import (
    compute_sliced_committor,
    compute_enriched_basin_moment_weights,
    evaluate_committor,
)

result = compute_sliced_committor(
    samples, in_A=in_A, in_B=in_B, n_directions=256,
)
ebmc = compute_enriched_basin_moment_weights(result, samples)
q = evaluate_committor(result, points, ebmc)
```

See [quickstart.md](quickstart.md) for the full path. The [theory primer](theory.md)
walks through how the method works and what each design knob controls.
