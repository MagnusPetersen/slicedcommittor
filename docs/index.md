# sliced-committor

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
umbrella_sampling
reproducibility
api/index
changelog
```

## At a glance

```python
import jax
jax.config.update("jax_enable_x64", True)

from sliced_committor import fit_committor

q = fit_committor(samples, in_A=in_A, in_B=in_B, weights="ebmc", n_directions=256)
q_vals = q(points)   # q is a callable
```

See [quickstart.md](quickstart.md) for the full path. The [theory primer](theory.md)
walks through how the method works and what each design knob controls.
