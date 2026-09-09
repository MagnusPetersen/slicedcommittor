# sliced-committor

Sample-based committor approximation via 1D projections, and the reaction
rates that follow from it. Given equilibrium samples and two basin labels the
library returns the committor `q(x)`, the probability that a trajectory
started at `x` reaches state B before state A, as a callable function; the
rates package turns it into `k_AB` and `k_BA`.

```{toctree}
:maxdepth: 2
:caption: Contents

quickstart
theory
rates
umbrella
advanced
design_decisions
reproducibility
api/index
changelog
```

## At a glance

```python
import jax

jax.config.update("jax_enable_x64", True)

from sliced_committor import fit_committor

q = fit_committor(samples, in_A=in_A, in_B=in_B, n_directions=256)
q_vals = q(points)  # q is a callable
```

[quickstart.md](quickstart.md) is the full path; the [theory primer](theory.md)
explains what each setting controls; [rates.md](rates.md) goes from the
committor to a rate.
