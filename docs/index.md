# sliced-committor

Sample-based committor approximation via 1D projections, and the reaction
rates that follow from it. Given equilibrium samples and two state labels
the library returns the committor `q(x)`, the probability that a trajectory
started at `x` reaches state B before state A, as a callable function; the
rates package turns it into `k_AB` and `k_BA`.

```python
import jax

jax.config.update("jax_enable_x64", True)

from sliced_committor import fit_committor

q = fit_committor(samples, in_A=in_A, in_B=in_B, n_directions=256)
q_vals = q(points)  # q is a callable
```

## Read in this order

1. [Quickstart](quickstart.md): what the library needs from you and why,
   the one call, what comes back and how to read it.
2. [Theory](theory.md): the model and its assumption, the sliced ansatz,
   the weight solve, the regularisation, and where each diagnostic comes
   from.
3. [Settings, diagnostics and uncertainty](settings.md): every setting,
   what it controls, how to choose it without a reference committor, and
   error bars.
4. [Rates](rates.md): from the committor to a rate, by three routes
   depending on the data you have.
5. [Umbrella sampling](umbrella.md): the dataset contract, reweighting, and
   the rate bundle, also step by step.
6. [Reproducibility](reproducibility.md) and
   [design decisions](design_decisions.md): what is pinned to the bit, and
   what was tried and is not here.

```{toctree}
:maxdepth: 2
:caption: Contents

quickstart
theory
settings
rates
umbrella
design_decisions
reproducibility
api/index
changelog
```
