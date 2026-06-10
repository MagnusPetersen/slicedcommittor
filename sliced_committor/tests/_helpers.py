"""Shared test helpers: a parametrizable two-basin sampler and named tolerances.

The suite already shares ``_equilibrium_samples`` via a relative import from
:mod:`test_validation_2d_double_well`; this module collects the *other* widely
duplicated setup, the Gaussian two-basin sampler, so its geometry lives in one
place. Import with the same relative style, e.g.::

    from ._helpers import two_basin_samples, TOL_SOLVER
"""

import jax.numpy as jnp
import numpy as np

# The one tolerance with genuine cross-file reuse: the "solver satisfied its
# constraint / normalisation" check (BMC residuals, full-Gram / diagonal
# sum-to-1). Naming it keeps those checks consistent. Single-use, locally
# meaningful tolerances (clip bounds, symmetry, exact reproducibility) stay
# inline at the assertion site, where the literal is the clearest documentation.
TOL_SOLVER = 1e-6


def two_basin_samples(n=500, *, dim=2, seed=0, radius=0.7, sep=2.0):
    """Gaussian samples with two spherical basins at ``±sep`` along axis 0.

    A standard-normal cloud of ``n`` points in ``dim`` dimensions; ``in_A`` /
    ``in_B`` are the points within ``radius`` of ``-sep·e_0`` / ``+sep·e_0``
    (made disjoint). Parametrized so the previously duplicated per-file samplers
    (2-D and 3-D, radius 0.6 / 0.7, n 200-600) all reduce to one call.

    Returns:
        ``(samples, in_A, in_B)`` with ``samples`` shape ``(n, dim)`` and the
        masks boolean ``(n,)``.
    """
    rng = np.random.default_rng(seed)
    samples = jnp.asarray(rng.standard_normal((n, dim)))
    center = jnp.zeros(dim).at[0].set(sep)
    in_A = jnp.linalg.norm(samples + center, axis=1) < radius
    in_B = jnp.linalg.norm(samples - center, axis=1) < radius
    in_A = in_A & ~in_B
    return samples, in_A, in_B
