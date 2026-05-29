"""RMSE-vs-n_directions convergence test for the 2D double-well.

This is the paper's central scaling claim: estimator error decreases
monotonically with the number of sampled directions M. The test asserts
the trend on the same 2D double-well + Jacobi PDE setup used in
``test_validation_2d_double_well``.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

import sliced_committor as sc

# Re-use the PDE solver and sampler from the sibling validation module.
# tests/__init__.py exists so this relative import resolves under pytest.
from .test_validation_2d_double_well import (
    _equilibrium_samples,
    _interpolate_pde,
    _solve_committor_2d_jacobi,
)


@pytest.fixture(scope="module")
def pde_and_samples():
    pde = _solve_committor_2d_jacobi(n_grid=64)
    samples, in_A, in_B = _equilibrium_samples(n_samples=2500, seed=1)
    return pde, samples, in_A, in_B


def test_rmse_decreases_with_n_directions(pde_and_samples):
    (q_grid, X, Y, _, _), samples, in_A, in_B = pde_and_samples

    eval_xs = np.linspace(-1.3, 1.3, 21)
    eval_ys = np.linspace(-0.8, 0.8, 13)
    EX, EY = np.meshgrid(eval_xs, eval_ys, indexing="xy")
    pts = np.stack([EX.ravel(), EY.ravel()], axis=1)
    interior_mask = ~(
        ((pts[:, 0] - (-1.0)) ** 2 + pts[:, 1] ** 2 < 0.30**2)
        | ((pts[:, 0] - 1.0) ** 2 + pts[:, 1] ** 2 < 0.30**2)
    )
    pts_int = pts[interior_mask]
    q_true = _interpolate_pde(q_grid, X, Y, pts_int)

    rmses = []
    for M in (16, 64, 256):
        result = sc.compute_sliced_committor(
            jnp.asarray(samples),
            in_A=jnp.asarray(in_A),
            in_B=jnp.asarray(in_B),
            n_directions=M,
            n_bins=120,
            seed=2,
        )
        ebmc = sc.compute_enriched_basin_moment_weights(result, jnp.asarray(samples))
        q_pred = np.asarray(sc.evaluate_committor(result, jnp.asarray(pts_int), ebmc))
        rmses.append(float(np.sqrt(np.mean((q_pred - q_true) ** 2))))

    # Strict monotone decrease across the three rungs we test. We allow a
    # tiny slack to absorb finite-sample noise: each step must shrink by
    # at least 5% of the previous error.
    assert rmses[1] < 0.97 * rmses[0], (
        f"RMSE did not decrease M=16->64: {rmses[0]:.3f} -> {rmses[1]:.3f}"
    )
    assert rmses[2] < 0.97 * rmses[1], (
        f"RMSE did not decrease M=64->256: {rmses[1]:.3f} -> {rmses[2]:.3f}"
    )
    # And the largest M must give a competitive absolute error.
    assert rmses[-1] < 0.1, f"final RMSE too large: {rmses[-1]:.3f}"
