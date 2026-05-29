"""Unit tests for the diagonal RD-corrected weight solver."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

import sliced_committor as sc


def _two_basin_samples(n=600, seed=0):
    rng = np.random.default_rng(seed)
    samples = jnp.asarray(rng.standard_normal((n, 2)))
    in_A = jnp.linalg.norm(samples - jnp.asarray([-2.0, 0.0]), axis=1) < 0.7
    in_B = jnp.linalg.norm(samples - jnp.asarray([2.0, 0.0]), axis=1) < 0.7
    in_A = in_A & ~in_B
    return samples, in_A, in_B


def test_diagonal_weights_normalised_and_nonnegative():
    samples, in_A, in_B = _two_basin_samples()
    result = sc.compute_sliced_committor(samples, in_A=in_A, in_B=in_B, n_directions=32, seed=0)
    weights = sc.compute_weights_multi(result, [sc.corrected_dirichlet_inv_rd])[
        "corrected_dirichlet_inv_rd"
    ]
    w = np.asarray(weights)
    # Diagonal RD weights sum to 1 (softmax of log-weights).
    np.testing.assert_allclose(w.sum(), 1.0, atol=1e-6)
    # And carry no negative mass (the (1-eps)+ correction guarantees this).
    assert (w >= -1e-12).all(), f"negative weights: min = {w.min()}"


def test_diagonal_weights_concentrate_on_aligned_direction():
    """A direction aligned with the A->B axis should win most of the weight."""
    samples, in_A, in_B = _two_basin_samples()
    # Force one direction to be the perfect A->B axis; the rest random.
    M = 8
    rng_jax = jax.random.PRNGKey(0)
    rand_dirs = jax.random.normal(rng_jax, shape=(M - 1, 2))
    rand_dirs = rand_dirs / jnp.linalg.norm(rand_dirs, axis=1, keepdims=True)
    aligned = jnp.asarray([[1.0, 0.0]])
    directions = jnp.concatenate([aligned, rand_dirs], axis=0)

    result = sc.compute_sliced_committor(
        samples,
        in_A=in_A,
        in_B=in_B,
        n_directions=M,
        directions=directions,
        seed=0,
    )
    w = np.asarray(
        sc.compute_weights_multi(result, [sc.corrected_dirichlet_inv_rd])[
            "corrected_dirichlet_inv_rd"
        ]
    )
    # The aligned axis is index 0 and should carry meaningful mass.
    assert w[0] > 1.0 / M, (
        f"aligned direction weight {w[0]:.3f} smaller than uniform 1/M={1 / M:.3f}"
    )
