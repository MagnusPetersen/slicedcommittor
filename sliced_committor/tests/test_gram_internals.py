"""Unit tests for the lowest-level math kernels in ``gram.py``.

These functions are shared by every Gram-based weight solver
(full-Gram, BMC, EBMC, PESB-EBMC). Errors here cascade silently into
every downstream estimator.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

import sliced_committor as sc
from sliced_committor.core.gram import (
    _assemble_gram_matrix,
    _compute_derivative_matrix,
    compute_shared_gram_diagnostics,
)

from ._helpers import two_basin_samples


def _build_ctx(n=200, M=12, seed=0):
    samples, in_A, in_B = two_basin_samples(n, seed=seed)
    result = sc.compute_sliced_committor(samples, in_A=in_A, in_B=in_B, n_directions=M, seed=seed)
    return sc.make_weighting_context(result), samples


def test_derivative_matrix_shape():
    ctx, samples = _build_ctx()
    projected = ctx.directions @ samples.T
    F = _compute_derivative_matrix(ctx, projected)
    assert F.shape == projected.shape  # (M, N)
    assert jnp.all(jnp.isfinite(F))


def test_derivative_matrix_linear_q_is_constant_slope():
    """If ``q(s) = a*s + b`` on a slice, the derivative matrix should pick up
    the slope ``a`` at every sample for that slice (modulo bin boundaries).
    """
    ctx, samples = _build_ctx(M=4)
    # Manually replace one slice's q-grid with a perfect linear ramp.
    s = jnp.asarray(ctx.slice_coords)
    new_q = jnp.copy(ctx.committors_1d)
    slope = 0.3
    linear_row = slope * (s[0] - s[0, 0])
    new_q = new_q.at[0].set(linear_row)
    ctx2 = ctx._replace(committors_1d=new_q)
    projected = ctx2.directions @ samples.T
    F = _compute_derivative_matrix(ctx2, projected)
    # All bins on slice 0 should have slope == 0.3 (piecewise-constant slope).
    # We pick samples that fall strictly inside the grid (not the edges).
    interior_mask = (projected[0] > s[0, 1]) & (projected[0] < s[0, -2])
    np.testing.assert_allclose(np.asarray(F[0, interior_mask]), slope, atol=1e-10)


def test_assemble_gram_matrix_is_symmetric_psd():
    ctx, samples = _build_ctx(M=10)
    projected = ctx.directions @ samples.T
    F = _compute_derivative_matrix(ctx, projected)
    W = jnp.ones(samples.shape[0]) / samples.shape[0]
    cos = ctx.directions @ ctx.directions.T
    G = _assemble_gram_matrix(F, W, cos)
    # Symmetric.
    np.testing.assert_allclose(np.asarray(G), np.asarray(G.T), atol=1e-10)
    # PSD up to symmetrisation noise.
    eigs = np.linalg.eigvalsh(np.asarray(G + G.T) / 2.0)
    assert eigs.min() >= -1e-8, f"min eig = {eigs.min()}"


def test_shared_gram_diagnostics_populates_keys():
    ctx, samples = _build_ctx(M=8)
    projected = ctx.directions @ samples.T
    F = _compute_derivative_matrix(ctx, projected)
    W = jnp.ones(samples.shape[0]) / samples.shape[0]
    cos = ctx.directions @ ctx.directions.T
    G = _assemble_gram_matrix(F, W, cos)

    # Fake weight vector to populate the negative-weight diagnostic.
    result_dict = {"w": jnp.asarray(np.array([1.0, -0.2, 0.3, 0.4, 0.1, 0.0, 0.2, 0.2]))}
    compute_shared_gram_diagnostics(result_dict, G, ctx.valid_mask, ctx=ctx)
    for key in (
        "n_negative_weights",
        "negative_weight_mass",
        "off_diagonal_magnitude",
        "diagonal_sanity",
    ):
        assert key in result_dict, f"missing diagnostic key {key}"
    assert result_dict["n_negative_weights"] == 1
    assert result_dict["off_diagonal_magnitude"] >= 0
