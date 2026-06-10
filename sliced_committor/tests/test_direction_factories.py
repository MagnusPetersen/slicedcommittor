"""Smoke tests for the geometric direction-sampling factories.

Covers shape contracts, unit-norm guarantees, eigenvalue ordering, the
dispatcher's geometric-mode branches, and end-to-end use with EBMC.

Kernel-based factories (diffusion maps, RBF kernel PCA via RFF) are
intentionally not in the library and not exercised here.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from sliced_committor import (
    DirectionSamplingConfig,
    build_committor,
    compute_enriched_basin_moment_weights,
    compute_sliced_committor,
    directions_gcpca,
    directions_pca,
    gcpca_basis,
    pca_basis,
    sample_directions,
)


@pytest.fixture(scope="module")
def cluster_samples():
    """400 samples in 12 dimensions, separated into two clusters along axis 0."""
    key = jax.random.PRNGKey(0)
    N, dim = 400, 12
    X = jax.random.normal(key, (N, dim)) * 0.3
    X = X.at[: N // 2, 0].add(-1.0)
    X = X.at[N // 2 :, 0].add(+1.0)
    in_A = jnp.arange(N) < 50
    in_B = jnp.arange(N) >= N - 50
    return X, in_A, in_B


def _is_unit_norm(axes, tol=1e-6):
    norms = jnp.linalg.norm(axes, axis=-1)
    return bool(jnp.all(jnp.abs(norms - 1.0) < tol))


def test_pca_basis_shape_and_norms(cluster_samples):
    X, _, _ = cluster_samples
    axes, eigvals, info = pca_basis(X, K=3)
    assert axes.shape == (3, X.shape[1])
    assert eigvals.shape == (3,)
    assert _is_unit_norm(axes)
    assert info["method"] == "pca"
    assert info["variant"] == "top"
    # PCA eigenvalues are descending and non-negative.
    np.testing.assert_array_less(-1e-9, np.asarray(eigvals))
    assert float(eigvals[0]) >= float(eigvals[-1])


def test_pca_bottom_variant_smallest_first(cluster_samples):
    X, _, _ = cluster_samples
    _, top_evals, _ = pca_basis(X, K=3, variant="top")
    _, bot_evals, _ = pca_basis(X, K=3, variant="bottom")
    # Bottom variant returns smallest variances; should be <= top variance #3.
    assert float(bot_evals[0]) <= float(top_evals[-1]) + 1e-9


def test_directions_pca_wrapper(cluster_samples):
    X, _, _ = cluster_samples
    axes = directions_pca(X, K=2)
    assert axes.shape == (2, X.shape[1])
    assert _is_unit_norm(axes)


def test_gcpca_basis_shape_and_eigvals(cluster_samples):
    X, in_A, in_B = cluster_samples
    axes, eigvals, info = gcpca_basis(X, in_A, in_B, K=3, background="bulk")
    assert axes.shape == (3, X.shape[1])
    assert eigvals.shape == (3,)
    assert _is_unit_norm(axes)
    # Rayleigh quotient of the symmetric pencil lives in [-1, 1].
    assert bool(jnp.all(jnp.abs(eigvals) <= 1.0 + 1e-6))
    assert info["background"] == "bulk"
    assert info["n_target"] == int(jnp.sum(in_A | in_B))


def test_gcpca_background_all_matches_full_sample(cluster_samples):
    X, in_A, in_B = cluster_samples
    _, _, info = gcpca_basis(X, in_A, in_B, K=2, background="all")
    assert info["n_background"] == X.shape[0]


def test_directions_gcpca_wrapper(cluster_samples):
    X, in_A, in_B = cluster_samples
    axes = directions_gcpca(X, in_A, in_B, K=2)
    assert axes.shape == (2, X.shape[1])
    assert _is_unit_norm(axes)


@pytest.mark.parametrize("mode", ["uniform", "lda", "pca", "gcpca"])
def test_sample_directions_returns_unit_norm(cluster_samples, mode):
    X, in_A, in_B = cluster_samples
    cfg = DirectionSamplingConfig(mode=mode, n_bias_axes=3)
    dirs, _ = sample_directions(
        jax.random.PRNGKey(1),
        32,
        X.shape[1],
        X,
        in_A,
        in_B,
        cfg,
    )
    assert dirs.shape == (32, X.shape[1])
    norms = jnp.linalg.norm(dirs, axis=-1)
    assert bool(jnp.all(jnp.abs(norms - 1.0) < 1e-6))


@pytest.mark.parametrize("mode", ["kpca_rff", "dmap"])
def test_sample_directions_rejects_removed_modes(cluster_samples, mode):
    """Kernel-based modes were dropped; the dispatcher must reject them."""
    X, in_A, in_B = cluster_samples
    cfg = DirectionSamplingConfig(mode=mode)
    with pytest.raises(ValueError, match="Unknown direction_sampling mode"):
        sample_directions(
            jax.random.PRNGKey(0),
            16,
            X.shape[1],
            X,
            in_A,
            in_B,
            cfg,
        )


def test_sample_directions_geometric_axis_weights_simplex(cluster_samples):
    """Default axis_weights are the |λ|-simplex; verify dispatch metadata."""
    X, in_A, in_B = cluster_samples
    cfg = DirectionSamplingConfig(mode="gcpca", n_bias_axes=4)
    _, info = sample_directions(
        jax.random.PRNGKey(2),
        32,
        X.shape[1],
        X,
        in_A,
        in_B,
        cfg,
    )
    eigvals = np.asarray(info["eigenvalues"])
    assert eigvals.shape == (4,)
    assert info["n_bias_axes"] == 4


@pytest.mark.parametrize("mode", ["uniform", "pca", "gcpca"])
def test_geometric_mode_end_to_end_with_ebmc(cluster_samples, mode):
    """All shipped direction modes compose with EBMC and respect basin BCs."""
    X, in_A, in_B = cluster_samples
    cfg = DirectionSamplingConfig(mode=mode, n_bias_axes=3)
    r = compute_sliced_committor(
        X,
        in_A=in_A,
        in_B=in_B,
        n_directions=32,
        seed=0,
        direction_sampling=cfg,
    )
    ebmc = compute_enriched_basin_moment_weights(r, X)
    q = build_committor(r, ebmc)(X, in_A=in_A, in_B=in_B)
    assert bool(jnp.all(q[:50] == 0.0))
    assert bool(jnp.all(q[-50:] == 1.0))
