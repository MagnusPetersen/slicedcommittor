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
    compute_lda_axis,
    compute_sliced_committor,
    directions_gcpca,
    directions_pca,
    directions_tica_ema,
    directions_tica_ema_decomposed,
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


# ---------------------------------------------------------------------------
# LDA axis + TICA-EMA factories (previously untested public API)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def tica_traj():
    """A 4-D AR(1) trajectory with one slow mode along a fixed axis."""
    rng = np.random.default_rng(0)
    T, dim = 4000, 4
    x = np.zeros((T, dim))
    decay = np.array([0.98, 0.5, 0.4, 0.3])  # axis 0 is the slow mode
    for t in range(1, T):
        x[t] = decay * x[t - 1] + rng.standard_normal(dim) * 0.3
    return jnp.asarray(x)


@pytest.mark.parametrize("method", ["fisher", "mean_diff"])
def test_compute_lda_axis(cluster_samples, method):
    X, in_A, in_B = cluster_samples
    axis, info = compute_lda_axis(X, in_A, in_B, method=method)
    assert axis.shape == (X.shape[1],)
    assert abs(float(jnp.linalg.norm(axis)) - 1.0) < 1e-6
    # Clusters are separated along axis 0, so the discriminant aligns with it.
    assert abs(float(axis[0])) > 0.9
    assert info["method"] == method
    assert info["n_A"] == int(jnp.sum(in_A)) and info["n_B"] == int(jnp.sum(in_B))


def test_compute_lda_axis_empty_basin_raises(cluster_samples):
    X, _, in_B = cluster_samples
    with pytest.raises(ValueError, match="at least one sample"):
        compute_lda_axis(X, jnp.zeros(X.shape[0], dtype=bool), in_B)


def test_directions_tica_ema_shape_norm_and_reproducible(tica_traj):
    key = jax.random.PRNGKey(3)
    V1 = directions_tica_ema(key, 16, tica_traj, alpha_ratio=30.0)
    V2 = directions_tica_ema(key, 16, tica_traj, alpha_ratio=30.0)
    assert V1.shape == (16, tica_traj.shape[1])
    assert _is_unit_norm(V1)
    np.testing.assert_array_equal(np.asarray(V1), np.asarray(V2))  # deterministic in key


def test_directions_tica_ema_accepts_multiple_trajectories(tica_traj):
    key = jax.random.PRNGKey(4)
    trajs = [tica_traj[:1500], tica_traj[1500:]]
    V = directions_tica_ema(key, 8, trajs, alpha_ratio=30.0)
    assert V.shape == (8, tica_traj.shape[1])
    assert _is_unit_norm(V)


def test_tica_ema_ridge_is_trace_relative_near_noop(tica_traj):
    # §1.1: directions_tica_ema's ridge is now trace-relative (η = ridge·tr/d),
    # matching LDA/gcPCA. At the 1e-6 default this is a near-no-op vs the old
    # absolute-1e-6 path: the generalized eigenvalues are unchanged to tight tol.
    from sliced_committor.core.directions import (
        _ema_covariances,
        _solve_ema_gep,
        _tica_ema_basis,
    )

    C0, M_a = _ema_covariances(tica_traj, 30.0)
    _, mu_rel = _tica_ema_basis(tica_traj, 30.0, 1e-6)  # trace-relative
    _, mu_abs = _solve_ema_gep(C0, M_a, 1e-6)  # absolute (old default)
    np.testing.assert_allclose(np.asarray(mu_rel), np.asarray(mu_abs), rtol=1e-4, atol=1e-8)


def test_directions_tica_ema_decomposed_shape_norm(tica_traj):
    key = jax.random.PRNGKey(5)
    T = tica_traj.shape[0]
    windows = [tica_traj[:1300], tica_traj[1300:2600], tica_traj[2600:]]
    mbar = [jnp.ones(w.shape[0]) / T for w in windows]
    V = directions_tica_ema_decomposed(key, 6, windows, mbar, ridge=2.0)
    assert V.shape == (6, tica_traj.shape[1])
    assert _is_unit_norm(V)


def test_color_normalize_all_negative_falls_back_uniform():
    # §1.2 guard: noise-dominated TICA (all eigenvalues ≤ 0) must warn and fall
    # back to uniform unit-norm directions rather than return zero vectors.
    from sliced_committor.core.directions import _color_normalize_directions

    V = jnp.eye(4)
    mu = jnp.array([-1.0, -2.0, -0.5, -3.0])
    with pytest.warns(UserWarning, match="non-positive"):
        out = _color_normalize_directions(jax.random.PRNGKey(0), 5, V, mu)
    assert out.shape == (5, 4)
    assert _is_unit_norm(out)


def test_tica_decomposed_warns_on_collapsed_between_window_variance(tica_traj):
    # §1.2 guard: when all windows share the same MBAR-mean (no between-window
    # variance), beta = tr(M_within)/tr(C_between) explodes and must warn.
    key = jax.random.PRNGKey(6)
    block = tica_traj[:1000]
    windows = [block, block, block]  # identical means → C_between ≈ 0
    mbar = [jnp.ones(block.shape[0]) / (3 * block.shape[0]) for _ in windows]
    with pytest.warns(UserWarning, match="between-window variance"):
        directions_tica_ema_decomposed(key, 6, windows, mbar, ridge=2.0)
