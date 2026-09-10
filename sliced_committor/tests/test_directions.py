"""Direction sampling: uniform, the Fisher-LDA cone, and the config dispatch."""

import jax
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp

import sliced_committor as sc
from sliced_committor.core.directions import _sample_directions


@pytest.fixture(scope="module")
def clusters():
    """400 samples in 12 dimensions, two clusters separated along axis 0."""
    key = jax.random.PRNGKey(0)
    N, dim = 400, 12
    X = jax.random.normal(key, (N, dim)) * 0.3
    X = X.at[: N // 2, 0].add(-1.0)
    X = X.at[N // 2 :, 0].add(+1.0)
    return X, jnp.arange(N) < 50, jnp.arange(N) >= N - 50


def _unit(rows, tol=1e-6):
    return bool(jnp.all(jnp.abs(jnp.linalg.norm(rows, axis=-1) - 1.0) < tol))


def test_uniform_directions_are_unit_and_deterministic():
    a = sc.directions_uniform(jax.random.PRNGKey(1), 32, 5)
    b = sc.directions_uniform(jax.random.PRNGKey(1), 32, 5)
    assert a.shape == (32, 5) and _unit(a)
    np.testing.assert_array_equal(np.asarray(a), np.asarray(b))


def test_lda_axis_aligns_with_the_separation(clusters):
    X, in_A, in_B = clusters
    axis, info = sc.compute_lda_axis(X, in_A, in_B)
    assert axis.shape == (12,) and abs(float(jnp.linalg.norm(axis)) - 1.0) < 1e-6
    assert abs(float(axis[0])) > 0.9
    assert info["n_A"] == 50 and info["n_B"] == 50 and info["mean_mahalanobis"] > 1.0


def test_lda_axis_rejects_an_empty_basin(clusters):
    X, _, in_B = clusters
    with pytest.raises(ValueError, match="at least one sample"):
        sc.compute_lda_axis(X, jnp.zeros(X.shape[0], bool), in_B)


def test_mixture_is_stratified_and_concentrated(clusters):
    X, in_A, in_B = clusters
    axis, _ = sc.compute_lda_axis(X, in_A, in_B)
    key = jax.random.PRNGKey(2)
    dirs = sc.sample_power_spherical_mixture(key, axis, 400, 12, mu=0.6, alpha=0.5)
    assert dirs.shape == (400, 12) and _unit(dirs)
    cos = np.asarray(dirs @ axis)
    # first 200 rows uniform (mean cosine ~ 0), last 200 concentrated (mean cosine ~ mu)
    assert abs(cos[:200].mean()) < 0.15
    assert abs(cos[200:].mean() - 0.6) < 0.1
    # alpha = 1 is pure uniform, whatever the axis
    unif = sc.sample_power_spherical_mixture(key, axis, 64, 12, mu=0.6, alpha=1.0)
    np.testing.assert_array_equal(np.asarray(unif), np.asarray(sc.directions_uniform(key, 64, 12)))


def test_mixture_validates_its_arguments(clusters):
    X, in_A, in_B = clusters
    axis, _ = sc.compute_lda_axis(X, in_A, in_B)
    key = jax.random.PRNGKey(0)
    with pytest.raises(ValueError, match="mu"):
        sc.sample_power_spherical_mixture(key, axis, 8, 12, mu=1.0)
    with pytest.raises(ValueError, match="alpha"):
        sc.sample_power_spherical_mixture(key, axis, 8, 12, alpha=1.5)
    with pytest.raises(ValueError, match="axis"):
        sc.sample_power_spherical_mixture(key, axis[:3], 8, 12, alpha=0.5)


@pytest.mark.parametrize("mode", ["uniform", "lda"])
def test_config_dispatch_returns_unit_directions(clusters, mode):
    X, in_A, in_B = clusters
    dirs, info = _sample_directions(
        jax.random.PRNGKey(1), 32, 12, X, in_A, in_B, sc.DirectionSamplingConfig(mode=mode)
    )
    assert dirs.shape == (32, 12) and _unit(dirs)
    assert (info is None) == (mode == "uniform")


def test_config_rejects_unknown_mode(clusters):
    X, in_A, in_B = clusters
    with pytest.raises(ValueError, match="'uniform' or 'lda'"):
        _sample_directions(
            jax.random.PRNGKey(0), 8, 12, X, in_A, in_B, sc.DirectionSamplingConfig(mode="pca")
        )


def test_config_user_axis_and_thin_floor_warning(clusters):
    X, in_A, in_B = clusters
    axis = jnp.zeros(12).at[0].set(2.0)  # renormalised inside
    cfg = sc.DirectionSamplingConfig(mode="lda", axis=axis, alpha=0.05)
    with pytest.warns(UserWarning, match="coverage floor"):
        dirs, info = _sample_directions(jax.random.PRNGKey(3), 40, 12, X, in_A, in_B, cfg)
    assert info["method"] == "user_supplied" and abs(float(info["axis"][0]) - 1.0) < 1e-12
    assert _unit(dirs)


def test_lda_cone_end_to_end(clusters):
    X, in_A, in_B = clusters
    cfg = sc.DirectionSamplingConfig(mode="lda", mu=0.5, alpha=0.3)
    q, det = sc.fit_committor(
        X,
        in_A=in_A,
        in_B=in_B,
        n_directions=32,
        seed=0,
        direction_sampling=cfg,
        return_details=True,
    )
    assert det.result.axis is not None and det.result.lda_info["mu"] == 0.5
    qv = q(X, in_A=in_A, in_B=in_B)
    assert bool(jnp.all(qv[:50] == 0.0)) and bool(jnp.all(qv[-50:] == 1.0))
