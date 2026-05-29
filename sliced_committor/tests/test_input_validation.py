"""Tests for the public-API input validation surface (Phase 6 of the polish plan).

Every check is paired with the error message string fragment a user is
expected to see, so accidental regressions in the message land here too.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

import sliced_committor as sc
from sliced_committor.weights import compute_epsilon


def _toy_samples(n=400, seed=0):
    rng = np.random.default_rng(seed)
    samples = jnp.asarray(rng.standard_normal((n, 2)))
    in_A = jnp.linalg.norm(samples - jnp.asarray([-2.0, 0.0]), axis=1) < 0.6
    in_B = jnp.linalg.norm(samples - jnp.asarray([2.0, 0.0]), axis=1) < 0.6
    in_A = in_A & ~in_B
    return samples, in_A, in_B


def test_rejects_non_2d_samples():
    samples, in_A, in_B = _toy_samples()
    with pytest.raises(ValueError, match="2-D"):
        sc.compute_sliced_committor(samples[:, 0], in_A=in_A, in_B=in_B, n_directions=8)


def test_rejects_empty_basin_A():
    samples, _, in_B = _toy_samples()
    with pytest.raises(ValueError, match="basin A is empty"):
        sc.compute_sliced_committor(
            samples,
            in_A=jnp.zeros(samples.shape[0], bool),
            in_B=in_B,
            n_directions=8,
        )


def test_rejects_empty_basin_B():
    samples, in_A, _ = _toy_samples()
    with pytest.raises(ValueError, match="basin B is empty"):
        sc.compute_sliced_committor(
            samples,
            in_A=in_A,
            in_B=jnp.zeros(samples.shape[0], bool),
            n_directions=8,
        )


def test_rejects_overlapping_basins():
    samples, _, _ = _toy_samples()
    mask = jnp.ones(samples.shape[0], bool)
    with pytest.raises(ValueError, match="disjoint"):
        sc.compute_sliced_committor(samples, in_A=mask, in_B=mask, n_directions=8)


def test_rejects_nan_samples():
    samples, in_A, in_B = _toy_samples()
    samples = samples.at[0, 0].set(jnp.nan)
    with pytest.raises(ValueError, match="non-finite"):
        sc.compute_sliced_committor(samples, in_A=in_A, in_B=in_B, n_directions=8)


def test_rejects_inf_samples():
    samples, in_A, in_B = _toy_samples()
    samples = samples.at[3, 1].set(jnp.inf)
    with pytest.raises(ValueError, match="non-finite"):
        sc.compute_sliced_committor(samples, in_A=in_A, in_B=in_B, n_directions=8)


def test_rejects_mismatched_label_length():
    samples, _, in_B = _toy_samples()
    with pytest.raises(ValueError, match="bool arrays matching samples"):
        sc.compute_sliced_committor(
            samples,
            in_A=jnp.zeros(samples.shape[0] - 1, bool),
            in_B=in_B,
            n_directions=8,
        )


def test_high_d_warning():
    """dim > N triggers a warning, not an error."""
    rng = np.random.default_rng(0)
    samples = jnp.asarray(rng.standard_normal((20, 100)))
    in_A = jnp.zeros(20, bool).at[:5].set(True)
    in_B = jnp.zeros(20, bool).at[15:].set(True)
    with pytest.warns(UserWarning, match="dim .* exceeds N"):
        sc.compute_sliced_committor(samples, in_A=in_A, in_B=in_B, n_directions=8)


def test_unknown_epsilon_name():
    samples, in_A, in_B = _toy_samples()
    result = sc.compute_sliced_committor(samples, in_A=in_A, in_B=in_B, n_directions=8)
    ctx = sc.make_weighting_context(result)
    with pytest.raises(ValueError, match="unknown estimator name"):
        compute_epsilon(ctx, epsilon_fn="equil")  # typo of 'equilibrium'


def test_known_epsilon_names_all_resolve():
    samples, in_A, in_B = _toy_samples()
    result = sc.compute_sliced_committor(samples, in_A=in_A, in_B=in_B, n_directions=8)
    ctx = sc.make_weighting_context(result)
    for name in ("equilibrium", "rms", "flux1d"):
        eps = compute_epsilon(ctx, epsilon_fn=name)
        assert eps.shape == (8,), f"{name}: shape was {eps.shape}"


def test_epsilon_rejects_bad_type():
    samples, in_A, in_B = _toy_samples()
    result = sc.compute_sliced_committor(samples, in_A=in_A, in_B=in_B, n_directions=8)
    ctx = sc.make_weighting_context(result)
    with pytest.raises(TypeError, match="must be None, a string name"):
        compute_epsilon(ctx, epsilon_fn=42)


@pytest.mark.parametrize(
    "fn",
    [
        sc.compute_basin_moment_weights,
        sc.compute_enriched_basin_moment_weights,
    ],
)
def test_solver_rejects_float32_with_actionable_message(fn):
    """EBMC / BMC must fire a clear error when float64 was not enabled.

    We can't unset ``jax_enable_x64`` mid-process (every existing JAX array
    in the test session would break), so we monkeypatch ``jax.config.read``
    to simulate it. This validates the message contract.
    """
    samples, in_A, in_B = _toy_samples()
    result = sc.compute_sliced_committor(samples, in_A=in_A, in_B=in_B, n_directions=8)

    real_read = jax.config.read

    def fake_read(name):
        if name == "jax_enable_x64":
            return False
        return real_read(name)

    jax.config.read = fake_read
    try:
        with pytest.raises(ValueError, match="jax_enable_x64=True"):
            fn(result, samples)
    finally:
        jax.config.read = real_read
