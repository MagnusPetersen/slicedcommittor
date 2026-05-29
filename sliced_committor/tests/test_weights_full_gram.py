"""Unit tests for the full-Gram simplex solver."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

import sliced_committor as sc
from sliced_committor.weights import compute_epsilon_rms


def _make_samples(n=500, seed=0):
    rng = np.random.default_rng(seed)
    samples = jnp.asarray(rng.standard_normal((n, 2)))
    in_A = jnp.linalg.norm(samples - jnp.asarray([-2.0, 0.0]), axis=1) < 0.7
    in_B = jnp.linalg.norm(samples - jnp.asarray([2.0, 0.0]), axis=1) < 0.7
    in_A = in_A & ~in_B
    return samples, in_A, in_B


def test_full_gram_returns_simplex_constraint():
    samples, in_A, in_B = _make_samples()
    result = sc.compute_sliced_committor(samples, in_A=in_A, in_B=in_B, n_directions=24, seed=0)
    out = sc.compute_full_gram_weights(result, samples)
    assert "w" in out
    np.testing.assert_allclose(float(np.asarray(out["w"]).sum()), 1.0, atol=1e-6)


def test_full_gram_allows_negative_weights_legitimate():
    """The README documents that negative weights are legitimate coupling
    corrections, not pathologies. Confirm the solver returns the diagnostic
    counts rather than silently truncating.
    """
    samples, in_A, in_B = _make_samples()
    result = sc.compute_sliced_committor(samples, in_A=in_A, in_B=in_B, n_directions=32, seed=4)
    out = sc.compute_full_gram_weights(result, samples)
    assert "n_negative_weights" in out
    assert "negative_weight_mass" in out


def test_full_gram_eps_override_changes_result():
    """Switching to the RMS epsilon estimator should change the b vector and
    therefore the returned weights.
    """
    samples, in_A, in_B = _make_samples()
    result = sc.compute_sliced_committor(samples, in_A=in_A, in_B=in_B, n_directions=24, seed=1)
    out_eq = sc.compute_full_gram_weights(result, samples)
    out_rms = sc.compute_full_gram_weights(result, samples, epsilon_fn=compute_epsilon_rms)
    diff = float(jnp.linalg.norm(out_eq["w"] - out_rms["w"]))
    assert diff > 1e-6, (
        "ε override did not affect weights; either the cache leaked or the override is a no-op"
    )


def test_full_gram_diagnostics_keys_present():
    samples, in_A, in_B = _make_samples()
    result = sc.compute_sliced_committor(samples, in_A=in_A, in_B=in_B, n_directions=16, seed=2)
    out = sc.compute_full_gram_weights(result, samples)
    for key in (
        "w",
        "G",
        "off_diagonal_magnitude",
        "diagonal_sanity",
        "n_negative_weights",
        "constraint",
        "eta_used",
    ):
        assert key in out, f"missing key {key}"
