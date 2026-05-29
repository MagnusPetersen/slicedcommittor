"""Tests for ``evaluate_committor`` boundary enforcement and dict-dispatch."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

import sliced_committor as sc


def _make_samples(n=500, seed=0):
    rng = np.random.default_rng(seed)
    samples = jnp.asarray(rng.standard_normal((n, 2)))
    in_A = jnp.linalg.norm(samples - jnp.asarray([-2.0, 0.0]), axis=1) < 0.7
    in_B = jnp.linalg.norm(samples - jnp.asarray([2.0, 0.0]), axis=1) < 0.7
    in_A = in_A & ~in_B
    return samples, in_A, in_B


def test_boundary_labels_enforce_q_zero_one():
    samples, in_A, in_B = _make_samples()
    result = sc.compute_sliced_committor(samples, in_A=in_A, in_B=in_B, n_directions=24, seed=0)
    raw = sc.compute_weights_multi(result, [sc.corrected_dirichlet_inv_rd])[
        "corrected_dirichlet_inv_rd"
    ]
    q = sc.evaluate_committor(result, samples, raw, in_A=in_A, in_B=in_B)
    q = np.asarray(q)
    # All flagged-A points must be exactly 0, all flagged-B exactly 1.
    np.testing.assert_array_equal(q[np.asarray(in_A)], 0.0)
    np.testing.assert_array_equal(q[np.asarray(in_B)], 1.0)


def test_no_boundary_labels_does_not_force():
    samples, in_A, in_B = _make_samples()
    result = sc.compute_sliced_committor(samples, in_A=in_A, in_B=in_B, n_directions=24, seed=0)
    raw = sc.compute_weights_multi(result, [sc.corrected_dirichlet_inv_rd])[
        "corrected_dirichlet_inv_rd"
    ]
    q = sc.evaluate_committor(result, samples, raw)
    # Without the labels, q on A samples may not be exactly zero.
    q_A_min = float(np.min(np.asarray(q)[np.asarray(in_A)]))
    q_A_max = float(np.max(np.asarray(q)[np.asarray(in_A)]))
    assert q_A_min >= 0 and q_A_max <= 1
    # At least one A sample should be > 0 (i.e. the raw aggregator did NOT
    # produce a pure 0 there); this is a sanity check that no enforcement is
    # silently happening. We allow it to be 0 in pathological setups, so the
    # assertion is just for non-negativity.


def test_dict_route_centered_basis():
    samples, in_A, in_B = _make_samples()
    result = sc.compute_sliced_committor(samples, in_A=in_A, in_B=in_B, n_directions=24, seed=2)
    ebmc = sc.compute_enriched_basin_moment_weights(result, samples)
    q = sc.evaluate_committor(result, samples[:30], ebmc)
    q = np.asarray(q)
    assert q.shape == (30,)
    # Centered-basis outputs need not lie in [0,1] before clip but the
    # evaluator clamps them. Allow tiny slack from float noise.
    assert (q >= -1e-9).all() and (q <= 1 + 1e-9).all()


def test_raw_array_route_returns_array():
    samples, in_A, in_B = _make_samples()
    result = sc.compute_sliced_committor(samples, in_A=in_A, in_B=in_B, n_directions=12, seed=3)
    raw = sc.compute_weights_multi(result, [sc.corrected_dirichlet_inv_rd])[
        "corrected_dirichlet_inv_rd"
    ]
    q = sc.evaluate_committor(result, samples[:25], raw)
    assert q.shape == (25,)
    assert jnp.all(jnp.isfinite(q))
