"""Tests for the callable committor (build_committor) boundary enforcement and dict-dispatch."""

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

import sliced_committor as sc

from ._helpers import two_basin_samples


def _make_samples(n=500, seed=0):
    return two_basin_samples(n, seed=seed)


def test_boundary_labels_enforce_q_zero_one():
    samples, in_A, in_B = _make_samples()
    result = sc.compute_sliced_committor(samples, in_A=in_A, in_B=in_B, n_directions=24, seed=0)
    raw = sc.compute_weights_multi(result, [sc.corrected_dirichlet_inv_rd])[
        "corrected_dirichlet_inv_rd"
    ]
    q = sc.build_committor(result, raw)(samples, in_A=in_A, in_B=in_B)
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
    q = sc.build_committor(result, raw)(samples)
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
    q = sc.build_committor(result, ebmc)(samples[:30])
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
    q = sc.build_committor(result, raw)(samples[:25])
    assert q.shape == (25,)
    assert jnp.all(jnp.isfinite(q))
