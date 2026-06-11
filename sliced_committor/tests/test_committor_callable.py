"""Tests for the callable committor model: fit/build + autodiff gradient.

The committor is a pure JAX function ``q(x)``; ``build_committor`` must
reproduce the internal reference evaluator, ``fit_committor`` round-trips to it,
and ``committor_gradient`` (autodiff) must match a central finite difference.
"""

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

import sliced_committor as sc

# Internal reference evaluator (removed from the public API; kept for the
# build-matches-reference consistency check only).
from sliced_committor.core.solver import evaluate_committor as _evaluate_reference

# Shared equilibrium double-well sampler (also used by the validation test).
from .test_validation_2d_double_well import _equilibrium_samples


@pytest.fixture(scope="module")
def fit():
    s, in_A, in_B = _equilibrium_samples(n_samples=1500)
    q, det = sc.fit_committor(
        s, in_A=in_A, in_B=in_B, n_directions=128, n_bins=120, seed=1, return_details=True
    )
    return s, in_A, in_B, q, det


def test_fit_returns_callable_and_details(fit):
    s, in_A, in_B, q, det = fit
    assert callable(q)
    assert isinstance(det, sc.CommittorFit)
    assert det.committor is q
    assert det.result.directions.shape[0] == 128
    vals = np.asarray(q(s[:20]))
    assert vals.shape == (20,)
    assert (vals >= -1e-9).all() and (vals <= 1 + 1e-9).all()


def test_fit_default_returns_bare_callable():
    s, in_A, in_B = _equilibrium_samples(n_samples=600, seed=3)
    q = sc.fit_committor(s, in_A=in_A, in_B=in_B, n_directions=32, seed=3)
    assert callable(q)
    assert np.asarray(q(s[:5])).shape == (5,)


def test_build_matches_internal_evaluator(fit):
    s, in_A, in_B, q, det = fit
    q2 = sc.build_committor(det.result, det.weights, enforce_boundary_conditions=False)
    a = np.asarray(q2(s[:60]))
    b = np.asarray(
        _evaluate_reference(det.result, s[:60], det.weights, enforce_boundary_conditions=False)
    )
    np.testing.assert_allclose(a, b, atol=1e-9, rtol=0)


@pytest.mark.parametrize("solver", ["full_gram", "bmc"])
def test_build_matches_internal_evaluator_array_path(fit, solver):
    """build_committor's linear array combiner must match the internal log-space
    evaluator for full_gram / bmc weights -- the path the EBMC (centered) check
    above never exercises. Same result + weights, so any divergence is real."""
    s, in_A, in_B, _, det = fit
    w = (
        sc.compute_full_gram_weights(det.result, s)
        if solver == "full_gram"
        else sc.compute_basin_moment_weights(det.result, s)
    )
    q2 = sc.build_committor(det.result, w, enforce_boundary_conditions=False)
    a = np.asarray(q2(s[:60]))
    b = np.asarray(_evaluate_reference(det.result, s[:60], w, enforce_boundary_conditions=False))
    np.testing.assert_allclose(a, b, atol=1e-8, rtol=0)


def test_fit_roundtrip_matches_build(fit):
    s, in_A, in_B, q, det = fit
    q2 = sc.build_committor(det.result, det.weights)
    np.testing.assert_allclose(np.asarray(q(s[:60])), np.asarray(q2(s[:60])), atol=1e-12)


def test_single_point_returns_scalar(fit):
    s, in_A, in_B, q, det = fit
    val = q(s[0])
    assert np.ndim(np.asarray(val)) == 0


def test_gradient_matches_finite_difference(fit):
    s, in_A, in_B, q, det = fit
    # Interior transition points near the barrier (q strictly in (0, 1)).
    pts = jnp.asarray([[0.0, 0.0], [0.1, 0.2], [-0.2, 0.1], [0.3, -0.1], [-0.1, -0.2]])
    g = np.asarray(sc.committor_gradient(q, pts))
    assert g.shape == (5, 2)
    eps = 1e-4
    fd = np.zeros_like(g)
    for d in range(2):
        e = jnp.asarray(np.eye(2)[d] * eps)
        fd[:, d] = (np.asarray(q(pts + e)) - np.asarray(q(pts - e))) / (2 * eps)
    # Autodiff of the piecewise-linear interpolant equals the slice slope;
    # FD agrees except exactly on bin breakpoints, so a modest tolerance.
    np.testing.assert_allclose(g, fd, atol=1e-2, rtol=5e-2)


def test_boundary_enforcement_via_callable(fit):
    s, in_A, in_B, q, det = fit
    qv = np.asarray(q(s, in_A=in_A, in_B=in_B))
    np.testing.assert_array_equal(qv[np.asarray(in_A)], 0.0)
    np.testing.assert_array_equal(qv[np.asarray(in_B)], 1.0)


def test_diagonal_weights_string(fit):
    s, in_A, in_B, _, _ = fit
    q = sc.fit_committor(s, in_A=in_A, in_B=in_B, weights="diagonal", n_directions=64, seed=1)
    vals = np.asarray(q(s[:10]))
    assert vals.shape == (10,) and np.all(np.isfinite(vals))
