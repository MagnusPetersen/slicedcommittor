"""The feature-space metric where it acts: the Gram and the weight solve.

The metric enters at exactly one factor, ``theta_j^T Mbar theta_k``. The 1D
profiles, directions, binning and basin moments are untouched, so the trial
space is IDENTICAL with and without a metric; only the optimal element moves.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from sliced_committor import compute_sliced_committor, fit_committor
from sliced_committor.core.gram import (
    cos_matrix_from_metric,
    metric_factor,
    validate_feature_metric,
)

from ._helpers import two_basin_samples, unit_directions


def fit(samples, in_A, in_B, *, metric=None, directions=None, **kw):
    kwargs = dict(n_directions=24, n_bins=40, seed=7)
    kwargs.update(kw)
    if directions is not None:
        kwargs["directions"] = directions
    _, detail = fit_committor(
        samples, in_A=in_A, in_B=in_B, return_details=True, feature_metric=metric, **kwargs
    )
    return detail


def basins(n=1200, dim=3, seed=0):
    return two_basin_samples(n=n, dim=dim, seed=seed, radius=1.1, sep=1.9)


# ---------------------------------------------------------------------------
# the None / identity path is exact
# ---------------------------------------------------------------------------
def test_identity_metric_is_bit_identical_to_none():
    samples, in_A, in_B = basins(n=900, seed=1)
    base = fit(samples, in_A, in_B)
    same = fit(samples, in_A, in_B, metric=jnp.eye(3))
    np.testing.assert_array_equal(np.asarray(same.weights.w), np.asarray(base.weights.w))
    assert same.weights.c == base.weights.c


def test_cos_matrix_none_path_is_verbatim():
    theta = unit_directions(4, 9)
    np.testing.assert_array_equal(
        np.asarray(cos_matrix_from_metric(theta, None)), np.asarray(theta @ theta.T)
    )
    np.testing.assert_array_equal(
        np.asarray(cos_matrix_from_metric(theta, jnp.eye(4))), np.asarray(theta @ theta.T)
    )
    assert metric_factor(None) is None


def test_cos_matrix_carries_the_metric():
    theta = unit_directions(3, 12)
    M = np.diag([1.0, 2.0, 0.5])
    np.testing.assert_allclose(
        np.asarray(cos_matrix_from_metric(theta, jnp.asarray(M))),
        np.asarray(theta) @ M @ np.asarray(theta).T,
        rtol=1e-12,
        atol=1e-14,
    )


# ---------------------------------------------------------------------------
# THE FLAGSHIP: affine reparameterisation invariance
# ---------------------------------------------------------------------------
def _affine_setup():
    dim = 3
    samples, in_A, in_B = two_basin_samples(n=600, dim=dim, seed=4)
    rng = np.random.default_rng(5)
    Q, _ = np.linalg.qr(rng.standard_normal((dim, dim)))
    A = Q @ np.diag([0.6, 1.0, 1.7])
    theta_x = unit_directions(dim, 24)
    theta_u = jnp.asarray(np.asarray(theta_x) @ np.linalg.inv(A))
    samples_u = jnp.asarray(np.asarray(samples) @ A.T)
    return samples, samples_u, in_A, in_B, A, theta_x, theta_u


@pytest.mark.parametrize(
    "tikhonov, tol",
    [("auto", dict(rtol=1e-7, atol=1e-9)), ("halfset_eigen", dict(rtol=0.0, atol=2e-3))],
)
def test_affine_reparameterisation_invariance(tikhonov, tol):
    """Slicing ``u = A x`` under ``Mbar = A A^T`` reproduces the x-space fit.

    With ``theta_u = A^-T theta_x`` the projections coincide identically, so every
    1D object is the same function, and ``theta_u^T (A A^T) theta_u' = theta_x^T
    theta_x'`` makes the Gram the same matrix up to rounding. A scalar ridge then
    returns the same committor to solver precision. The half-set filter is a
    data-driven spectral rule whose bands are eigenvectors of a Gram with
    cond ~ 1e12; the last-ulp difference between the two assemblies rotates its
    near-null bands, so it inherits the invariance only to regularisation
    accuracy (observed 4e-4 in q).
    """
    samples, samples_u, in_A, in_B, A, theta_x, theta_u = _affine_setup()
    ref = fit(samples, in_A, in_B, directions=theta_x, tikhonov=tikhonov)
    got = fit(samples_u, in_A, in_B, directions=theta_u, metric=A @ A.T, tikhonov=tikhonov)
    np.testing.assert_allclose(
        np.asarray(got.result.slice_coords),
        np.asarray(ref.result.slice_coords),
        rtol=1e-9,
        atol=1e-11,
    )
    np.testing.assert_allclose(
        np.asarray(got.result.committors_1d),
        np.asarray(ref.result.committors_1d),
        rtol=1e-8,
        atol=1e-10,
    )
    np.testing.assert_allclose(
        np.asarray(got.committor(samples_u)), np.asarray(ref.committor(samples)), **tol
    )


def test_without_the_metric_an_affine_change_of_features_changes_the_answer():
    samples, samples_u, in_A, in_B, _, theta_x, theta_u = _affine_setup()
    ref = fit(samples, in_A, in_B, directions=theta_x)
    naive = fit(samples_u, in_A, in_B, directions=theta_u)
    delta = float(jnp.max(jnp.abs(naive.committor(samples_u) - ref.committor(samples))))
    assert delta > 1e-3, delta


# ---------------------------------------------------------------------------
# scale: only the SHAPE of the metric is physical
# ---------------------------------------------------------------------------
def _random_metric(seed=7, dim=3):
    B = np.random.default_rng(seed).standard_normal((dim, dim))
    M = B @ B.T
    return M * (dim / np.trace(M))


def test_auto_ridge_is_invariant_to_the_metric_scale():
    samples, in_A, in_B = basins(n=1000, seed=6)
    M = _random_metric()
    ws = [
        np.asarray(fit(samples, in_A, in_B, metric=c * M, tikhonov="auto").weights.w)
        for c in (1e-3, 1.0, 1e3)
    ]
    np.testing.assert_allclose(ws[0], ws[1], rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(ws[2], ws[1], rtol=1e-9, atol=1e-12)


def test_halfset_is_only_approximately_scale_free():
    """The half-set filter reads a band correlation off an eigenbasis whose
    eigenvectors are ill-determined where the spectrum is closely spaced, so
    rescaling perturbs the rounding: the weights jitter, the committor barely
    moves. Pass the metric trace-normalised (the default)."""
    samples, in_A, in_B = basins(n=1000, seed=6)
    M = _random_metric()
    ref = fit(samples, in_A, in_B, metric=M)
    q0 = np.asarray(ref.committor(samples))
    worst_q = 0.0
    for c in (1e-3, 1e3):
        got = fit(samples, in_A, in_B, metric=c * M)
        worst_q = max(worst_q, np.max(np.abs(np.asarray(got.committor(samples)) - q0)))
    assert worst_q < 5e-2, worst_q


def test_an_absolute_ridge_does_not_co_scale():
    samples, in_A, in_B = basins(n=1000, seed=6)
    M = np.diag([0.5, 1.0, 1.5])
    w1 = np.asarray(fit(samples, in_A, in_B, metric=M, tikhonov=1e-3).weights.w)
    w2 = np.asarray(fit(samples, in_A, in_B, metric=1e4 * M, tikhonov=1e-3).weights.w)
    assert not np.allclose(w1, w2, rtol=1e-3)


# ---------------------------------------------------------------------------
# validation and plumbing
# ---------------------------------------------------------------------------
def test_metric_validation_rejects_bad_input():
    samples, in_A, in_B = basins(n=600, seed=14)
    bad = {
        "square": (np.zeros((3, 4)), "square"),
        "dim": (np.eye(4), "dim"),
        "finite": (np.array([[1.0, np.nan, 0], [np.nan, 1, 0], [0, 0, 1]]), "non-finite"),
        "asym": (np.array([[1.0, 0.9, 0], [-0.9, 1, 0], [0, 0, 1]]), "not symmetric"),
        "psd": (np.diag([1.0, -1.0, 1.0]), "positive semi-definite"),
        "zero": (np.zeros((3, 3)), "all zeros"),
    }
    for _, (M, msg) in bad.items():
        with pytest.raises(ValueError, match=msg):
            compute_sliced_committor(
                samples, in_A=in_A, in_B=in_B, n_directions=8, n_bins=20, feature_metric=M
            )


def test_ill_conditioned_metric_warns():
    with pytest.warns(UserWarning, match="ill-conditioned"):
        validate_feature_metric(np.diag([1.0, 1e-9, 1.0]), 3)


def test_rank_deficient_metric_is_accepted_and_keeps_the_gram_psd():
    v = np.array([1.0, -0.5, 0.25])
    M = np.outer(v, v)
    with pytest.warns(UserWarning, match="ill-conditioned"):
        Mv = validate_feature_metric(M, 3)
    cos = np.asarray(cos_matrix_from_metric(unit_directions(3, 12), Mv))
    assert np.linalg.eigvalsh(cos).min() > -1e-12
    samples, in_A, in_B = basins(n=800, seed=15)
    with pytest.warns(UserWarning):
        res = compute_sliced_committor(
            samples, in_A=in_A, in_B=in_B, n_directions=12, n_bins=30, feature_metric=M
        )
    assert res.feature_metric is not None


def test_metric_is_promoted_never_demoted():
    assert jax.config.read("jax_enable_x64")
    M = validate_feature_metric(np.eye(3, dtype=np.float32) * 2.0, 3)
    assert M.dtype == jnp.float64
    assert cos_matrix_from_metric(unit_directions(3, 8), M).dtype == jnp.float64


def test_metric_rides_solver_kwargs_and_reaches_the_result():
    samples, in_A, in_B = basins(n=800, seed=16)
    M = np.diag([0.6, 1.0, 1.4])
    out = fit(samples, in_A, in_B, metric=M)
    np.testing.assert_allclose(np.asarray(out.result.feature_metric), M, rtol=1e-12)


def test_metric_changes_the_weights_but_not_the_trial_space():
    samples, in_A, in_B = basins(n=600, seed=17)
    M = np.diag([0.3, 1.0, 1.7])
    theta = unit_directions(3, 24)
    ref = fit(samples, in_A, in_B, directions=theta)
    got = fit(samples, in_A, in_B, directions=theta, metric=M)
    np.testing.assert_array_equal(
        np.asarray(got.result.committors_1d), np.asarray(ref.result.committors_1d)
    )
    np.testing.assert_array_equal(
        np.asarray(got.result.slice_coords), np.asarray(ref.result.slice_coords)
    )
    assert not np.allclose(np.asarray(got.weights.w), np.asarray(ref.weights.w), rtol=1e-3)
