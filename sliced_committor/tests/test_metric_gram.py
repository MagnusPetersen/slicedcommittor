"""The feature-space metric where it acts: the Gram, the weight solve, the energy.

The metric enters the estimator at exactly one factor, ``theta_j^T Mbar theta_k``.
The 1D slice profiles, the directions, the binning and the basin moments are all
untouched, so the trial space is IDENTICAL with and without a metric -- only
which element of it the weight solve picks moves. These tests pin that, the
exactness of the ``feature_metric=None`` path, and the two invariances the
construction rests on.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from sliced_committor import (
    build_committor,
    committor_dirichlet_energy,
    compute_sliced_committor,
    fit_committor,
)
from sliced_committor.core.gram import (
    cos_matrix_from_metric,
    metric_diagonal,
    metric_factor,
    validate_feature_metric,
)
from sliced_committor.core.solver import make_weighting_context

from ._helpers import two_basin_samples

SOLVERS = ["ebmc", "full_gram", "diagonal"]


def fit(samples, in_A, in_B, *, weights="ebmc", metric=None, directions=None, **kw):
    """``fit_committor`` -> the ``CommittorFit`` detail bundle."""
    kwargs = dict(n_directions=24, n_bins=40, seed=7)
    kwargs.update(kw)
    if directions is not None:
        kwargs["directions"] = directions
    _, detail = fit_committor(
        samples,
        in_A=in_A,
        in_B=in_B,
        weights=weights,
        return_details=True,
        feature_metric=metric,
        **kwargs,
    )
    return detail


def basins(n=1200, dim=3, seed=0):
    """Two-basin samples with enough population per basin for a stable solve."""
    return two_basin_samples(n=n, dim=dim, seed=seed, radius=1.1, sep=1.9)


def unit_directions(dim, m, seed=11):
    rng = np.random.default_rng(seed)
    raw = rng.standard_normal((m, dim))
    return jnp.asarray(raw / np.linalg.norm(raw, axis=1, keepdims=True))


# ---------------------------------------------------------------------------
# the None / identity path must be exact, not merely close
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("solver", SOLVERS)
def test_identity_metric_is_bit_identical_to_none(solver):
    """``feature_metric=eye(d)`` reproduces the default bit for bit.

    Guards the identity short-circuit in :func:`metric_factor`: without it the
    result would ride on ``eigh`` returning exactly ``V = I``.
    """
    samples, in_A, in_B = basins(n=900, seed=1)
    base = fit(samples, in_A, in_B, weights=solver)
    same = fit(samples, in_A, in_B, weights=solver, metric=jnp.eye(3))
    wb, ws = base.weights, same.weights
    if isinstance(wb, dict):
        for k in ("w", "G", "c"):
            if k in wb and wb[k] is not None and hasattr(wb[k], "shape"):
                np.testing.assert_array_equal(np.asarray(ws[k]), np.asarray(wb[k]))
    else:
        np.testing.assert_array_equal(np.asarray(ws), np.asarray(wb))


def test_cos_matrix_none_path_is_verbatim():
    theta = unit_directions(4, 9)
    np.testing.assert_array_equal(
        np.asarray(cos_matrix_from_metric(theta, None)), np.asarray(theta @ theta.T)
    )
    np.testing.assert_array_equal(
        np.asarray(cos_matrix_from_metric(theta, jnp.eye(4))), np.asarray(theta @ theta.T)
    )
    assert metric_diagonal(theta, None) is None
    assert metric_factor(None) is None


def test_context_carries_the_metric():
    samples, in_A, in_B = basins(n=800, seed=2)
    M = np.diag([1.0, 2.0, 0.5])
    res = compute_sliced_committor(
        samples, in_A=in_A, in_B=in_B, n_directions=12, n_bins=30, seed=3, feature_metric=M
    )
    assert res.feature_metric is not None
    ctx = make_weighting_context(res)
    assert ctx.feature_metric is not None
    np.testing.assert_allclose(
        np.asarray(ctx.cos_matrix),
        np.asarray(res.directions) @ M @ np.asarray(res.directions).T,
        rtol=1e-12,
        atol=1e-14,
    )
    np.testing.assert_allclose(
        np.asarray(metric_diagonal(res.directions, M)), np.diag(np.asarray(ctx.cos_matrix)),
        rtol=1e-12,
    )


# ---------------------------------------------------------------------------
# THE FLAGSHIP: affine reparameterisation invariance
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("solver", SOLVERS)
def test_affine_reparameterisation_invariance(solver):
    """Slicing ``u = Ax`` under ``Mbar = A A^T`` reproduces the x-space fit exactly.

    With ``theta_u = A^-T theta_x`` the projections coincide identically,
    ``theta_u . u = theta_x . x``, so the 1D profiles are the same functions; and
    ``theta_u^T (A A^T) theta_u' = theta_x^T theta_x'``, so the Gram is the same
    matrix. The committor field must therefore be identical.

    This is the property the method GAINS. Today, with ``D = I`` asserted in
    whatever space the features happen to live in, a linear change of features
    silently changes the answer -- see the companion test below.
    """
    dim = 3
    samples, in_A, in_B = two_basin_samples(n=600, dim=dim, seed=4)
    rng = np.random.default_rng(5)
    Q, _ = np.linalg.qr(rng.standard_normal((dim, dim)))
    A = Q @ np.diag([0.6, 1.0, 1.7])  # cond ~2.8; well away from amplifying fp error

    theta_x = unit_directions(dim, 24)
    theta_u = jnp.asarray(np.asarray(theta_x) @ np.linalg.inv(A))
    samples_u = jnp.asarray(np.asarray(samples) @ A.T)

    ref = fit(samples, in_A, in_B, weights=solver, directions=theta_x)
    got = fit(samples_u, in_A, in_B, weights=solver, directions=theta_u, metric=A @ A.T)

    # the projections, hence every 1D object, coincide
    np.testing.assert_allclose(
        np.asarray(got.result.slice_coords),
        np.asarray(ref.result.slice_coords),
        rtol=1e-9,
        atol=1e-11,
    )
    np.testing.assert_array_equal(
        np.asarray(got.result.valid_mask), np.asarray(ref.result.valid_mask)
    )
    np.testing.assert_allclose(
        np.asarray(got.result.committors_1d),
        np.asarray(ref.result.committors_1d),
        rtol=1e-8,
        atol=1e-10,
    )
    # and so does the committor field
    np.testing.assert_allclose(
        np.asarray(got.committor(samples_u)),
        np.asarray(ref.committor(samples)),
        rtol=1e-7,
        atol=1e-9,
    )


def test_without_the_metric_an_affine_change_of_features_changes_the_answer():
    """The control that makes the flagship test meaningful."""
    dim = 3
    samples, in_A, in_B = two_basin_samples(n=600, dim=dim, seed=4)
    rng = np.random.default_rng(5)
    Q, _ = np.linalg.qr(rng.standard_normal((dim, dim)))
    A = Q @ np.diag([0.6, 1.0, 1.7])
    theta_x = unit_directions(dim, 24)
    theta_u = jnp.asarray(np.asarray(theta_x) @ np.linalg.inv(A))
    samples_u = jnp.asarray(np.asarray(samples) @ A.T)

    ref = fit(samples, in_A, in_B, directions=theta_x)
    naive = fit(samples_u, in_A, in_B, directions=theta_u)  # metric=None
    delta = float(
        jnp.max(jnp.abs(naive.committor(samples_u) - ref.committor(samples)))
    )
    assert delta > 1e-3, f"expected the D=I fit to move under an affine map; got {delta:.2e}"


# ---------------------------------------------------------------------------
# scale invariance: only the SHAPE of the metric is physical
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "tikhonov", ["auto", "auto_m", "auto_lambda", ("lambda", 3e-4), 1e-3, "cv"]
)
def test_weight_solve_is_invariant_to_the_metric_scale(tikhonov):
    """``Mbar -> c Mbar`` scales ``G`` and cancels in ``w = G^-1(b-a)/M_gap``.

    This is why the metric can be reported trace-normalised and no unit question
    arises: only the SHAPE of M0 is physical. Exact to machine precision for
    every ridge rule whose anchor is a RATIO of diagonal statistics of the same
    G -- which is all of them except the two pinned below.
    """
    samples, in_A, in_B = basins(n=1000, seed=6)
    rng = np.random.default_rng(7)
    B = rng.standard_normal((3, 3))
    M = B @ B.T
    M = M * (3.0 / np.trace(M))
    ws = [
        np.asarray(
            fit(samples, in_A, in_B, metric=c * M, weight_kwargs={"tikhonov": tikhonov}).weights["w"]
        )
        for c in (1e-3, 1.0, 1e3)
    ]
    np.testing.assert_allclose(ws[0], ws[1], rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(ws[2], ws[1], rtol=1e-9, atol=1e-12)


def test_halfset_eigen_is_only_approximately_scale_free():
    """The deployed ridge is scale-equivariant in exact arithmetic, not in floats.

    ``_halfset_eigen_regularize`` is homogeneous of degree 1 in G on paper, and
    ``np.corrcoef`` makes its band SSNR scale-free. But it reads that correlation
    off the EIGENBASIS of ``(G1+G2)/2``, and a Gram spectrum with closely spaced
    eigenvalues (measured here: cond ~1.7e2 with a minimum gap of 0.6% of the
    spread) has correspondingly ill-determined eigenvectors. Rescaling perturbs
    the rounding, the band partition shifts, and the SSNR moves a few percent.

    So the rule is: ALWAYS pass the metric trace-normalised --
    :func:`sincos_pullback_metric` does so by default. Pinned rather than fixed,
    because the sensitivity belongs to the estimator, not to the metric, and the
    committor barely moves (~5e-3 over six decades of scale) even where the
    weights do.
    """
    samples, in_A, in_B = basins(n=1000, seed=6)
    rng = np.random.default_rng(7)
    B = rng.standard_normal((3, 3))
    M = B @ B.T
    M = M * (3.0 / np.trace(M))
    kw = {"tikhonov": "halfset_eigen"}
    ref = fit(samples, in_A, in_B, metric=M, weight_kwargs=kw)
    q0 = np.asarray(ref.committor(samples))
    worst_w, worst_q = 0.0, 0.0
    for c in (1e-3, 1e3):
        got = fit(samples, in_A, in_B, metric=c * M, weight_kwargs=kw)
        w0, w1 = np.asarray(ref.weights["w"]), np.asarray(got.weights["w"])
        worst_w = max(worst_w, np.max(np.abs(w1 - w0) / np.maximum(np.abs(w0), 1e-12)))
        worst_q = max(worst_q, np.max(np.abs(np.asarray(got.committor(samples)) - q0)))
    assert worst_w > 1e-6, "expected halfset_eigen to jitter; did the regulariser change?"
    assert worst_q < 5e-2, f"committor moved {worst_q:.2e} over 1e6 in metric scale"


def test_ridge_abs_is_the_documented_scale_exception():
    """A hand-supplied ABSOLUTE ridge does not co-scale with G, by design.

    Pinned so nobody "fixes" it later. Every machine-generated ``ridge_abs`` (the
    'cv' and 'cv_refit' round-trips) is derived from the same G and is therefore
    safe; only a literal carried over from a D = I calibration is not -- and the
    effect is large (measured: ~85x on the weights over six decades), not subtle.
    """
    samples, in_A, in_B = basins(n=1000, seed=6)
    M = np.diag([0.5, 1.0, 1.5])
    M = M * (3.0 / np.trace(M))
    kw = {"tikhonov": ("ridge_abs", 1e-3)}
    w1 = np.asarray(fit(samples, in_A, in_B, metric=M, weight_kwargs=kw).weights["w"])
    w2 = np.asarray(fit(samples, in_A, in_B, metric=1e4 * M, weight_kwargs=kw).weights["w"])
    assert not np.allclose(w1, w2, rtol=1e-3)


# ---------------------------------------------------------------------------
# the d_j paths that the cosine matrix does not cover
# ---------------------------------------------------------------------------
def test_diagonal_weights_pick_up_the_metric_diagonal():
    """``w ~ (1-eps)+ / (d_j D_j)`` -- the diagonal solver's own metric term."""
    samples, in_A, in_B = basins(n=900, seed=8)
    M = np.diag([0.4, 1.0, 1.6])
    M = M * (3.0 / np.trace(M))
    res = compute_sliced_committor(
        samples, in_A=in_A, in_B=in_B, n_directions=24, n_bins=40, seed=7, feature_metric=M
    )
    ctx = make_weighting_context(res)
    d_j = np.asarray(metric_diagonal(res.directions, M))
    log_w = (
        np.log(np.maximum(1.0 - np.asarray(ctx.boundary_errors), 0.0))
        - np.asarray(ctx.log_dirichlet)
        - np.log(d_j)
    )
    log_w = np.where(np.asarray(ctx.valid_mask), log_w, -np.inf)
    expect = np.asarray(jax.nn.softmax(jnp.asarray(log_w)))
    got = np.asarray(
        fit(samples, in_A, in_B, weights="diagonal", metric=M, n_bins=40).weights
    )
    # same basis, so compare against the closed form on the same result object
    from sliced_committor import compute_weights_multi
    from sliced_committor.core.weights import corrected_dirichlet_inv_rd

    direct = np.asarray(
        compute_weights_multi(res, [corrected_dirichlet_inv_rd], samples=samples)[
            "corrected_dirichlet_inv_rd"
        ]
    )
    np.testing.assert_allclose(direct, expect, rtol=1e-10)
    assert got.shape == direct.shape


def test_diagonal_sanity_stays_calibrated_under_a_metric():
    """``diag(G)`` gains ``d_j``; the reference integral must gain it too.

    Without the ``metric_diag`` fix, ``diagonal_sanity`` reports a spurious
    ``|d_j - 1|`` and the ``R_M_ratio`` derived from it inverts -- and both are
    printed as quality scores.
    """
    samples, in_A, in_B = basins(n=800, seed=10)
    M = np.diag([0.35, 1.0, 1.65])
    M = M * (3.0 / np.trace(M))
    plain = fit(samples, in_A, in_B, n_bins=60).weights["diagonal_sanity"]
    metric = fit(samples, in_A, in_B, metric=M, n_bins=60).weights["diagonal_sanity"]
    plain = np.asarray(plain)[np.isfinite(np.asarray(plain))]
    metric = np.asarray(metric)[np.isfinite(np.asarray(metric))]
    assert metric.max() < 5 * max(plain.max(), 1e-3) + 1e-2, (plain.max(), metric.max())


def test_dirichlet_energy_diagonal_fallback_carries_d_j():
    """The Gram branch already carries the metric; the diagonal fallback must too."""
    samples, in_A, in_B = basins(n=1000, seed=12)
    M = np.diag([0.5, 1.0, 1.5])
    M = M * (3.0 / np.trace(M))
    res = compute_sliced_committor(
        samples, in_A=in_A, in_B=in_B, n_directions=16, n_bins=40, seed=13, feature_metric=M
    )
    from sliced_committor import compute_weights_multi
    from sliced_committor.core.weights import corrected_dirichlet_inv_rd

    w = compute_weights_multi(res, [corrected_dirichlet_inv_rd], samples=samples)[
        "corrected_dirichlet_inv_rd"
    ]
    energy = committor_dirichlet_energy(res, w, mode="diagonal")
    d_j = np.asarray(metric_diagonal(res.directions, M))
    wn = np.asarray(w) * np.asarray(res.valid_mask)
    wn = wn / wn.sum()
    log_D = np.asarray(res.log_dirichlet)
    ok = np.isfinite(log_D) & (wn != 0)
    expect = float(np.sum(np.where(ok, wn**2 * np.exp(np.where(ok, log_D, 0.0)) * d_j, 0.0)))
    assert energy == pytest.approx(expect, rel=1e-10)


# ---------------------------------------------------------------------------
# validation
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
    for name, (M, msg) in bad.items():
        with pytest.raises(ValueError, match=msg):
            compute_sliced_committor(
                samples, in_A=in_A, in_B=in_B, n_directions=8, n_bins=20, feature_metric=M
            )


def test_ill_conditioned_metric_warns():
    with pytest.warns(UserWarning, match="ill-conditioned"):
        validate_feature_metric(np.diag([1.0, 1e-9, 1.0]), 3)


def test_rank_deficient_metric_is_accepted_and_keeps_the_gram_psd():
    """A pull-back can be singular; ``eigh`` handles it where ``cholesky`` would not."""
    v = np.array([1.0, -0.5, 0.25])
    M = np.outer(v, v)  # rank 1, exactly singular
    with pytest.warns(UserWarning, match="ill-conditioned"):
        Mv = validate_feature_metric(M, 3)
    theta = unit_directions(3, 12)
    cos = np.asarray(cos_matrix_from_metric(theta, Mv))
    assert np.linalg.eigvalsh(cos).min() > -1e-12
    samples, in_A, in_B = basins(n=800, seed=15)
    with pytest.warns(UserWarning):
        res = compute_sliced_committor(
            samples, in_A=in_A, in_B=in_B, n_directions=12, n_bins=30, feature_metric=M
        )
    assert res.feature_metric is not None


def test_metric_is_promoted_never_demoted():
    """A float32 metric must not silently demote the whole Gram to single precision."""
    assert jax.config.read("jax_enable_x64")
    M = validate_feature_metric(np.eye(3, dtype=np.float32) * 2.0, 3)
    assert M.dtype == jnp.float64
    theta = unit_directions(3, 8)
    assert cos_matrix_from_metric(theta, M).dtype == jnp.float64


def test_metric_rides_solver_kwargs_and_reaches_the_result():
    """``fit_committor`` forwards it through ``**solver_kwargs`` with no plumbing."""
    samples, in_A, in_B = basins(n=800, seed=16)
    M = np.diag([0.6, 1.0, 1.4])
    out = fit(samples, in_A, in_B, metric=M)
    np.testing.assert_allclose(np.asarray(out.result.feature_metric), M, rtol=1e-12)


def test_metric_changes_the_weights_but_not_the_trial_space():
    """The headline structural claim: same basis, different optimal element."""
    samples, in_A, in_B = basins(n=600, seed=17)
    M = np.diag([0.3, 1.0, 1.7])
    M = M * (3.0 / np.trace(M))
    theta = unit_directions(3, 24)
    ref = fit(samples, in_A, in_B, directions=theta)
    got = fit(samples, in_A, in_B, directions=theta, metric=M)
    # identical basis
    np.testing.assert_array_equal(
        np.asarray(got.result.committors_1d), np.asarray(ref.result.committors_1d)
    )
    np.testing.assert_array_equal(
        np.asarray(got.result.slice_coords), np.asarray(ref.result.slice_coords)
    )
    # different weights
    assert not np.allclose(
        np.asarray(got.weights["w"]), np.asarray(ref.weights["w"]), rtol=1e-3
    )
