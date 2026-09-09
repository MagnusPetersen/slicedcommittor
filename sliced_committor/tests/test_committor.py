"""The callable committor: fit, build, evaluate, differentiate, post-process."""

import jax
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp

import sliced_committor as sc

from ._helpers import double_well_samples, two_basin_samples


@pytest.fixture(scope="module")
def fit():
    s, in_A, in_B = double_well_samples(n=1500)
    q, det = sc.fit_committor(
        s, in_A=in_A, in_B=in_B, n_directions=128, n_bins=120, seed=1, return_details=True
    )
    return jnp.asarray(s), jnp.asarray(in_A), jnp.asarray(in_B), q, det


def test_fit_returns_callable_and_details(fit):
    s, in_A, in_B, q, det = fit
    assert callable(q)
    assert isinstance(det, sc.CommittorFit)
    assert det.committor is q
    assert isinstance(det.weights, sc.Weights)
    assert det.result.directions.shape[0] == 128
    assert det.dirichlet_energy == det.weights.dirichlet_energy > 0
    vals = np.asarray(q(s[:20]))
    assert vals.shape == (20,)
    assert (vals >= 0).all() and (vals <= 1).all()


def test_fit_default_returns_bare_callable():
    s, in_A, in_B = two_basin_samples(n=600, seed=3)
    q = sc.fit_committor(s, in_A=in_A, in_B=in_B, n_directions=32, seed=3)
    assert callable(q)
    assert np.asarray(q(s[:5])).shape == (5,)


def test_fit_roundtrip_matches_build(fit):
    s, _, _, q, det = fit
    q2 = sc.build_committor(det.result, det.weights)
    np.testing.assert_array_equal(np.asarray(q(s[:60])), np.asarray(q2(s[:60])))


def test_single_point_returns_scalar(fit):
    s, _, _, q, _ = fit
    assert np.ndim(np.asarray(q(s[0]))) == 0


def test_committor_is_a_pure_function_of_the_point(fit):
    """The value at a point does not depend on what else is in the batch."""
    s, _, _, q, _ = fit
    batch = np.asarray(q(s[:9]))
    for i in (0, 4, 8):
        assert float(q(s[i])) == batch[i]
    np.testing.assert_array_equal(np.asarray(q(s[3:6])), batch[3:6])


def test_boundary_snapping_via_masks(fit):
    s, in_A, in_B, q, _ = fit
    qv = np.asarray(q(s, in_A=in_A, in_B=in_B))
    np.testing.assert_array_equal(qv[np.asarray(in_A)], 0.0)
    np.testing.assert_array_equal(qv[np.asarray(in_B)], 1.0)
    free = np.asarray(q(s))
    assert free.min() >= 0.0 and free.max() <= 1.0


def test_clip_off_exposes_the_raw_combination(fit):
    _, _, _, _, det = fit
    raw = sc.build_committor(det.result, det.weights, clip=False)
    clipped = sc.build_committor(det.result, det.weights)
    pts = jnp.asarray([[-1.9, 0.0], [1.9, 0.0], [0.0, 0.0]])
    r, c = np.asarray(raw(pts)), np.asarray(clipped(pts))
    np.testing.assert_array_equal(c, np.clip(r, 0.0, 1.0))


def test_gradient_matches_finite_difference(fit):
    _, _, _, q, _ = fit
    pts = jnp.asarray([[0.0, 0.0], [0.1, 0.2], [-0.2, 0.1], [0.3, -0.1], [-0.1, -0.2]])
    g = np.asarray(sc.committor_gradient(q, pts))
    assert g.shape == (5, 2)
    eps = 1e-4
    fd = np.zeros_like(g)
    for d in range(2):
        e = jnp.asarray(np.eye(2)[d] * eps)
        fd[:, d] = (np.asarray(q(pts + e)) - np.asarray(q(pts - e))) / (2 * eps)
    np.testing.assert_allclose(g, fd, atol=1e-2, rtol=5e-2)


def test_rescale_transition_spans_the_unit_interval(fit):
    s, in_A, in_B, q, _ = fit
    raw = np.asarray(q(s))
    scaled = np.asarray(sc.rescale_transition(raw, in_A, in_B))
    trans = ~(np.asarray(in_A) | np.asarray(in_B))
    assert scaled[trans].min() == pytest.approx(0.0, abs=1e-12)
    assert scaled[trans].max() == pytest.approx(1.0, abs=1e-12)
    np.testing.assert_array_equal(scaled[np.asarray(in_A)], 0.0)
    np.testing.assert_array_equal(scaled[np.asarray(in_B)], 1.0)
    # an affine map of the transition values: the ordering is preserved
    order_raw = np.argsort(raw[trans])
    order_scaled = np.argsort(scaled[trans])
    np.testing.assert_array_equal(order_raw, order_scaled)


def test_heldout_cap_through_fit():
    s, in_A, in_B = double_well_samples(n=1500, seed=4)
    _, det = sc.fit_committor(
        s,
        in_A=in_A,
        in_B=in_B,
        n_directions=32,
        n_bins=60,
        seed=4,
        heldout_cap=True,
        return_details=True,
    )
    hc = det.weights.heldout_cap
    assert hc is not None and np.isfinite(hc["cap"]) and hc["cap"] > 0
    assert hc["n_ok"] == hc["n_folds"] == 10


def test_solver_rejects_float32_with_actionable_message():
    """The weight solve must fail clearly when float64 is off (simulated)."""
    s, in_A, in_B = two_basin_samples(400)
    result = sc.compute_sliced_committor(s, in_A=in_A, in_B=in_B, n_directions=8)
    real_read = jax.config.read

    def fake_read(name):
        return False if name == "jax_enable_x64" else real_read(name)

    jax.config.read = fake_read
    try:
        with pytest.raises(ValueError, match="jax_enable_x64"):
            sc.solve_weights(result)
    finally:
        jax.config.read = real_read
