"""Tests for the 1D recalibration curve along the aggregate q-bar."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

import sliced_committor as sc
from sliced_committor.calibration import (
    apply_recalibration,
    compute_recalibration_curve,
    evaluate_committor_with_recal,
)


def _make_samples(n=700, seed=0):
    rng = np.random.default_rng(seed)
    samples = jnp.asarray(rng.standard_normal((n, 2)))
    in_A = jnp.linalg.norm(samples - jnp.asarray([-2.0, 0.0]), axis=1) < 0.7
    in_B = jnp.linalg.norm(samples - jnp.asarray([2.0, 0.0]), axis=1) < 0.7
    in_A = in_A & ~in_B
    return samples, in_A, in_B


def test_recal_curve_endpoints_pin_to_unit_interval():
    samples, in_A, in_B = _make_samples()
    result = sc.compute_sliced_committor(samples, in_A=in_A, in_B=in_B, n_directions=24, seed=0)
    raw = sc.compute_weights_multi(result, [sc.corrected_dirichlet_inv_rd])[
        "corrected_dirichlet_inv_rd"
    ]
    q_at_samples = sc.evaluate_committor(result, samples, raw)
    s_centers, q_recal = compute_recalibration_curve(q_at_samples, in_A, in_B, n_bins=80)
    q_recal = np.asarray(q_recal)
    assert q_recal.shape == s_centers.shape
    # The endpoints of the recalibration curve should land on [0, 1] within
    # the absorption tolerance of the RD solver.
    assert q_recal[0] < 0.05, f"curve start = {q_recal[0]}"
    assert q_recal[-1] > 0.95, f"curve end = {q_recal[-1]}"


def test_recal_curve_is_monotone_in_s():
    samples, in_A, in_B = _make_samples()
    result = sc.compute_sliced_committor(samples, in_A=in_A, in_B=in_B, n_directions=24, seed=1)
    raw = sc.compute_weights_multi(result, [sc.corrected_dirichlet_inv_rd])[
        "corrected_dirichlet_inv_rd"
    ]
    q_at_samples = sc.evaluate_committor(result, samples, raw)
    s_centers, q_recal = compute_recalibration_curve(q_at_samples, in_A, in_B, n_bins=120)
    q_recal = np.asarray(q_recal)
    diffs = np.diff(q_recal)
    # Allow a tiny non-monotone wiggle near the absorbing boundaries.
    assert (diffs >= -1e-3).all(), f"recalibration curve non-monotone: min step = {diffs.min():.3f}"


def test_apply_recalibration_preserves_shape():
    samples, in_A, in_B = _make_samples()
    result = sc.compute_sliced_committor(samples, in_A=in_A, in_B=in_B, n_directions=24, seed=2)
    raw = sc.compute_weights_multi(result, [sc.corrected_dirichlet_inv_rd])[
        "corrected_dirichlet_inv_rd"
    ]
    q_at_samples = sc.evaluate_committor(result, samples, raw)
    s_centers, q_recal = compute_recalibration_curve(q_at_samples, in_A, in_B)

    q_new = jnp.linspace(0.0, 1.0, 17).reshape(17, 1).repeat(3, axis=1)
    out = apply_recalibration(q_new, s_centers, q_recal)
    assert out.shape == q_new.shape


def test_evaluate_committor_with_recal_runs():
    samples, in_A, in_B = _make_samples()
    result = sc.compute_sliced_committor(samples, in_A=in_A, in_B=in_B, n_directions=24, seed=3)
    raw = sc.compute_weights_multi(result, [sc.corrected_dirichlet_inv_rd])[
        "corrected_dirichlet_inv_rd"
    ]
    q = evaluate_committor_with_recal(
        result,
        samples[:50],
        raw,
        samples,
        in_A_samples=in_A,
        in_B_samples=in_B,
    )
    assert q.shape == (50,)
    q = np.asarray(q)
    assert (q >= 0).all() and (q <= 1).all()
