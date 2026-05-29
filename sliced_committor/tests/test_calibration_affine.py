"""Tests for affine label-mean calibration (centered-basis) and ABC v2."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

import sliced_committor as sc
from sliced_committor.calibration import (
    apply_affine,
    calibrate_weights_affine,
    compute_affine_calibration,
    evaluate_committor_calibrated,
)


def _make_samples(n=500, seed=0):
    rng = np.random.default_rng(seed)
    samples = jnp.asarray(rng.standard_normal((n, 2)))
    in_A = jnp.linalg.norm(samples - jnp.asarray([-2.0, 0.0]), axis=1) < 0.7
    in_B = jnp.linalg.norm(samples - jnp.asarray([2.0, 0.0]), axis=1) < 0.7
    in_A = in_A & ~in_B
    return samples, in_A, in_B


def _basin_means(q, in_A, in_B):
    q = np.asarray(q)
    in_A = np.asarray(in_A)
    in_B = np.asarray(in_B)
    return float(q[in_A].mean()), float(q[in_B].mean())


def test_calibrate_weights_affine_pins_basin_means_global():
    samples, in_A, in_B = _make_samples()
    result = sc.compute_sliced_committor(samples, in_A=in_A, in_B=in_B, n_directions=24, seed=0)
    raw = sc.compute_weights_multi(result, [sc.corrected_dirichlet_inv_rd])[
        "corrected_dirichlet_inv_rd"
    ]
    cal = calibrate_weights_affine(result, raw, mode="global")
    q_cal = sc.evaluate_committor(result, samples, cal)
    muA, muB = _basin_means(q_cal, in_A, in_B)
    # Mean pinning is exact pre-clip; the clip applied inside evaluate_committor
    # can shrink the basin means slightly so we allow a small tolerance.
    assert abs(muA) < 0.05, f"mu_A={muA:.3f}; should pin to 0"
    assert abs(muB - 1.0) < 0.05, f"mu_B={muB:.3f}; should pin to 1"


def test_calibrate_weights_affine_per_slice_runs():
    samples, in_A, in_B = _make_samples()
    result = sc.compute_sliced_committor(samples, in_A=in_A, in_B=in_B, n_directions=24, seed=2)
    raw = sc.compute_weights_multi(result, [sc.corrected_dirichlet_inv_rd])[
        "corrected_dirichlet_inv_rd"
    ]
    cal = calibrate_weights_affine(result, raw, mode="per_slice")
    # The dict should be evaluator-consumable.
    q = sc.evaluate_committor(result, samples[:50], cal)
    assert q.shape == (50,)


def test_abc_v2_affine_calibration_pins_means():
    samples, in_A, in_B = _make_samples(n=800)
    result = sc.compute_sliced_committor(samples, in_A=in_A, in_B=in_B, n_directions=24, seed=3)
    raw = sc.compute_weights_multi(result, [sc.corrected_dirichlet_inv_rd])[
        "corrected_dirichlet_inv_rd"
    ]
    cal = compute_affine_calibration(
        result, raw, samples=samples, in_A=in_A, in_B=in_B, n_calib=400, seed=1
    )
    assert not cal.is_degenerate

    q_raw = sc.evaluate_committor(result, samples, raw)
    q_cal = apply_affine(q_raw, cal.alpha, cal.beta, clip=True)
    muA, muB = _basin_means(q_cal, in_A, in_B)
    assert abs(muA) < 0.15
    assert abs(muB - 1.0) < 0.15


def test_abc_v2_degenerate_fallback_identity():
    """When the basin set is too small to calibrate, ABC v2 must fall back to
    identity rather than crash."""
    samples, in_A, in_B = _make_samples()
    # Reduce basin A to <10 samples.
    a_indices = np.nonzero(np.asarray(in_A))[0]
    keep = a_indices[: max(1, len(a_indices) // 100)]
    new_in_A = np.zeros_like(np.asarray(in_A))
    new_in_A[keep] = True
    in_A_small = jnp.asarray(new_in_A)
    result = sc.compute_sliced_committor(
        samples, in_A=in_A_small, in_B=in_B, n_directions=12, seed=0
    )
    raw = sc.compute_weights_multi(result, [sc.corrected_dirichlet_inv_rd])[
        "corrected_dirichlet_inv_rd"
    ]
    with pytest.warns(UserWarning, match="too few basin samples"):
        cal = compute_affine_calibration(
            result,
            raw,
            samples=samples,
            in_A=in_A_small,
            in_B=in_B,
        )
    assert cal.is_degenerate
    assert cal.alpha == 1.0 and cal.beta == 0.0


def test_evaluate_committor_calibrated_dispatch():
    samples, in_A, in_B = _make_samples(n=700)
    result = sc.compute_sliced_committor(samples, in_A=in_A, in_B=in_B, n_directions=24, seed=4)
    raw = sc.compute_weights_multi(result, [sc.corrected_dirichlet_inv_rd])[
        "corrected_dirichlet_inv_rd"
    ]
    # ``evaluate_committor_calibrated`` internally calls compute_affine_calibration
    # and returns ``(q, AffineCalibration)``.
    q, cal = evaluate_committor_calibrated(
        result,
        samples[:50],
        raw,
        samples,
        in_A=in_A,
        in_B=in_B,
    )
    assert q.shape == (50,)
    assert not cal.is_degenerate
