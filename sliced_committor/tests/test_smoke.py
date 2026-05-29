"""End-to-end smoke test for the library's public surface.

Exercises every public entry point (solver, all three weight solvers, all
three ε estimators, calibration, recalibration). Does not check numerical
correctness; that's covered by ``test_beta_invariance.py``.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from sliced_committor import (
    compute_basin_moment_weights,
    compute_epsilon,
    compute_epsilon_equilibrium,
    compute_epsilon_flux1d,
    compute_epsilon_rms,
    compute_full_gram_weights,
    compute_sliced_committor,
    compute_weights_multi,
    evaluate_committor,
    get_all_weight_functions,
    get_default_weight_functions,
    make_weighting_context,
)
from sliced_committor.calibration import (
    apply_affine,
    apply_recalibration,
    calibrate_weights_affine,
    compute_affine_calibration,
    compute_recalibration_curve,
    evaluate_committor_calibrated,
    evaluate_committor_with_recal,
)


@pytest.fixture(scope="module")
def fit(golden_samples, golden_labels):
    samples = golden_samples
    in_A, in_B = golden_labels
    return (
        samples,
        in_A,
        in_B,
        compute_sliced_committor(
            samples,
            in_A=in_A,
            in_B=in_B,
            n_directions=16,
            seed=0,
            store_projected_samples=True,
        ),
    )


def test_solver_runs(fit):
    samples, _, _, result = fit
    M, N, dim = 16, samples.shape[0], samples.shape[1]
    assert result.directions.shape[0] == M
    assert result.directions.shape[1] == dim
    assert result.committors_1d.shape[0] == M
    assert result.in_A.shape == (N,)
    assert result.in_B.shape == (N,)


def test_diagonal_weights(fit):
    samples, _, _, result = fit
    w = compute_weights_multi(result, get_default_weight_functions())
    assert "corrected_dirichlet_inv_rd" in w
    assert w["corrected_dirichlet_inv_rd"].shape == (16,)


def test_full_gram_solver(fit):
    samples, _, _, result = fit
    out = compute_full_gram_weights(result, samples)
    assert out["w"].shape == (16,)
    assert "G" in out


def test_bmc_solver(fit):
    samples, _, _, result = fit
    out = compute_basin_moment_weights(result, samples)
    assert out["w"].shape == (16,)
    assert "a" in out and "b" in out


def test_epsilon_estimators(fit):
    _, _, _, result = fit
    ctx = make_weighting_context(result)
    eps_eq = compute_epsilon_equilibrium(ctx)
    eps_rms = compute_epsilon_rms(ctx)
    eps_flux = compute_epsilon_flux1d(ctx)
    eps_disp = compute_epsilon(ctx)
    assert eps_eq.shape == (16,)
    assert eps_rms.shape == (16,)
    assert eps_flux.shape == (16,)
    assert eps_disp.shape == (16,)


def test_evaluate_with_boundary_labels(fit):
    samples, in_A, in_B, result = fit
    w = compute_weights_multi(result, get_default_weight_functions())
    q = evaluate_committor(
        result,
        samples,
        w["corrected_dirichlet_inv_rd"],
        in_A=in_A,
        in_B=in_B,
    )
    # `evaluate_committor` writes 0.0 / 1.0 exactly via jnp.where.
    assert bool(jnp.all(q[in_A] == 0.0))
    assert bool(jnp.all(q[in_B] == 1.0))


def test_evaluate_without_boundary_labels_is_smooth(fit):
    samples, _, _, result = fit
    w = compute_weights_multi(result, get_default_weight_functions())
    # Without in_A / in_B, no enforcement; values may not be exactly 0/1.
    q = evaluate_committor(result, samples, w["corrected_dirichlet_inv_rd"])
    assert q.shape == (500,)
    assert float(jnp.min(q)) >= 0.0
    assert float(jnp.max(q)) <= 1.0


def test_calibration_affine(fit):
    samples, in_A, in_B, result = fit
    w = compute_weights_multi(result, get_default_weight_functions())
    abc_dict = calibrate_weights_affine(
        result,
        w["corrected_dirichlet_inv_rd"],
        samples=samples,
    )
    assert "w" in abc_dict and "c" in abc_dict and "q_bar" in abc_dict


def test_abc_v2(fit):
    samples, in_A, in_B, result = fit
    w = compute_weights_multi(result, get_default_weight_functions())
    cal = compute_affine_calibration(
        result,
        w["corrected_dirichlet_inv_rd"],
        samples=samples,
    )
    assert np.isfinite(cal.alpha)
    assert np.isfinite(cal.beta)
    q = apply_affine(jnp.linspace(0.0, 1.0, 10), cal.alpha, cal.beta)
    assert q.shape == (10,)


def test_recalibration(fit):
    samples, in_A, in_B, result = fit
    w = compute_weights_multi(result, get_default_weight_functions())
    q = evaluate_committor(result, samples, w["corrected_dirichlet_inv_rd"])
    s_centers, q_recal = compute_recalibration_curve(q, in_A, in_B)
    assert s_centers.shape == (200,)
    assert q_recal.shape == (200,)
    q_applied = apply_recalibration(q, s_centers, q_recal)
    assert q_applied.shape == q.shape


def test_disjoint_states_check():
    """Solver must reject in_A and in_B that overlap."""
    samples = jnp.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    in_A = jnp.array([True, True, False])
    in_B = jnp.array([True, False, True])  # first sample is in both
    with pytest.raises(ValueError, match="disjoint"):
        compute_sliced_committor(samples, in_A=in_A, in_B=in_B, n_directions=4)


def test_evaluator_accepts_bool_labels_at_eval_points(fit):
    samples, _, _, result = fit
    w = compute_weights_multi(result, get_default_weight_functions())
    points = jnp.array([[-1.5, 0.0], [0.0, 0.0], [1.5, 0.0]])
    in_A_pts = jnp.array([True, False, False])
    in_B_pts = jnp.array([False, False, True])
    q = evaluate_committor(
        result,
        points,
        w["corrected_dirichlet_inv_rd"],
        in_A=in_A_pts,
        in_B=in_B_pts,
    )
    assert q.shape == (3,)
    assert float(q[0]) == 0.0
    assert float(q[2]) == 1.0
