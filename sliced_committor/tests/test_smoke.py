"""End-to-end smoke test for the library's public surface.

Exercises every public entry point (solver, all three weight solvers, all
three ε estimators). Does not check numerical correctness; that's covered
by ``test_beta_invariance.py``.
"""

import jax.numpy as jnp
import pytest

from sliced_committor import (
    build_committor,
    compute_basin_moment_weights,
    compute_epsilon,
    compute_epsilon_equilibrium,
    compute_epsilon_flux1d,
    compute_epsilon_rms,
    compute_full_gram_weights,
    compute_sliced_committor,
    compute_weights_multi,
    get_all_weight_functions,
    get_default_weight_functions,
    make_weighting_context,
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
    q = build_committor(result, w["corrected_dirichlet_inv_rd"])(
        samples,
        in_A=in_A,
        in_B=in_B,
    )
    # The callable committor writes 0.0 / 1.0 exactly via jnp.where.
    assert bool(jnp.all(q[in_A] == 0.0))
    assert bool(jnp.all(q[in_B] == 1.0))


def test_evaluate_without_boundary_labels_is_smooth(fit):
    samples, _, _, result = fit
    w = compute_weights_multi(result, get_default_weight_functions())
    # Without in_A / in_B, no enforcement; values may not be exactly 0/1.
    q = build_committor(result, w["corrected_dirichlet_inv_rd"])(samples)
    assert q.shape == (500,)
    assert float(jnp.min(q)) >= 0.0
    assert float(jnp.max(q)) <= 1.0


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
    q = build_committor(result, w["corrected_dirichlet_inv_rd"])(
        points,
        in_A=in_A_pts,
        in_B=in_B_pts,
    )
    assert q.shape == (3,)
    assert float(q[0]) == 0.0
    assert float(q[2]) == 1.0
