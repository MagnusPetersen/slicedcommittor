"""Tests for the three boundary-error (epsilon) estimators and dispatcher."""

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

import sliced_committor as sc
from sliced_committor.core.weights import (
    compute_epsilon,
    compute_epsilon_equilibrium,
    compute_epsilon_flux1d,
    compute_epsilon_rms,
)


def _ctx():
    rng = np.random.default_rng(0)
    samples = jnp.asarray(rng.standard_normal((500, 2)))
    in_A = jnp.linalg.norm(samples - jnp.asarray([-2.0, 0.0]), axis=1) < 0.7
    in_B = jnp.linalg.norm(samples - jnp.asarray([2.0, 0.0]), axis=1) < 0.7
    in_A = in_A & ~in_B
    result = sc.compute_sliced_committor(samples, in_A=in_A, in_B=in_B, n_directions=16, seed=0)
    return sc.make_weighting_context(result)


def test_three_estimators_return_M_vector():
    ctx = _ctx()
    M = ctx.directions.shape[0]
    for fn in (compute_epsilon_equilibrium, compute_epsilon_rms, compute_epsilon_flux1d):
        eps = fn(ctx)
        assert eps.shape == (M,), f"{fn.__name__}: shape {eps.shape}"


def test_rms_ge_equilibrium_pointwise():
    """Cauchy-Schwarz: the RMS epsilon dominates the equilibrium one on every
    direction (README guarantee)."""
    ctx = _ctx()
    eps_eq = np.asarray(compute_epsilon_equilibrium(ctx))
    eps_rms = np.asarray(compute_epsilon_rms(ctx))
    # Allow tiny float slack: RMS ≥ equilibrium on every valid direction.
    valid = np.asarray(ctx.valid_mask)
    diff = (eps_rms - eps_eq)[valid]
    assert (diff > -1e-9).all(), (
        f"RMS < equilibrium on {(diff < -1e-9).sum()} directions; min diff = {diff.min()}"
    )


def test_dispatcher_default_is_equilibrium():
    ctx = _ctx()
    eps_default = compute_epsilon(ctx)
    eps_eq = compute_epsilon_equilibrium(ctx)
    np.testing.assert_allclose(np.asarray(eps_default), np.asarray(eps_eq))


def test_dispatcher_string_name_routes_correctly():
    ctx = _ctx()
    eps_rms_named = compute_epsilon(ctx, epsilon_fn="rms")
    eps_rms_direct = compute_epsilon_rms(ctx)
    np.testing.assert_allclose(np.asarray(eps_rms_named), np.asarray(eps_rms_direct))


def test_dispatcher_callable_passthrough():
    ctx = _ctx()
    eps = compute_epsilon(ctx, epsilon_fn=compute_epsilon_rms)
    assert eps.shape == (ctx.directions.shape[0],)


def test_dispatcher_clamps_above_one_with_warning():
    """clamp=True caps epsilon at 1.0 (per docstring)."""
    ctx = _ctx()
    eps = np.asarray(compute_epsilon(ctx, clamp=True))
    assert (eps <= 1.0 + 1e-9).all()
