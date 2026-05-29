"""Unit tests for the plain basin-moment-constrained (BMC) solver.

Adapted from the main-repo ``tests/test_bmc.py``; copies the math rather
than imports the upstream module.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

import sliced_committor as sc
from sliced_committor import EnrichedBMCRepresentationError  # for parallel error type
from sliced_committor._bmc import BMCRepresentationError


def _make_samples(n=600, seed=0):
    rng = np.random.default_rng(seed)
    samples = jnp.asarray(rng.standard_normal((n, 2)))
    in_A = jnp.linalg.norm(samples - jnp.asarray([-2.0, 0.0]), axis=1) < 0.7
    in_B = jnp.linalg.norm(samples - jnp.asarray([2.0, 0.0]), axis=1) < 0.7
    in_A = in_A & ~in_B
    return samples, in_A, in_B


def test_bmc_constraint_residuals_machine_epsilon():
    samples, in_A, in_B = _make_samples()
    result = sc.compute_sliced_committor(samples, in_A=in_A, in_B=in_B, n_directions=24, seed=0)
    out = sc.compute_basin_moment_weights(result, samples)
    assert out["constraint_residual_A"] < 1e-6
    assert out["constraint_residual_B"] < 1e-6


def test_bmc_returns_solver_provenance():
    samples, in_A, in_B = _make_samples()
    result = sc.compute_sliced_committor(samples, in_A=in_A, in_B=in_B, n_directions=20, seed=2)
    out = sc.compute_basin_moment_weights(result, samples)
    for key in (
        "w",
        "a",
        "b",
        "G",
        "A",
        "B",
        "C",
        "det",
        "cond_basin",
        "condition_number",
        "constraint_residual_A",
        "constraint_residual_B",
        "sum_w",
    ):
        assert key in out, f"BMC result missing key {key}"


def test_bmc_reports_cond_basin_diagnostic():
    """Even on a clean setup, the cond_basin diagnostic must be populated and
    sit in [0, 1] so callers can decide whether the projection basis is
    discriminating enough.
    """
    samples, in_A, in_B = _make_samples()
    result = sc.compute_sliced_committor(samples, in_A=in_A, in_B=in_B, n_directions=24, seed=5)
    out = sc.compute_basin_moment_weights(result, samples)
    assert 0.0 <= out["cond_basin"] <= 1.0 + 1e-9


def test_bmc_allows_override_with_raise_on_degenerate_false():
    """`raise_on_degenerate=False` should always return a weight vector
    regardless of conditioning.
    """
    samples, in_A, in_B = _make_samples(n=400, seed=7)
    result = sc.compute_sliced_committor(samples, in_A=in_A, in_B=in_B, n_directions=16, seed=7)
    out = sc.compute_basin_moment_weights(result, samples, raise_on_degenerate=False)
    assert out["w"].shape == (16,)
