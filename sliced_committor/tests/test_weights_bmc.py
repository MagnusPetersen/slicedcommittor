"""Unit tests for the plain basin-moment-constrained (BMC) solver.

Adapted from the main-repo ``tests/test_bmc.py``; copies the math rather
than imports the upstream module.
"""

import jax

jax.config.update("jax_enable_x64", True)

import sliced_committor as sc

from ._helpers import TOL_SOLVER, two_basin_samples


def _make_samples(n=600, seed=0):
    return two_basin_samples(n, seed=seed)


def test_bmc_constraint_residuals_machine_epsilon():
    samples, in_A, in_B = _make_samples()
    result = sc.compute_sliced_committor(samples, in_A=in_A, in_B=in_B, n_directions=24, seed=0)
    out = sc.compute_basin_moment_weights(result, samples)
    assert out["constraint_residual_A"] < TOL_SOLVER
    assert out["constraint_residual_B"] < TOL_SOLVER


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
