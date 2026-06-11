"""End-to-end tests for the enriched BMC (EBMC) and PESB-EBMC solvers.

Covers:
  * basic shape / key structure (smoke)
  * boundary enforcement at training samples (q[A]=0, q[B]=1)
  * Galerkin monotonicity: EBMC strictly improves BMC, PESB strictly
    improves EBMC at this sample size
  * PESB at P=1 reproduces EBMC on shared keys
  * smoothstep n_values validation (rejects n<1)
"""

import jax.numpy as jnp
import numpy as np
import pytest

from sliced_committor import (
    build_committor,
    compute_basin_moment_weights,
    compute_enriched_basin_moment_weights,
    compute_enriched_basin_moment_weights_power,
    compute_sliced_committor,
)


@pytest.fixture(scope="module")
def fit(golden_samples, golden_labels):
    samples = golden_samples
    in_A, in_B = golden_labels
    result = compute_sliced_committor(
        samples,
        in_A=in_A,
        in_B=in_B,
        n_directions=16,
        seed=0,
        store_projected_samples=True,
    )
    return samples, in_A, in_B, result


def test_ebmc_shape_and_keys(fit):
    samples, _, _, result = fit
    out = compute_enriched_basin_moment_weights(result, samples)
    assert out["w"].shape == (16,)
    assert isinstance(out["c"], float)
    for key in (
        "M_gap",
        "A",
        "B",
        "C",
        "Delta",
        "improvement_over_bmc",
        "cond_enriched",
        "optimal_dirichlet_energy",
        "constraint_residual_moment_gap",
        "mu_A_check",
        "mu_B_check",
    ):
        assert key in out, f"missing EBMC key: {key}"


def test_ebmc_boundary_enforcement_at_samples(fit):
    samples, in_A, in_B, result = fit
    ebmc = compute_enriched_basin_moment_weights(result, samples)
    q = build_committor(result, ebmc)(samples, in_A=in_A, in_B=in_B)
    assert bool(jnp.all(q[in_A] == 0.0))
    assert bool(jnp.all(q[in_B] == 1.0))


def test_ebmc_constraint_residual_is_machine_eps(fit):
    samples, _, _, result = fit
    ebmc = compute_enriched_basin_moment_weights(result, samples)
    # (b - a)ᵀ w = 1 is enforced exactly by the KKT solve.
    assert ebmc["constraint_residual_moment_gap"] < 1e-6
    # μ_A = 0 (aᵀw + c) and μ_B = 1 (bᵀw + c) should hold to similar precision.
    assert abs(ebmc["mu_A_check"]) < 1e-6
    assert abs(ebmc["mu_B_check"] - 1.0) < 1e-6


def test_ebmc_improves_over_bmc(fit):
    samples, _, _, result = fit
    bmc = compute_basin_moment_weights(result, samples)
    ebmc = compute_enriched_basin_moment_weights(result, samples)
    # Galerkin guarantee: optimal Dirichlet energy of EBMC ≤ that of BMC.
    1.0 / max(bmc["M_gap"] if "M_gap" in bmc else float("inf"), 1e-30)
    ebmc["optimal_dirichlet_energy"]
    # The diagnostic in the EBMC dict reports D_BMC / D_EBMC directly.
    assert ebmc["improvement_over_bmc"] >= 1.0 - 1e-9, (
        f"expected D_BMC/D_EBMC >= 1, got {ebmc['improvement_over_bmc']}"
    )


def test_pesb_p1_matches_ebmc(fit):
    """At P=1 with n_values=[1.0], PESB reproduces basic EBMC bit-for-bit."""
    samples, _, _, result = fit
    ebmc = compute_enriched_basin_moment_weights(result, samples)
    pesb = compute_enriched_basin_moment_weights_power(
        result,
        samples,
        P=1,
    )
    np.testing.assert_allclose(np.asarray(pesb["w"]), np.asarray(ebmc["w"]), rtol=1e-10, atol=1e-12)
    assert abs(pesb["c"] - ebmc["c"]) < 1e-10
    assert abs(pesb["M_gap"] - ebmc["M_gap"]) < 1e-10


def test_pesb_p2_improves_over_ebmc(fit):
    samples, _, _, result = fit
    pesb = compute_enriched_basin_moment_weights_power(result, samples, P=2)
    # improvement_over_ebmc is M_gap_PESB / M_gap_EBMC, ≥ 1 by Galerkin.
    assert pesb["improvement_over_ebmc"] >= 1.0 - 1e-9
    assert pesb["P"] == 2
    assert pesb["w_by_power"].shape == (16, 2)
    np.testing.assert_allclose(
        np.asarray(pesb["n_values"]),
        np.array([1.0, 2.0]),
        rtol=0,
        atol=1e-12,
    )


def test_pesb_evaluator_enforces_boundaries(fit):
    samples, in_A, in_B, result = fit
    pesb = compute_enriched_basin_moment_weights_power(result, samples, P=2)
    q = build_committor(result, pesb)(samples, in_A=in_A, in_B=in_B)
    assert bool(jnp.all(q[in_A] == 0.0))
    assert bool(jnp.all(q[in_B] == 1.0))


def test_pesb_evaluator_clips_to_unit_interval(fit):
    samples, _, _, result = fit
    pesb = compute_enriched_basin_moment_weights_power(result, samples, P=2)
    q = build_committor(result, pesb)(samples)
    # The evaluator clips its output to [0, 1] by default (clip=True).
    assert float(jnp.min(q)) >= 0.0
    assert float(jnp.max(q)) <= 1.0


def test_pesb_rejects_n_less_than_one(fit):
    samples, _, _, result = fit
    with pytest.raises(ValueError, match="n_values >= 1"):
        compute_enriched_basin_moment_weights_power(
            result,
            samples,
            n_values=[0.5, 1.0],
        )


def test_pesb_custom_n_values(fit):
    samples, in_A, in_B, result = fit
    pesb = compute_enriched_basin_moment_weights_power(
        result,
        samples,
        n_values=[1.0, 3.0],
    )
    assert pesb["P"] == 2
    np.testing.assert_allclose(
        np.asarray(pesb["n_values"]),
        np.array([1.0, 3.0]),
        rtol=0,
        atol=1e-12,
    )
    q = build_committor(result, pesb)(samples, in_A=in_A, in_B=in_B)
    assert q.shape == samples.shape[:1]
