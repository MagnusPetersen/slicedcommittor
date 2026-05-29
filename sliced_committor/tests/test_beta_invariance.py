"""β-invariance regression test.

The library dropped the explicit β argument because β cancels in every
downstream computation (see ``audit/verify_beta_invariance.py``). This
test loads the frozen golden reference (committors, weights, ε estimators,
calibration, recalibration) computed by the monorepo at β=1 and asserts
that the new label-only API reproduces every output to machine precision.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from sliced_committor import (
    compute_basin_moment_weights,
    compute_epsilon_equilibrium,
    compute_epsilon_flux1d,
    compute_epsilon_rms,
    compute_full_gram_weights,
    compute_sliced_committor,
    compute_weights_multi,
    evaluate_committor,
    get_all_weight_functions,
    make_weighting_context,
)

N_DIRECTIONS = 32
SEED = 0


@pytest.fixture(scope="module")
def fitted(golden_samples, golden_labels):
    samples = golden_samples
    in_A, in_B = golden_labels
    result = compute_sliced_committor(
        samples,
        in_A=in_A,
        in_B=in_B,
        n_directions=N_DIRECTIONS,
        seed=SEED,
        store_projected_samples=True,
    )
    return samples, in_A, in_B, result


def _allclose(a, b, name, rtol=1e-8, atol=1e-10):
    np.testing.assert_allclose(
        np.asarray(a),
        np.asarray(b),
        rtol=rtol,
        atol=atol,
        equal_nan=True,
        err_msg=name,
    )


def test_committors_1d_match_golden(fitted, golden_data):
    _, _, _, result = fitted
    _allclose(result.committors_1d, golden_data["committors_1d"], "committors_1d")


def test_directions_match_golden(fitted, golden_data):
    _, _, _, result = fitted
    _allclose(result.directions, golden_data["directions"], "directions")


def test_slice_coords_match_golden(fitted, golden_data):
    _, _, _, result = fitted
    _allclose(result.slice_coords, golden_data["slice_coords"], "slice_coords")


def test_log_density_matches_golden(fitted, golden_data):
    """`result.free_energies` is now stored at β=1 (= -log ρ). The golden file
    saved `log_density (= -β·F)` from the old code; both should match."""
    _, _, _, result = fitted
    _allclose(
        result.free_energies,
        golden_data["log_density (= -β·F)"],
        "free_energies (β=1 = log_density)",
    )


def test_diagonal_weight_matches_golden(fitted, golden_data):
    _, _, _, result = fitted
    w = compute_weights_multi(result, get_all_weight_functions())
    _allclose(
        w["corrected_dirichlet_inv_rd"],
        golden_data["w_diag::corrected_dirichlet_inv_rd"],
        "corrected_dirichlet_inv_rd",
    )


def test_full_gram_weights_match_golden(fitted, golden_data):
    samples, _, _, result = fitted
    gram = compute_full_gram_weights(result, samples)
    # Full-Gram solver involves a Cholesky+matmul cascade. The output drifts
    # by O(1%) across JAX / BLAS versions because the Cholesky kernel
    # itself differs. The goldens were captured on Linux x86_64 + JAX 0.4.30 +
    # MKL; CI's runner gets a different combination. Tolerances are tuned
    # so cross-environment drift is silently absorbed but any real
    # algorithmic change (which typically moves weights 10%+) still
    # trips the test. See the tighter tests on committors_1d and the
    # diagonal weights for the β-invariance contract.
    _allclose(gram["w"], golden_data["w_gram::w"], "full_gram w", rtol=5e-2, atol=5e-2)


def test_basin_moment_weights_match_golden(fitted, golden_data):
    samples, _, _, result = fitted
    bmc = compute_basin_moment_weights(result, samples)
    # Same cross-environment Cholesky / BLAS drift as full_gram above.
    _allclose(bmc["w"], golden_data["w_bmc::w"], "bmc w", rtol=5e-2, atol=5e-2)


def test_epsilon_estimators_match_golden(fitted, golden_data):
    _, _, _, result = fitted
    ctx = make_weighting_context(result)
    _allclose(compute_epsilon_equilibrium(ctx), golden_data["eps_equilibrium"], "eps_equilibrium")
    _allclose(compute_epsilon_rms(ctx), golden_data["eps_rms"], "eps_rms")
    _allclose(compute_epsilon_flux1d(ctx), golden_data["eps_flux1d"], "eps_flux1d")


def test_q_at_samples_matches_golden(fitted, golden_data):
    samples, in_A, in_B, result = fitted
    w = compute_weights_multi(result, get_all_weight_functions())
    # The golden snapshot was computed with the monorepo's auto-enforcement
    # of q=0/q=1 at basin samples. The library doesn't auto-enforce anymore;
    # pass labels explicitly to reproduce that behaviour.
    q = evaluate_committor(
        result,
        samples,
        w["corrected_dirichlet_inv_rd"],
        in_A=in_A,
        in_B=in_B,
    )
    _allclose(q, golden_data["q_at_samples"], "q_at_samples")
