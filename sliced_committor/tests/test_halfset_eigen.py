"""Gates for the halfset-eigen Gram regularization (``tikhonov='halfset_eigen'``).

The mode is the RECOVAR sec. 1 transfer graduated into the production library:
two interleaved contiguous basin-stratified half-Grams measure the per-eigenband
SSNR of G, a Wiener factor damps each band, and the EBMC closed form runs on the
regularized Gram with zero additional ridge. Validated in
``docs/recovar_transfers.md`` (EXP-A); the vendored reference implementation
lives in ``lib/recovar/regularize.py``, which these tests compare against.
"""

import jax
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp

from sliced_committor.core.committor import fit_committor

HALFSET_KW = {"tikhonov": "halfset_eigen"}


def _wolfe_quapp_fit(n=4000, m=64, seed=0):
    """Small 2D rotated Wolfe-Quapp fit at beta=1.0 (exact Boltzmann samples).

    The sampler and state definitions come from ``recovar.systems`` (the
    frozen package under comparison); the fit itself is the production
    ``fit_committor`` one-liner with the new tikhonov mode.
    """
    systems = pytest.importorskip("recovar.systems")

    cfg = systems.wolfe_quapp_config(beta=1.0, dim=2)
    X = systems.grid_boltzmann_samples(cfg, n, rng=seed)
    in_A, in_B = cfg.in_A(X), cfg.in_B(X)
    assert in_A.sum() > 50 and in_B.sum() > 50, "basins under-populated"
    q, fit = fit_committor(
        jnp.asarray(X),
        in_A=jnp.asarray(in_A),
        in_B=jnp.asarray(in_B),
        n_directions=m,
        seed=seed,
        return_details=True,
        n_bins=50,
        n_min=10,
        binning_method="quantile",
        weights="ebmc",
        weight_kwargs=dict(HALFSET_KW),
    )
    return q, fit, X, in_A, in_B


# --------------------------------------------------------------------------- #
# parity with the lib/recovar reference path
# --------------------------------------------------------------------------- #
def test_parity_with_the_recovar_reference_path():
    """The one-liner must reproduce the validated EXP-A arm (iv) arithmetic.

    Reference path: ``halfset_grams_from_arrays`` (interleaved contiguous
    stratified fold blocks) -> valid-restricted ``halfset_eigen_regularize``
    -> ``solve_with_G``, applied to the SAME extracted (F, W, cos, a, b).
    """
    recovar = pytest.importorskip("recovar")
    from sliced_committor.core.gram import _compute_derivative_matrix
    from sliced_committor.core.solver import make_weighting_context

    _q, fit, _X, in_A, in_B = _wolfe_quapp_fit(n=4000, m=64, seed=0)
    res = fit.result
    ctx = make_weighting_context(res)
    F = _compute_derivative_matrix(ctx, res.projected_samples)

    G1, G2 = recovar.halfset_grams_from_arrays(
        F, None, np.asarray(ctx.cos_matrix, np.float64), in_A, in_B, n_folds=10
    )
    valid = np.asarray(res.valid_mask, bool)
    iv = np.flatnonzero(valid)
    G_reg_ref, info = recovar.halfset_eigen_regularize(
        np.asarray(G1)[np.ix_(iv, iv)], np.asarray(G2)[np.ix_(iv, iv)], n_bands=12
    )
    a = np.asarray(fit.weights["a"], np.float64)
    b = np.asarray(fit.weights["b"], np.float64)
    sol = recovar.solve_with_G(G_reg_ref, a[iv], b[iv])
    w_ref = np.zeros(len(valid))
    w_ref[iv] = sol["w"]

    w_lib = np.asarray(fit.weights["w"], np.float64)
    assert np.allclose(w_lib, w_ref, rtol=1e-8, atol=0.0)
    assert np.isclose(float(fit.weights["M_gap"]), sol["M_gap"], rtol=1e-8)
    assert np.isclose(float(fit.weights["c"]), sol["c"], rtol=1e-8)

    # The regularized Gram and the recorded diagnostics agree with the
    # reference regularizer on the valid block.
    G_reg_lib = np.asarray(fit.weights["G_reg"], np.float64)
    assert np.allclose(G_reg_lib[np.ix_(iv, iv)], G_reg_ref, rtol=1e-10, atol=1e-14)
    diag = fit.weights["halfset_eigen"]
    assert diag["n_folds"] == 10 and diag["n_bands"] == 12
    assert np.allclose(diag["band_ssnr"], info["band_ssnr"], rtol=1e-10)
    assert np.isclose(diag["lam_max"], float(np.max(info["lam"])), rtol=1e-12)
    assert np.isclose(diag["lam_min_reg"], float(np.min(info["lam_reg"])), rtol=1e-12)
    # No scalar ridge is applied on top of the regularized Gram.
    assert float(fit.weights["eta_used"]) == 0.0


# --------------------------------------------------------------------------- #
# invariants
# --------------------------------------------------------------------------- #
def test_regularized_gram_is_psd_and_diagnostics_are_finite():
    _q, fit, *_ = _wolfe_quapp_fit(n=4000, m=32, seed=1)
    G_reg = np.asarray(fit.weights["G_reg"], np.float64)
    eigs = np.linalg.eigvalsh(0.5 * (G_reg + G_reg.T))
    assert eigs.min() > 0.0, "halfset-eigen G_reg must be strictly PSD (floored)"
    diag = fit.weights["halfset_eigen"]
    assert diag["lam_min_reg"] > 0.0
    assert np.all(np.isfinite(diag["band_ssnr"]))
    assert np.isfinite(float(fit.weights["M_gap"])) and float(fit.weights["M_gap"]) > 0


def test_runs_with_invalid_directions_and_zeroes_their_weights():
    """The valid restriction must happen BEFORE the eigendecomposition (the
    zero rows of masked directions would poison the low eigen-bands), the
    masked weights stay zero, and the embedding uses decoupled identity rows
    (the ``_mask_and_regularize_gram`` contract)."""
    from sliced_committor.core.solver import (
        compute_enriched_basin_moment_weights,
        compute_sliced_committor,
    )

    from ._helpers import two_basin_samples

    samples, in_A, in_B = two_basin_samples(n=3000, dim=2, seed=0)
    res = compute_sliced_committor(
        samples,
        in_A=in_A,
        in_B=in_B,
        n_directions=12,
        seed=0,
        n_bins=30,
        n_min=10,
        binning_method="quantile",
    )
    vm = np.asarray(res.valid_mask, bool).copy()
    vm[[0, 5]] = False  # force invalid directions (suite convention)
    res2 = res._replace(valid_mask=jnp.asarray(vm))

    w_dict = compute_enriched_basin_moment_weights(res2, samples, tikhonov="halfset_eigen")
    valid = np.asarray(res2.valid_mask, bool)
    assert not valid.all() and valid.sum() >= 2
    w = np.asarray(w_dict["w"])
    assert np.all(w[~valid] == 0.0)
    assert np.all(np.isfinite(w))
    assert np.isfinite(float(w_dict["M_gap"])) and float(w_dict["M_gap"]) > 0
    G_reg = np.asarray(w_dict["G_reg"], np.float64)
    for j in np.flatnonzero(~valid):
        assert G_reg[j, j] == 1.0
        assert np.all(G_reg[j, np.arange(len(valid)) != j] == 0.0)
    eigs = np.linalg.eigvalsh(0.5 * (G_reg + G_reg.T))
    assert eigs.min() > 0.0


def test_deterministic_across_repeated_fits():
    _q1, f1, *_ = _wolfe_quapp_fit(n=3000, m=24, seed=2)
    _q2, f2, *_ = _wolfe_quapp_fit(n=3000, m=24, seed=2)
    assert np.array_equal(np.asarray(f1.weights["w"]), np.asarray(f2.weights["w"]))
    assert float(f1.weights["M_gap"]) == float(f2.weights["M_gap"])
    assert f1.weights["halfset_eigen"]["band_ssnr"] == f2.weights["halfset_eigen"]["band_ssnr"]


# --------------------------------------------------------------------------- #
# smoke: the one-liner
# --------------------------------------------------------------------------- #
def test_fit_committor_one_liner_returns_callable_with_finite_energy():
    rng = np.random.default_rng(0)
    N = 6000
    X = rng.normal(size=(N, 2)) * 0.6
    X[: N // 2, 0] -= 1.5
    X[N // 2 :, 0] += 1.5
    in_A, in_B = X[:, 0] < -1.2, X[:, 0] > 1.2

    q, fit = fit_committor(
        X,
        in_A=in_A,
        in_B=in_B,
        n_directions=24,
        seed=0,
        return_details=True,
        n_bins=40,
        n_min=10,
        binning_method="quantile",
        weights="ebmc",
        weight_kwargs=dict(HALFSET_KW),
    )
    assert callable(q)
    assert np.isfinite(fit.dirichlet_energy) and fit.dirichlet_energy > 0
    vals = np.asarray(q(np.array([[-2.0, 0.0], [0.0, 0.0], [2.0, 0.0]])))
    assert vals[0] < 0.15 < vals[1] < 0.85 < vals[2]
