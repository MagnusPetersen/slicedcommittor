"""Frozen-reference gates: the numerics the published results depend on.

Three reference files, all produced ONCE at tag ``v0.6.0`` by
``golden/freeze_1_0_references.py`` (which cannot run against this version):

* ``beta_invariance_golden.npz``: the slice basis on a 500-sample fixture.
* ``ebmc_golden.npz``: the weight solve (``auto`` and ``halfset_eigen``), the
  committor at fixed points, and the held-out cap on that fixture.
* ``halfset_reference.npz``: the eigenband regulariser against the numpy
  prototype it was ported from, the Wolfe-Quapp sample generator, and a
  full Wolfe-Quapp fit (weights, committor on a grid, held-out cap).

The half-set path is the paper's; it must stay bit-for-bit. The ``auto``
ridge and the basis are allowed rounding drift across JAX/LAPACK versions.
"""

import os

import jax
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp

import sliced_committor as sc
from sliced_committor.core import _halfset as hs

from ._helpers import wolfe_quapp_samples

GOLDEN = os.path.join(os.path.dirname(__file__), "golden")
EXACT = dict(rtol=0.0, atol=0.0)
ROUNDING = dict(rtol=1e-8, atol=1e-12)


def _load(name):
    return np.load(os.path.join(GOLDEN, name), allow_pickle=False)


def close(a, b, **tol):
    np.testing.assert_allclose(np.asarray(a, np.float64), np.asarray(b, np.float64), **tol)


@pytest.fixture(scope="module")
def basis(golden_samples, golden_labels):
    in_A, in_B = golden_labels
    return sc.compute_sliced_committor(
        golden_samples, in_A=in_A, in_B=in_B, n_directions=32, seed=0
    )


# --------------------------------------------------------------------------- #
# the slice basis
# --------------------------------------------------------------------------- #
def test_basis_matches_golden(basis, golden_data):
    close(basis.directions, golden_data["directions"], **EXACT)
    close(basis.slice_coords, golden_data["slice_coords"], **ROUNDING)
    close(basis.free_energies, golden_data["log_density (= -β·F)"], **ROUNDING)
    close(basis.committors_1d, golden_data["committors_1d"], **ROUNDING)
    assert bool(jnp.all(basis.valid_mask))


# --------------------------------------------------------------------------- #
# the weight solve
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("tikhonov,tol", [("auto", ROUNDING), ("halfset_eigen", EXACT)])
def test_weights_match_golden(basis, golden_samples, golden_labels, tikhonov, tol):
    e = _load("ebmc_golden.npz")
    k = f"ebmc_{tikhonov}"
    w = sc.solve_weights(basis, tikhonov=tikhonov)
    close(w.w, e[f"{k}::w"], **tol)
    close(w.c, e[f"{k}::c"], **tol)
    close(w.moment_gap, e[f"{k}::M_gap"], **tol)
    close(w.dirichlet_energy, e[f"{k}::energy"], **tol)
    close(w.diagnostics["a"], e[f"{k}::a"], **EXACT)
    close(w.diagnostics["b"], e[f"{k}::b"], **EXACT)
    q = sc.build_committor(basis, w)
    pts = jnp.asarray(e["eval_points"])
    in_A, in_B = golden_labels
    close(q(pts), e[f"{k}::q_ref_eval_points_free"], **tol)
    close(q(golden_samples, in_A=in_A, in_B=in_B), e[f"{k}::q_at_samples_snapped"], **tol)


def test_halfset_diagnostics_match_golden(basis):
    e = _load("ebmc_golden.npz")
    w = sc.solve_weights(basis, tikhonov="halfset_eigen")
    assert w.ridge == 0.0
    close(w.diagnostics["G_reg"], e["ebmc_halfset_eigen::G_reg"], **EXACT)
    close(w.diagnostics["band_ssnr"], e["ebmc_halfset_eigen::band_ssnr"], **EXACT)


def test_heldout_cap_matches_golden(basis):
    e = _load("ebmc_golden.npz")
    hc = sc.solve_weights(basis, tikhonov="halfset_eigen", heldout_cap=True).heldout_cap
    close(hc["cap"], e["ebmc_halfset_eigen::heldout_cap"], **EXACT)
    close(hc["per_fold"], e["ebmc_halfset_eigen::heldout_per_fold"], **EXACT)
    close(hc["se"], e["ebmc_halfset_eigen::heldout_se"], **ROUNDING)
    assert np.isfinite(hc["gap"])


# --------------------------------------------------------------------------- #
# the eigenband regulariser against the prototype it was ported from
# --------------------------------------------------------------------------- #
def test_eigenband_regulariser_matches_reference():
    h = _load("halfset_reference.npz")
    for M, nb, seed in h["spd_cases"]:
        k = f"spd::{M}_{nb}_{seed}"
        R, info = hs.halfset_eigen_regularize(h[k + "::G1"], h[k + "::G2"], n_bands=int(nb))
        close(R, h[k + "::G_reg"], rtol=1e-10, atol=1e-14)
        close(info["band_ssnr"], h[k + "::band_ssnr"], rtol=1e-10, atol=0)
        ev = np.linalg.eigvalsh(R)
        assert ev.min() >= -1e-10 * ev.max()
        assert np.all(info["lam_reg"] >= np.maximum(info["lam"], 0.0) - 1e-12 * ev.max())


# --------------------------------------------------------------------------- #
# Wolfe-Quapp: generator, fit, cap
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("n,seed", [(4000, 0), (3000, 2), (6000, 0), (20000, 0)])
def test_wolfe_quapp_generator_matches_reference(n, seed):
    h = _load("halfset_reference.npz")
    X, in_A, in_B = wolfe_quapp_samples(n, seed)
    close(X, h[f"wq::X_{n}_{seed}"], **EXACT)
    np.testing.assert_array_equal(in_A, h[f"wq::in_A_{n}_{seed}"])
    np.testing.assert_array_equal(in_B, h[f"wq::in_B_{n}_{seed}"])


def test_wolfe_quapp_fit_matches_reference():
    """The full one-liner at the paper's ridge, bit for bit, and against the
    independent numpy reference path (``halfset_grams`` -> regularise -> solve)."""
    h = _load("halfset_reference.npz")
    X, in_A, in_B = wolfe_quapp_samples(4000, 0)
    q, fit = sc.fit_committor(
        jnp.asarray(X),
        in_A=jnp.asarray(in_A),
        in_B=jnp.asarray(in_B),
        n_directions=64,
        seed=0,
        n_bins=50,
        n_min=10,
        binning_method="quantile",
        tikhonov="halfset_eigen",
        return_details=True,
    )
    close(fit.result.committors_1d, h["wq_fit::committors_1d"], **ROUNDING)
    close(fit.weights.w, h["wq_fit::w_lib"], **EXACT)
    close(fit.weights.w, h["wq_fit::w_ref"], rtol=1e-8, atol=0)
    close(fit.weights.moment_gap, h["wq_fit::M_gap_ref"], rtol=1e-8, atol=0)
    close(fit.weights.c, h["wq_fit::c_lib"], **EXACT)
    close(fit.weights.dirichlet_energy, h["wq_fit::energy_lib"], **EXACT)
    close(fit.weights.diagnostics["G_reg"], h["wq_fit::G_reg_lib"], **EXACT)
    close(fit.weights.diagnostics["band_ssnr"], h["wq_fit::band_ssnr_ref"], rtol=1e-10, atol=0)
    close(q(jnp.asarray(h["wq_fit::grid"])), h["wq_fit::q_grid_lib"], **EXACT)


def test_wolfe_quapp_heldout_cap_matches_reference():
    h = _load("halfset_reference.npz")
    X, in_A, in_B = wolfe_quapp_samples(6000, 0)
    res = sc.compute_sliced_committor(
        jnp.asarray(X),
        in_A=jnp.asarray(in_A),
        in_B=jnp.asarray(in_B),
        n_directions=64,
        seed=0,
        n_bins=50,
        n_min=10,
        binning_method="equal_width",
    )
    w = sc.solve_weights(res, tikhonov="halfset_eigen", heldout_cap=True)
    close(w.w, h["wq_cap::w"], **EXACT)
    close(w.heldout_cap["cap"], h["wq_cap::cap"], **EXACT)
    close(w.heldout_cap["per_fold"], h["wq_cap::per_fold"], **EXACT)
