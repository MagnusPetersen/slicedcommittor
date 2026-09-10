"""Frozen-reference gates: the numerics the published results depend on.

Three reference files, all produced ONCE at tag ``v0.6.0`` by
``golden/freeze_1_0_references.py`` (which cannot run against this version):

* ``beta_invariance_golden.npz``: the slice basis on a 500-sample fixture.
* ``ebmc_golden.npz``: the weight solve (``auto`` and ``halfset_eigen``), the
  committor at fixed points, and the held-out cap on that fixture.
* ``halfset_reference.npz``: the eigenband regulariser against the numpy
  prototype it was ported from, the Wolfe-Quapp sample generator, and a
  full Wolfe-Quapp fit (weights, committor on a grid, held-out cap).

The references were produced with jax ``0.5.3`` (``FROZEN_JAX``). In that
environment the half-set path, the paper's, is gated bit for bit, and the
``auto`` ridge and the basis are allowed rounding drift. Under any other
JAX/XLA the same gates run at the measured cross-version drift instead. Two
mechanisms move the numbers between XLA versions, neither of them a bug: a
sample whose projection coincides with a grid point of its slice (quantile
bins are built from the sample values, so duplicated frames put bin centres
exactly on samples) takes the slope of one adjacent bin or the other by the
last bit of rounding, which moves a few Gram entries by order one; and the
half-set filter reads a band correlation off a nearly degenerate spectrum,
which turns ``1e-13`` in the Gram into ``1e-3`` in individual weights.
Measured between 0.5.3 and 0.10.2: the slice basis moves by ``1e-13``,
individual weights by up to ``1e-2`` relative, the committor by up to
``1e-5`` (half-set) and ``1e-3`` (scalar ridge, on the 500-sample fixture),
energies and caps by ``1e-4`` relative. CI runs the strict tier in the
``numerics`` job and the drift tier in the version matrix.
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
FROZEN_JAX = "0.5.3"
SAME_ENV = jax.__version__ == FROZEN_JAX

EXACT = dict(rtol=0.0, atol=0.0)
ROUNDING = dict(rtol=1e-8, atol=1e-12)
# tier: (frozen environment, any other JAX/XLA). The drift tolerances are the
# measured 0.5.3 -> 0.6.2 and 0.5.3 -> 0.10.2 differences with a margin of ten
# to a hundred (the noisy intermediates, band SSNR and per-fold caps, move
# by a few percent on the 500-sample fixture under 0.6.2).
_TIERS = {
    "directions": (EXACT, dict(rtol=0.0, atol=1e-12)),
    "basis": (ROUNDING, dict(rtol=1e-8, atol=1e-9)),
    "moments": (EXACT, dict(rtol=1e-9, atol=1e-12)),
    "weights_auto": (ROUNDING, dict(rtol=0.0, atol=5e-2)),
    "weights_halfset": (EXACT, dict(rtol=0.0, atol=1e-4)),
    "committor_auto": (ROUNDING, dict(rtol=0.0, atol=5e-3)),
    "committor_halfset": (EXACT, dict(rtol=0.0, atol=1e-4)),
    "scalar_auto": (ROUNDING, dict(rtol=1e-2, atol=1e-5)),
    "scalar_halfset": (EXACT, dict(rtol=1e-2, atol=1e-5)),
    "per_fold": (EXACT, dict(rtol=1e-1, atol=0.0)),
    "fold_se": (ROUNDING, dict(rtol=1e-1, atol=1e-6)),
    "gram_reg": (EXACT, dict(rtol=0.25, atol=0.0)),
    "band_ssnr": (EXACT, dict(rtol=5e-2, atol=1e-2)),
    "reference": (dict(rtol=1e-8, atol=0.0), dict(rtol=0.0, atol=1e-4)),
    "reference_scalar": (dict(rtol=1e-8, atol=0.0), dict(rtol=1e-2, atol=0.0)),
    "reference_ssnr": (dict(rtol=1e-10, atol=0.0), dict(rtol=5e-2, atol=1e-2)),
}


def tol(name):
    """The tolerance of a golden comparison: strict in the frozen environment."""
    return _TIERS[name][0 if SAME_ENV else 1]


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
    close(basis.directions, golden_data["directions"], **tol("directions"))
    close(basis.slice_coords, golden_data["slice_coords"], **tol("basis"))
    close(basis.free_energies, golden_data["log_density (= -β·F)"], **tol("basis"))
    close(basis.committors_1d, golden_data["committors_1d"], **tol("basis"))
    assert bool(jnp.all(basis.valid_mask))


# --------------------------------------------------------------------------- #
# the weight solve
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("tikhonov,rule", [("auto", "auto"), ("halfset_eigen", "halfset")])
def test_weights_match_golden(basis, golden_samples, golden_labels, tikhonov, rule):
    e = _load("ebmc_golden.npz")
    k = f"ebmc_{tikhonov}"
    w = sc.solve_weights(basis, tikhonov=tikhonov)
    close(w.w, e[f"{k}::w"], **tol(f"weights_{rule}"))
    close(w.c, e[f"{k}::c"], **tol(f"scalar_{rule}"))
    close(w.moment_gap, e[f"{k}::M_gap"], **tol(f"scalar_{rule}"))
    close(w.dirichlet_energy, e[f"{k}::energy"], **tol(f"scalar_{rule}"))
    close(w.diagnostics["a"], e[f"{k}::a"], **tol("moments"))
    close(w.diagnostics["b"], e[f"{k}::b"], **tol("moments"))
    q = sc.build_committor(basis, w)
    pts = jnp.asarray(e["eval_points"])
    in_A, in_B = golden_labels
    close(q(pts), e[f"{k}::q_ref_eval_points_free"], **tol(f"committor_{rule}"))
    snapped = q(golden_samples, in_A=in_A, in_B=in_B)
    close(snapped, e[f"{k}::q_at_samples_snapped"], **tol(f"committor_{rule}"))


def test_halfset_diagnostics_match_golden(basis):
    e = _load("ebmc_golden.npz")
    w = sc.solve_weights(basis, tikhonov="halfset_eigen")
    assert w.ridge == 0.0
    close(w.diagnostics["G_reg"], e["ebmc_halfset_eigen::G_reg"], **tol("gram_reg"))
    close(w.diagnostics["band_ssnr"], e["ebmc_halfset_eigen::band_ssnr"], **tol("band_ssnr"))


def test_heldout_cap_matches_golden(basis):
    e = _load("ebmc_golden.npz")
    hc = sc.solve_weights(basis, tikhonov="halfset_eigen", heldout_cap=True).heldout_cap
    close(hc["cap"], e["ebmc_halfset_eigen::heldout_cap"], **tol("scalar_halfset"))
    close(hc["per_fold"], e["ebmc_halfset_eigen::heldout_per_fold"], **tol("per_fold"))
    close(hc["se"], e["ebmc_halfset_eigen::heldout_se"], **tol("fold_se"))
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
    """The full one-liner at the paper's ridge, bit for bit in the frozen
    environment, and against the independent numpy reference path
    (``halfset_grams`` -> regularise -> solve)."""
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
    close(fit.result.committors_1d, h["wq_fit::committors_1d"], **tol("basis"))
    close(fit.weights.w, h["wq_fit::w_lib"], **tol("weights_halfset"))
    close(fit.weights.w, h["wq_fit::w_ref"], **tol("reference"))
    close(fit.weights.moment_gap, h["wq_fit::M_gap_ref"], **tol("reference_scalar"))
    close(fit.weights.c, h["wq_fit::c_lib"], **tol("scalar_halfset"))
    close(fit.weights.dirichlet_energy, h["wq_fit::energy_lib"], **tol("scalar_halfset"))
    close(fit.weights.diagnostics["G_reg"], h["wq_fit::G_reg_lib"], **tol("gram_reg"))
    close(fit.weights.diagnostics["band_ssnr"], h["wq_fit::band_ssnr_ref"], **tol("reference_ssnr"))
    close(q(jnp.asarray(h["wq_fit::grid"])), h["wq_fit::q_grid_lib"], **tol("committor_halfset"))


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
    close(w.w, h["wq_cap::w"], **tol("weights_halfset"))
    close(w.heldout_cap["cap"], h["wq_cap::cap"], **tol("scalar_halfset"))
    close(w.heldout_cap["per_fold"], h["wq_cap::per_fold"], **tol("per_fold"))
