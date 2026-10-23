"""Frozen-reference gates: the numerics the published results depend on.

Three reference files:

* ``beta_invariance_golden.npz``: the slice basis on a 500-sample fixture.
* ``ebmc_golden.npz``: the weight solve (``auto`` and ``halfset_eigen``), the
  committor at fixed points, and the held-out cap on that fixture.
* ``halfset_reference.npz``: the eigenband regulariser against the filter
  written from its definition, the Wolfe-Quapp sample generator, and a
  full Wolfe-Quapp fit (weights, committor on a grid, held-out cap).

Everything that depends on the slice basis is frozen from this library by
``golden/freeze_references.py``, which also reruns the weight path
independently on the library's own slopes for the Wolfe-Quapp fit: the
half-Grams of the numpy prototype the library was ported from, then the
filter and the solve written from their definitions. The inputs of the
regulariser cases, the Wolfe-Quapp samples and the fixture itself come from
tag ``v0.6.0`` (``golden/freeze_1_0_references.py``, kept for provenance).

Two tiers. The STRICT tier is the release gate: the half-set path, the
paper's, bit for bit, the ``auto`` ridge and the basis to rounding. It holds
in the environment that produced the references, jax ``0.5.3``
(``FROZEN_JAX``) on the machine that froze them, and only there: XLA
compiles for the host CPU and the BLAS kernels are chosen by it and by the
matrix size, so another machine rounds the last bit differently even with
the same versions (an AVX-512 Xeon with jax 0.6.2 reproduces the half-set
weights to ``3e-13``, not bit for bit). No probe can
certify that in advance, so the strict tier is selected explicitly:
``SLICED_COMMITTOR_GOLDEN_TIER=strict`` on the freezing machine.

By default, CI included, the DRIFT tier runs. One mechanism carries the last
bit of rounding into the weights without being a bug: a sample whose
projection coincides with a grid point of its slice (quantile bins are built
from the sample values, so duplicated frames put bin centres exactly on
samples; the 500-sample fixture has 40 duplicated rows) takes the slope of
one adjacent bin or the other, which moves a few Gram entries by order one,
under either rule; rounding decides, so it fires on some machines and not
on others. Where it does not fire the path is stable: between jax 0.5.3,
0.6.2 and 0.10.1 on the freezing machine, and with jax 0.6.2 on an AVX-512
Xeon, the slice basis moves by ``1e-13``, the weights by ``3e-13``
relative, the committor by ``2e-14`` and the energies, caps and band SSNR
by ``2e-14`` relative or less. The drift tier compares the outputs, the
committor values, the half-set weights and the scalar summaries, at
tolerances set for that mechanism, and leaves the intermediates it moves
most to the strict tier. CI runs the drift tier on the newest JAX and on the
oldest supported one.
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
TIER_ENV = "SLICED_COMMITTOR_GOLDEN_TIER"

EXACT = dict(rtol=0.0, atol=0.0)
ROUNDING = dict(rtol=1e-8, atol=1e-12)
# name: (strict tier, drift tier); None = strict tier only (ill-conditioned);
# scaled_atol = absolute tolerance as a fraction of the reference's largest entry.
_TIERS = {
    "directions": (EXACT, dict(rtol=0.0, atol=1e-12)),
    "basis": (ROUNDING, dict(rtol=1e-8, atol=1e-9)),
    "moments": (EXACT, dict(rtol=1e-9, atol=1e-12)),
    "weights_auto": (ROUNDING, None),
    "weights_halfset": (EXACT, dict(rtol=0.0, atol=1e-3)),
    "committor_auto": (ROUNDING, dict(rtol=0.0, atol=1e-2)),
    "committor_halfset": (EXACT, dict(rtol=0.0, atol=1e-3)),
    "scalar_auto": (ROUNDING, dict(rtol=5e-2, atol=1e-4)),
    "scalar_halfset": (EXACT, dict(rtol=5e-2, atol=1e-4)),
    "per_fold": (EXACT, None),
    "fold_se": (ROUNDING, None),
    "gram_reg": (EXACT, None),
    "band_ssnr": (EXACT, None),
    "reference": (dict(rtol=1e-8, atol=0.0), dict(rtol=0.0, atol=1e-3)),
    "reference_scalar": (dict(rtol=1e-8, atol=0.0), dict(rtol=5e-2, atol=0.0)),
    "reference_ssnr": (dict(rtol=1e-10, atol=0.0), None),
    "regulariser": (dict(rtol=1e-10, atol=1e-14), dict(scaled_atol=1e-2)),
    "regulariser_ssnr": (dict(rtol=1e-10, atol=0.0), None),
}


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


@pytest.fixture(scope="module")
def strict():
    """The strict tier, selected with ``SLICED_COMMITTOR_GOLDEN_TIER=strict`` on
    the machine that froze the references (with jax ``FROZEN_JAX``)."""
    return os.environ.get(TIER_ENV) == "strict"


@pytest.fixture(scope="module")
def check(strict):
    """``check(name, got, ref)``: compare at the tier's tolerance, or skip a
    quantity the drift tier does not compare."""

    def _check(name, got, ref):
        tol = _TIERS[name][0 if strict else 1]
        if tol is None:
            return
        if "scaled_atol" in tol:  # absolute, relative to the reference's largest entry
            scale = float(np.max(np.abs(np.asarray(ref, np.float64))))
            tol = dict(rtol=0.0, atol=tol["scaled_atol"] * scale)
        try:
            close(got, ref, **tol)
        except AssertionError as exc:
            hint = (
                f"strict tier ({TIER_ENV}=strict): differs from the frozen reference. On the "
                f"machine that froze it with jax {FROZEN_JAX} this is a real change of the "
                "numerics; anywhere else it is the machine's rounding, and the default drift "
                "tier is the comparison to make."
                if strict
                else "drift tier: beyond the measured cross-environment drift, a real change"
            )
            raise AssertionError(f"{name}: {hint}\n{exc}") from None

    return _check


# --------------------------------------------------------------------------- #
# the slice basis
# --------------------------------------------------------------------------- #
def test_basis_matches_golden(basis, golden_data, check):
    check("directions", basis.directions, golden_data["directions"])
    check("basis", basis.slice_coords, golden_data["slice_coords"])
    check("basis", basis.free_energies, golden_data["log_density (= -β·F)"])
    check("basis", basis.committors_1d, golden_data["committors_1d"])
    assert bool(jnp.all(basis.valid_mask))


# --------------------------------------------------------------------------- #
# the weight solve
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("tikhonov,rule", [("auto", "auto"), ("halfset_eigen", "halfset")])
def test_weights_match_golden(basis, golden_samples, golden_labels, check, tikhonov, rule):
    e = _load("ebmc_golden.npz")
    k = f"ebmc_{tikhonov}"
    w = sc.solve_weights(basis, tikhonov=tikhonov)
    check(f"weights_{rule}", w.w, e[f"{k}::w"])
    check(f"scalar_{rule}", w.c, e[f"{k}::c"])
    check(f"scalar_{rule}", w.moment_gap, e[f"{k}::M_gap"])
    check(f"scalar_{rule}", w.dirichlet_energy, e[f"{k}::energy"])
    check("moments", w.diagnostics["a"], e[f"{k}::a"])
    check("moments", w.diagnostics["b"], e[f"{k}::b"])
    q = sc.build_committor(basis, w)
    pts = jnp.asarray(e["eval_points"])
    in_A, in_B = golden_labels
    check(f"committor_{rule}", q(pts), e[f"{k}::q_ref_eval_points_free"])
    snapped = q(golden_samples, in_A=in_A, in_B=in_B)
    check(f"committor_{rule}", snapped, e[f"{k}::q_at_samples_snapped"])


def test_halfset_diagnostics_match_golden(basis, check):
    e = _load("ebmc_golden.npz")
    w = sc.solve_weights(basis, tikhonov="halfset_eigen")
    assert w.ridge == 0.0
    check("gram_reg", w.diagnostics["G_reg"], e["ebmc_halfset_eigen::G_reg"])
    check("band_ssnr", w.diagnostics["band_ssnr"], e["ebmc_halfset_eigen::band_ssnr"])


def test_heldout_cap_matches_golden(basis, check):
    e = _load("ebmc_golden.npz")
    hc = sc.solve_weights(basis, tikhonov="halfset_eigen", heldout_cap=True).heldout_cap
    check("scalar_halfset", hc["cap"], e["ebmc_halfset_eigen::heldout_cap"])
    check("per_fold", hc["per_fold"], e["ebmc_halfset_eigen::heldout_per_fold"])
    check("fold_se", hc["se"], e["ebmc_halfset_eigen::heldout_se"])
    assert np.isfinite(hc["gap"])


# --------------------------------------------------------------------------- #
# the eigenband regulariser against its definition
# --------------------------------------------------------------------------- #
def test_eigenband_regulariser_matches_reference(check):
    """numpy and scipy only, on stored matrices, against the filter written
    from its definition (``freeze_references.FILTER_SPEC``). The band
    statistic depends only on each band's eigenspace, so what can move
    between machines is a nearly degenerate pair straddling a band edge."""
    h = _load("halfset_reference.npz")
    for M, nb, seed in h["spd_cases"]:
        k = f"spd::{M}_{nb}_{seed}"
        R, info = hs.halfset_eigen_regularize(h[k + "::G1"], h[k + "::G2"], n_bands=int(nb))
        check("regulariser", R, h[k + "::G_reg"])
        check("regulariser_ssnr", info["band_ssnr"], h[k + "::band_ssnr"])
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


def test_wolfe_quapp_fit_matches_reference(check):
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
    check("basis", fit.result.committors_1d, h["wq_fit::committors_1d"])
    check("weights_halfset", fit.weights.w, h["wq_fit::w_lib"])
    check("reference", fit.weights.w, h["wq_fit::w_ref"])
    check("reference_scalar", fit.weights.moment_gap, h["wq_fit::M_gap_ref"])
    check("scalar_halfset", fit.weights.c, h["wq_fit::c_lib"])
    check("scalar_halfset", fit.weights.dirichlet_energy, h["wq_fit::energy_lib"])
    check("gram_reg", fit.weights.diagnostics["G_reg"], h["wq_fit::G_reg_lib"])
    check("reference_ssnr", fit.weights.diagnostics["band_ssnr"], h["wq_fit::band_ssnr_ref"])
    check("committor_halfset", q(jnp.asarray(h["wq_fit::grid"])), h["wq_fit::q_grid_lib"])


def test_wolfe_quapp_heldout_cap_matches_reference(check):
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
    check("weights_halfset", w.w, h["wq_cap::w"])
    check("scalar_halfset", w.heldout_cap["cap"], h["wq_cap::cap"])
    check("per_fold", w.heldout_cap["per_fold"], h["wq_cap::per_fold"])
