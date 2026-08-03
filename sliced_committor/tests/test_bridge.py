"""Tests for the v2 diffusion bridge (rates/bridge.py): position-dependent mapping,
regressed landscape, committor-quality reductions, and uncertainty.

Self-contained: reuses the 2D double-well equilibrium sampler and the windowed-OU
fixture; no external umbrella data needed. The full known-D0 validation ladder lives
in ``experiments/bridge_v2.py``.
"""

import jax

jax.config.update("jax_enable_x64", True)

import numpy as np
import pytest

import sliced_committor as sc

from .test_rates import _windowed_ou
from .test_validation_2d_double_well import _equilibrium_samples


@pytest.fixture(scope="module")
def committor2d():
    s, in_A, in_B = _equilibrium_samples(n_samples=3000, seed=7)
    q = sc.fit_committor(s, in_A=in_A, in_B=in_B, n_directions=256, n_bins=120, seed=7)
    return q, np.asarray(s), np.asarray(in_A), np.asarray(in_B)


# ---------------------------------------------------------------------------
# conditional_mean (Axis 3 landscape regression)
# ---------------------------------------------------------------------------
def test_conditional_mean_recovers_linear():
    x = np.linspace(0, 1, 400)
    y = 2.0 + 3.0 * x
    for method in ("kernel", "local_linear", "pspline"):
        reg = sc.conditional_mean(x, y, None, np.array([0.3, 0.7]), method=method)
        # local_linear/pspline are exact for a line; kernel has O(h) boundary bias only
        atol = 0.05 if method == "kernel" else 1e-3
        np.testing.assert_allclose(reg.value, [2.9, 4.1], atol=atol)


def test_conditional_mean_local_linear_slope():
    x = np.linspace(0, 1, 400)
    reg = sc.conditional_mean(x, 5.0 * x, None, np.array([0.5]), method="local_linear")
    assert reg.slope[0] == pytest.approx(5.0, abs=1e-3)


# ---------------------------------------------------------------------------
# committor_populations = TPT rate normaliser
# ---------------------------------------------------------------------------
def test_committor_populations_sum_to_one(committor2d):
    q, s, in_A, in_B = committor2d
    qx = np.asarray(q(s))
    rho_A, rho_B = sc.committor_populations(qx)
    assert rho_A + rho_B == pytest.approx(1.0, abs=1e-12)
    assert 0.0 < rho_A < 1.0


# ---------------------------------------------------------------------------
# flux_reductions matches the library committor_rate (arithmetic + harmonic)
# ---------------------------------------------------------------------------
def test_flux_reductions_match_committor_rate(committor2d):
    q, s, in_A, in_B = committor2d
    n = 200
    Dq = sc.mapped_committor_diffusion(q, s, D_s=0.05, cv_grad_sq=1.0, n_bins=n)
    pi = np.asarray(sc.density(q, s, n_bins=n).values)
    grid = np.asarray(sc.density(q, s, n_bins=n).levels)
    rho_A, rho_B = sc.committor_populations(np.asarray(q(s)))
    red = sc.flux_reductions(grid, pi, np.asarray(Dq.values), rho_A, rho_B, band=(0.3, 0.7))
    for reduction, key in (("arithmetic", "arithmetic"), ("harmonic", "harmonic")):
        lib = sc.committor_rate(
            q,
            s,
            s,
            dt=0.01,
            D_profile=Dq,
            reduction=reduction,
            at=None,
            n_bins=n,
            in_A=in_A,
            in_B=in_B,
        )
        assert red[key]["k_AB"] == pytest.approx(lib["k_AB"], rel=0.05), reduction
    assert np.isfinite(red["const_flux_mle"]["k_AB"])
    assert 0.0 <= red["flux_cv"] < 2.0
    lo, hi = red["bracket_k_AB"]
    assert lo <= hi  # harmonic (bottleneck) <= arithmetic (Dirichlet upper bound)


# ---------------------------------------------------------------------------
# field bridge reduces to the scalar bridge when D_s (and g_s) are constant
# ---------------------------------------------------------------------------
def test_field_reduces_to_scalar_when_D_constant(committor2d):
    q, s, in_A, in_B = committor2d
    n = 120
    D0 = 0.05
    scalar = sc.mapped_committor_diffusion(q, s, D_s=D0, cv_grad_sq=1.0, n_bins=n)
    s_cv = np.asarray(s)[:, 0]  # arbitrary CV
    s_centers = np.linspace(s_cv.min(), s_cv.max(), 20)
    D_s_prof = (s_centers, np.full_like(s_centers, D0))  # constant D_s
    field = sc.mapped_committor_diffusion_field(
        q, s, D_s_profile=D_s_prof, cv_grad_sq=1.0, s_values=s_cv, n_bins=n, method="local_linear"
    )
    gs = np.asarray(scalar.values)
    gf = np.asarray(field.values)
    grid = np.asarray(scalar.levels)
    m = (grid >= 0.3) & (grid <= 0.7) & np.isfinite(gs) & np.isfinite(gf) & (gs > 0)
    ratio = gf[m] / gs[m]
    # same underlying <|grad q|^2> * D0, up to the smoothing method -> agree to ~25%
    assert 0.7 < float(np.median(ratio)) < 1.4


# ---------------------------------------------------------------------------
# reparam runs and reports the monotonicity R^2 gate
# ---------------------------------------------------------------------------
def test_reparam_runs_and_reports_r2(committor2d):
    q, s, in_A, in_B = committor2d
    s_cv = np.asarray(s)[:, 0]
    s_centers = np.linspace(s_cv.min(), s_cv.max(), 20)
    D_s_prof = (s_centers, np.full_like(s_centers, 0.05))
    prof, info = sc.mapped_committor_diffusion_reparam(
        q, s, D_s_profile=D_s_prof, s_values=s_cv, n_bins=120
    )
    assert "r2_monotone" in info and 0.0 <= info["r2_monotone"] <= 1.0
    good = np.isfinite(np.asarray(prof.values))
    assert good.sum() > 20


# ---------------------------------------------------------------------------
# Hummer D_s profile + window bootstrap on confined OU (known D=0.05)
# ---------------------------------------------------------------------------
def test_hummer_Ds_profile_and_bootstrap_on_ou():
    x, wid = _windowed_ou(n_win=6, n_per=20000, D_true=0.05)
    s_c, D_s = sc.hummer_Ds_profile(x, wid, dt=0.01)
    assert D_s.shape[0] == 6
    assert float(np.median(D_s)) == pytest.approx(0.05, rel=0.3)
    # window bootstrap: q irrelevant here (all windows at same coordinate), use a dummy
    qx = np.full_like(x, 0.5)
    D_hat, boots, lo, hi = sc.bootstrap_barrier_Ds(
        qx, x, np.ones_like(x), wid, dt=0.01, n_boot=50, seed=0
    )
    assert D_hat > 0 and boots.size > 0 and 0.0 < lo <= 1.0 <= hi


# ---------------------------------------------------------------------------
# committor_grad_profile is positive on the interior and finite
# ---------------------------------------------------------------------------
def test_committor_grad_profile_positive(committor2d):
    q, s, in_A, in_B = committor2d
    prof, reg = sc.committor_grad_profile(q, s, n_bins=100, method="local_linear")
    grid = np.asarray(prof.levels)
    vals = np.asarray(prof.values)
    m = (grid >= 0.3) & (grid <= 0.7)
    assert np.all(np.isfinite(vals[m])) and np.all(vals[m] > 0)
