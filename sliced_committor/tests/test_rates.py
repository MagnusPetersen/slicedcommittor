"""Rates: the pair ``{D_q, pi}``, its constructors, and its reductions.

Analytic gates first. On the exact committor of a 1D double well the flux is
constant and every reduction returns the same rate to rounding (and the
harmonic value is shown NOT to be a bound in general). The Kramers-Moyal,
Hummer and pooled-ACF estimators recover the known ``D`` of an
Ornstein-Uhlenbeck process. The Jacobian map is linear in ``D_s``, inverse-
linear in the CV gradient, and invariant under a rotation of the features;
the co-area identity holds on a fitted committor. Then the argument discipline
of ``rate_from_profiles`` / ``committor_rate``, the plateau finder, and the
layering rule that ``rates`` never imports the umbrella subpackage.
"""

import ast
import pathlib

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

import sliced_committor as sc
from sliced_committor.rates.baselines import pmf_kramers_rate

from ._helpers import (
    double_well_1d_profiles,
    double_well_samples,
    ou_trajectory,
    overdamped_double_well_trajectory,
    unit_directions,
    windowed_ou,
)

D0 = 0.05
DT = 0.01


@pytest.fixture(scope="module")
def fit():
    samples, in_A, in_B = double_well_samples(n=2500, seed=0)
    q = sc.fit_committor(
        jnp.asarray(samples),
        in_A=jnp.asarray(in_A),
        in_B=jnp.asarray(in_B),
        n_directions=128,
        n_bins=100,
        seed=2,
    )
    return q, jnp.asarray(samples), np.asarray(in_A), np.asarray(in_B)


# ---------------------------------------------------------------------------
# the exact committor: constant flux, every reduction agrees
# ---------------------------------------------------------------------------
def test_exact_committor_flux_is_constant_and_every_reduction_agrees():
    pi, Dq, rho_A, rho_B, nu = double_well_1d_profiles(n_bins=2000, D=D0)
    for reduction in ("arithmetic", "harmonic", "plateau", "local"):
        out = sc.rate_from_profiles(pi, Dq, rho_A, rho_B, reduction=reduction)
        assert out["k_AB"] == pytest.approx(nu / rho_A, rel=1e-9), reduction
        assert out["k_BA"] == pytest.approx(nu / rho_B, rel=1e-9), reduction
        assert out["flatness"] < 1e-12
        assert out["k_AB_harmonic"] == pytest.approx(out["k_AB_arithmetic"], rel=1e-9)


def test_auto_plateau_and_harmonic_window_on_the_exact_committor():
    pi, Dq, rho_A, rho_B, nu = double_well_1d_profiles(n_bins=500, D=D0)
    auto = sc.rate_from_profiles(pi, Dq, rho_A, rho_B, reduction="plateau", band="auto")
    assert auto["plateau_ok"] and auto["nu_R"] == pytest.approx(nu, rel=1e-9)
    lo, hi = auto["band"]
    assert 0.0 <= lo < hi <= 1.0
    # restricting the MFPT integral to a window drops positive mass: the rate goes up
    windowed = sc.rate_from_profiles(pi, Dq, rho_A, rho_B, reduction="harmonic", window=(0.2, 0.8))
    assert windowed["k_AB"] > nu / rho_A and np.isnan(windowed["nu_R"])


def test_harmonic_is_not_a_bound_on_the_arithmetic_rate():
    # uniform pi and nu = sqrt(q): the harmonic rate (3/2) exceeds the arithmetic one (4/3)
    n = 4000
    centers = (np.arange(n) + 0.5) / n
    ones = np.ones(n, dtype=np.int64)
    out = sc.rate_from_profiles(
        sc.Profile(centers, np.ones(n), ones),
        sc.Profile(centers, np.sqrt(centers), ones),
        0.5,
        0.5,
        reduction="arithmetic",
    )
    assert out["k_AB_arithmetic"] == pytest.approx(4.0 / 3.0, rel=1e-3)
    assert out["k_AB_harmonic"] == pytest.approx(1.5, rel=1e-3)
    assert out["k_AB_harmonic"] > out["k_AB_arithmetic"]


def test_rate_from_profiles_interpolates_a_coarse_D_q_onto_the_density_grid():
    pi, _, rho_A, rho_B, _ = double_well_1d_profiles(n_bins=100)
    coarse = sc.Profile(
        np.array([0.1, 0.5, 0.9]), np.array([1.0, np.nan, 3.0]), np.ones(3, dtype=np.int64)
    )
    out = sc.rate_from_profiles(pi, coarse, rho_A, rho_B, reduction="arithmetic")
    expect = np.interp(pi.levels, [0.1, 0.9], [1.0, 3.0]) * pi.values
    np.testing.assert_allclose(out["nu"].values, expect, rtol=1e-12)


# ---------------------------------------------------------------------------
# argument discipline
# ---------------------------------------------------------------------------
def test_rate_from_profiles_rejects_foreign_parameters_and_bad_bands():
    pi, Dq, rho_A, rho_B, _ = double_well_1d_profiles(n_bins=200)
    with pytest.raises(TypeError, match="band= belongs to"):
        sc.rate_from_profiles(pi, Dq, rho_A, rho_B, reduction="harmonic", band=(0.3, 0.7))
    with pytest.raises(TypeError, match="q_star= belongs to"):
        sc.rate_from_profiles(pi, Dq, rho_A, rho_B, reduction="plateau", q_star=0.5)
    with pytest.raises(TypeError, match="window= belongs to"):
        sc.rate_from_profiles(pi, Dq, rho_A, rho_B, reduction="arithmetic", window=(0.2, 0.8))
    with pytest.raises(ValueError, match="unknown reduction"):
        sc.rate_from_profiles(pi, Dq, rho_A, rho_B, reduction="bogus")
    with pytest.raises(ValueError, match="lo < hi"):
        sc.rate_from_profiles(pi, Dq, rho_A, rho_B, reduction="plateau", band=(0.5, 0.5))
    with pytest.raises(ValueError, match="a band is"):
        sc.rate_from_profiles(pi, Dq, rho_A, rho_B, reduction="plateau", band=0.5)


def test_committor_rate_routes_are_exclusive(fit):
    q, X, in_A, in_B = fit
    Dq = sc.committor_diffusion_from_cv(
        sc.committor_grad_sq(q, X, n_bins=50), D_s=D0, cv_grad_sq=1.0
    )
    with pytest.raises(ValueError, match="exactly one"):
        sc.committor_rate(q, X)
    with pytest.raises(ValueError, match="exactly one"):
        sc.committor_rate(q, X, D_q=Dq, trajectory=X, dt=DT, lag=1)
    with pytest.raises(ValueError, match="belong to the trajectory route"):
        sc.committor_rate(q, X, D_q=Dq, dt=DT)
    with pytest.raises(ValueError, match="explicit lag"):
        sc.committor_rate(q, X, trajectory=X, dt=DT)


# ---------------------------------------------------------------------------
# the static ensemble
# ---------------------------------------------------------------------------
def test_density_integrates_to_one_and_a_cv_density_matches_the_histogram(fit):
    q, X, _, _ = fit
    prof = sc.density(np.asarray(q(X)), n_bins=50)
    h = prof.levels[1] - prof.levels[0]
    assert float(np.sum(prof.values) * h) == pytest.approx(1.0, abs=1e-9)
    cv = np.asarray(X)[:, 0]
    prof = sc.density(cv, n_bins=30, span=(cv.min(), cv.max()))
    hist, edges = np.histogram(cv, bins=np.linspace(cv.min(), cv.max(), 31))
    np.testing.assert_allclose(
        prof.values, hist / (hist.sum() * (edges[1] - edges[0])), atol=1e-9, rtol=0
    )
    np.testing.assert_array_equal(prof.counts, hist)


def test_basin_populations_sum_to_one_snap_and_reweight(fit):
    q, X, in_A, in_B = fit
    qx = np.asarray(q(X))
    rho_A, rho_B = sc.basin_populations(qx, in_A=in_A, in_B=in_B)
    assert rho_A + rho_B == pytest.approx(1.0, abs=1e-12) and 0.0 < rho_A < 1.0
    snapped = np.where(in_A, 0.0, np.where(in_B, 1.0, np.clip(qx, 0.0, 1.0)))
    assert rho_B == pytest.approx(float(np.mean(snapped)), abs=1e-12)
    w = np.ones(qx.shape[0])
    w[:10] = 100.0
    rho_A_w, _ = sc.basin_populations(qx, sample_weights=w)
    expect = float(np.sum(w * (1.0 - np.clip(qx, 0.0, 1.0))) / w.sum())
    assert rho_A_w == pytest.approx(expect, abs=1e-12)


def test_committor_grad_sq_satisfies_the_coarea_identity(fit):
    # int_0^1 pi(q) <|grad q|^2>_q dq == <|grad q|^2>_pi, the Dirichlet form
    q, X, _, _ = fit
    g = sc.committor_grad_sq(q, X, n_bins=60)
    pi = sc.density(np.asarray(q(X)), n_bins=60)
    lhs = float(np.nansum(g.values * pi.values) / 60)
    grads = np.asarray(jax.vmap(jax.grad(q))(X))
    rhs = float(np.mean(np.sum(grads**2, axis=1)))
    assert lhs == pytest.approx(rhs, rel=1e-9)
    interior = np.isfinite(g.values) & (g.levels > 0.2) & (g.levels < 0.8)
    assert interior.sum() > 5 and np.all(g.values[interior] > 0)


def test_committor_diffusion_from_cv_is_linear_in_D_s_and_inverse_in_the_cv_gradient(fit):
    q, X, _, _ = fit
    g = sc.committor_grad_sq(q, X, n_bins=50)
    d1 = sc.committor_diffusion_from_cv(g, D_s=1.0, cv_grad_sq=1.0)
    d2 = sc.committor_diffusion_from_cv(g, D_s=2.0, cv_grad_sq=1.0)
    d3 = sc.committor_diffusion_from_cv(g, D_s=2.0, cv_grad_sq=2.0)
    good = np.isfinite(d1.values)
    assert good.sum() > 5 and np.all(d1.values[good] > 0)
    np.testing.assert_allclose(d2.values[good], 2.0 * d1.values[good], rtol=1e-12)
    np.testing.assert_allclose(d3.values[good], d1.values[good], rtol=1e-12)
    with pytest.raises(ValueError, match="cv_grad_sq"):
        sc.committor_diffusion_from_cv(g, D_s=1.0, cv_grad_sq=0.0)
    with pytest.raises(ValueError, match="D_s"):
        sc.committor_diffusion_from_cv(g, D_s=-1.0, cv_grad_sq=1.0)


def test_grad_sq_profile_and_rate_are_invariant_under_a_rotation_of_the_features():
    samples, in_A, in_B = double_well_samples(n=1500, seed=3)
    rng = np.random.default_rng(5)
    R, _ = np.linalg.qr(rng.standard_normal((2, 2)))
    theta = unit_directions(2, 32, seed=5)
    kw = dict(
        in_A=jnp.asarray(in_A),
        in_B=jnp.asarray(in_B),
        n_directions=32,
        n_bins=80,
        seed=1,
        tikhonov="auto",
    )
    X1, X2 = jnp.asarray(samples), jnp.asarray(samples @ R.T)
    q1 = sc.fit_committor(X1, directions=jnp.asarray(theta), **kw)
    q2 = sc.fit_committor(X2, directions=jnp.asarray(theta @ R.T), **kw)
    g1 = sc.committor_grad_sq(q1, X1, n_bins=40)
    g2 = sc.committor_grad_sq(q2, X2, n_bins=40)
    good = np.isfinite(g1.values) & np.isfinite(g2.values)
    assert good.sum() > 10
    np.testing.assert_allclose(g2.values[good], g1.values[good], rtol=1e-6)
    r1 = sc.committor_rate(
        q1, X1, D_q=sc.committor_diffusion_from_cv(g1, D_s=D0, cv_grad_sq=1.0), n_bins=40
    )
    r2 = sc.committor_rate(
        q2, X2, D_q=sc.committor_diffusion_from_cv(g2, D_s=D0, cv_grad_sq=1.0), n_bins=40
    )
    assert r2["k_AB"] == pytest.approx(r1["k_AB"], rel=1e-6)


def test_trajectory_and_mapped_routes_agree_on_the_double_well(fit):
    q, X, in_A, in_B = fit
    traj = overdamped_double_well_trajectory(T=60000, dt=DT, D0=D0, seed=1)
    km = sc.committor_rate(
        q, X, trajectory=jnp.asarray(traj), dt=DT, lag=1, in_A=in_A, in_B=in_B, n_bins=40
    )
    g = sc.committor_grad_sq(q, X, n_bins=40)
    Dq = sc.committor_diffusion_from_cv(g, D_s=D0, cv_grad_sq=1.0)
    mapped = sc.committor_rate(q, X, D_q=Dq, in_A=in_A, in_B=in_B, n_bins=40)
    for out in (km, mapped):
        assert np.isfinite(out["k_AB"]) and out["k_AB"] > 0
        assert out["rho_A"] + out["rho_B"] == pytest.approx(1.0, abs=1e-12)
        assert out["nu"].values.shape == (40,) and out["band"] == (0.3, 0.7)
    assert 0.4 < km["k_AB"] / mapped["k_AB"] < 2.5
    ks = [
        sc.committor_rate(q, X, D_q=Dq, in_A=in_A, in_B=in_B, n_bins=40, reduction=r)["k_AB"]
        for r in ("arithmetic", "harmonic", "plateau", "local")
    ]
    assert max(ks) / min(ks) < 10.0


# ---------------------------------------------------------------------------
# diffusion estimators on processes with a known D
# ---------------------------------------------------------------------------
def test_diffusion_profile_recovers_D_on_an_ou_process():
    x = ou_trajectory(200_000, dt=DT, k=1.0, D=D0, seed=0)
    prof = sc.diffusion_profile(x, dt=DT, lag=1, n_bins=20, span=(x.min(), x.max()))
    assert sc.value_at(prof, 0.0) == pytest.approx(D0, rel=0.1)
    with pytest.raises(ValueError, match="lag must be"):
        sc.diffusion_profile(x, dt=DT, lag=0)
    with pytest.raises(ValueError, match="not shorter"):
        sc.diffusion_profile(x[:5], dt=DT, lag=5)


def test_window_stratified_diffusion_profile_recovers_D_on_windowed_ou():
    x, wid = windowed_ou(n_win=4, n_per=20000, dt=DT, D=D0)
    prof = sc.diffusion_profile(x, dt=DT, lag=1, window_ids=wid, n_bins=15, span=(x.min(), x.max()))
    assert sc.value_at(prof, 0.0) == pytest.approx(D0, rel=0.1)
    with pytest.raises(ValueError, match="window_ids length"):
        sc.diffusion_profile(x, dt=DT, lag=1, window_ids=wid[:-1])


def test_lag_scan_levels_are_the_lags_and_the_short_lag_reads_D():
    x = ou_trajectory(100_000, dt=DT, k=1.0, D=D0, seed=1)
    scan = sc.lag_scan(
        x, dt=DT, lags=(1, 2, 5), n_bins=20, span=(x.min(), x.max()), band=(-0.5, 0.5)
    )
    np.testing.assert_array_equal(scan.levels, [1.0, 2.0, 5.0])
    assert scan.values[0] == pytest.approx(D0, rel=0.1)
    assert np.all(scan.counts > 0) and np.all(np.isfinite(scan.values))


def test_hummer_diffusion_recovers_D_per_window_with_and_without_runs():
    x, wid = windowed_ou(n_win=6, n_per=20000, dt=DT, D=D0)
    prof = sc.hummer_diffusion(x, wid, dt=DT)
    assert prof.values.shape[0] == 6
    assert float(np.median(prof.values)) == pytest.approx(D0, rel=0.3)
    # splitting every window into two runs moves the estimate only by its own noise
    runs = (np.arange(x.shape[0]) % 20000) // 10000
    split = sc.hummer_diffusion(x, wid, dt=DT, run_ids=runs)
    np.testing.assert_allclose(split.values, prof.values, rtol=0.5)
    np.testing.assert_array_equal(split.counts, prof.counts)
    with pytest.raises(RuntimeError, match="no window"):
        sc.hummer_diffusion(np.zeros(100), np.zeros(100, dtype=int), dt=DT)


def test_pooled_acf_diffusion_recovers_D_and_selects_windows_by_their_mean():
    centers = np.array([-1.0, -0.5, 0.0, 0.5, 1.0, 1.5])
    x, wid = windowed_ou(n_win=6, n_per=20000, dt=DT, D=D0, centers=centers)
    pooled = sc.pooled_acf_diffusion(x, wid, dt=DT, n_boot=40, seed=0)
    assert pooled.n_windows == 6 and pooled.n_runs == 6
    assert pytest.approx(D0, rel=0.2) == pooled.D
    lo, hi = pooled.ci
    assert lo <= pooled.D <= hi
    band = sc.pooled_acf_diffusion(x, wid, dt=DT, window_band=(-0.6, 0.6))
    assert band.n_windows == 3 and band.ci is None
    runs = (np.arange(x.shape[0]) % 20000) // 10000
    split = sc.pooled_acf_diffusion(x, wid, dt=DT, run_ids=runs)
    assert split.n_runs == 12 and pytest.approx(pooled.D, rel=0.3) == split.D
    with pytest.raises(ValueError, match="long enough"):
        sc.pooled_acf_diffusion(x[:10], wid[:10], dt=DT)


# ---------------------------------------------------------------------------
# the CV gradient and the reparametrisation route
# ---------------------------------------------------------------------------
def test_linear_response_grad_sq_is_exact_for_a_linear_cv_and_window_dependent_otherwise():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((4000, 3))
    a = np.array([0.5, -1.0, 2.0])
    s = X @ a + 0.3
    assert sc.linear_response_grad_sq(s, X) == pytest.approx(float(a @ a), rel=1e-5)
    M = np.array([1.0, 2.0, 0.5])
    expect = float(a @ (M * a))
    assert sc.linear_response_grad_sq(s, X, metric=M) == pytest.approx(expect, rel=1e-5)
    assert sc.linear_response_grad_sq(s, X, metric=np.diag(M)) == pytest.approx(expect, rel=1e-5)
    # the caveat: a best-linear-predictor coefficient depends on the fitting window
    x = rng.uniform(-3.0, 3.0, size=(20000, 1))
    s = np.tanh(3.0 * x[:, 0])
    inner = np.abs(x[:, 0]) < 0.2
    ratio = sc.linear_response_grad_sq(s[inner], x[inner]) / sc.linear_response_grad_sq(s, x)
    assert ratio > 5.0


def test_reparam_recovers_D_q_for_a_monotone_map_and_gates_a_non_monotone_one():
    rng = np.random.default_rng(1)
    s = rng.standard_normal(30000)
    q = 0.5 * (1.0 + np.tanh(2.0 * s))  # a monotone committor of s
    prof = sc.committor_diffusion_from_cv_reparam(q, s, D0, n_bins=50)
    s_of_q = np.arctanh(2.0 * prof.levels - 1.0) / 2.0  # dq/ds = sech^2(2 s)
    exact = D0 * (1.0 - np.tanh(2.0 * s_of_q) ** 2) ** 2
    mid = (prof.levels > 0.25) & (prof.levels < 0.75)
    np.testing.assert_allclose(prof.values[mid], exact[mid], rtol=0.2)
    Ds_prof = sc.Profile(np.linspace(-3, 3, 7), np.full(7, D0), np.ones(7, dtype=np.int64))
    prof2 = sc.committor_diffusion_from_cv_reparam(q, s, Ds_prof, n_bins=50)
    np.testing.assert_allclose(prof2.values[mid], prof.values[mid], rtol=1e-12)
    with pytest.raises(ValueError, match="not monotonically related"):
        sc.committor_diffusion_from_cv_reparam(np.clip(s**2 / 4.0, 0.0, 1.0), s, D0, n_bins=50)


# ---------------------------------------------------------------------------
# the committor-free baseline
# ---------------------------------------------------------------------------
def test_pmf_kramers_baseline_on_the_double_well_coordinate():
    centers = np.linspace(-1.8, 1.8, 181)  # includes the wells at +-1 and the barrier at 0
    p = np.exp(-((centers**2 - 1.0) ** 2))
    p /= p.sum() * (centers[1] - centers[0])
    pi = sc.Profile(centers, p, np.ones(181, dtype=np.int64))
    D = sc.Profile(np.linspace(-1.8, 1.8, 5), np.full(5, D0), np.ones(5, dtype=np.int64))
    out = pmf_kramers_rate(pi, D)
    assert out["delta_F_AB"] == pytest.approx(1.0, abs=1e-9)
    assert out["delta_F_BA"] == pytest.approx(1.0, abs=1e-9)
    assert out["s_barrier"] == pytest.approx(0.0, abs=1e-12)
    assert out["D_barrier"] == pytest.approx(D0)
    kramers = D0 / (2.0 * np.pi) * np.sqrt(8.0 * 4.0) * np.exp(-1.0)
    assert out["k_AB"] == pytest.approx(kramers, rel=0.05)
    assert out["k_BA"] == pytest.approx(kramers, rel=0.05)


# ---------------------------------------------------------------------------
# profiles and their reducers
# ---------------------------------------------------------------------------
def _flux_profile(values, n=50):
    centers = (np.arange(n) + 0.5) / n
    return sc.Profile(centers, np.asarray(values, dtype=float), np.full(n, 100))


def test_value_at_point_array_and_range():
    prof = _flux_profile(np.arange(50, dtype=float))
    assert sc.value_at(prof, 0.5) == pytest.approx(24.5)
    np.testing.assert_allclose(sc.value_at(prof, np.array([0.01, 0.99])), [0.0, 49.0])
    assert sc.value_at(prof, (0.0, 1.0)) == pytest.approx(24.5)
    with pytest.raises(ValueError, match="no finite"):
        sc.value_at(_flux_profile(np.full(50, np.nan)), 0.5)


def test_flux_flatness_is_zero_for_a_constant_and_nan_below_two_bins():
    assert sc.flux_flatness(_flux_profile(np.ones(50)), (0.2, 0.8)) == 0.0
    assert np.isnan(sc.flux_flatness(_flux_profile(np.ones(50)), (0.495, 0.505)))
    vals = np.ones(50)
    vals[:10] = 5.0
    assert sc.flux_flatness(_flux_profile(vals), (0.3, 0.7)) == 0.0
    assert sc.flux_flatness(_flux_profile(vals), (0.0, 1.0)) > 0.5


def test_find_plateau_flat_profile_takes_the_widest_interior_window():
    rng = np.random.default_rng(0)
    vals = 1.0 + 0.01 * rng.standard_normal(50)
    pw = sc.find_plateau(_flux_profile(vals), margin=0.05)
    assert pw.ok and pw.flatness < 0.1
    assert pw.lo <= 0.15 and pw.hi >= 0.85
    assert pw.value == pytest.approx(1.0, abs=0.1)


def test_find_plateau_excludes_basin_spikes():
    vals = np.ones(50)
    vals[:6] = 12.0
    vals[-6:] = 12.0
    pw = sc.find_plateau(_flux_profile(vals), margin=0.02)
    assert pw.ok and pw.lo >= 0.1 and pw.hi <= 0.9
    assert pw.value == pytest.approx(1.0, abs=0.2)


def test_find_plateau_with_nothing_valid_is_not_ok_and_the_rate_warns():
    pw = sc.find_plateau(_flux_profile(np.full(50, np.nan)))
    assert not pw.ok and pw.n_valid == 0 and not np.isfinite(pw.value)
    # a flux that is nowhere flat: the auto plateau warns and flags plateau_ok=False
    centers = (np.arange(50) + 0.5) / 50
    ones = np.ones(50, dtype=np.int64)
    pi = sc.Profile(centers, np.ones(50), ones)
    Dq = sc.Profile(centers, np.exp(6.0 * centers), ones)
    with pytest.warns(UserWarning, match="no band"):
        out = sc.rate_from_profiles(pi, Dq, 0.5, 0.5, reduction="plateau", band="auto")
    assert out["plateau_ok"] is False and np.isfinite(out["k_AB"])


# ---------------------------------------------------------------------------
# layering
# ---------------------------------------------------------------------------
def test_rates_never_imports_the_umbrella_subpackage():
    root = pathlib.Path(sc.__file__).parent / "rates"
    for path in root.glob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            else:
                continue
            for name in names:
                assert "umbrella" not in name and "workflows" not in name, (path.name, name)
