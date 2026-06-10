"""Tests for the rate computations: quantities + rate formulas.

Covers populations normalisation, the ``coordinate=`` abstraction, Kramers-Moyal
diffusion recovery on a 1D OU process, reactive-flux plateau flatness (TPT flux
conservation), the Dirichlet volume-integral / co-area identity, and that the
Berezhkovskii-Szabo and Kramers estimates produce finite positive rates.
"""

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

import sliced_committor as sc

# Shared equilibrium double-well sampler (also used by the validation test).
from .test_validation_2d_double_well import _equilibrium_samples


def _overdamped_double_well_trajectory(T=60000, dt=0.01, D0=0.05, seed=1):
    """Overdamped Langevin on V=(x^2-1)^2 + 0.5 y^2 with mobility D0."""
    rng = np.random.default_rng(seed)
    x = np.array([-1.0, 0.0])
    out = np.empty((T, 2))
    noise = np.sqrt(2.0 * D0 * dt)
    for t in range(T):
        grad = np.array([4.0 * x[0] * (x[0] ** 2 - 1.0), x[1]])
        x = x - D0 * grad * dt + noise * rng.standard_normal(2)
        out[t] = x
    return out


@pytest.fixture(scope="module")
def committor():
    s, in_A, in_B = _equilibrium_samples(n_samples=2500)
    q = sc.fit_committor(s, in_A=in_A, in_B=in_B, n_directions=256, n_bins=120, seed=2)
    return q, s, in_A, in_B


# ---------------------------------------------------------------------------
# Populations
# ---------------------------------------------------------------------------


def test_populations_sum_to_one(committor):
    q, s, in_A, in_B = committor
    out = sc.dirichlet_rate(q, s, D=0.05, in_A=in_A, in_B=in_B)
    assert out["rho_A"] + out["rho_B"] == pytest.approx(1.0, abs=1e-12)
    assert 0.0 < out["rho_A"] < 1.0


# ---------------------------------------------------------------------------
# Density + coordinate= abstraction
# ---------------------------------------------------------------------------


def test_density_along_committor_integrates_to_one(committor):
    q, s, in_A, in_B = committor
    prof = sc.density(q, s, n_bins=50)
    dq = prof.levels[1] - prof.levels[0]
    assert float(np.sum(prof.values) * dq) == pytest.approx(1.0, abs=1e-6)


def test_density_with_cv_coordinate_matches_histogram(committor):
    q, s, in_A, in_B = committor
    cv = np.asarray(s)[:, 0]  # use x as an arbitrary collective variable
    dummy = lambda p: jnp.zeros(jnp.asarray(p).shape[0])  # noqa: E731 (unused for coordinate=)
    prof = sc.density(dummy, np.asarray(s), coordinate=cv, n_bins=30)
    edges = np.linspace(cv.min(), cv.max(), 31)
    dq = edges[1] - edges[0]
    hist, _ = np.histogram(cv, bins=edges)
    expected = hist / (hist.sum() * dq)
    np.testing.assert_allclose(prof.values, expected, atol=1e-9, rtol=0)


# ---------------------------------------------------------------------------
# Diffusion (Kramers-Moyal) on a 1D OU process with known D
# ---------------------------------------------------------------------------


def test_diffusion_recovers_constant_D_on_ou():
    # 1D OU: dx = -k x dt + sqrt(2 D) dW. Drift-corrected KM second moment -> D.
    rng = np.random.default_rng(0)
    T, dt, k, D_true = 200_000, 0.01, 1.0, 0.05
    x = np.empty(T)
    xi = 0.0
    noise = np.sqrt(2.0 * D_true * dt)
    for t in range(T):
        xi = xi - k * xi * dt + noise * rng.standard_normal()
        x[t] = xi
    dummy = lambda p: jnp.zeros(jnp.asarray(p).shape[0])  # noqa: E731
    D_est = sc.diffusion_coefficient(
        dummy, x[:, None], dt=dt, coordinate=x, at=0.0, lag=1, n_bins=20
    )
    assert float(D_est) == pytest.approx(D_true, rel=0.25)


# ---------------------------------------------------------------------------
# Reactive flux: TPT flux conservation (plateau flatness)
# ---------------------------------------------------------------------------


def test_reactive_flux_plateau_is_flat(committor):
    q, s, in_A, in_B = committor
    prof = sc.reactive_flux(q, s, D=0.05, n_bins=30)
    sel = (prof.levels >= 0.3) & (prof.levels <= 0.7) & (prof.counts > 0)
    vals = prof.values[sel]
    assert vals.size >= 3
    flatness = float(np.std(vals) / max(abs(np.mean(vals)), 1e-30))
    assert flatness < 0.35, f"plateau flatness {flatness:.3f} too large"


def test_dirichlet_full_matches_coarea_integral(committor):
    q, s, in_A, in_B = committor
    # Volume integral D<|grad q|^2>_pi == integral of the co-area flux Phi(c) dc.
    nu_full = sc.dirichlet_rate(q, s, D=0.05, at=None)["nu_R"]
    prof = sc.reactive_flux(q, s, D=0.05, n_bins=60)
    dc = prof.levels[1] - prof.levels[0]
    nu_coarea = float(np.sum(prof.values) * dc)
    assert nu_full == pytest.approx(nu_coarea, rel=1e-6)


def test_dirichlet_point_and_range_positive(committor):
    q, s, in_A, in_B = committor
    full = sc.dirichlet_rate(q, s, D=0.05, at=None, in_A=in_A, in_B=in_B)
    pt = sc.dirichlet_rate(q, s, D=0.05, at=0.5, in_A=in_A, in_B=in_B)
    rng_ = sc.dirichlet_rate(q, s, D=0.05, at=(0.3, 0.7), in_A=in_A, in_B=in_B)
    for out in (full, pt, rng_):
        assert out["nu_R"] > 0 and np.isfinite(out["k_AB"]) and out["k_AB"] > 0
    # point / range / full agree to within an order of magnitude (flat flux).
    assert 0.1 < pt["nu_R"] / full["nu_R"] < 10.0


def test_tpt_rate_runs(committor):
    q, s, in_A, in_B = committor
    out = sc.tpt_rate(q, s, D=0.05, in_A=in_A, in_B=in_B)
    assert out["nu_R"] > 0 and np.isfinite(out["k_AB"]) and out["k_AB"] > 0
    assert "plateau_flatness" in out


def test_dirichlet_plateau_keys_are_flat_and_consistent(committor):
    # The bespoke nested `section` dict was replaced by flat plateau keys shared
    # with tpt_rate / committor_rate. Lock that contract here.
    q, s, in_A, in_B = committor
    rng_ = sc.dirichlet_rate(q, s, D=0.05, at=(0.3, 0.7), in_A=in_A, in_B=in_B)
    assert "section" not in rng_
    assert rng_["kind"] == "range"
    assert rng_["plateau"] == (0.3, 0.7)
    assert np.isfinite(rng_["plateau_flatness"])

    auto = sc.dirichlet_rate(q, s, D=0.05, at="auto", in_A=in_A, in_B=in_B)
    assert "section" not in auto
    assert auto["kind"] == "auto"
    assert auto["plateau_auto"] is True
    assert isinstance(auto["plateau_ok"], bool)
    lo, hi = auto["plateau"]
    assert 0.0 <= lo < hi <= 1.0
    assert np.isfinite(auto["plateau_flatness"])

    full = sc.dirichlet_rate(q, s, D=0.05, at=None, in_A=in_A, in_B=in_B)
    assert full["kind"] == "full" and "plateau" not in full


# ---------------------------------------------------------------------------
# Trajectory-based rates: Berezhkovskii-Szabo + Kramers
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def committor_and_trajectory():
    s, in_A, in_B = _equilibrium_samples(n_samples=3000, seed=4)
    q = sc.fit_committor(s, in_A=in_A, in_B=in_B, n_directions=256, n_bins=120, seed=4)
    traj = _overdamped_double_well_trajectory()
    return q, s, in_A, in_B, traj


def test_berezhkovskii_szabo_local_and_mfpt_finite(committor_and_trajectory):
    q, s, in_A, in_B, traj = committor_and_trajectory
    local = sc.berezhkovskii_szabo_rate(
        q, s, traj, dt=0.01, at=0.5, mode="local", n_bins=50, in_A=in_A, in_B=in_B
    )
    assert local["D_at_q_star"] > 0 and local["pi_at_q_star"] > 0
    assert local["nu_R"] > 0 and np.isfinite(local["k_AB"]) and local["k_AB"] > 0
    assert local["mode"] == "local"  # back-compat label preserved by the wrapper

    mfpt = sc.berezhkovskii_szabo_rate(
        q, s, traj, dt=0.01, mode="mfpt", at=None, n_bins=50, in_A=in_A, in_B=in_B
    )
    assert mfpt["mfpt_AB"] > 0 and np.isfinite(mfpt["k_AB"]) and mfpt["k_AB"] > 0
    assert mfpt["mode"] == "mfpt"


def test_kramers_rate_on_physical_cv(committor_and_trajectory):
    # Kramers is well-posed on a physical CV (here x) where F(x) is a genuine
    # double well with interior minima and a central barrier.
    q, s, in_A, in_B, traj = committor_and_trajectory
    cv_s = np.asarray(s)[:, 0]
    cv_t = np.asarray(traj)[:, 0]
    out = sc.kramers_rate(
        q, s, traj, dt=0.01, n_bins=60, coordinate=cv_s, traj_coordinate=cv_t, in_A=in_A, in_B=in_B
    )
    assert np.isfinite(out["k_AB"]) and out["k_AB"] > 0
    assert out["delta_F_AB"] > 0  # barrier (≈1 kT) between the x=±1 wells
    assert out["D_barrier"] > 0


def test_kramers_rate_on_committor_runs(committor_and_trajectory):
    # On the bare committor the harmonic fit is crude but must not crash.
    q, s, in_A, in_B, traj = committor_and_trajectory
    out = sc.kramers_rate(q, s, traj, dt=0.01, n_bins=50, in_A=in_A, in_B=in_B)
    assert set(out) >= {"k_AB", "k_BA", "rho_A", "rho_B", "q_barrier"}
    assert out["rho_A"] + out["rho_B"] == pytest.approx(1.0, abs=1e-12)


# ---------------------------------------------------------------------------
# Hummer (Var/tau_int) diffusion estimator
# ---------------------------------------------------------------------------


def _windowed_ou(n_win=6, n_per=20000, dt=0.01, k=1.0, D_true=0.05, seed=3):
    """K independent confined-OU segments around 0; var/tau_corr = D_true."""
    rng = np.random.default_rng(seed)
    noise = np.sqrt(2.0 * D_true * dt)
    xs, wid = [], []
    for w in range(n_win):
        x = np.empty(n_per)
        xi = 0.0
        for t in range(n_per):
            xi = xi - k * xi * dt + noise * rng.standard_normal()
            x[t] = xi
        xs.append(x)
        wid.append(np.full(n_per, w))
    return np.concatenate(xs), np.concatenate(wid)


def test_hummer_recovers_constant_D_on_windowed_ou():
    # Confined OU: D = var / tau_corr. The Hummer Var/tau_int estimator recovers
    # it per window WITHOUT a lag scan (lag-robust).
    x, wid = _windowed_ou()
    dummy = lambda p: jnp.zeros(jnp.asarray(p).shape[0])  # noqa: E731 (coordinate= path)
    prof = sc.diffusion_coefficient(
        dummy, x[:, None], dt=0.01, coordinate=x, window_ids=wid, method="hummer"
    )
    assert prof.values.shape[0] == 6  # one D per window
    assert float(np.median(prof.values)) == pytest.approx(0.05, rel=0.3)


def test_hummer_requires_window_ids():
    x, _ = _windowed_ou(n_win=1, n_per=2000)
    dummy = lambda p: jnp.zeros(jnp.asarray(p).shape[0])  # noqa: E731
    with pytest.raises(ValueError, match="requires window_ids"):
        sc.diffusion_coefficient(dummy, x[:, None], dt=0.01, coordinate=x, method="hummer")


def test_diffusion_method_unknown_raises(committor_and_trajectory):
    q, s, in_A, in_B, traj = committor_and_trajectory
    with pytest.raises(ValueError, match="unknown method"):
        sc.diffusion_coefficient(q, traj, dt=0.01, method="bogus")


# ---------------------------------------------------------------------------
# committor_rate: unified {D_q, pi} reductions
# ---------------------------------------------------------------------------


def test_committor_rate_harmonic_matches_bs_mfpt(committor_and_trajectory):
    # The harmonic reduction IS the Berezhkovskii-Szabo MFPT (shared machinery).
    q, s, in_A, in_B, traj = committor_and_trajectory
    cr = sc.committor_rate(
        q, s, traj, dt=0.01, reduction="harmonic", at=None, n_bins=50, in_A=in_A, in_B=in_B
    )
    bs = sc.berezhkovskii_szabo_rate(
        q, s, traj, dt=0.01, mode="mfpt", at=None, n_bins=50, in_A=in_A, in_B=in_B
    )
    assert cr["k_AB"] == pytest.approx(bs["k_AB"], rel=1e-12)
    assert cr["k_BA"] == pytest.approx(bs["k_BA"], rel=1e-12)


def test_committor_rate_local_matches_bs_local(committor_and_trajectory):
    q, s, in_A, in_B, traj = committor_and_trajectory
    cr = sc.committor_rate(
        q, s, traj, dt=0.01, reduction="local", at=0.5, n_bins=50, in_A=in_A, in_B=in_B
    )
    bs = sc.berezhkovskii_szabo_rate(
        q, s, traj, dt=0.01, mode="local", at=0.5, n_bins=50, in_A=in_A, in_B=in_B
    )
    assert cr["nu_R"] == pytest.approx(bs["nu_R"], rel=1e-12)


def test_committor_rate_reductions_finite_and_consistent(committor_and_trajectory):
    # All four reductions of {D_q, pi} are finite + positive, and for this
    # well-behaved toy committor (near-flat flux) they agree within an order.
    q, s, in_A, in_B, traj = committor_and_trajectory
    ks = {}
    for red in ("arithmetic", "plateau", "local", "harmonic"):
        out = sc.committor_rate(
            q, s, traj, dt=0.01, reduction=red, at=None, n_bins=50, in_A=in_A, in_B=in_B
        )
        assert np.isfinite(out["k_AB"]) and out["k_AB"] > 0, red
        assert out["rho_A"] + out["rho_B"] == pytest.approx(1.0, abs=1e-12)
        ks[red] = out["k_AB"]
    vals = np.array(list(ks.values()))
    assert vals.max() / vals.min() < 10.0


def test_committor_rate_unknown_reduction_raises(committor_and_trajectory):
    q, s, in_A, in_B, traj = committor_and_trajectory
    with pytest.raises(ValueError, match="unknown reduction"):
        sc.committor_rate(q, s, traj, dt=0.01, reduction="bogus", n_bins=50)


# ---------------------------------------------------------------------------
# find_plateau: automatic flux-flatness range detection
# ---------------------------------------------------------------------------


def _flux_profile(values, n=50):
    centers = (np.arange(n) + 0.5) / n
    return sc.Profile(
        levels=centers, values=np.asarray(values, float), counts=np.full(n, 100), name="committor"
    )


def test_find_plateau_flat_profile_takes_widest_interior():
    rng = np.random.default_rng(0)
    n = 50
    vals = 1.0 + 0.01 * rng.standard_normal(n)  # near-constant flux
    pw = sc.find_plateau(_flux_profile(vals, n), margin=0.05)
    assert pw.ok and pw.flatness < 0.1
    assert pw.lo <= 0.15 and pw.hi >= 0.85  # near-full interior window
    assert pw.value == pytest.approx(1.0, abs=0.1)


def test_find_plateau_excludes_basin_spikes():
    n = 50
    vals = np.ones(n)
    vals[:6] = 12.0  # basin-side inflation (q≈0)
    vals[-6:] = 12.0  # basin-side inflation (q≈1)
    pw = sc.find_plateau(_flux_profile(vals, n), margin=0.02)
    assert pw.ok
    assert pw.lo >= 0.1 and pw.hi <= 0.9  # spikes excluded
    assert pw.value == pytest.approx(1.0, abs=0.2)


def test_find_plateau_all_invalid_returns_not_ok():
    pw = sc.find_plateau(_flux_profile(np.full(50, np.nan)))
    assert not pw.ok and pw.n_valid == 0 and not np.isfinite(pw.value)


def test_tpt_auto_plateau_runs(committor):
    q, s, in_A, in_B = committor
    out = sc.tpt_rate(q, s, D=0.05, at="auto", in_A=in_A, in_B=in_B)
    assert out["plateau_auto"] is True and "plateau_ok" in out
    assert out["nu_R"] > 0 and np.isfinite(out["k_AB"]) and out["k_AB"] > 0
    lo, hi = out["plateau"]
    assert 0.0 <= lo < hi <= 1.0


# ---------------------------------------------------------------------------
# saddle_bridge_D: calibrated configurational D bridging the two rate families
# ---------------------------------------------------------------------------


def test_saddle_bridge_local_calibration_identity(committor_and_trajectory):
    # By construction D = D_q(q*) / ⟨|∇q̄|²⟩_{q*}, and the object is a
    # callable constant-D usable directly as the D= argument.
    q, s, in_A, in_B, traj = committor_and_trajectory
    bridge = sc.saddle_bridge_D(q, s, traj, dt=0.01, q_star=0.5, n_bins=50)
    assert bridge.D > 0 and np.isfinite(bridge.D)
    assert bridge.D * bridge.g_ref == pytest.approx(bridge.D_q_ref, rel=1e-9)
    np.testing.assert_allclose(bridge(np.array([0.1, 0.5, 0.9])), bridge.D)


def test_saddle_bridge_calibrates_feature_space_rate(committor_and_trajectory):
    # Feeding the bridge into tpt_rate yields a committor-coordinate-consistent
    # rate: it lands within an order of magnitude of committor_rate(plateau),
    # whereas an uncalibrated scalar D=1 is off by ~1/bridge.D.
    q, s, in_A, in_B, traj = committor_and_trajectory
    bridge = sc.saddle_bridge_D(q, s, traj, dt=0.01, q_star=0.5, n_bins=50)
    tpt = sc.tpt_rate(q, s, D=bridge, at="auto", in_A=in_A, in_B=in_B)
    cr = sc.committor_rate(
        q, s, traj, dt=0.01, reduction="plateau", at="auto", n_bins=50, in_A=in_A, in_B=in_B
    )
    assert tpt["k_AB"] > 0 and cr["k_AB"] > 0
    assert 0.1 < tpt["k_AB"] / cr["k_AB"] < 10.0


def test_saddle_bridge_volume_average_runs(committor_and_trajectory):
    q, s, in_A, in_B, traj = committor_and_trajectory
    bridge = sc.saddle_bridge_D(q, s, traj, dt=0.01, mode="volume_average", n_bins=50)
    assert bridge.D > 0 and np.isfinite(bridge.D)
    assert bridge.mode == "volume_average"


def test_saddle_bridge_unknown_mode_raises(committor_and_trajectory):
    q, s, in_A, in_B, traj = committor_and_trajectory
    with pytest.raises(ValueError, match="unknown mode"):
        sc.saddle_bridge_D(q, s, traj, dt=0.01, mode="bogus", n_bins=50)
