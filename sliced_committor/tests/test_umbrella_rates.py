"""The umbrella rate bundle on a synthetic double-well umbrella run with a known rate.

Overdamped Langevin windows on ``V = (x^2 - 1)^2 + y^2 / 2`` restrained in
``x``: the exact committor depends on ``x`` alone, the mobility ``D0`` is
known, and the exact rate follows from the 1D flux formula. WHAM supplies the
weights (pymbar is optional), the sliced committor is fitted on ``(x, y)``,
and the CV-mapped plateau rate must land on the exact one.
"""

import jax

jax.config.update("jax_enable_x64", True)

import numpy as np
import pytest

from sliced_committor.umbrella import (
    USDataset,
    fit_and_rate,
    pmf_kramers_rate,
    progress_coordinate,
    reweight,
)

from ._helpers import umbrella_double_well

D0 = 0.05
DT = 0.01
A = 0.9  # basins: x < -A and x > A


def _exact_rates(D0, a):
    """``(k_AB, k_BA)`` of 1D diffusion in ``V = (x^2 - 1)^2`` between ``|x| < a``."""
    x = np.linspace(-2.5, 2.5, 100_001)
    dx = x[1] - x[0]
    V = (x**2 - 1.0) ** 2
    w = np.exp(-V)
    Z_pi = float(np.sum(w) * dx)
    eV = np.where(np.abs(x) < a, np.exp(V), 0.0)
    Z_q = float(np.sum(eV) * dx)
    q = np.clip(np.cumsum(eV) * dx / Z_q, 0.0, 1.0)
    rho_A = float(np.sum((1.0 - q) * w) * dx / Z_pi)
    nu = D0 / (Z_q * Z_pi)
    return nu / rho_A, nu / (1.0 - rho_A)


@pytest.fixture(scope="module")
def dataset():
    x, y, wid, centers, kappa = umbrella_double_well(n_windows=8, dt=DT, D0=D0, seed=0)
    return USDataset(
        features=np.stack([x, y], axis=1),
        cvs=x[:, None],
        window_ids=wid,
        window_centers=centers[:, None],
        window_kappa=np.full((centers.size, 1), kappa),
        beta=1.0,
        dt=DT,
        in_A=x < -A,
        in_B=x > A,
        cv_periodic=(None,),
        meta={},
    )


@pytest.fixture(scope="module")
def bundle(dataset):
    rw = reweight(dataset, method="wham", n_bins=80)
    out = fit_and_rate(
        dataset,
        dataset.features,
        rw.sample_weights,
        n_directions=64,
        n_bins=60,
        seed=0,
        diffusion=("cvmap", "hummer_q", "km_q"),
        n_diff_bins=25,
    )
    return rw, out


def test_cvmap_plateau_rate_matches_the_exact_double_well_rate(bundle):
    _, out = bundle
    k_AB, k_BA = _exact_rates(D0, A)
    got = out["rates"]["cvmap"]["plateau"]
    assert 0.6 < got["k_AB"] / k_AB < 1.5, (got["k_AB"], k_AB)
    assert 0.6 < got["k_BA"] / k_BA < 1.5, (got["k_BA"], k_BA)
    assert out["D_s"]["value"] == pytest.approx(D0, rel=0.3)
    assert out["cv_grad_sq"] == pytest.approx(1.0, rel=1e-3)  # x is itself a feature


def test_bundle_carries_every_constructor_reduction_and_the_baseline(bundle):
    rw, out = bundle
    assert set(out["rates"]) == {"cvmap", "hummer_q", "km_q"}
    for name, by_reduction in out["rates"].items():
        assert set(by_reduction) == {"plateau", "harmonic", "arithmetic"}, name
        for reduction, rate in by_reduction.items():
            assert np.isfinite(rate["k_AB"]) and rate["k_AB"] > 0, (name, reduction)
            assert "nu" not in rate  # profiles live under out["profiles"]
        assert np.isfinite(out["flux_cv"][name])
        assert out["profiles"]["D_q"][name].shape == (25,)
        assert out["profiles"]["flux"][name].shape == (25,)
    assert out["profiles"]["levels"].shape == (25,) and out["profiles"]["pi"].shape == (25,)
    assert out["q_samples"].shape == (rw.sample_weights.shape[0],)
    assert out["committor"]["n_directions"] == 64 and out["committor"]["valid_fraction"] == 1.0
    assert np.isfinite(out["kramers"]["k_AB"]) and out["kramers"]["k_AB"] > 0
    assert out["errors"] == {}
    assert out["D_s"]["n_windows"] == 8 and out["D_s"]["tau_int"] > 1.0


def test_unknown_constructors_and_reductions_are_rejected(dataset):
    w = np.full(dataset.n_frames, 1.0 / dataset.n_frames)
    with pytest.raises(ValueError, match="diffusion"):
        fit_and_rate(dataset, dataset.features, w, diffusion=("bogus",), n_directions=8)
    with pytest.raises(ValueError, match="reduction"):
        fit_and_rate(dataset, dataset.features, w, reductions=("bogus",), n_directions=8)


def test_strict_false_records_a_failed_constructor_instead_of_raising(dataset):
    w = np.full(dataset.n_frames, 1.0 / dataset.n_frames)
    kw = dict(diffusion=("km_q",), lag=10**7, n_directions=16, n_bins=40, n_diff_bins=10)
    with pytest.raises(ValueError, match="not shorter"):
        fit_and_rate(dataset, dataset.features, w, **kw)
    out = fit_and_rate(dataset, dataset.features, w, strict=False, **kw)
    assert "km_q" in out["errors"] and "not shorter" in out["errors"]["km_q"]
    assert out["rates"] == {} and np.isfinite(out["kramers"]["k_AB"])


def test_kramers_baseline_reads_the_barrier_of_the_pmf(dataset, bundle):
    rw, _ = bundle
    out = pmf_kramers_rate(dataset, rw.sample_weights)
    assert np.isfinite(out["k_AB"]) and out["k_AB"] > 0
    assert out["delta_F_AB"] == pytest.approx(1.0, abs=0.3)
    assert abs(out["s_barrier"]) < 0.15
    assert out["D_barrier"] == pytest.approx(D0, rel=0.4)


def test_progress_coordinate_projects_a_2d_cv_onto_the_AB_axis():
    rng = np.random.default_rng(1)
    cvs = rng.normal(size=(2000, 2))
    in_A = (cvs[:, 0] < -0.8) & (cvs[:, 1] < -0.8)
    in_B = (cvs[:, 0] > 0.8) & (cvs[:, 1] > 0.8)
    ds = USDataset(
        features=cvs,
        cvs=cvs,
        window_ids=np.zeros(len(cvs), int),
        window_centers=np.zeros((1, 2)),
        window_kappa=np.zeros((1, 2)),
        beta=1.0,
        dt=1.0,
        in_A=in_A,
        in_B=in_B,
        cv_periodic=(None, None),
        meta={},
    )
    s = progress_coordinate(ds)
    assert s.shape == (len(cvs),)
    assert s[in_A].mean() < s[in_B].mean()
    assert np.corrcoef(s, cvs[:, 0] + cvs[:, 1])[0, 1] > 0.99
