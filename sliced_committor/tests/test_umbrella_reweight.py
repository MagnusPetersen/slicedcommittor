"""Reweighting (WHAM always, MBAR when pymbar is present) and the dataset container."""

import numpy as np
import pytest

from sliced_committor.umbrella import (
    USDataset,
    mbar_weights,
    reduced_harmonic,
    reweight,
    split_by_window,
    wham_weights,
)


def _dataset(cvs, wid, centers, kappa):
    n = cvs.shape[0]
    return USDataset(
        features=cvs,
        cvs=cvs,
        window_ids=wid,
        window_centers=centers,
        window_kappa=kappa,
        beta=1.0,
        dt=1.0,
        in_A=np.zeros(n, bool),
        in_B=np.zeros(n, bool),
        cv_periodic=tuple([None] * cvs.shape[1]),
        meta={},
    )


# ---------------------------------------------------------------------------
# 1D
# ---------------------------------------------------------------------------
def _synthetic_windows(n_per=300, seed=0):
    rng = np.random.default_rng(seed)
    centers = np.linspace(-2, 2, 9)[:, None]
    kappa = np.full((9, 1), 20.0)
    cvs, wid = [], []
    for k, c in enumerate(centers[:, 0]):
        cvs.append(rng.normal(c, 1 / np.sqrt(20.0), size=n_per))
        wid += [k] * n_per
    return np.concatenate(cvs)[:, None], np.array(wid), centers, kappa


def test_wham_normalised_and_pmf():
    cvs, wid, centers, kappa = _synthetic_windows()
    rw = wham_weights(cvs, wid, centers, kappa, 1.0, (None,), n_bins=50)
    assert rw.method == "wham"
    assert abs(rw.sample_weights.sum() - 1.0) < 1e-9
    assert rw.n_eff > 0
    assert rw.pmf is not None


def test_mbar_matches_wham():
    pytest.importorskip("pymbar")
    cvs, wid, centers, kappa = _synthetic_windows()
    mb = mbar_weights(cvs, wid, centers, kappa, 1.0, (None,))
    wh = wham_weights(cvs, wid, centers, kappa, 1.0, (None,), n_bins=60)
    assert abs(mb.sample_weights.sum() - 1.0) < 1e-9
    fmb = mb.f_k - mb.f_k[0]
    fwh = wh.f_k - wh.f_k[0]
    assert np.max(np.abs(fmb - fwh)) < 0.25  # free energies agree within 0.25 kT


def test_reweight_auto_and_forced_engines():
    cvs, wid, centers, kappa = _synthetic_windows()
    ds = _dataset(cvs, wid, centers, kappa)
    rw = reweight(ds, method="auto", n_bins=40)
    assert rw.method in ("mbar", "wham")  # 9 windows: MBAR when available
    assert reweight(ds, method="wham", n_bins=40).method == "wham"
    with pytest.raises(ValueError, match="unknown reweight method"):
        reweight(ds, method="bogus")


def test_reduced_harmonic_periodic():
    pts = np.array([[np.pi - 0.1]])
    centers = np.array([[-np.pi + 0.1]])
    kappa = np.array([[1.0]])
    # without periodicity the displacement is ~2 pi; with it, ~0.2 (minimum image)
    u_plain = reduced_harmonic(pts, centers, kappa, 1.0, (None,))
    u_per = reduced_harmonic(pts, centers, kappa, 1.0, ((-np.pi, np.pi),))
    assert u_per[0, 0] < u_plain[0, 0]
    assert u_per[0, 0] < 0.05


# ---------------------------------------------------------------------------
# 2D
# ---------------------------------------------------------------------------
def _gaussian_well_2d(seed=0, n_per=4000, kappa=6.0):
    """Umbrella windows on a 2D grid sampling the biased Gaussian of the known
    potential ``V(x, y) = (x^2 + y^2) / 2``: for window centre ``c`` the biased
    density is Gaussian with per-dim precision ``1 + kappa`` and mean
    ``kappa c / (1 + kappa)``, so the samples are exact and WHAM must recover
    ``V`` up to a constant."""
    rng = np.random.default_rng(seed)
    grid = np.linspace(-1.2, 1.2, 5)
    centers = np.stack(np.meshgrid(grid, grid), axis=-1).reshape(-1, 2)  # (25, 2)
    K = centers.shape[0]
    kap = np.full((K, 2), kappa)
    prec = 1.0 + kappa
    cvs, wid = [], []
    for k, c in enumerate(centers):
        mean = kappa * c / prec
        cvs.append(mean[None, :] + rng.normal(scale=1.0 / np.sqrt(prec), size=(n_per, 2)))
        wid.append(np.full(n_per, k))
    return np.concatenate(cvs), np.concatenate(wid), centers, kap


def _reweighted_pmf_2d(cvs, weights, lo=-1.0, hi=1.0, nb=16):
    edges = np.linspace(lo, hi, nb + 1)
    H, _, _ = np.histogram2d(cvs[:, 0], cvs[:, 1], bins=[edges, edges], weights=weights)
    cen = 0.5 * (edges[1:] + edges[:-1])
    XX, YY = np.meshgrid(cen, cen, indexing="ij")
    occ = H.max() * 1e-3 < H
    F = np.full_like(H, np.nan)
    F[occ] = -np.log(H[occ])
    return F, 0.5 * (XX**2 + YY**2), occ


def test_wham_2d_recovers_known_pmf():
    cvs, wid, centers, kappa = _gaussian_well_2d()
    rw = wham_weights(cvs, wid, centers, kappa, 1.0, (None, None), n_bins=40)
    w = np.asarray(rw.sample_weights)
    assert abs(w.sum() - 1.0) < 1e-8 and (w >= 0).all()
    F, Vtrue, occ = _reweighted_pmf_2d(cvs, w)
    dev = (F - Vtrue)[occ]
    dev -= dev.mean()  # a PMF is defined up to a constant
    assert np.nanmax(np.abs(dev)) < 0.4, f"2D WHAM PMF off by {np.nanmax(np.abs(dev)):.3f} kT"


def test_mbar_matches_wham_2d():
    pytest.importorskip("pymbar")
    cvs, wid, centers, kappa = _gaussian_well_2d()
    mb = mbar_weights(cvs, wid, centers, kappa, 1.0, (None, None))
    wh = wham_weights(cvs, wid, centers, kappa, 1.0, (None, None), n_bins=40)
    Fm, _, occ = _reweighted_pmf_2d(cvs, np.asarray(mb.sample_weights))
    Fw, _, _ = _reweighted_pmf_2d(cvs, np.asarray(wh.sample_weights))
    d = (Fm - Fw)[occ]
    d -= d.mean()
    assert np.nanmax(np.abs(d)) < 0.4, "2D MBAR and WHAM PMFs disagree"


# ---------------------------------------------------------------------------
# the container
# ---------------------------------------------------------------------------
def test_split_by_window_and_subsample():
    arr = np.arange(10)
    wid = np.array([0, 0, 0, 1, 1, 1, 1, 2, 2, 2])
    assert [len(p) for p in split_by_window(arr, wid)] == [3, 4, 3]
    feats = np.arange(20).reshape(10, 2).astype(float)
    ds = _dataset(feats, wid, np.zeros((3, 2)), np.ones((3, 2)))
    assert (ds.n_frames, ds.n_features, ds.n_cv, ds.n_windows) == (10, 2, 2, 3)
    sub, kept = ds.subsample(6, seed=0)
    assert 3 <= sub.n_frames <= 10
    assert set(np.unique(sub.window_ids)) == {0, 1, 2}  # every window represented
    assert kept.shape[0] == sub.n_frames
    same, idx = ds.subsample(100)
    assert same is ds and idx.shape == (10,)
