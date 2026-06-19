"""2D collective-variable umbrella support: 2D WHAM/MBAR reweighting, the A->B
progress-coordinate projection, and the toy loader's 2D-bias branch.

These guard the 2D path the c-Src activation system (2D, 384 windows -> WHAM) and
the wolfe_quapp_stiff MEP-tube system rely on. Synthetic fixtures only.
"""

from __future__ import annotations

import numpy as np
import pytest

from sliced_committor.workflows import reweight
from sliced_committor.workflows._containers import USDataset
from sliced_committor.workflows.loaders import load_toy_dataset
from sliced_committor.workflows.pmf_kramers import progress_coordinate


def _gaussian_well_2d(seed=0, n_per=4000, kappa=6.0):
    """Umbrella windows on a 2D grid sampling the biased Gaussian for a known
    unbiased potential V(x,y) = 0.5 (x^2 + y^2). For window centre c the biased
    density is Gaussian with per-dim precision (1 + kappa) and mean kappa c/(1+kappa),
    so the samples are exact and WHAM must recover V up to a constant.
    """
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
    Vtrue = 0.5 * (XX**2 + YY**2)
    return F, Vtrue, occ


def test_wham_2d_recovers_known_pmf():
    cvs, wid, centers, kappa = _gaussian_well_2d()
    rw = reweight.wham_weights(cvs, wid, centers, kappa, 1.0, (None, None), n_bins=40)
    assert rw.method == "wham"
    w = np.asarray(rw.sample_weights)
    assert abs(w.sum() - 1.0) < 1e-8 and (w >= 0).all()
    F, Vtrue, occ = _reweighted_pmf_2d(cvs, w)
    dev = (F - Vtrue)[occ]
    dev -= dev.mean()  # PMF defined up to a constant
    assert np.nanmax(np.abs(dev)) < 0.4, f"2D WHAM PMF off by {np.nanmax(np.abs(dev)):.3f} kT"


def test_mbar_matches_wham_2d():
    pytest.importorskip("pymbar")
    cvs, wid, centers, kappa = _gaussian_well_2d()
    mb = reweight.mbar_weights(cvs, wid, centers, kappa, 1.0, (None, None))
    wh = reweight.wham_weights(cvs, wid, centers, kappa, 1.0, (None, None), n_bins=40)
    Fm, _, occ = _reweighted_pmf_2d(cvs, np.asarray(mb.sample_weights))
    Fw, _, _ = _reweighted_pmf_2d(cvs, np.asarray(wh.sample_weights))
    d = (Fm - Fw)[occ]
    d -= d.mean()
    assert np.nanmax(np.abs(d)) < 0.4, "2D MBAR and WHAM PMFs disagree"


def test_progress_coordinate_2d_projects_onto_AB_axis():
    rng = np.random.default_rng(1)
    # cloud spanning a tilted A->B axis; A near (-1,-1), B near (1,1)
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
        system_name="t",
        cv_names=("a", "b"),
        cv_periodic=(None, None),
        meta={},
    )
    s = progress_coordinate(ds)
    assert s.shape == (len(cvs),)
    assert s[in_A].mean() < s[in_B].mean()  # oriented A -> B
    # progress is the projection onto (1,1)/sqrt2: correlates with x+y
    assert np.corrcoef(s, cvs[:, 0] + cvs[:, 1])[0, 1] > 0.99


def test_toy_loader_2d_bias_branch(tmp_path):
    """A 2D-bias toy us_traj.npz (centers (K,2), k_bias [kx,ky]) loads as n_cv=2."""
    K, n = 3, 200
    rng = np.random.default_rng(2)
    cx = np.array([-1.863, 0.0, 1.887])
    cy = np.array([0.02, 1.0, 0.02])
    samples = np.stack(
        [
            np.stack([cx[k] + 0.1 * rng.normal(size=n), cy[k] + 0.1 * rng.normal(size=n)], -1)
            for k in range(K)
        ]
    )
    d = tmp_path / "wolfe_quapp_stiff"
    d.mkdir()
    np.savez(
        d / "us_traj.npz",
        samples=samples,
        centers=np.stack([cx, cy], axis=1),
        k_bias=np.array([30.0, 50.0]),
        beta=1.0,
        dt=0.01,
        state_A=np.array([-1.863, 0.022]),
        state_B=np.array([1.887, 0.022]),
        state_radius=0.4,
    )
    ds, _ = load_toy_dataset("wolfe_quapp_stiff", data_root=tmp_path)
    assert ds.n_cv == 2
    assert np.asarray(ds.cvs).shape == (K * n, 2)
    assert np.asarray(ds.window_centers).shape == (K, 2)
    assert np.asarray(ds.window_kappa).shape == (K, 2)
    np.testing.assert_allclose(np.asarray(ds.window_kappa)[0], [30.0, 50.0])
