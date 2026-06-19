"""Reweighting (WHAM always, MBAR if pymbar present), basins, and containers."""

import numpy as np
import pytest

from sliced_committor.workflows import reweight
from sliced_committor.workflows._containers import USDataset, split_by_window
from sliced_committor.workflows.config import Region, compute_basins, get_config


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
    rw = reweight.wham_weights(cvs, wid, centers, kappa, 1.0, (None,), n_bins=50)
    assert rw.method == "wham"
    assert abs(rw.sample_weights.sum() - 1.0) < 1e-9
    assert rw.n_eff > 0
    assert rw.pmf is not None


def test_mbar_matches_wham():
    pytest.importorskip("pymbar")
    cvs, wid, centers, kappa = _synthetic_windows()
    mb = reweight.mbar_weights(cvs, wid, centers, kappa, 1.0, (None,))
    wh = reweight.wham_weights(cvs, wid, centers, kappa, 1.0, (None,), n_bins=60)
    assert abs(mb.sample_weights.sum() - 1.0) < 1e-9
    fmb = mb.f_k - mb.f_k[0]
    fwh = wh.f_k - wh.f_k[0]
    assert np.max(np.abs(fmb - fwh)) < 0.25  # free energies agree within 0.25 kT


def test_reweight_auto_picks_wham_for_many_windows():
    cvs, wid, centers, kappa = _synthetic_windows()
    ds = USDataset(
        features=cvs,
        cvs=cvs,
        window_ids=wid,
        window_centers=centers,
        window_kappa=kappa,
        beta=1.0,
        dt=1.0,
        in_A=np.zeros(cvs.shape[0], bool),
        in_B=np.zeros(cvs.shape[0], bool),
        system_name="syn",
        cv_names=("x",),
        cv_periodic=(None,),
        meta={},
    )
    rw = reweight.reweight(ds, method="auto", n_bins=40)
    assert rw.method in ("mbar", "wham")  # 9 windows -> MBAR if available


def test_reduced_harmonic_periodic():
    pts = np.array([[np.pi - 0.1]])
    centers = np.array([[-np.pi + 0.1]])
    kappa = np.array([[1.0]])
    # Without periodicity the displacement is ~2pi; with it, ~0.2 (minimum image).
    u_plain = reweight.reduced_harmonic(pts, centers, kappa, 1.0, (None,))
    u_per = reweight.reduced_harmonic(pts, centers, kappa, 1.0, ((-np.pi, np.pi),))
    assert u_per[0, 0] < u_plain[0, 0]
    assert u_per[0, 0] < 0.05


def test_compute_basins_1d_and_2d():
    cfg = get_config("chignolin")
    # chignolin basins are flipped: A = folded (Q > 0.85), B = unfolded (Q < 0.30)
    cvs = np.array([[0.1], [0.9], [0.5]])
    in_A, in_B = compute_basins(np.zeros((3, 1)), cvs, cfg.region_A, cfg.region_B)
    assert list(in_A) == [False, True, False]  # Q=0.9 folded -> A
    assert list(in_B) == [True, False, False]  # Q=0.1 unfolded -> B

    rA = Region("ball", "features", center=(-1.0, 0.0), radius=0.3)
    rB = Region("ball", "features", center=(1.0, 0.0), radius=0.3)
    feats = np.array([[-1.0, 0.0], [1.0, 0.05], [0.0, 0.0]])
    a, b = compute_basins(feats, np.zeros((3, 1)), rA, rB)
    assert list(a) == [True, False, False]
    assert list(b) == [False, True, False]


def test_basins_made_disjoint():
    # Overlapping regions: A must lose the overlap.
    rA = Region("box", "cvs", lo=(0.0,), hi=(1.0,), dims=(0,))
    rB = Region("box", "cvs", lo=(0.5,), hi=(1.0,), dims=(0,))
    cvs = np.array([[0.7]])
    a, b = compute_basins(np.zeros((1, 1)), cvs, rA, rB)
    assert not (a[0] and b[0])
    assert b[0]


def test_split_by_window_and_subsample():
    arr = np.arange(10)
    wid = np.array([0, 0, 0, 1, 1, 1, 1, 2, 2, 2])
    parts = split_by_window(arr, wid)
    assert [len(p) for p in parts] == [3, 4, 3]

    feats = np.arange(20).reshape(10, 2).astype(float)
    ds = USDataset(
        features=feats,
        cvs=feats[:, :1],
        window_ids=wid,
        window_centers=np.zeros((3, 1)),
        window_kappa=np.ones((3, 1)),
        beta=1.0,
        dt=1.0,
        in_A=np.zeros(10, bool),
        in_B=np.zeros(10, bool),
        system_name="s",
        cv_names=("x",),
        cv_periodic=(None,),
        meta={},
    )
    sub, kept = ds.subsample(6, seed=0)
    assert sub.n_frames <= 10 and sub.n_frames >= 3
    assert set(np.unique(sub.window_ids)) == {0, 1, 2}  # every window represented
    assert kept.shape[0] == sub.n_frames
