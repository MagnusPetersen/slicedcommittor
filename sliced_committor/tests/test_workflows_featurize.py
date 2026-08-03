"""Featurization (mdtraj) and stride reconciliation (pure numpy)."""

import numpy as np
import pytest

from sliced_committor.workflows import trajectory as traj_mod


def test_align_same_cadence():
    t = np.arange(10.0)
    ci, ti = traj_mod.align_colvar_traj(t, t, 10)
    np.testing.assert_array_equal(ci, np.arange(10))
    np.testing.assert_array_equal(ti, np.arange(10))


def test_align_traj_coarser():
    colvar_t = np.arange(0, 10, 1.0)  # 1 ps cadence
    traj_t = np.arange(0, 10, 2.0)  # 2 ps cadence (coarser, fewer frames)
    ci, ti = traj_mod.align_colvar_traj(colvar_t, traj_t, traj_t.size)
    assert ci.size == ti.size == traj_t.size
    np.testing.assert_allclose(colvar_t[ci], traj_t, atol=0.5)


def test_align_no_timestamps_falls_back():
    colvar_t = np.arange(20.0)
    ci, ti = traj_mod.align_colvar_traj(colvar_t, None, 10)
    assert ci.size == ti.size == 10


def test_dihedral_features_shape(synthetic_traj):
    traj = synthetic_traj(n_frames=40, n_res=4)
    feats = pytest.importorskip("sliced_committor.workflows.featurize")
    from sliced_committor.workflows.featurize import dihedral_features

    f = dihedral_features(traj, kinds=("phi", "psi"))
    assert f.shape[0] == 40
    assert f.shape[1] % 2 == 0  # sin/cos pairs
    assert np.all(np.abs(f) <= 1.0 + 1e-6)


def test_aligned_cartesian_and_distances(synthetic_traj):
    from sliced_committor.workflows.featurize import (
        aligned_cartesian_features,
        pairwise_distance_features,
    )

    traj = synthetic_traj(n_frames=30, n_res=4)
    cart = aligned_cartesian_features(traj, selection="name CA")
    assert cart.shape == (30, 3 * 4)  # 4 CA atoms
    dist = pairwise_distance_features(traj, selection="name CA", max_pairs=100)
    assert dist.shape == (30, 6)  # C(4,2) = 6 pairs


def test_pairwise_distance_cap(synthetic_traj):
    from sliced_committor.workflows.featurize import pairwise_distance_features

    traj = synthetic_traj(n_frames=10, n_res=4)
    dist = pairwise_distance_features(traj, selection="all", max_pairs=5)
    assert dist.shape == (10, 5)  # capped


def test_correlation_filter():
    from sliced_committor.workflows.featurize import correlation_filter

    rng = np.random.default_rng(0)
    cv = rng.normal(size=500)
    good = cv + 0.1 * rng.normal(size=500)  # correlated
    bad = rng.normal(size=500)  # uncorrelated
    features = np.column_stack([good, bad])
    kept, mask, corr = correlation_filter(features, cv, threshold=0.5)
    assert mask[0] and not mask[1]
    assert kept.shape[1] == 1
