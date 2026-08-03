"""Flexible US-folder discovery + loading on a synthetic in-memory campaign."""

import numpy as np
import pytest

from sliced_committor.workflows.config import Region

pytest.importorskip("mdtraj")


def test_load_synthetic_us_folder(synthetic_us_folder):
    from sliced_committor.workflows.loaders import load_us_dataset

    sidecar = {
        "region_A": {"kind": "box", "on": "cvs", "lo": [-1.0], "hi": [0.4], "dims": [0]},
        "region_B": {"kind": "box", "on": "cvs", "lo": [0.6], "hi": [2.0], "dims": [0]},
        "beta": 1.0,
        "cv_columns": (0,),
    }
    ds = load_us_dataset(synthetic_us_folder, system=None, sidecar=sidecar, atom_selection="all")
    assert ds.n_windows == 2
    assert ds.n_cv == 1
    assert ds.n_frames > 0
    # window 0 centered at q=0.2 -> A, window 1 at q=0.8 -> B
    assert int(ds.in_A.sum()) > 0
    assert int(ds.in_B.sum()) > 0
    assert ds.meta["trajectory"] is not None
    # bias centers recovered from plumed.dat
    np.testing.assert_allclose(sorted(ds.window_centers[:, 0]), [0.2, 0.8], atol=1e-6)


def test_dot_path_resolves_to_registry_name(tmp_path, monkeypatch):
    """``load_us_dataset(".")`` from INSIDE a folder must match the registry by the
    RESOLVED folder name, not the empty string ``Path(".").name`` returns.

    Regression: the bare ``./`` invocation previously failed with "could not match
    a system config for ''". With the fix the resolved name ("us_chignolin_gmx" ->
    chignolin) matches, so the run gets PAST config-matching and only then fails on
    the (here intentionally absent) windows.
    """
    from sliced_committor.workflows.loaders import load_us_dataset

    folder = tmp_path / "us_chignolin_gmx"  # a registry alias
    folder.mkdir()
    monkeypatch.chdir(folder)
    with pytest.raises(ValueError, match="no umbrella windows"):
        load_us_dataset(".", atom_selection="all")


def test_loader_then_reweight(synthetic_us_folder):
    from sliced_committor.workflows.loaders import load_us_dataset
    from sliced_committor.workflows.reweight import reweight

    sidecar = {
        "region_A": Region("box", "cvs", lo=(-1.0,), hi=(0.4,), dims=(0,)),
        "region_B": Region("box", "cvs", lo=(0.6,), hi=(2.0,), dims=(0,)),
        "beta": 1.0,
    }
    ds = load_us_dataset(synthetic_us_folder, sidecar=sidecar, atom_selection="all")
    rw = reweight(ds, method="wham", n_bins=20)
    assert abs(rw.sample_weights.sum() - 1.0) < 1e-9
    assert rw.sample_weights.shape[0] == ds.n_frames
