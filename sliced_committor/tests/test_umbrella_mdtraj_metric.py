"""The mdtraj-side inputs of the feature-space metric (skipped without mdtraj)."""

import numpy as np
import pytest

pytest.importorskip("mdtraj")

from sliced_committor.umbrella import mdtraj_metric as mm


def test_dihedral_index_arrays_follow_the_featuriser_order(synthetic_traj):
    traj = synthetic_traj(n_frames=3, n_res=4)
    quads, counts = mm.dihedral_index_arrays(traj, kinds=("phi", "psi"))
    assert counts == {"phi": 3, "psi": 3} and quads.shape == (6, 4)
    np.testing.assert_array_equal(quads[:3], mm.dihedral_index_arrays(traj, kinds=("phi",))[0])
    with pytest.raises(ValueError, match="unknown dihedral kind"):
        mm.dihedral_index_arrays(traj, kinds=("bogus",))
    with pytest.raises(ValueError, match="no torsions"):
        mm.dihedral_index_arrays(traj, kinds=("chi1",))  # alanine has no chi1


def test_atom_masses_and_the_heavy_atom_guard(synthetic_traj):
    traj = synthetic_traj(n_frames=1, n_res=4)
    quads, _ = mm.dihedral_index_arrays(traj, kinds=("phi", "psi"))
    masses = mm.atom_masses(traj)
    assert masses.shape == (16,) and np.all(masses > 5.0)
    mm.assert_heavy_atom_quads(masses, quads)
    light = masses.copy()
    light[int(quads[0, 0])] = 1.008
    with pytest.raises(ValueError, match="heavy atoms only"):
        mm.assert_heavy_atom_quads(light, quads)
