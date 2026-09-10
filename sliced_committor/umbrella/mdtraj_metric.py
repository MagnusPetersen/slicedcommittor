"""mdtraj-side inputs for the feature-space metric.

:func:`sliced_committor.sincos_pullback_metric` is free of mdtraj: it takes
coordinates, an ``(n, 4)`` index array of the torsions' atom quadruples and
optional per-atom weights. This module supplies the quadruples and the masses
from a trajectory.

The index array is the piece a featuriser throws away: ``md.compute_phi`` and
friends return the atom quadruples with the angles, and only the angles are
kept, so the atoms that define the features (and hence the Jacobian) are lost.
:func:`dihedral_index_arrays` reconstructs them in the SAME concatenation
order the featuriser used, which is load-bearing: the metric must be expressed
in the basis the features are in.
"""

from __future__ import annotations

import numpy as np

#: the seven-family concatenation order of the AIB9 and villin featurisation
TORSION_KINDS_ALL = ("phi", "psi", "omega", "chi1", "chi2", "chi3", "chi4")
#: the five-family concatenation order of the chignolin featurisation
TORSION_KINDS_FEATURIZE = ("phi", "psi", "omega", "chi1", "chi2")


def _compute_fns():
    import mdtraj as md

    return {
        "phi": md.compute_phi,
        "psi": md.compute_psi,
        "omega": md.compute_omega,
        "chi1": md.compute_chi1,
        "chi2": md.compute_chi2,
        "chi3": md.compute_chi3,
        "chi4": md.compute_chi4,
    }


def dihedral_index_arrays(traj, kinds=TORSION_KINDS_FEATURIZE):
    """``(n_total, 4)`` atom quadruples in the featuriser's concatenation order.

    Args:
        traj: mdtraj Trajectory (one frame is enough; only the topology is used).
        kinds: torsion families, in the order they are concatenated:
            :data:`TORSION_KINDS_ALL` (seven families) or
            :data:`TORSION_KINDS_FEATURIZE` (five).

    Returns:
        ``(indices (n_total, 4) int64, counts dict)``. Families that yield no
        torsion are skipped, exactly as a featuriser skips them.
    """
    fns = _compute_fns()
    single = traj[0] if traj.n_frames > 1 else traj
    cols, counts = [], {}
    for k in kinds:
        if k not in fns:
            raise ValueError(f"unknown dihedral kind {k!r}; choose from {sorted(fns)}")
        idx, _ = fns[k](single)
        idx = np.asarray(idx, dtype=np.int64).reshape(-1, 4)
        if idx.shape[0]:
            cols.append(idx)
            counts[k] = int(idx.shape[0])
    if not cols:
        raise ValueError(f"no torsions of kinds {tuple(kinds)} found in this topology.")
    return np.concatenate(cols, axis=0), counts


def atom_masses(traj, atom_indices=None) -> np.ndarray:
    """Per-atom masses in amu, from the mdtraj topology."""
    top = traj.topology
    idx = range(top.n_atoms) if atom_indices is None else np.asarray(atom_indices).ravel()
    m = np.array([top.atom(int(i)).element.mass for i in idx], dtype=np.float64)
    if not np.all(np.isfinite(m)) or m.min() <= 0:
        raise ValueError("topology yielded a non-positive or non-finite atomic mass.")
    return m


def assert_heavy_atom_quads(masses, quad_index, min_mass: float = 5.0) -> None:
    """Guard: proper dihedrals must not involve hydrogens.

    phi/psi/omega/chi1-4 are defined on backbone and sidechain heavy atoms only
    (mass 12-32), which is why a per-atom mass weighting of the metric changes
    it by only a few percent on the peptides. A hydrogen appearing here means
    the index array does not describe the torsions the features were built from.
    """
    m = np.asarray(masses, dtype=np.float64)
    touched = np.unique(np.asarray(quad_index).ravel())
    worst = float(m[touched].min())
    if worst < float(min_mass):
        raise ValueError(
            f"lightest atom in a dihedral quadruple is {worst:.3f} amu (< {min_mass}); "
            "proper torsions should involve heavy atoms only. Check the index array."
        )
