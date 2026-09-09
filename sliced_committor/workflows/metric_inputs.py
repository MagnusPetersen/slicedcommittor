"""mdtraj-side inputs for the feature-space metric.

:mod:`sliced_committor.core.metric` is deliberately free of mdtraj and of any
repo-local import: it takes coordinates, an ``(n, 4)`` index array and a per-atom
diffusion shape. This module supplies those three things from a trajectory.

The index array is the piece the featurisers throw away.
:func:`sliced_committor.workflows.featurize.dihedral_features` calls
``md.compute_phi`` and friends and keeps only the angles, so the atom quadruples
that define the features -- and hence the Jacobian -- are lost. Both functions
here reconstruct them in the SAME concatenation order the featurisers use, which
is load-bearing: the metric must be expressed in the basis the features are in.
"""

from __future__ import annotations

import numpy as np

from ..core.metric import M0Kind

#: Concatenation order used by ``src.domains.aib9._TORSION_TYPES`` (AIB9, villin).
TORSION_KINDS_ALL = ("phi", "psi", "omega", "chi1", "chi2", "chi3", "chi4")
#: Concatenation order used by :func:`..workflows.featurize.dihedral_features`.
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
        traj: mdtraj Trajectory (one frame is enough -- only the topology is used).
        kinds: torsion families, in the order they are concatenated. Use
            :data:`TORSION_KINDS_ALL` for the AIB9/villin 7-family convention and
            :data:`TORSION_KINDS_FEATURIZE` for the chignolin 5-family one.

    Returns:
        ``(indices (n_total, 4) int64, counts dict)``. Families that yield no
        torsion are skipped, exactly as the featurisers skip them.
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


def m0_atom_diag(masses, kind: M0Kind | str) -> np.ndarray:
    """Per-atom diffusion shape ``M0``: ones, or ``1/m_a``.

    Returned per ATOM, not repeated three times: the pull-back contracts the
    Cartesian component index before weighting. Only the shape matters, so the
    overall scale is irrelevant.
    """
    kind = M0Kind(kind)
    m = np.asarray(masses, dtype=np.float64)
    if kind is M0Kind.IDENTITY:
        return np.ones_like(m)
    return 1.0 / m


def assert_heavy_atom_quads(masses, quad_index, min_mass: float = 5.0) -> None:
    """Guard: proper dihedrals must not involve hydrogens.

    phi/psi/omega/chi1-4 are defined on backbone and sidechain heavy atoms only
    (mass 12-32), which is exactly why ``M0 = I`` and ``M0 = diag(1/m)`` differ by
    only a few percent on the peptides. A hydrogen appearing here means the index
    array does not describe the torsions the features were built from.
    """
    m = np.asarray(masses, dtype=np.float64)
    touched = np.unique(np.asarray(quad_index).ravel())
    worst = float(m[touched].min())
    if worst < float(min_mass):
        raise ValueError(
            f"lightest atom in a dihedral quadruple is {worst:.3f} amu (< {min_mass}); "
            "proper torsions should involve heavy atoms only. Check the index array."
        )
