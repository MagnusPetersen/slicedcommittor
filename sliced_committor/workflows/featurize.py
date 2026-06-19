"""Featurization of MD frames into committor-input feature vectors.

Four families, all returning a ``(N, d)`` float array aligned with the input
frames:

* ``cv_only`` - the umbrella bias CV(s) themselves (read from COLVAR; passed in).
* ``dihedrals`` - backbone/sidechain torsions (mdtraj phi/psi/chi1), sin/cos
  encoded so the periodic angle becomes a smooth Euclidean feature.
* ``aligned_cartesian`` - Kabsch-superposed Cartesian coordinates of a selection
  (default CA), flattened.
* ``pairwise_distances`` - inter-atom distances over a selection (capped, with
  deterministic pair subsampling logged rather than silently truncated).

Plus :func:`best_hummer_q` (fraction of native contacts) and
:func:`correlation_filter` (drop features weakly correlated with the CV), both
mirroring the prior pipeline but expressed with numpy / mdtraj only.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


def _sincos(angles: np.ndarray) -> np.ndarray:
    """Encode ``(N, n)`` angles (radians) as ``(N, 2n)`` ``[sin, cos]``."""
    return np.concatenate([np.sin(angles), np.cos(angles)], axis=1)


def dihedral_features(traj, kinds=("phi", "psi")) -> np.ndarray:
    """sin/cos-encoded backbone/sidechain torsions.

    Args:
        traj: mdtraj Trajectory.
        kinds: any of ``"phi"``, ``"psi"``, ``"omega"``, ``"chi1"``, ``"chi2"``.

    Returns:
        ``(N, 2 * n_torsions)`` features (empty-safe: returns ``(N, 0)`` if none).
    """
    import mdtraj as md

    fns = {
        "phi": md.compute_phi,
        "psi": md.compute_psi,
        "omega": md.compute_omega,
        "chi1": md.compute_chi1,
        "chi2": md.compute_chi2,
    }
    cols = []
    for k in kinds:
        if k not in fns:
            raise ValueError(f"unknown dihedral kind {k!r}; choose from {sorted(fns)}")
        _, ang = fns[k](traj)
        if ang.shape[1]:
            cols.append(ang)
    if not cols:
        return np.zeros((traj.n_frames, 0))
    return _sincos(np.concatenate(cols, axis=1))


def aligned_cartesian_features(traj, selection="name CA", ref=None) -> np.ndarray:
    """Kabsch-superposed Cartesian coordinates of a selection, flattened.

    Args:
        traj: mdtraj Trajectory.
        selection: mdtraj atom-selection string.
        ref: reference Trajectory (single frame) to align to; defaults to the
            first frame of ``traj``.

    Returns:
        ``(N, 3 * n_selected)`` features.
    """
    sel = traj.top.select(selection)
    if sel.size == 0:
        raise ValueError(f"selection {selection!r} matched no atoms")
    reference = ref if ref is not None else traj[0]
    aligned = traj.superpose(reference, atom_indices=sel, ref_atom_indices=sel)
    return aligned.xyz[:, sel, :].reshape(traj.n_frames, -1)


def pairwise_distance_features(traj, selection="name CA", *, max_pairs=4000, seed=0) -> np.ndarray:
    """Inter-atom distances over a selection, with capped, logged pair subsampling.

    Args:
        traj: mdtraj Trajectory.
        selection: mdtraj atom-selection string.
        max_pairs: cap on the number of atom pairs; if exceeded, a deterministic
            random subset is used and the reduction is logged (not silent).
        seed: RNG seed for pair subsampling.

    Returns:
        ``(N, n_pairs)`` distance features (nm).
    """
    import mdtraj as md

    sel = traj.top.select(selection)
    if sel.size < 2:
        raise ValueError(f"selection {selection!r} matched <2 atoms")
    i, j = np.triu_indices(sel.size, k=1)
    pairs = np.stack([sel[i], sel[j]], axis=1)
    if pairs.shape[0] > max_pairs:
        rng = np.random.default_rng(seed)
        keep = np.sort(rng.choice(pairs.shape[0], size=max_pairs, replace=False))
        logger.warning(
            "pairwise_distance_features: %d pairs > max_pairs=%d; using a "
            "deterministic random subset of %d pairs (seed=%d).",
            pairs.shape[0],
            max_pairs,
            max_pairs,
            seed,
        )
        pairs = pairs[keep]
    return md.compute_distances(traj, pairs)


def contact_map_features(
    traj, selection="name CA", *, r0_nm=0.8, steep=6, max_pairs=4000, seed=0
) -> np.ndarray:
    """Soft contact map over a selection: c_ij = 1 / (1 + (r_ij / r0)^steep).

    A smooth (differentiable) 0..1 contact indicator per atom pair, the standard
    contact-map featurization. Distinct from :func:`pairwise_distance_features`
    (raw distances): contacts saturate, emphasising formed/broken contacts over
    absolute distances. Pairs are capped + deterministically subsampled (logged).

    Args:
        traj: mdtraj Trajectory.
        selection: mdtraj atom-selection string.
        r0_nm: contact threshold distance (nm).
        steep: switching-function steepness exponent.
        max_pairs: cap on atom pairs (deterministic subsample if exceeded).
        seed: RNG seed for pair subsampling.

    Returns:
        ``(N, n_pairs)`` soft-contact features in [0, 1].
    """
    r = pairwise_distance_features(traj, selection=selection, max_pairs=max_pairs, seed=seed)
    return 1.0 / (1.0 + (r / float(r0_nm)) ** steep)


def best_hummer_q(traj, native_pairs, r0_nm, *, beta_per_nm=50.0, lam=1.8) -> np.ndarray:
    """Best-Hummer fraction of native contacts Q per frame.

    Q = mean over native pairs of ``1 / (1 + exp[beta (r - lam r0)])``.

    Args:
        traj: mdtraj Trajectory.
        native_pairs: ``(P, 2)`` atom-index pairs.
        r0_nm: ``(P,)`` native distances (nm).
        beta_per_nm: switching sharpness (1/nm).
        lam: native-distance tolerance factor.

    Returns:
        ``(N,)`` Q values in [0, 1].
    """
    import mdtraj as md

    pairs = np.asarray(native_pairs)
    r = md.compute_distances(traj, pairs)  # (N, P) nm
    r0 = np.asarray(r0_nm, dtype=float)[None, :]
    s = 1.0 / (1.0 + np.exp(beta_per_nm * (r - lam * r0)))
    return s.mean(axis=1)


def correlation_filter(
    features: np.ndarray, cv: np.ndarray, weights: np.ndarray | None = None, *, threshold=0.1
):
    """Keep feature columns whose |weighted Pearson r| with the CV exceeds a threshold.

    Args:
        features: ``(N, d)`` features.
        cv: ``(N,)`` collective variable (the 1D umbrella CV).
        weights: optional ``(N,)`` MBAR weights (uniform if None).
        threshold: minimum absolute correlation to keep.

    Returns:
        ``(kept_features (N, d'), keep_mask (d,), abs_corr (d,))``.
    """
    features = np.asarray(features, dtype=float)
    cv = np.asarray(cv, dtype=float).reshape(-1)
    n = features.shape[0]
    w = np.ones(n) / n if weights is None else np.asarray(weights, dtype=float)
    w = w / w.sum()

    def wmean(a):
        return np.sum(w[:, None] * a, axis=0) if a.ndim == 2 else np.sum(w * a)

    fmu = wmean(features)
    cmu = wmean(cv)
    fc = features - fmu[None, :]
    cc = cv - cmu
    cov = np.sum(w[:, None] * fc * cc[:, None], axis=0)
    var_f = np.sum(w[:, None] * fc**2, axis=0)
    var_c = np.sum(w * cc**2)
    denom = np.sqrt(np.maximum(var_f * var_c, 1e-30))
    corr = np.abs(cov / denom)
    keep = corr >= threshold
    if not np.any(keep):  # never return an empty feature matrix
        keep = corr >= np.sort(corr)[-1]  # keep the single best
    return features[:, keep], keep, corr


# Featurization names that require an mdtraj trajectory (everything but cv_only /
# identity). The orchestrator skips these when no topology/trajectory is present.
TRAJECTORY_FEATURIZATIONS = (
    "dihedrals",
    "aligned_cartesian",
    "pairwise_distances",
    "distance_matrix",
    "contact_map",
    "best_hummer_q",
)


def featurize(name, *, traj=None, cvs=None, params=None, ref=None) -> np.ndarray:
    """Dispatch to a featurizer by name.

    Args:
        name: ``"cv_only"`` | ``"dihedrals"`` | ``"aligned_cartesian"`` |
            ``"pairwise_distances"`` | ``"best_hummer_q"``.
        traj: mdtraj Trajectory (required for all but ``cv_only``).
        cvs: ``(N, n_cv)`` CV array (required for ``cv_only``).
        params: per-featurizer keyword params.
        ref: reference frame for ``aligned_cartesian``.

    Returns:
        ``(N, d)`` features.
    """
    params = dict(params or {})
    if name == "cv_only":
        if cvs is None:
            raise ValueError("cv_only featurization needs cvs")
        return np.atleast_2d(np.asarray(cvs, dtype=float))
    if traj is None:
        raise ValueError(f"featurization {name!r} requires a trajectory")
    if name == "dihedrals":
        return dihedral_features(traj, kinds=params.get("kinds", ("phi", "psi")))
    if name == "aligned_cartesian":
        return aligned_cartesian_features(
            traj, selection=params.get("selection", "name CA"), ref=ref
        )
    if name in ("pairwise_distances", "distance_matrix"):
        return pairwise_distance_features(
            traj,
            selection=params.get("selection", "name CA"),
            max_pairs=params.get("max_pairs", 4000),
            seed=params.get("seed", 0),
        )
    if name == "contact_map":
        return contact_map_features(
            traj,
            selection=params.get("selection", "name CA"),
            r0_nm=params.get("r0_nm", 0.8),
            steep=params.get("steep", 6),
            max_pairs=params.get("max_pairs", 4000),
            seed=params.get("seed", 0),
        )
    if name == "best_hummer_q":
        return best_hummer_q(
            traj,
            params["native_pairs"],
            params["r0_nm"],
            beta_per_nm=params.get("beta_per_nm", 50.0),
            lam=params.get("lam", 1.8),
        )[:, None]
    raise ValueError(f"unknown featurization {name!r}")
