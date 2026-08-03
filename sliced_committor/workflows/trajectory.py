"""Trajectory loading (mdtraj) and COLVAR<->trajectory stride reconciliation.

mdtraj reads every format we need (GROMACS ``.xtc``/``.trr``, OpenMM ``.dcd``/
``.h5``) given a topology (``.pdb``/``.gro``/``.prmtop``/``.psf``). PLUMED writes
COLVAR at its own ``STRIDE`` and the MD engine writes frames at another, so the
two series rarely line up one-to-one. :func:`align_colvar_traj` matches them by
timestamp at the coarser cadence, returning paired index arrays so featurization
and CV values stay in register.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


def _require_mdtraj():
    try:
        import mdtraj
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "mdtraj is required for trajectory loading / featurization. Install "
            "with 'pip install -e .[workflows]'."
        ) from exc
    return mdtraj


def load_trajectory(traj_path: str | Path, top_path: str | Path, *, stride: int = 1):
    """Load a trajectory with mdtraj.

    Args:
        traj_path: trajectory file (.xtc/.trr/.dcd/.h5/...).
        top_path: topology file (.pdb/.gro/.prmtop/.psf/...).
        stride: read every ``stride``-th frame (cheap pre-thinning).

    Returns:
        an ``mdtraj.Trajectory``.
    """
    md = _require_mdtraj()
    return md.load(str(traj_path), top=str(top_path), stride=stride)


def align_colvar_traj(
    colvar_time: np.ndarray,
    traj_time: np.ndarray | None,
    n_traj_frames: int,
    *,
    tol_frac: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Match COLVAR rows to trajectory frames by timestamp at the coarser cadence.

    Both series are assumed monotonically increasing in time. The coarser series
    (larger median step, fewer frames) anchors the alignment; for each anchor
    time the nearest time in the other series within ``tol_frac * max(cadence)``
    is paired. When ``traj_time`` is unavailable (or degenerate), falls back to an
    integer-stride subsample of the finer series onto the coarser length.

    Args:
        colvar_time: ``(T_c,)`` COLVAR times.
        traj_time: ``(T_t,)`` trajectory frame times, or ``None``.
        n_traj_frames: number of trajectory frames (used for the fallback).
        tol_frac: pairing tolerance as a fraction of the coarser cadence.

    Returns:
        ``(colvar_idx, traj_idx)`` equal-length index arrays.
    """
    colvar_time = np.asarray(colvar_time, dtype=float).reshape(-1)
    tc = colvar_time.size

    degenerate = (
        traj_time is None
        or np.asarray(traj_time).size != n_traj_frames
        or n_traj_frames < 2
        or float(np.ptp(np.asarray(traj_time, dtype=float))) <= 0.0
    )
    if degenerate:
        # No usable timestamps: assume both regular, subsample the finer one.
        n = min(tc, n_traj_frames)
        if tc >= n_traj_frames:
            ci = np.linspace(0, tc - 1, n).round().astype(int)
            ti = np.arange(n)
        else:
            ci = np.arange(n)
            ti = np.linspace(0, n_traj_frames - 1, n).round().astype(int)
        logger.warning(
            "align_colvar_traj: no trajectory timestamps; subsampling by length "
            "(colvar=%d, traj=%d -> %d paired frames)",
            tc,
            n_traj_frames,
            n,
        )
        return ci, ti

    traj_time = np.asarray(traj_time, dtype=float).reshape(-1)
    cad_c = float(np.median(np.diff(colvar_time))) if tc > 1 else np.inf
    cad_t = float(np.median(np.diff(traj_time))) if traj_time.size > 1 else np.inf
    tol = tol_frac * max(cad_c, cad_t)

    if cad_t >= cad_c:  # trajectory is coarser -> anchor on it
        nn = np.searchsorted(colvar_time, traj_time)
        nn = np.clip(nn, 1, tc - 1)
        left = colvar_time[nn - 1]
        right = colvar_time[nn]
        choose_left = np.abs(traj_time - left) <= np.abs(traj_time - right)
        ci = np.where(choose_left, nn - 1, nn)
        ti = np.arange(traj_time.size)
        good = np.abs(colvar_time[ci] - traj_time) <= tol
        return ci[good], ti[good]
    # COLVAR is coarser -> anchor on it
    nn = np.searchsorted(traj_time, colvar_time)
    nn = np.clip(nn, 1, traj_time.size - 1)
    left = traj_time[nn - 1]
    right = traj_time[nn]
    choose_left = np.abs(colvar_time - left) <= np.abs(colvar_time - right)
    ti = np.where(choose_left, nn - 1, nn)
    ci = np.arange(tc)
    good = np.abs(traj_time[ti] - colvar_time) <= tol
    return ci[good], ti[good]
