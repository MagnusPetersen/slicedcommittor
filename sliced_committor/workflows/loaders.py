"""Dataset loaders: analytic-toy ``.npz`` reuse and flexible US-folder discovery.

Two entry points:

* :func:`load_toy_dataset` reuses the prior pipeline's stored analytic-system
  samples (``us_samples.npz``) plus their exact PDE / MFPT / Kramers reference
  rates. No integrator is run.
* :func:`load_us_dataset` discovers a real umbrella-sampling folder (per-window
  ``COLVAR`` + trajectory + ``plumed.dat`` + topology), reconciles COLVAR vs
  trajectory stride, labels basins from the registry, and returns a
  :class:`USDataset`. The protein trajectories (protein atoms only, optionally
  thinned) are kept on the dataset ``meta`` for the sweep to featurize from.

Discovery is heuristic and leaves the data in place; a sidecar config (passed as
a dict) can override any auto-detected field.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

import numpy as np

from . import bias as bias_mod
from . import colvar as colvar_mod
from ._containers import USDataset
from .config import Region, SystemConfig, compute_basins, get_config, match_config

logger = logging.getLogger(__name__)

# Regenerated analytic-toy umbrella data (fine save interval, diffusion-capable).
# Produced by US_data/toy_us/generate_toy_us.py; lives with the data, not in the
# package. Each system dir holds us_traj.npz + the reference rate JSONs.
#
# Point SLICED_COMMITTOR_TOY_DATA at that directory to use the toy loaders. This
# was an absolute path into the authors' shared filesystem, which is useless to
# anyone else and is not ours to advertise; the data is not distributed.
TOY_DATA_ROOT = Path(os.environ.get("SLICED_COMMITTOR_TOY_DATA", ""))
_TRAJ_EXTS = (".xtc", ".trr", ".dcd", ".h5", ".nc", ".dtr")
_COLVAR_NAMES = ("COLVAR", "colvar.dat", "COLVAR.dat", "colvar")


# ---------------------------------------------------------------------------
# Analytic toys
# ---------------------------------------------------------------------------
def _load_reference_rates(data_dir: Path) -> dict[str, dict[str, Any]]:
    refs: dict[str, dict[str, Any]] = {}
    mapping = {
        "pde_rate.json": ("PDE", "exact committor PDE reactive flux"),
        "mfpt.json": ("MFPT", "direct unbiased mean-first-passage time"),
        "kramers_rate.json": ("Kramers_analytic", "analytic Kramers (1D)"),
    }
    for fname, (label, source) in mapping.items():
        p = data_dir / fname
        if not p.is_file():
            continue
        with p.open() as fh:
            payload = json.load(fh)
        if "rate" in payload:
            refs[label] = {
                "k": float(payload["rate"]),
                "units": "1/time(reduced)",
                "source": source,
            }
    return refs


def load_toy_dataset(
    name: str, *, data_root: str | Path | None = None
) -> tuple[USDataset, dict[str, dict[str, Any]]]:
    """Load an analytic toy system from stored umbrella samples + references.

    Args:
        name: ``"double_well"``, ``"double_well_high"`` or ``"wolfe_quapp"``.
        data_root: override for :data:`TOY_DATA_ROOT`.

    Returns:
        ``(dataset, references)`` where ``references`` maps a label to
        ``{"k", "units", "source"}`` (PDE / MFPT / analytic Kramers).
    """
    cfg = get_config(name)
    root = Path(data_root) if data_root is not None else TOY_DATA_ROOT
    if str(root) in ("", "."):
        raise RuntimeError(
            "The analytic-toy umbrella data is not bundled. Set "
            "SLICED_COMMITTOR_TOY_DATA to its root directory, or pass data_root=."
        )
    data_dir = root / cfg.name
    npz = np.load(data_dir / "us_traj.npz")
    samples = np.asarray(npz["samples"], dtype=float)  # (K, n_per, 2), time-ordered
    centers = np.asarray(npz["centers"], dtype=float)  # (K,) or (K, n_cv) for a 2D-bias tube
    k_bias = np.asarray(npz["k_bias"], dtype=float)
    beta = float(npz["beta"])
    dt = float(npz["dt"])  # save interval -> valid for diffusion estimation
    K, n_per, dim = samples.shape

    features = samples.reshape(K * n_per, dim)  # (N, 2), per-window blocks
    window_ids = np.repeat(np.arange(K), n_per)
    if centers.ndim == 2:
        # 2D "tube" umbrella (e.g. wolfe_quapp_stiff): bias on the first n_cv
        # feature coordinates (x, y), so the orthogonal restraint is reweighted.
        n_cv = centers.shape[1]
        cvs = features[:, :n_cv]
        window_centers = centers
        kap = k_bias if k_bias.ndim else np.full(n_cv, float(k_bias))
        window_kappa = np.broadcast_to(np.asarray(kap, float), (K, n_cv)).copy()
    else:
        # 1D bias on the CV (x).
        cvs = features[:, :1]
        window_centers = centers[:, None]
        window_kappa = np.full((K, 1), float(k_bias))

    in_A, in_B = compute_basins(features, cvs, cfg.region_A, cfg.region_B)
    refs = _load_reference_rates(data_dir)

    dataset = USDataset(
        features=features,
        cvs=cvs,
        window_ids=window_ids,
        window_centers=window_centers,
        window_kappa=window_kappa,
        beta=beta,
        dt=dt,
        in_A=in_A,
        in_B=in_B,
        system_name=cfg.name,
        cv_names=cfg.cv_names,
        cv_periodic=cfg.cv_periodic,
        meta={
            "kind": "toy",
            "source": str(data_dir),
            "featurizations": ("identity",),
            "trajectory": None,
        },
    )
    return dataset, refs


# ---------------------------------------------------------------------------
# Real umbrella-sampling folders
# ---------------------------------------------------------------------------
def _find_window_dirs(folder: Path) -> list[Path]:
    base = folder / "windows" if (folder / "windows").is_dir() else folder
    dirs = [
        d
        for d in sorted(base.iterdir())
        if d.is_dir()
        and (d.name.startswith("window") or d.name.startswith("W_") or d.name.startswith("w"))
    ]
    # Keep only window dirs that actually contain a COLVAR file.
    return [d for d in dirs if _find_colvar(d) is not None]


def _find_colvar(window_dir: Path) -> Path | None:
    for name in _COLVAR_NAMES:
        p = window_dir / name
        if p.is_file():
            return p
    hits = sorted(window_dir.glob("*olvar*"))
    return hits[0] if hits else None


def _find_trajectory(window_dir: Path) -> Path | None:
    preferred = ["prod.xtc", "prod.trr", "prod.dcd", "prod.h5"]
    for name in preferred:
        if (window_dir / name).is_file():
            return window_dir / name
    for ext in _TRAJ_EXTS:
        hits = sorted(window_dir.glob(f"*{ext}"))
        if hits:
            return hits[0]
    return None


def _topology_candidates(window_dir: Path, folder: Path, cfg: SystemConfig | None) -> list[Path]:
    """Ordered topology candidates (cfg globs first, then common defaults)."""
    globs = list(cfg.topology_globs) if cfg else []
    globs += ["*.pdb", "*.gro", "*.prmtop", "*.psf", "conf.gro", "topol.top"]
    out: list[Path] = []
    for g in globs:
        for base in (window_dir, folder):
            for h in sorted(base.glob(g)):
                if h.is_file() and h not in out:
                    out.append(h)
    return out


def _resolve_topology(candidates: list[Path], traj_path: Path | None) -> Path | None:
    """Pick the first candidate whose atom count matches the trajectory.

    A protein-only topology (e.g. ``proc.pdb``) often sits next to a full-system
    solvated trajectory (``prod.xtc``); using it would silently load the wrong
    atoms. We validate by loading a single trajectory frame against each candidate
    and keeping the first that succeeds; if none do (or no trajectory), fall back
    to the first candidate.
    """
    if not candidates:
        return None
    if traj_path is None:
        return candidates[0]
    # one representative per filename (US windows share a topology), so we do not
    # retry 384 identical proc.pdb before reaching the matching full-system .gro
    seen, unique = set(), []
    for c in candidates:
        if c.name not in seen:
            seen.add(c.name)
            unique.append(c)
    import mdtraj as md

    for cand in unique:
        try:
            md.load_frame(str(traj_path), 0, top=str(cand))
            return cand
        except Exception:  # atom-count / format mismatch
            continue
    return unique[0]


def _cv_columns(data: colvar_mod.ColvarData, cfg: SystemConfig | None) -> tuple[int, ...]:
    """Indices (into non-time fields) of the bias CV columns."""
    non_time = data.non_time_fields()
    if cfg is not None and cfg.colvar_cv_columns is not None:
        return cfg.colvar_cv_columns
    # Auto: every non-time field that is not a bias term.
    cols = [i for i, f in enumerate(non_time) if "bias" not in f.lower()]
    return tuple(cols) if cols else (0,)


def load_us_dataset(
    folder: str | Path,
    *,
    system: str | None = None,
    sidecar: dict | None = None,
    atom_selection: str = "protein",
    traj_stride: int = 1,
    max_frames: int | None = None,
    window_fraction: float | None = None,
    seed: int = 0,
) -> USDataset:
    """Discover and load a real umbrella-sampling folder into a :class:`USDataset`.

    Args:
        folder: the US campaign root (contains ``windows/`` or window dirs).
        system: registry name; if ``None`` it is matched from the folder name.
        sidecar: optional dict overriding fields (``kappa``, ``cv_columns``,
            ``basins`` regions, ``beta``, ``topology``).
        atom_selection: mdtraj selection of atoms to load (default ``"protein"``)
            to keep memory bounded; CVs come from COLVAR regardless.
        traj_stride: read every ``traj_stride``-th trajectory frame.
        max_frames: cap total frames (stratified per window) after loading.
        window_fraction: if given (``0 < f < 1``), keep only the FIRST fraction
            ``f`` of each window's time-ordered frames (contiguous from t=0),
            BEFORE reweighting -- i.e. emulate a shorter simulation of every
            window (e.g. ``0.1`` = the first 10% of each run). Distinct from
            ``traj_stride`` / ``max_frames``, which thin uniformly over the FULL
            time span (fewer frames, same wall-clock coverage); truncation
            instead shortens the wall-clock coverage. ``None`` or ``f >= 1``
            keeps every frame. Use it to probe how the rate estimators degrade
            with less sampling.
        seed: subsampling seed.

    Returns:
        a :class:`USDataset` with the (protein-atom, thinned) trajectory and a
        reference frame on ``meta`` for the sweep to featurize from.
    """
    import mdtraj as md

    from . import trajectory as traj_mod  # local import (mdtraj heavy)

    # Resolve to a canonical absolute path so the folder name is recoverable for
    # the registry heuristic even when invoked as "." / "./" from inside the folder
    # (Path("./").name is "", which matches nothing).
    folder = Path(folder).resolve()
    cfg = get_config(system) if system else match_config(folder.name)
    if cfg is None and sidecar is None:
        raise ValueError(
            f"could not match a system config for {folder.name!r}. Pass --system NAME "
            "(a registered system), or a --sidecar YAML supplying the basin/CV "
            "definitions discovery cannot infer: region_A and region_B (dicts of "
            "{kind, on, lo, hi} for axis-aligned boxes or {center, radius} for balls), "
            "cv_columns (list of COLVAR column indices), kappa (float or per-CV list), "
            "beta (float = 1/kT), and topology (path). The windows, COLVAR, trajectory "
            "and restraint are auto-discovered; run with -v to see what WAS found."
        )
    if cfg is not None:
        logger.info(
            "config: matched %r (%s)",
            cfg.name,
            "--system override" if system else "folder-name registry match",
        )
    else:
        logger.info("config: no registry match for %r; using --sidecar overrides", folder.name)
    sidecar = sidecar or {}

    window_dirs = _find_window_dirs(folder)
    if not window_dirs:
        raise ValueError(f"no umbrella windows with COLVAR found under {folder}")
    _names = ", ".join(d.name for d in window_dirs[:3]) + (", ..." if len(window_dirs) > 3 else "")
    logger.info("discovered %d umbrella windows under %s (%s)", len(window_dirs), folder, _names)

    # Resolve a topology once (shared atom ordering is assumed across windows),
    # preferring one whose atom count matches the trajectory (proc.pdb is often
    # protein-only while prod.xtc is the full solvated system).
    if "topology" in sidecar:
        topo_path = Path(sidecar["topology"])
    else:
        topo_path = _resolve_topology(
            _topology_candidates(window_dirs[0], folder, cfg), _find_trajectory(window_dirs[0])
        )
    if topo_path is None:
        raise ValueError(
            f"no topology (.pdb/.gro/.prmtop/.psf) found near {window_dirs[0]}; "
            "pass sidecar={'topology': ...}."
        )
    top = md.load_topology(str(topo_path))
    logger.info("topology: %s (%d atoms)", topo_path, top.n_atoms)
    atom_indices = top.select(atom_selection)
    if atom_indices.size == 0:
        logger.warning("atom_selection %r matched no atoms; loading all atoms", atom_selection)
        atom_indices = None
    else:
        logger.info("atom_selection %r -> %d atoms", atom_selection, atom_indices.size)

    cvs_parts: list[np.ndarray] = []
    wid_parts: list[np.ndarray] = []
    centers: list[np.ndarray] = []
    kappa: list[np.ndarray] = []
    subtrajs = []
    dts: list[float] = []
    n_cv_expected = None

    for wid, wdir in enumerate(window_dirs):
        cpath = _find_colvar(wdir)
        cdata = colvar_mod.read_colvar(cpath)
        cv_cols = (
            _cv_columns(cdata, cfg) if "cv_columns" not in sidecar else tuple(sidecar["cv_columns"])
        )
        non_time = cdata.non_time_fields()
        cv_full = np.stack([cdata.column(non_time[i]) for i in cv_cols], axis=1)  # (T_c, n_cv)
        ctime = cdata.time
        if n_cv_expected is None:
            n_cv_expected = cv_full.shape[1]

        restraint = bias_mod.parse_restraint(wdir / "plumed.dat")
        if restraint is not None:
            at = np.asarray(restraint.at, dtype=float)
            kp = np.asarray(restraint.kappa, dtype=float)
        else:
            at = bias_mod.center_from_window_name(wdir.name)
            kp = np.asarray(sidecar.get("kappa", []), dtype=float)
            if at is None or kp.size == 0:
                raise ValueError(
                    f"window {wdir.name}: no plumed.dat RESTRAINT and no name-decodable "
                    "center / sidecar kappa. Provide sidecar={'kappa': [...]}."
                )
        if kp.size == 1 and at.size > 1:
            kp = np.full(at.size, kp[0])
        _restr_src = "plumed.dat" if restraint is not None else "window-name/sidecar"
        if wid == 0:
            logger.info(
                "window[0] %s: COLVAR=%s, cv_cols=%s (n_cv=%d); restraint(%s) center=%s kappa=%s",
                wdir.name,
                cpath.name,
                list(cv_cols),
                n_cv_expected,
                _restr_src,
                np.round(at, 3).tolist(),
                np.round(kp, 1).tolist(),
            )
        else:
            logger.debug(
                "window %s: center=%s kappa=%s (%s)",
                wdir.name,
                np.round(at, 3).tolist(),
                np.round(kp, 1).tolist(),
                _restr_src,
            )

        traj_path = _find_trajectory(wdir)
        if traj_path is None:
            logger.warning("window %s has no trajectory; using CV-only frames", wdir.name)
            ci = np.arange(ctime.size)
            ti = None
            sub = None
        else:
            traj = md.load(
                str(traj_path), top=str(topo_path), stride=traj_stride, atom_indices=atom_indices
            )
            ci, ti = traj_mod.align_colvar_traj(ctime, getattr(traj, "time", None), traj.n_frames)
            sub = traj[ti] if ti is not None and len(ti) else traj[: len(ci)]
            if wid == 0:
                logger.info(
                    "window[0] trajectory: %s (stride=%d) -> %d frames aligned with COLVAR",
                    traj_path.name,
                    traj_stride,
                    len(ci),
                )

        # Emulate a shorter simulation: keep only the FIRST `window_fraction` of
        # this window's time-ordered frames (contiguous from t=0), BEFORE any
        # reweighting. `ci` (kept COLVAR indices) and `sub` (aligned subtraj) are
        # equal-length and time-ordered, so a head slice of both shortens the run
        # while preserving the frame spacing the diffusion estimator assumes.
        if window_fraction is not None and 0.0 < window_fraction < 1.0:
            n_keep = max(2, int(round(window_fraction * ci.size)))
            if n_keep < ci.size:
                if wid == 0:
                    logger.info(
                        "window truncation: window_fraction=%.4g -> first %d/%d frames "
                        "per window (emulating a shorter simulation)",
                        window_fraction,
                        n_keep,
                        ci.size,
                    )
                ci = ci[:n_keep]
                if sub is not None:
                    sub = sub[:n_keep]

        cvs_parts.append(cv_full[ci])
        wid_parts.append(np.full(ci.size, wid))
        centers.append(at)
        kappa.append(kp)
        if sub is not None:
            subtrajs.append(sub)
        # dt for diffusion = spacing of the frames we actually KEEP (after the
        # traj_stride load + COLVAR/traj reconciliation), NOT the raw COLVAR
        # cadence. Using the raw cadence with traj_stride>1 would inflate D_q.
        kept_t = ctime[ci]
        if kept_t.size > 1:
            dts.append(float(np.median(np.diff(kept_t))))

    cvs = np.concatenate(cvs_parts, axis=0)
    window_ids = np.concatenate(wid_parts)
    window_centers = np.stack(centers, axis=0)
    window_kappa = np.stack(kappa, axis=0)
    dt = float(np.median(dts)) if dts else 1.0

    trajectory = None
    if subtrajs and len(subtrajs) == len(window_dirs):
        trajectory = subtrajs[0].join(subtrajs[1:]) if len(subtrajs) > 1 else subtrajs[0]

    # Optional frame cap: uniform per-window thinning (preserves the constant
    # within-window frame spacing the diffusion estimator assumes) and scale dt.
    if max_frames is not None and cvs.shape[0] > max_frames:
        stride = int(np.ceil(cvs.shape[0] / max_frames))
        keep = np.sort(
            np.concatenate(
                [np.nonzero(window_ids == w)[0][::stride] for w in np.unique(window_ids)]
            )
        )
        cvs = cvs[keep]
        window_ids = window_ids[keep]
        if trajectory is not None:
            trajectory = trajectory[keep]
        dt *= stride

    features = cvs.copy()  # base features = cv_only; sweep recomputes others

    def _as_region(r):
        if r is None or isinstance(r, Region):
            return r
        return Region(**r)  # coerce a sidecar/YAML dict into a Region

    region_A = _as_region(sidecar.get("region_A", cfg.region_A if cfg else None))
    region_B = _as_region(sidecar.get("region_B", cfg.region_B if cfg else None))
    if region_A is None or region_B is None:
        raise ValueError("no basin regions: provide a known system or sidecar region_A/region_B.")
    in_A, in_B = compute_basins(features, cvs, region_A, region_B)

    from .config import beta_from_temperature

    if "beta" in sidecar:
        beta_val = float(sidecar["beta"])
    elif cfg is not None:
        beta_val = beta_from_temperature(cfg.temperature_K)
    else:
        beta_val = 1.0

    dataset = USDataset(
        features=features,
        cvs=cvs,
        window_ids=window_ids,
        window_centers=window_centers,
        window_kappa=window_kappa,
        beta=beta_val,
        dt=dt,
        in_A=in_A,
        in_B=in_B,
        system_name=cfg.name if cfg else folder.name,
        cv_names=cfg.cv_names if cfg else tuple(f"cv{i}" for i in range(cvs.shape[1])),
        cv_periodic=cfg.cv_periodic if cfg else tuple(None for _ in range(cvs.shape[1])),
        meta={
            "kind": "us",
            "source": str(folder),
            "topology": str(topo_path),
            "atom_selection": atom_selection,
            "n_windows": int(window_centers.shape[0]),
            "trajectory": trajectory,
            "ref_frame": trajectory[0] if trajectory is not None else None,
            "featurizations": cfg.featurizations if cfg else ("cv_only",),
            "featurize_params": dict(cfg.featurize_params) if cfg else {},
            "config": cfg,
        },
    )
    logger.info(
        "loaded %s: %d frames, %d windows, n_cv=%d, dt=%g, beta=%g",
        dataset.system_name,
        dataset.n_frames,
        dataset.n_windows,
        dataset.n_cv,
        dt,
        beta_val,
    )
    return dataset
