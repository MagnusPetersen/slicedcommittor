"""Sweep orchestrator: committors and the three rate estimators over a grid.

The grid is featurization x direction-sampling. For each cell a committor is fit
(auto-tuned per feature space by default: EBMC vs PESB ranked by the label-free
Dirichlet energy) and every estimator (TPT / Szabo, see :mod:`committor_rates`)
is read off a local diffusion in the space it needs (CV-space for TPT, q-space
for Szabo). ``diffusion_mode`` sets the diffusion: ``"bins_hummer"`` (default,
per-window Hummer on an n_diff_bins committor-bin rate grid), ``"bins"`` (per-bin
Kramers-Moyal), or ``"per_window"`` (Hummer on a fine grid), using a matched
local flux. Each cell also yields the diffusion/flux/PMF profiles and a
``(CV, q_sliced)`` scatter sample for the diagnostic grid. The committor-free
Kramers rate (PMF + CV-space D) is added once per system.

Modes: ``"fast"`` (a small featurization x direction subset) and ``"exhaustive"``
(all featurizations the trial used x all direction-sampling methods).
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from ._containers import Reweighting, USDataset
from .committor_rates import fit_and_rate
from .config import SystemConfig, get_config
from .pmf_kramers import pmf_kramers_rate
from .reweight import reweight as run_reweight

logger = logging.getLogger(__name__)

ALL_DIRECTION_MODES = ("uniform", "lda", "pca", "gcpca", "tica_ema", "tica_ema_decomposed")


def _sweep_axes(mode: str, cfg: SystemConfig, has_trajectory: bool) -> dict[str, tuple]:
    """Return the featurization and direction-mode option tuples for the mode."""
    toy = cfg.featurizations == ("identity",)
    traj_feats = tuple(f for f in cfg.featurizations if f not in ("cv_only", "identity"))
    if mode == "fast":
        if toy:
            featurizations = ("identity",)
        elif has_trajectory:
            featurizations = ("cv_only", "dihedrals")
        else:
            featurizations = ("cv_only",)
        return {
            "featurizations": featurizations,
            # LDA is the fast-mode informed default: supervised (uses the A/B
            # labels) and dynamics-free, so it is cheaper than the TICA modes
            # (which need per-window trajectories) for a quick run.
            "direction_modes": ("uniform", "lda"),
        }
    if mode == "exhaustive":
        if toy:
            featurizations = ("identity",)
        elif has_trajectory:
            featurizations = ("cv_only", *traj_feats)
        else:
            featurizations = ("cv_only",)
        return {
            "featurizations": featurizations,
            "direction_modes": ALL_DIRECTION_MODES,
        }
    raise ValueError(f"unknown sweep mode {mode!r}; use 'fast' or 'exhaustive'")


def _compute_features(dataset: USDataset, name: str, cfg: SystemConfig) -> np.ndarray | None:
    """Materialise a named featurization for the dataset (or None on failure)."""
    if name == "identity":
        return np.asarray(dataset.features, dtype=float)
    if name == "cv_only":
        return np.asarray(dataset.cvs, dtype=float)
    traj = dataset.meta.get("trajectory")
    if traj is None:
        logger.warning("featurization %s needs a trajectory; skipping", name)
        return None
    from .featurize import featurize

    params = dict(cfg.featurize_params.get(name, {}))
    try:
        feats = featurize(
            name, traj=traj, cvs=dataset.cvs, params=params, ref=dataset.meta.get("ref_frame")
        )
        return np.asarray(feats, dtype=float)
    except Exception as exc:
        logger.warning("featurization %s failed: %s", name, exc)
        return None


def run_sweep(
    dataset: USDataset,
    *,
    mode: str = "fast",
    n_directions: int = 256,
    seed: int = 0,
    reweighting: Reweighting | None = None,
    reweight_method: str = "auto",
    references: dict[str, dict[str, Any]] | None = None,
    n_bins: int = 200,
    diffusion_mode: str = "bins_hummer",
    n_diff_bins: int = 25,
    committor_sweep_grid: dict | str | None = "auto",
    precomputed_features: dict[str, np.ndarray] | None = None,
) -> dict[str, Any]:
    """Run the featurization x direction sweep and collect the three estimators.

    Returns a result dict with ``methods`` (flat ``feat/dir :: ESTIMATOR`` ->
    rate dict, for the bar chart / tables), ``combos`` (per-cell committor
    diagnostics, profiles, and the CV-vs-q scatter, for the diagnostic grids),
    plus ``reweighting``, ``references``, ``axes`` and ``diagnostics``.
    """
    cfg = dataset.meta.get("config") or get_config(dataset.system_name)
    has_trajectory = dataset.meta.get("trajectory") is not None or not cfg.has_dynamics

    if reweighting is None:
        reweighting = run_reweight(dataset, method=reweight_method, n_bins=min(n_bins, 80))
    weights = np.asarray(reweighting.sample_weights, dtype=float)

    axes = _sweep_axes(mode, cfg, has_trajectory)
    refs = references if references is not None else dict(cfg.reference_rates)

    # Features come either from precomputed arrays (e.g. a one-time trajectory
    # featurization cache, so the sweep avoids re-reading large trajectories) or
    # are materialised on demand from the dataset / trajectory.
    feature_cache: dict[str, np.ndarray] = {}
    if precomputed_features:
        for fname, arr in precomputed_features.items():
            feature_cache[fname] = np.asarray(arr, dtype=float)
        axes = {**axes, "featurizations": tuple(feature_cache)}
    else:
        for fname in axes["featurizations"]:
            feats = _compute_features(dataset, fname, cfg)
            if feats is not None:
                feature_cache[fname] = feats

    methods: dict[str, dict[str, Any]] = {}
    combos: list[dict[str, Any]] = []
    for fname, feats in feature_cache.items():
        for dmode in axes["direction_modes"]:
            label = f"{fname}/{dmode}"
            try:
                res = fit_and_rate(
                    dataset,
                    feats,
                    weights,
                    direction_mode=dmode,
                    weight_solver="ebmc",
                    n_directions=n_directions,
                    has_dynamics=cfg.has_dynamics,
                    n_bins=n_bins,
                    diffusion_mode=diffusion_mode,
                    n_diff_bins=n_diff_bins,
                    solver_kwargs=cfg.solver_kwargs,
                    committor_sweep_grid=committor_sweep_grid,
                    seed=seed,
                )
            except Exception as exc:
                logger.warning("combo %s failed: %s", label, exc)
                methods[f"{label} :: error"] = {"error": str(exc)}
                continue
            for ekey, eval_ in res.get("estimators", {}).items():
                methods[f"{label} :: {ekey}"] = eval_
            combos.append(
                {
                    "featurization": fname,
                    "direction": dmode,
                    "committor": res.get("committor", {}),
                    "scatter": res.get("scatter", {}),
                    "profiles": res.get("profiles", {}),
                }
            )

    # Committor-free PMF-Kramers baseline (once per system): PMF along the CV +
    # the CV-space diffusion (per-window Hummer, or binned Kramers-Moyal).
    binned = diffusion_mode == "bins"
    baseline: dict[str, Any] = {}
    try:
        if cfg.has_dynamics:
            baseline["pmf_kramers"] = {
                k: (float(v) if np.isscalar(v) else v)
                for k, v in pmf_kramers_rate(
                    dataset,
                    weights,
                    diffusion_method="kramers_moyal" if binned else "hummer",
                    n_bins=n_diff_bins if binned else n_bins,
                    per_window=not binned,
                ).items()
            }
        else:
            baseline["pmf_kramers"] = {"note": "needs dynamics; skipped"}
    except Exception as exc:
        logger.warning("pmf_kramers baseline failed: %s", exc)
        baseline["pmf_kramers"] = {"error": str(exc)}

    return {
        "system": dataset.system_name,
        "mode": mode,
        "reweighting": {
            "method": reweighting.method,
            "n_eff": float(reweighting.n_eff),
            "n_frames": int(dataset.n_frames),
            "n_windows": int(dataset.n_windows),
            **{
                k: v
                for k, v in reweighting.meta.items()
                if np.isscalar(v) or isinstance(v, (str, list))
            },
        },
        "methods": methods,
        "baseline": baseline,
        "combos": combos,
        "references": refs,
        "axes": {k: list(v) for k, v in axes.items()},
        "diagnostics": {
            "n_features_by_featurization": {k: int(v.shape[1]) for k, v in feature_cache.items()},
            "n_combos": len(combos),
            "has_dynamics": cfg.has_dynamics,
            "dt": dataset.dt,
            "diffusion_mode": diffusion_mode,
            "n_diff_bins": n_diff_bins if diffusion_mode in ("bins", "bins_hummer") else None,
        },
    }
