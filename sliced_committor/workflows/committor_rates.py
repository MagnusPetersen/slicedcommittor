"""Fit one sliced committor and read off the rate-estimator matrix.

For a given (featurization, direction-sampling) combination this fits the
committor once (reusing :func:`sliced_committor.fit_committor`) and reads the
rates off a local diffusion + matched local flux, never a barrier-band median
broadcast across the coordinate (as an earlier prototype did).

``diffusion_mode`` decouples the rate-integral GRID from the diffusion ESTIMATOR:
``"bins_hummer"`` (DEFAULT) integrates on ``n_diff_bins`` evenly-spaced committor
bins but takes the per-window **Hummer** Var/tau_int diffusion interpolated onto
that grid (Hummer is rate-relevant; resampled, not re-estimated per bin);
``"bins"`` uses the per-bin drift-corrected **Kramers-Moyal** diffusion (the only
genuine per-bin estimator, since Hummer needs a confined per-window segment);
``"per_window"`` keeps the Hummer D per window on a fine density grid. Either way
the SAME reactive current is read off several COORDINATE-INVARIANT ways on the
committor coordinate (``D_q * pi``), nothing here re-implements committor or rate
math:

* **TPT flux plateau, q-space D** -- committor-coordinate flux ``D_q * pi``
  plateau (reusing :func:`sliced_committor.committor_rate`, reduction "plateau").
* **Szabo MFPT, q-space D** -- the 1D-Smoluchowski mean-first-passage rate
  ``k = 1/int dq/(D_q pi)`` (reduction "harmonic").
* **BS local, q-space D** -- the Berezhkovskii-Szabo LOCAL flux ``D_q * pi`` read
  at the single ``q=0.5`` iso-committor surface (no q-integration; reduction
  "local").
* **Szabo MFPT / TPT plateau, CV-MAPPED q-space D** (``BS_mfpt_cvmap`` /
  ``TPT_cvmap``) -- the same harmonic / plateau readings but with the committor
  diffusion MAPPED from the clean umbrella-CV diffusion,
  ``D_q(q) = D_s <|grad q|^2>_q / <|grad s|^2>``, via
  :func:`sliced_committor.mapped_committor_diffusion`.

The earlier feature-space **TPT_cv** (scalar CV-space ``D`` times the FEATURE-space
``|grad q|^2``) was removed: it is not coordinate-invariant (it scales as
``1/alpha^2`` under a feature rescaling ``f -> alpha*f``). The CV-mapped flux above
is its coordinate-invariant replacement.

The committor-free **Kramers** rate (PMF + CV-space D) is added once per system
by :func:`sweep.run_sweep`. The committor itself is auto-tuned per feature space
by default (EBMC vs PESB ranked by the label-free Dirichlet energy, reusing
:func:`sliced_committor.sweep_committor`).

It also returns the per-window diffusion / flux / PMF profiles (for plotting over
CV and q) and a subsample of ``(CV, q_sliced)`` pairs (for the featurization x
direction scatter grid).
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from ._containers import USDataset

logger = logging.getLogger(__name__)

DIRECTION_MODES = ("uniform", "lda", "pca", "gcpca", "tica_ema", "tica_ema_decomposed")


def _enable_x64():
    import jax

    jax.config.update("jax_enable_x64", True)


def build_directions(
    dataset: USDataset,
    features: np.ndarray,
    sample_weights: np.ndarray,
    mode: str,
    n_directions: int,
    seed: int,
):
    """Return an ``(M, d)`` direction array for the mode, or ``None`` for uniform.

    ``None`` lets :func:`compute_sliced_committor` sample uniform directions
    itself. PCA/gcPCA go through ``DirectionSamplingConfig`` (returned as a config
    object, not an array); TICA modes return explicit arrays.
    """
    import jax
    from jax import random

    from sliced_committor import (
        DirectionSamplingConfig,
        directions_tica_ema,
        directions_tica_ema_decomposed,
    )

    from ._containers import split_by_window

    key = random.PRNGKey(seed)
    if mode == "uniform":
        return None
    # Direction sampling is a no-op for a 1D feature (e.g. cv_only): there is a
    # single axis up to sign and the committor is sign-invariant, so every informed
    # mode collapses to the trivial axis. Fall back to the uniform single-axis path
    # (the power-spherical mixture the informed modes use requires dim >= 2).
    if np.asarray(features).shape[1] < 2:
        return None
    if mode == "lda":
        # Single supervised discriminant axis. Needs only the A/B labels (no
        # dynamics), which compute_sliced_committor feeds from in_A/in_B at fit
        # time; alpha=0.5 keeps a uniform coverage floor (config default).
        return DirectionSamplingConfig(mode="lda", lda_method="fisher")
    if mode in ("pca", "gcpca"):
        return DirectionSamplingConfig(mode=mode, n_bias_axes=min(4, features.shape[1]))
    # TICA modes must split the SAME feature matrix used for the committor fit
    # (not dataset.features, which may be a different featurization).
    feats_np = np.asarray(features, dtype=float)
    if mode == "tica_ema":
        per_window = [jax.numpy.asarray(w) for w in split_by_window(feats_np, dataset.window_ids)]
        return np.asarray(directions_tica_ema(key, n_directions, per_window))
    if mode == "tica_ema_decomposed":
        per_window = [jax.numpy.asarray(w) for w in split_by_window(feats_np, dataset.window_ids)]
        per_w = [jax.numpy.asarray(w) for w in dataset.per_window_weights(sample_weights)]
        return np.asarray(directions_tica_ema_decomposed(key, n_directions, per_window, per_w))
    raise ValueError(f"unknown direction mode {mode!r}; choose from {DIRECTION_MODES}")


def _finite_list(arr) -> list:
    """Float list with non-finite entries mapped to None (JSON-safe profile)."""
    out = []
    for v in np.asarray(arr).reshape(-1):
        out.append(float(v) if np.isfinite(v) else None)
    return out


def _plateau_marker(est: dict | None) -> dict | None:
    """``{lo, hi, value}`` of the plateau actually taken by a plateau estimator.

    ``value`` is the flux ``nu_R`` read off the ``[lo, hi]`` window (the number
    that becomes the rate). Returns ``None`` if no auto-located plateau is present.
    """
    if not isinstance(est, dict):
        return None
    pl = est.get("plateau")
    nu = est.get("nu_R")
    if pl is None or nu is None or not np.isfinite(nu):
        return None
    try:
        lo, hi = float(pl[0]), float(pl[1])
    except (TypeError, ValueError, IndexError):
        return None
    return {"lo": lo, "hi": hi, "value": float(nu)}


# Default committor auto-tuning grid: rank these boundary-anchored Gram-family
# solvers by the (label-free) Dirichlet energy and keep the best per feature
# space. Both anchor the committor to [0, 1] via the free bias c (EBMC) / its
# enrichment (PESB), so the Dirichlet energy is comparable and the winner is
# never the compressed/flattened full_gram committor. PESB is only chosen when
# its slice enrichment genuinely lowers the energy (the flattening regime).
DEFAULT_COMMITTOR_SWEEP: dict[str, tuple] = {"weights": ("ebmc", "pesb")}


def _hummer_per_window(x: np.ndarray, dt: float) -> float:
    """Within-window Hummer diffusion ``Var(x)/(tau_int dt)`` (one window).

    Mirrors the package :func:`diffusion_coefficient` ``method="hummer"`` path
    (Geyer integrated autocorrelation time), so the per-window value plotted
    matches the value the Szabo/Kramers rate consumes.
    """
    from sliced_committor.rates.quantities import _tau_int_geyer_frames

    var = float(np.var(x))
    if not np.isfinite(var) or var <= 0.0:
        return float("nan")
    D = var / (_tau_int_geyer_frames(x) * dt)
    return D if (np.isfinite(D) and D > 0) else float("nan")


def _hummer_cv_barrier_scalar(qv, cvp, sample_weights, window_ids, dt, min_count=5) -> float:
    """Barrier-band median of the per-window HUMMER CV diffusion ``D_s``.

    The CV-mapped committor diffusion (:func:`sliced_committor.mapped_committor_diffusion`)
    is built on the clean umbrella-CV Hummer measurement REGARDLESS of how the
    committor-space diffusion is binned, so ``D_s`` is always the per-window
    Var/tau_int CV diffusion, median over windows whose mean committor sits in the
    barrier band [0.3, 0.7] (all finite windows if none do).
    """
    wid = np.asarray(window_ids).reshape(-1)
    qv = np.asarray(qv, dtype=float)
    cvp = np.asarray(cvp, dtype=float)
    w = np.asarray(sample_weights, dtype=float)
    Ds, qcs = [], []
    for k in np.unique(wid):
        m = wid == k
        ck = cvp[m]
        if ck.shape[0] < max(int(min_count), 4):
            continue
        D = _hummer_per_window(ck, dt)
        if not np.isfinite(D):
            continue
        wk = w[m]
        Ds.append(D)
        qcs.append(float(np.average(qv[m], weights=wk if wk.sum() > 0 else None)))
    Ds = np.asarray(Ds)
    qcs = np.asarray(qcs)
    if Ds.size == 0:
        return float("nan")
    band = (qcs >= 0.3) & (qcs <= 0.7)
    return float(np.median(Ds[band])) if band.any() else float(np.median(Ds))


def _per_window_quantities(
    qv, cvp, grad_sq, sample_weights, window_ids, dt, *, pi_q_prof, pi_cv_prof, min_count=5
) -> dict[str, np.ndarray]:
    """Per-umbrella-window local Hummer diffusion, density and gradient.

    Each window is ONE data point: its committor- and CV-space Hummer diffusion
    (Var/tau_int), its reweighted population, its mean squared committor gradient,
    and the committor / CV density evaluated at its mean coordinate. These feed
    the matched per-window fluxes (no barrier-band median is ever broadcast across
    windows). Windows are returned sorted by mean committor value so the q-indexed
    arrays form a valid 1D profile.
    """
    from sliced_committor.rates._coordinate import value_at

    wid = np.asarray(window_ids).reshape(-1)
    rows = []
    for k in np.unique(wid):
        m = wid == k
        qk = np.asarray(qv[m], dtype=float)
        ck = np.asarray(cvp[m], dtype=float)
        gk = np.asarray(grad_sq[m], dtype=float)
        wk = np.asarray(sample_weights[m], dtype=float)
        n = int(qk.shape[0])
        if n < max(int(min_count), 4):
            continue
        wsum = float(wk.sum())
        ww = wk if wsum > 0 else None
        rows.append(
            {
                "n": n,
                "wsum": wsum,
                "q_center": float(np.average(qk, weights=ww)),
                "cv_center": float(np.average(ck, weights=ww)),
                "g": float(np.average(gk, weights=ww)),
                "D_q": _hummer_per_window(qk, dt),
                "D_cv": _hummer_per_window(ck, dt),
            }
        )
    if not rows:
        raise RuntimeError("no umbrella window had enough samples for per-window estimates")
    rows.sort(key=lambda r: r["q_center"])
    keys = ("n", "wsum", "q_center", "cv_center", "g", "D_q", "D_cv")
    out = {key: np.asarray([r[key] for r in rows], dtype=float) for key in keys}
    out["n"] = out["n"].astype(np.int64)
    # Committor / CV density evaluated at each window's mean coordinate (the same
    # reweighted density the rate integrals use), for the matched per-window flux.
    out["pi_q"] = np.asarray([value_at(pi_q_prof, c) for c in out["q_center"]], dtype=float)
    out["pi_cv"] = np.asarray([value_at(pi_cv_prof, c) for c in out["cv_center"]], dtype=float)
    return out


def _binned_quantities(
    qv,
    cvp,
    grad_sq,
    sample_weights,
    window_ids,
    dt,
    *,
    n_bins,
    pi_q_prof,
    pi_cv_prof,
    lag=1,
    min_count=5,
) -> dict[str, np.ndarray]:
    """Per-bin local Kramers-Moyal diffusion, density and gradient (bins along q).

    The committor range [0, 1] is split into ``n_bins`` evenly-spaced bins. Within
    each bin the drift-corrected Kramers-Moyal diffusion is estimated from
    within-window consecutive pairs (reusing the package's window-stratified KM
    kernel) -- one D per bin, for both the committor coordinate (``D_q``) and the
    CV (``D_cv``, indexed by the same committor bins so it feeds the TPT flux). The
    Hummer estimator is NOT available here: it needs a confined per-window segment,
    not an arbitrary committor bin, so the binned mode is Kramers-Moyal only. This
    is the per-bin counterpart of :func:`_per_window_quantities`, returning the
    same keys so the downstream estimators / profiles are mode-agnostic.
    """
    from sliced_committor.rates._coordinate import value_at
    from sliced_committor.rates.quantities import _estimate_D_q_stratified

    qv = np.asarray(qv, dtype=float)
    cvp = np.asarray(cvp, dtype=float)
    w = np.asarray(sample_weights, dtype=float)
    wid = np.asarray(window_ids).reshape(-1)
    L = max(int(lag), 1)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    bin_idx = np.clip(np.digitize(np.clip(qv, 0.0, 1.0 - 1e-12), edges) - 1, 0, n_bins - 1)
    # Drift-corrected KM D of q and of the CV, binned by committor, within-window.
    D_q, cts = _estimate_D_q_stratified(qv, bin_idx, wid, L, dt, n_bins, min_count)
    D_cv, _ = _estimate_D_q_stratified(cvp, bin_idx, wid, L, dt, n_bins, min_count)
    sw_bin = np.bincount(bin_idx, weights=w, minlength=n_bins)
    sw_safe = np.where(sw_bin > 0, sw_bin, 1.0)
    g = np.bincount(bin_idx, weights=w * grad_sq, minlength=n_bins) / sw_safe
    cv_center = np.bincount(bin_idx, weights=w * cvp, minlength=n_bins) / sw_safe
    # Empty bins get NaN centers so they drop out of the profiles / interpolation.
    cv_center = np.where(sw_bin > 0, cv_center, np.nan)
    out = {
        "n": np.asarray(cts, dtype=np.int64),
        "q_center": centers,
        "cv_center": cv_center,
        "g": np.where(sw_bin > 0, g, np.nan),
        "D_q": np.asarray(D_q, dtype=float),
        "D_cv": np.asarray(D_cv, dtype=float),
    }
    out["pi_q"] = np.asarray([value_at(pi_q_prof, c) for c in centers], dtype=float)
    out["pi_cv"] = np.asarray(
        [value_at(pi_cv_prof, c) if np.isfinite(c) else np.nan for c in cv_center], dtype=float
    )
    return out


def _value_at_safe(profile, level) -> float:
    """``value_at(profile, level)`` returning NaN instead of raising."""
    from sliced_committor.rates._coordinate import value_at

    try:
        return float(value_at(profile, float(level)))
    except (ValueError, TypeError):
        return float("nan")


def _cv_feature_grad_sq(cvp, features, weights, metric=None) -> float:
    """``<|grad s|^2>`` of the umbrella CV in the feature space (M0 = I).

    Fits ``s ~ a.x + b`` by the MBAR-weighted, lightly-ridged least squares so the
    linear gradient is ``a`` and ``<|grad s|^2> = |a|^2``. Exact for ``cv_only`` /
    toy ``identity`` (where ``s`` IS a feature, recovering ``|a|^2 = 1``); a
    linear-response approximation for richer featurizations (the single scalar that
    :func:`sliced_committor.mapped_committor_diffusion` divides ``D_s`` by).
    """
    X = np.asarray(features, dtype=float)
    s = np.asarray(cvp, dtype=float)
    w = np.asarray(weights, dtype=float)
    w = w / w.sum() if w.sum() > 0 else np.full(s.shape[0], 1.0 / s.shape[0])
    xbar = (w[:, None] * X).sum(axis=0)
    Xc = X - xbar
    sc = s - float((w * s).sum())
    d = Xc.shape[1]
    A = Xc.T @ (w[:, None] * Xc)
    lam = 1e-6 * (float(np.trace(A)) / max(d, 1) + 1e-30)
    coef = np.linalg.solve(A + lam * np.eye(d), Xc.T @ (w * sc))
    if metric is None:
        return float(coef @ coef)
    M = np.asarray(metric, dtype=float)
    return float(coef @ (M * coef if M.ndim == 1 else M @ coef))


def _mapped_committor_diffusion_profile(
    q, feats, cvp, features, weights, *, D_s, n_bins, mapped_fn, metric=None
):
    """Committor-coordinate diffusion ``D_q(q)`` MAPPED from the umbrella-CV ``D_s``.

    Returns a :class:`Profile` (or ``None`` when ``D_s`` / the CV gradient is not
    usable). See :func:`sliced_committor.mapped_committor_diffusion`.

    ``metric`` is applied to BOTH mean-squared gradients: the committor's co-area
    numerator and the CV's denominator. Applying it to one only would silently
    rescale every rate, since ``D_q`` is exactly linear in the numerator and
    inverse-linear in the denominator.
    """
    if not (np.isfinite(D_s) and D_s > 0):
        return None
    try:
        g_s = _cv_feature_grad_sq(cvp, features, weights, metric=metric)
        if not (np.isfinite(g_s) and g_s > 0):
            return None
        return mapped_fn(
            q, feats, D_s=float(D_s), cv_grad_sq=g_s, sample_weights=weights,
            n_bins=n_bins, metric=metric,
        )
    except Exception as exc:
        logger.warning("CV-mapped diffusion profile failed: %s", exc)
        return None


def fit_and_rate(
    dataset: USDataset,
    features: np.ndarray,
    sample_weights: np.ndarray,
    *,
    direction_mode: str = "uniform",
    weight_solver: str = "ebmc",
    n_directions: int = 256,
    has_dynamics: bool = True,
    n_bins: int = 200,
    diffusion_mode: str = "bins_hummer",
    n_diff_bins: int = 25,
    solver_kwargs: dict | None = None,
    weight_kwargs: dict | None = None,
    bridge_metric=None,
    committor_sweep_grid: dict | str | None = "auto",
    committor_select_by: str = "dirichlet",
    scatter_n: int = 4000,
    seed: int = 0,
) -> dict[str, Any]:
    """Fit a committor and return the per-window rate estimators + profiles.

    The rate processing is PER WINDOW: each umbrella window is one data point
    with its own local diffusion and matched local flux (no barrier-band median
    is broadcast across windows). The diffusion is the per-window Hummer estimate
    (Var/tau_int) throughout; the same reactive current is read off several
    COORDINATE-INVARIANT ways on the committor coordinate (all are ``D_q*pi``):

        TPT flux plateau (q-space D)            -- ``TPT_q``
        Szabo 1D-Smoluchowski MFPT (q-space D)  -- ``Szabo_mfpt``
        BS local flux at q*=0.5 (q-space D, no q-integration) -- ``BS_local``
        Szabo MFPT (CV-mapped q-space D)        -- ``BS_mfpt_cvmap``
        TPT flux plateau (CV-mapped q-space D)  -- ``TPT_cvmap``
        BS local flux at q*=0.5 (CV-mapped D, no q-integration) -- ``BS_local_cvmap``

    (the committor-free Kramers rate, PMF + CV-space D, is added once per system
    by :func:`sweep.run_sweep`). The committor itself is auto-tuned per feature
    space by default: ``committor_sweep_grid="auto"`` ranks the boundary-anchored
    Gram solvers (EBMC / PESB) by their label-free Dirichlet energy and keeps the
    best fit.

    Args:
        dataset: the umbrella dataset (window_ids, dt, basins, CVs).
        features: ``(N, d)`` features, row-aligned with ``dataset`` frames.
        sample_weights: ``(N,)`` MBAR/WHAM weights.
        direction_mode: see :data:`DIRECTION_MODES`.
        weight_kwargs: forwarded to the weight solver, e.g.
            ``{"tikhonov": "cv"}`` for the calibration-free ridge.
        weight_solver: committor weight solver used when auto-tuning is disabled
            (``committor_sweep_grid`` falsy); ``"ebmc"`` default.
        n_directions: number of projection directions.
        has_dynamics: whether a real trajectory exists for diffusion estimation.
        n_bins: committor / density bin count for the per-window mode (the
            per-window D is interpolated onto this density grid by the rate integrals).
        diffusion_mode: ``"bins_hummer"`` (default) -- ``n_diff_bins`` committor-bin
            rate grid, per-window Hummer D interpolated onto it; ``"bins"`` --
            per-bin drift-corrected Kramers-Moyal D on the ``n_diff_bins`` grid (the
            only genuine per-bin estimator); ``"per_window"`` -- Hummer D per window
            on the fine ``n_bins`` density grid.
        n_diff_bins: number of evenly-spaced committor bins for the binned modes.
        solver_kwargs: extra committor-solver kwargs (merged over equal-width).
        committor_sweep_grid: ``"auto"`` (default) auto-tunes the committor per
            feature space over :data:`DEFAULT_COMMITTOR_SWEEP` (EBMC vs PESB,
            ranked by Dirichlet energy via :func:`sliced_committor.sweep_committor`,
            keeping the best fit). A ``{fit_committor_setting: [values]}`` dict
            sweeps that grid instead; ``None`` / ``{}`` disables tuning and fits a
            single committor with ``weight_solver``. Keep any custom grid within
            one weight-solver family (Dirichlet energy is only comparable there).
        committor_select_by: sweep selector forwarded to
            :func:`sliced_committor.sweep_committor` (``"dirichlet"`` default).
        scatter_n: number of ``(CV, q)`` pairs to keep for the scatter grid.
        seed: RNG seed.

    Returns:
        dict with ``committor`` diagnostics (incl. ``settings_sweep`` when tuned),
        ``estimators`` (TPT_q / Szabo_mfpt / BS_local on the measured q-space D, plus
        BS_mfpt_cvmap / TPT_cvmap on the CV-mapped D; all coordinate-invariant),
        ``profiles`` (per-window D / flux / PMF over CV and q), and ``scatter``.
    """
    if committor_sweep_grid == "auto":
        committor_sweep_grid = dict(DEFAULT_COMMITTOR_SWEEP)
    _enable_x64()
    import jax
    import jax.numpy as jnp

    from sliced_committor import (
        committor_rate,
        density,
        fit_committor,
        mapped_committor_diffusion,
    )

    from .pmf_kramers import progress_coordinate

    feats = jnp.asarray(np.asarray(features, dtype=float))
    in_A = jnp.asarray(np.asarray(dataset.in_A, dtype=bool))
    in_B = jnp.asarray(np.asarray(dataset.in_B, dtype=bool))
    sw = jnp.asarray(np.asarray(sample_weights, dtype=float))

    if int(in_A.sum()) == 0 or int(in_B.sum()) == 0:
        raise ValueError(
            f"empty basin(s): |A|={int(in_A.sum())}, |B|={int(in_B.sum())}. "
            "Check basin definitions / CV coverage."
        )

    directions = build_directions(
        dataset, features, sample_weights, direction_mode, n_directions, seed
    )
    # equal_width binning is the default: quantile binning puts too few bins in
    # the low-density barrier region, under-resolving the committor exactly where
    # it is steep and inflating the rate (verified ~3x on Wolfe-Quapp).
    base_solver: dict[str, Any] = {
        "n_bins": n_bins,
        "sample_weights": sw,
        "binning_method": "equal_width",
    }
    if solver_kwargs:
        base_solver.update(solver_kwargs)
    solver_kwargs = base_solver
    fit_kwargs: dict[str, Any] = {"weights": weight_solver, "seed": seed}
    if weight_kwargs:
        # Reaches the weight solver, not compute_sliced_committor -- this is how
        # `tikhonov='cv'` (docs/ridge_rule.md) is selected for a rate run.
        fit_kwargs["weight_kwargs"] = dict(weight_kwargs)
    if directions is None:
        fit_kwargs["n_directions"] = n_directions
    elif hasattr(directions, "mode"):  # DirectionSamplingConfig
        fit_kwargs["n_directions"] = n_directions
        solver_kwargs["direction_sampling"] = directions
    else:  # explicit (M, d) array
        directions = jnp.asarray(directions)
        fit_kwargs["n_directions"] = int(directions.shape[0])
        solver_kwargs["directions"] = directions

    sweep_info: dict[str, Any] | None = None
    if committor_sweep_grid:
        # Tune committor settings by the label-free Dirichlet-energy objective and
        # keep the best fit (reusing sliced_committor.sweep_committor). The fixed
        # kwargs are everything not being swept; grid keys override them.
        from sliced_committor import sweep_committor

        fixed = {**fit_kwargs, **solver_kwargs}
        for k in committor_sweep_grid:
            fixed.pop(k, None)
        swept = sweep_committor(
            feats,
            in_A=in_A,
            in_B=in_B,
            grid=committor_sweep_grid,
            select_by=committor_select_by,
            **fixed,
        )
        best = swept.best_fit
        q, result = best.committor, best.result
        sweep_info = {
            "grid": {k: [str(v) for v in vals] for k, vals in committor_sweep_grid.items()},
            "best_config": {k: str(v) for k, v in swept.configs[swept.best_idx].items()},
            "best_dirichlet_energy": float(swept.dirichlet_energies[swept.best_idx]),
            "select_by": committor_select_by
            if isinstance(committor_select_by, str)
            else "callable",
        }
    else:
        q, details = fit_committor(
            feats, in_A=in_A, in_B=in_B, return_details=True, **fit_kwargs, **solver_kwargs
        )
        result = details.result

    cvp = np.asarray(progress_coordinate(dataset))  # (N,) 1D CV progress coordinate
    qv = np.asarray(q(feats))  # (N,) committor at samples

    out: dict[str, Any] = {
        "committor": {
            "n_directions": int(np.asarray(result.directions).shape[0]),
            "valid_fraction": float(np.mean(np.asarray(result.valid_mask))),
            "direction_mode": direction_mode,
            "weight_solver": weight_solver,
            "n_features": int(feats.shape[1]),
        },
    }
    if sweep_info is not None:
        out["committor"]["settings_sweep"] = sweep_info
    # (CV, q) scatter for the featurization x direction diagnostic grid.
    n = qv.shape[0]
    idx = np.linspace(0, n - 1, min(scatter_n, n)).astype(int)
    out["scatter"] = {"cv": cvp[idx].tolist(), "q": qv[idx].tolist()}
    # Full per-sample committor (for downstream display, e.g. colouring a TICA
    # scatter). Kept as an array (not JSON-serialised) on the returned dict.
    out["q_samples"] = np.asarray(qv, dtype=float)

    if not has_dynamics:
        return out  # no trajectory -> no diffusion-based estimators

    wid = np.asarray(dataset.window_ids)
    dt = dataset.dt

    # |grad q|^2 per frame (feature space) -- the geometric factor in the TPT flux.
    grads = jax.vmap(jax.grad(q))(feats)
    grad_sq = np.asarray(jnp.sum(jnp.asarray(grads) ** 2, axis=-1), dtype=float)

    # Three diffusion modes, decoupling the rate-integral GRID from the diffusion
    # ESTIMATOR:
    #   "per_window"  -- Hummer Var/tau_int D per umbrella window; fine density grid.
    #   "bins"        -- n_diff_bins evenly-spaced committor bins, per-bin drift-
    #                    corrected Kramers-Moyal D (KM is the only per-bin estimator;
    #                    Hummer needs a confined per-window segment).
    #   "bins_hummer" -- n_diff_bins committor grid for the rate integrals, but the
    #                    diffusion is the per-window Hummer D interpolated onto that
    #                    grid (Hummer is rate-relevant; resampled, not re-estimated).
    if diffusion_mode not in ("per_window", "bins", "bins_hummer"):
        raise ValueError(
            f"diffusion_mode must be 'per_window', 'bins' or 'bins_hummer'; got {diffusion_mode!r}"
        )
    binned_grid = diffusion_mode in ("bins", "bins_hummer")  # rate-integral grid
    use_km = diffusion_mode == "bins"  # per-bin KM, else per-window Hummer
    rate_nbins = n_diff_bins if binned_grid else n_bins
    diff_method = "kramers_moyal" if use_km else "hummer"

    # Reweighted committor / CV densities (the same pi the rate integrals use),
    # sampled at each window/bin centre for the matched local flux.
    pi_q_prof = density(q, feats, sample_weights=sw, n_bins=rate_nbins)
    pi_cv_prof = density(q, feats, sample_weights=sw, n_bins=rate_nbins, coordinate=cvp)

    # === Local diffusion + matched local flux (ONE point per window OR per bin) ===
    # Each window/bin is a data point with its OWN local D and flux; we never
    # broadcast a barrier-band median D across them (as the earlier trial did).
    # KM is per-bin; Hummer (per_window / bins_hummer) is per-window.
    pw = (
        _binned_quantities(
            qv,
            cvp,
            grad_sq,
            np.asarray(sample_weights, dtype=float),
            wid,
            dt,
            n_bins=n_diff_bins,
            pi_q_prof=pi_q_prof,
            pi_cv_prof=pi_cv_prof,
        )
        if use_km
        else _per_window_quantities(
            qv,
            cvp,
            grad_sq,
            np.asarray(sample_weights, dtype=float),
            wid,
            dt,
            pi_q_prof=pi_q_prof,
            pi_cv_prof=pi_cv_prof,
        )
    )
    # Barrier-band median of the CV diffusion: a scalar reference only (the rate
    # uses the full per-window/per-bin profile, not this number).
    band = (pw["q_center"] >= 0.3) & (pw["q_center"] <= 0.7)
    dcv_band = pw["D_cv"][band & np.isfinite(pw["D_cv"])]
    out["committor"]["D_cv_scalar"] = float(np.median(dcv_band)) if dcv_band.size else float("nan")

    estimators: dict[str, Any] = {}
    # The reactive current read off several COORDINATE-INVARIANT ways on the
    # committor coordinate (D_q*pi), all from the SAME local diffusion (per-window
    # Hummer, or per-bin Kramers-Moyal in "bins" mode): the TPT flux PLATEAU and the
    # Szabo 1D-Smoluchowski MFPT with the directly-measured q-space D_q, the
    # Berezhkovskii-Szabo LOCAL flux at the q=0.5 surface (no q-integration), and --
    # using the CV-MAPPED diffusion (built below where mapped_Dq is constructed) --
    # the MFPT (BS_mfpt_cvmap) and the TPT flux plateau (TPT_cvmap). For the exact
    # committor all agree; their spread is a committor-quality diagnostic.
    # NOTE: the old feature-space "TPT_cv" (a frozen scalar CV-space D times the
    # FEATURE-space |grad q|^2) was dropped because it is NOT coordinate-invariant:
    # under a feature rescaling f -> alpha*f it scales as 1/alpha^2 (the committor is
    # invariant, |grad q|^2 -> |grad q|^2/alpha^2, but the scalar D_cv does not
    # transform). The CV-mapped flux below fixes this -- the configurational scale
    # D0 = D_s/<|grad s|^2> transforms as alpha^2 and cancels the gradient, so
    # D_q = D0*<|grad q|^2> (and hence the rate) is invariant.
    # (1-3) committor-coordinate flux D_q*pi with the local q-diffusion: the flux
    #       PLATEAU (TPT, q-space D), the LOCAL flux at q*=0.5 (Berezhkovskii-Szabo,
    #       no q-integration), and the 1D-Smoluchowski MFPT.
    szabo_common = dict(
        sample_weights=sw,
        window_ids=wid,
        in_A=in_A,
        in_B=in_B,
        n_bins=rate_nbins,
        diffusion_method=diff_method,
    )
    try:
        estimators["TPT_q"] = _clean(
            committor_rate(q, feats, feats, dt=dt, reduction="plateau", at="auto", **szabo_common)
        )
        estimators["BS_local"] = _clean(
            committor_rate(q, feats, feats, dt=dt, reduction="local", at=0.5, **szabo_common)
        )
        estimators["Szabo_mfpt"] = _clean(
            committor_rate(q, feats, feats, dt=dt, reduction="harmonic", **szabo_common)
        )
    except Exception as exc:
        logger.warning("q-space estimators failed: %s", exc)
        estimators["TPT_q"] = {"error": str(exc)}
    # (5) Berezhkovskii-Szabo MFPT on the CV-MAPPED q-space diffusion: instead of
    #     measuring D_q on the (barrier-unstable) committor, map the clean umbrella-CV
    #     diffusion D_s onto the committor under D = D0*I,
    #         D_q(q) = D_s * <|grad q|^2>_q / <|grad s|^2>,
    #     i.e. one configurational scale D0 = D_s/<|grad s|^2> pushed through the
    #     committor gradient geometry (reuses sliced_committor.mapped_committor_diffusion).
    #     <|grad s|^2> is the CV's mean squared gradient in the feature/slicing space
    #     (best-linear-predictor of s on the features; ~1 for cv_only / toy identity).
    # The CV-mapped diffusion ALWAYS uses the clean Hummer CV diffusion as D_s (the
    # note's premise), independent of the q-space binning mode -- not the
    # mode-dependent D_cv_scalar (which is KM in "bins" mode).
    D_s_hummer = _hummer_cv_barrier_scalar(
        qv, cvp, np.asarray(sample_weights, dtype=float), wid, dt
    )
    out["committor"]["D_s_hummer"] = D_s_hummer
    mapped_Dq = _mapped_committor_diffusion_profile(
        q,
        feats,
        cvp,
        np.asarray(features, dtype=float),
        np.asarray(sample_weights, dtype=float),
        D_s=D_s_hummer,
        n_bins=rate_nbins,
        mapped_fn=mapped_committor_diffusion,
        metric=bridge_metric,
    )
    if mapped_Dq is not None:
        try:
            estimators["BS_mfpt_cvmap"] = _clean(
                committor_rate(
                    q,
                    feats,
                    feats,
                    dt=dt,
                    reduction="harmonic",
                    sample_weights=sw,
                    in_A=in_A,
                    in_B=in_B,
                    n_bins=rate_nbins,
                    D_profile=mapped_Dq,
                )
            )
        except Exception as exc:
            logger.warning("CV-mapped BS MFPT failed: %s", exc)
        # The TPT flux PLATEAU on the same CV-mapped diffusion. This replaces the
        # dropped feature-space TPT_cv: it is the coordinate-invariant flux-plateau
        # reading (D_q*pi, CV-mapped D_q), matching BS_mfpt_cvmap's diffusion but
        # using the plateau reduction instead of the harmonic MFPT.
        try:
            estimators["TPT_cvmap"] = _clean(
                committor_rate(
                    q,
                    feats,
                    feats,
                    dt=dt,
                    reduction="plateau",
                    at="auto",
                    sample_weights=sw,
                    in_A=in_A,
                    in_B=in_B,
                    n_bins=rate_nbins,
                    D_profile=mapped_Dq,
                )
            )
        except Exception as exc:
            logger.warning("CV-mapped TPT plateau failed: %s", exc)
        # The Berezhkovskii-Szabo LOCAL flux read at the single q*=0.5 iso-committor
        # surface, on the CV-mapped diffusion (D_q at q~0.5 only; no q-integration).
        # This is the "diffusion that maps to around q=0.5" reading: unlike the
        # harmonic MFPT it is NOT exposed to the poorly-sampled q~0,1 basins/tails,
        # where the committor ansatz is least accurate.
        try:
            estimators["BS_local_cvmap"] = _clean(
                committor_rate(
                    q,
                    feats,
                    feats,
                    dt=dt,
                    reduction="local",
                    at=0.5,
                    sample_weights=sw,
                    in_A=in_A,
                    in_B=in_B,
                    n_bins=rate_nbins,
                    D_profile=mapped_Dq,
                )
            )
        except Exception as exc:
            logger.warning("CV-mapped BS local (q*=0.5) failed: %s", exc)
    # NOTE: the Kramers rate (PMF along the CV + CV-space diffusion) is
    # committor-independent and added ONCE per system as the PMF-Kramers baseline
    # in sweep.run_sweep / pmf_kramers.
    out["estimators"] = estimators

    # === Per-window profiles for plotting: ONE point per window; each flux uses
    # its own matched per-window Hummer diffusion (CV-space and q-space).
    try:
        pmf = -np.log(np.where(pw["pi_cv"] > 0, pw["pi_cv"], np.nan))
        pmf = pmf - np.nanmin(pmf) if np.isfinite(pmf).any() else pmf
        bs = estimators.get("BS_local", {})
        bs_val = bs.get("nu_R") if isinstance(bs, dict) else None
        bs_local = (
            {"q_star": float(bs.get("q_star", 0.5)), "value": float(bs_val)}
            if bs_val is not None and np.isfinite(bs_val) and bs_val > 0
            else None
        )
        out["profiles"] = {
            "cv": {
                "levels": _finite_list(pw["cv_center"]),
                "D_cv": _finite_list(pw["D_cv"]),
                "pmf": _finite_list(pmf),
            },
            "q": {
                "levels": _finite_list(pw["q_center"]),
                "D_q": _finite_list(pw["D_q"]),
                # The CV-MAPPED committor diffusion D_q(q) = D_s <|grad q|^2>_q/<|grad s|^2>,
                # sampled at the same q centres (overlaid on the measured D_q).
                "D_q_mapped": _finite_list(
                    [_value_at_safe(mapped_Dq, c) for c in pw["q_center"]]
                    if mapped_Dq is not None
                    else []
                ),
                "pi": _finite_list(pw["pi_q"]),
                # TPT flux Phi(q) = D_q*pi, both coordinate-invariant: the measured
                # q-space D_q, and the CV-MAPPED D_q (D_s<|grad q|^2>/<|grad s|^2>).
                "flux_tpt_cvmap": _finite_list(
                    np.asarray([_value_at_safe(mapped_Dq, c) for c in pw["q_center"]]) * pw["pi_q"]
                    if mapped_Dq is not None
                    else []
                ),
                "flux_tpt_q": _finite_list(pw["D_q"] * pw["pi_q"]),
                # The (lo, hi, value) plateau taken by each plateau rate, and the
                # BS local point read at the q=0.5 surface.
                "tpt_cvmap_plateau": _plateau_marker(estimators.get("TPT_cvmap")),
                "tpt_q_plateau": _plateau_marker(estimators.get("TPT_q")),
                "bs_local": bs_local,
            },
        }
    except Exception as exc:
        logger.warning("profile computation failed: %s", exc)
        out["profiles"] = {}
    return out


def _clean(rate_dict: dict) -> dict:
    """Convert a rate dict's array/JAX scalars to plain Python floats for JSON."""
    out = {}
    for k, v in rate_dict.items():
        if isinstance(v, (tuple, list)):
            out[k] = [float(x) for x in v]
        elif np.isscalar(v) or (hasattr(v, "ndim") and getattr(v, "ndim", 1) == 0):
            try:
                out[k] = float(v)
            except (TypeError, ValueError):
                out[k] = v
        else:
            out[k] = v
    return out
