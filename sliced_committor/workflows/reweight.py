"""Umbrella reweighting: MBAR (pymbar) with a WHAM fallback.

Both estimators turn per-frame CV values plus the harmonic-bias parameters into
per-frame equilibrium weights (summing to 1), per-window free energies, and a
PMF. MBAR is exact but materialises a ``(K, N)`` reduced-potential matrix, which
is infeasible for the 384-window 2D c-Src set (confirmed: MBAR OOMs there). WHAM
is binned and iterative, with cost independent of ``N`` once histogrammed, so it
scales to many windows. :func:`reweight` auto-selects (WHAM when ``K`` is large
or MBAR fails) but the engine can be forced.

The reduced harmonic bias of window ``k`` at point ``x`` is
``0.5 * beta * sum_d kappa[k,d] * dist(x_d, center[k,d])**2`` with minimum-image
distance for periodic CVs (e.g. a torsion). ``beta`` and ``kappa`` carry the
energy units; the library downstream stays beta-invariant.
"""

from __future__ import annotations

import logging
import warnings

import numpy as np
from scipy.special import logsumexp

from ._containers import Reweighting, USDataset

logger = logging.getLogger(__name__)

# Above this many windows MBAR's (K, N) reduced-potential matrix is assumed
# infeasible (c-Src: K=384, N~5e5 -> ~1.6 GB just for u_kn). Auto picks WHAM.
MBAR_MAX_WINDOWS = 128


def _periodic_delta(delta: np.ndarray, periodic) -> np.ndarray:
    """Minimum-image displacement per CV dim given per-dim periods (or None)."""
    delta = np.array(delta, dtype=float, copy=True)
    for d, per in enumerate(periodic):
        if per is None:
            continue
        lo, hi = per
        length = hi - lo
        delta[..., d] -= length * np.round(delta[..., d] / length)
    return delta


def reduced_harmonic(
    points: np.ndarray,
    centers: np.ndarray,
    kappa: np.ndarray,
    beta: float,
    periodic,
) -> np.ndarray:
    """Reduced harmonic bias of every window at every point.

    Args:
        points: ``(P, n_cv)`` CV coordinates.
        centers: ``(K, n_cv)`` window centers.
        kappa: ``(K, n_cv)`` force constants (energy / CV-unit^2).
        beta: inverse temperature ``1/kT`` in matching energy units.
        periodic: length-``n_cv`` sequence of ``(min, max)`` or ``None`` per dim.

    Returns:
        ``(K, P)`` reduced (dimensionless) bias energies.
    """
    points = np.atleast_2d(np.asarray(points, dtype=float))
    centers = np.atleast_2d(np.asarray(centers, dtype=float))
    kappa = np.atleast_2d(np.asarray(kappa, dtype=float))
    # delta[k, p, d] = points[p, d] - centers[k, d]
    delta = points[None, :, :] - centers[:, None, :]
    delta = _periodic_delta(delta, periodic)
    return 0.5 * beta * np.sum(kappa[:, None, :] * delta**2, axis=2)


def _counts_per_window(window_ids: np.ndarray, n_windows: int) -> np.ndarray:
    return np.bincount(window_ids, minlength=n_windows).astype(float)


def _kish_n_eff(weights: np.ndarray) -> float:
    w = np.asarray(weights, dtype=float)
    s2 = float(np.sum(w * w))
    return 1.0 / s2 if s2 > 0 else 0.0


def mbar_weights(
    cvs: np.ndarray,
    window_ids: np.ndarray,
    centers: np.ndarray,
    kappa: np.ndarray,
    beta: float,
    periodic,
) -> Reweighting:
    """Per-frame equilibrium weights via pymbar MBAR.

    Returns a :class:`Reweighting` with ``method="mbar"``. The unbiased weights
    use the standard log-denominator form ``w_n proportional to
    1 / sum_k N_k exp(f_k - u_kn)``.

    Raises:
        ImportError: if pymbar is not installed.
    """
    try:
        from pymbar import MBAR
    except ImportError as exc:  # pragma: no cover - exercised only without pymbar
        raise ImportError(
            "pymbar is required for MBAR reweighting. Install with "
            "'pip install -e .[workflows]' or pass reweight method='wham'."
        ) from exc

    cvs = np.atleast_2d(np.asarray(cvs, dtype=float))
    window_ids = np.asarray(window_ids).reshape(-1)
    K = int(centers.shape[0])
    N_k = _counts_per_window(window_ids, K)
    # Order samples by window so MBAR's N_k blocks line up, then invert later.
    order = np.argsort(window_ids, kind="stable")
    inv = np.empty_like(order)
    inv[order] = np.arange(order.size)
    u_kn = reduced_harmonic(cvs[order], centers, kappa, beta, periodic)  # (K, N)

    mbar = MBAR(u_kn, N_k)
    f_k = np.asarray(mbar.f_k, dtype=float)
    log_terms = np.log(np.maximum(N_k, 1e-300))[:, None] + f_k[:, None] - u_kn  # (K, N)
    log_denom = logsumexp(log_terms, axis=0)  # (N,)
    log_w = -log_denom
    log_w -= logsumexp(log_w)
    w_ordered = np.exp(log_w)
    weights = w_ordered[inv]
    return Reweighting(
        sample_weights=weights,
        f_k=f_k,
        n_eff=_kish_n_eff(weights),
        method="mbar",
        pmf_edges=None,
        pmf=None,
        meta={"n_windows": K, "n_frames": int(cvs.shape[0])},
    )


def _histogram_grid(cvs: np.ndarray, n_bins, pad_frac: float = 0.02):
    """Build per-dim bin edges and the flat bin index of every sample."""
    n_cv = cvs.shape[1]
    if np.isscalar(n_bins):
        n_bins = [int(n_bins)] * n_cv
    edges = []
    for d in range(n_cv):
        lo, hi = float(cvs[:, d].min()), float(cvs[:, d].max())
        span = hi - lo
        pad = pad_frac * span if span > 0 else 1.0
        edges.append(np.linspace(lo - pad, hi + pad, n_bins[d] + 1))
    # Per-dim bin index (clipped into range), then ravel to a flat bin id.
    idx_per_dim = []
    shape = []
    for d in range(n_cv):
        bi = np.clip(np.digitize(cvs[:, d], edges[d]) - 1, 0, n_bins[d] - 1)
        idx_per_dim.append(bi)
        shape.append(n_bins[d])
    flat_idx = np.ravel_multi_index(idx_per_dim, shape)
    centers_per_dim = [0.5 * (e[:-1] + e[1:]) for e in edges]
    mesh = np.meshgrid(*centers_per_dim, indexing="ij")
    bin_centers = np.stack([m.reshape(-1) for m in mesh], axis=1)  # (B, n_cv)
    return edges, flat_idx, bin_centers, tuple(shape)


def wham_weights(
    cvs: np.ndarray,
    window_ids: np.ndarray,
    centers: np.ndarray,
    kappa: np.ndarray,
    beta: float,
    periodic,
    *,
    n_bins=60,
    tol: float = 1e-6,
    max_iter: int = 10000,
) -> Reweighting:
    """Per-frame equilibrium weights via binned, iterative WHAM (1D or 2D).

    Solves the standard WHAM equations in log space:
    ``log p_b = log h_b - logsumexp_k(log N_k + f_k - bias_kb)`` and
    ``f_k = -logsumexp_b(log p_b - bias_kb)`` to convergence, then assigns each
    frame the weight of its bin (bin probability divided equally among the
    frames it contains) and renormalises.

    Args:
        cvs, window_ids, centers, kappa, beta, periodic: as in :func:`mbar_weights`.
        n_bins: bins per CV dim (int or per-dim sequence).
        tol: convergence tolerance on ``max|delta f_k|``.
        max_iter: iteration cap.

    Returns:
        A :class:`Reweighting` with ``method="wham"`` and a filled PMF.
    """
    cvs = np.atleast_2d(np.asarray(cvs, dtype=float))
    window_ids = np.asarray(window_ids).reshape(-1)
    K = int(centers.shape[0])
    N_k = _counts_per_window(window_ids, K)
    log_N_k = np.log(np.maximum(N_k, 1e-300))

    edges, flat_idx, bin_centers, shape = _histogram_grid(cvs, n_bins)
    B = bin_centers.shape[0]
    h_b = np.bincount(flat_idx, minlength=B).astype(float)  # (B,)
    occupied = h_b > 0
    log_h = np.where(occupied, np.log(np.maximum(h_b, 1e-300)), -np.inf)

    bias_kb = reduced_harmonic(bin_centers, centers, kappa, beta, periodic)  # (K, B)

    f_k = np.zeros(K)
    for it in range(max_iter):
        # log p_b (unnormalised) over occupied bins.
        log_denom_b = logsumexp(log_N_k[:, None] + f_k[:, None] - bias_kb, axis=0)  # (B,)
        log_p = np.where(occupied, log_h - log_denom_b, -np.inf)
        log_p -= logsumexp(log_p)  # normalise the density
        f_new = -logsumexp(log_p[None, :] - bias_kb, axis=1)  # (K,)
        f_new -= f_new[0]  # gauge: anchor window 0
        delta = np.max(np.abs(f_new - f_k))
        f_k = f_new
        if delta < tol:
            break
    else:
        warnings.warn(
            f"WHAM did not converge in {max_iter} iters (last |df|={delta:.2e})",
            stacklevel=2,
        )

    # Per-frame weight = bin probability shared equally among the bin's frames.
    p_b = np.where(occupied, np.exp(log_p), 0.0)
    per_frame = np.zeros(cvs.shape[0])
    nonzero = h_b > 0
    w_b = np.zeros(B)
    w_b[nonzero] = p_b[nonzero] / h_b[nonzero]
    per_frame = w_b[flat_idx]
    total = per_frame.sum()
    if total > 0:
        per_frame /= total

    pmf = np.full(B, np.nan)
    pmf[occupied] = -(log_p[occupied] - np.max(log_p[occupied]))  # kT units, min 0
    pmf = pmf.reshape(shape)
    return Reweighting(
        sample_weights=per_frame,
        f_k=f_k,
        n_eff=_kish_n_eff(per_frame),
        method="wham",
        pmf_edges=edges,
        pmf=pmf,
        meta={
            "n_windows": K,
            "n_frames": int(cvs.shape[0]),
            "n_bins": list(shape),
            "iterations": it + 1,
            "final_df": float(delta),
            "frac_bins_occupied": float(np.mean(occupied)),
        },
    )


def reweight(
    dataset: USDataset,
    *,
    method: str = "auto",
    n_bins=60,
    **wham_kwargs,
) -> Reweighting:
    """Reweight a :class:`USDataset`, auto-selecting MBAR or WHAM.

    Args:
        dataset: the umbrella dataset (provides cvs, window ids, bias params,
            beta, periodicity).
        method: ``"mbar"``, ``"wham"``, or ``"auto"`` (WHAM when
            ``n_windows > MBAR_MAX_WINDOWS`` or MBAR raises / runs out of memory).
        n_bins: WHAM bins per CV dim.
        **wham_kwargs: forwarded to :func:`wham_weights` (``tol``, ``max_iter``).

    Returns:
        A :class:`Reweighting`; ``method`` records which engine actually ran and
        ``meta["fallback"]`` is set when MBAR was abandoned for WHAM.
    """
    args = (
        dataset.cvs,
        dataset.window_ids,
        dataset.window_centers,
        dataset.window_kappa,
        dataset.beta,
        dataset.cv_periodic,
    )
    method = method.lower()
    if method == "mbar":
        return mbar_weights(*args)
    if method == "wham":
        return wham_weights(*args, n_bins=n_bins, **wham_kwargs)
    if method != "auto":
        raise ValueError(f"unknown reweight method {method!r}; use mbar|wham|auto")

    if dataset.n_windows > MBAR_MAX_WINDOWS:
        logger.info(
            "reweight: %d windows > %d -> using WHAM (MBAR matrix infeasible)",
            dataset.n_windows,
            MBAR_MAX_WINDOWS,
        )
        rw = wham_weights(*args, n_bins=n_bins, **wham_kwargs)
        return rw._replace(meta={**rw.meta, "auto_choice": "wham_large_K"})
    try:
        return mbar_weights(*args)
    except Exception as exc:
        logger.warning("reweight: MBAR failed (%s); falling back to WHAM", exc)
        rw = wham_weights(*args, n_bins=n_bins, **wham_kwargs)
        return rw._replace(meta={**rw.meta, "fallback": f"mbar_failed: {type(exc).__name__}"})
