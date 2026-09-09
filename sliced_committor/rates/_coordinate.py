"""Profiles along a reaction coordinate, and the reducers that read them.

Every rate quantity is a :class:`Profile`: one value per bin along a coordinate
(the committor by default, with levels in ``[0, 1]``). Three reducers read a
profile: :func:`value_at` samples it at a level or averages it over a range;
:func:`flux_flatness` is the constancy statistic of a flux profile over a band;
:func:`find_plateau` locates the flattest band of a flux profile.
"""

from typing import NamedTuple

import numpy as np


class Profile(NamedTuple):
    """A quantity profiled along a coordinate.

    Attributes:
        levels: ``(n,)`` bin centres along the coordinate (lags, for a lag scan).
        values: ``(n,)`` the quantity per bin, NaN where undefined.
        counts: ``(n,)`` per-bin sample counts, for undersampling guards.
    """

    levels: np.ndarray
    values: np.ndarray
    counts: np.ndarray


def _is_range(level) -> bool:
    """A 2-tuple is a ``(lo, hi)`` range; a length-2 array is two query levels."""
    return isinstance(level, tuple) and len(level) == 2


def value_at(profile: Profile, level):
    """Read a profile at a level, at an array of levels, or averaged over a range.

    ``level`` is a scalar (linear interpolation between the finite bins), a 1D
    array of scalars (elementwise), or a ``(lo, hi)`` tuple (the mean of the
    finite bins whose centre lies in the range; NaN when there is none).
    """
    centers = np.asarray(profile.levels, dtype=np.float64)
    values = np.asarray(profile.values, dtype=np.float64)
    good = np.isfinite(values)
    if not np.any(good):
        raise ValueError("profile has no finite values to sample")
    if _is_range(level):
        lo, hi = float(level[0]), float(level[1])
        mask = good & (centers >= lo) & (centers <= hi)
        return float(np.mean(values[mask])) if np.any(mask) else float("nan")
    arr = np.atleast_1d(np.asarray(level, dtype=np.float64))
    out = np.interp(arr, centers[good], values[good])
    return float(out[0]) if np.ndim(level) == 0 else out


def flux_flatness(profile: Profile, band) -> float:
    """``sigma / mu`` of a flux profile over ``band``: the constancy statistic.

    The committor-coordinate reactive flux ``nu_R(q) = D_q(q) pi(q)`` is constant
    in ``q`` for the exact committor (current conservation), so its relative
    spread over a band measures how far a fitted committor is from it. Only
    finite, positive bins count; NaN with fewer than two of them. The paper's
    ``flux_cv`` is this statistic on the fixed band ``(0.2, 0.8)``, so that fits
    can be compared on one band.
    """
    lo, hi = _check_band(band)
    levels = np.asarray(profile.levels, dtype=np.float64)
    values = np.asarray(profile.values, dtype=np.float64)
    sel = (levels >= lo) & (levels <= hi) & np.isfinite(values) & (values > 0)
    if int(sel.sum()) < 2:
        return float("nan")
    v = values[sel]
    return float(np.std(v) / np.mean(v))


def _check_band(band):
    if not _is_range(band):
        raise ValueError(f"a band is a (lo, hi) tuple of committor levels; got {band!r}")
    lo, hi = float(band[0]), float(band[1])
    if not lo < hi:
        raise ValueError(f"a band needs lo < hi; got ({lo}, {hi})")
    return lo, hi


class PlateauWindow(NamedTuple):
    """The band located by :func:`find_plateau`.

    lo, hi:    window bounds along the committor coordinate.
    flatness:  ``sigma/|mu|`` of the flux over the window's valid bins.
    value:     median flux over the window.
    n_valid:   number of valid bins inside the window.
    ok:        whether the window met the flatness tolerance. ``False`` means the
               global minimum-flatness window was returned instead; treat the
               rate read on it as indicative, not converged.
    """

    lo: float
    hi: float
    flatness: float
    value: float
    n_valid: int
    ok: bool


def find_plateau(
    profile: Profile, *, min_width=0.2, tol=0.25, min_bins=3, min_count=1, margin=0.05
) -> PlateauWindow:
    """Locate the flattest band of a flux profile: the automatic plateau.

    For an approximate committor the flux ``D_q pi`` is flat on a saddle band
    near ``q = 0.5`` and inflates in the basins, where committor error
    dominates. This scans every window ``[lo, hi]`` within ``[margin, 1 -
    margin]`` of width at least ``min_width`` and returns the WIDEST whose
    relative flatness ``sigma/|mu|`` over its valid bins (finite, positive, at
    least ``min_count`` samples) is at most ``tol``, ties broken toward the
    flattest. If no window meets ``tol`` it returns the global minimum-flatness
    window with ``ok=False``.
    """
    centers = np.asarray(profile.levels, dtype=np.float64)
    values = np.asarray(profile.values, dtype=np.float64)
    counts = np.asarray(profile.counts, dtype=np.float64)
    valid = (
        np.isfinite(values)
        & (values > 0)
        & (counts >= min_count)
        & (centers >= margin)
        & (centers <= 1.0 - margin)
    )
    if int(valid.sum()) < min_bins:
        return PlateauWindow(margin, 1.0 - margin, float("nan"), float("nan"), 0, False)

    vc = centers[valid]
    vy = values[valid]
    n = vc.shape[0]
    p1 = np.concatenate(([0.0], np.cumsum(vy)))
    p2 = np.concatenate(([0.0], np.cumsum(vy * vy)))

    best_ok = None  # (width, -flatness, i, j) among windows meeting tol
    best_any = None  # (flatness, -width, i, j) global fallback
    for i in range(n):
        for j in range(i + min_bins - 1, n):
            width = vc[j] - vc[i]
            if width < min_width:
                continue
            cnt = j - i + 1
            mean = (p1[j + 1] - p1[i]) / cnt
            var = max((p2[j + 1] - p2[i]) / cnt - mean * mean, 0.0)
            flat = float(np.sqrt(var) / max(abs(mean), 1e-30))
            cand_any = (flat, -width, i, j)
            if best_any is None or cand_any < best_any:
                best_any = cand_any
            if flat <= tol:
                cand_ok = (width, -flat, i, j)
                if best_ok is None or cand_ok > best_ok:
                    best_ok = cand_ok

    if best_ok is not None:
        _, _, i, j = best_ok
        ok = True
    else:
        _, _, i, j = best_any
        ok = False
    win = vy[i : j + 1]
    mean = float(np.mean(win))
    flatness = float(np.std(win) / max(abs(mean), 1e-30))
    return PlateauWindow(
        float(vc[i]), float(vc[j]), flatness, float(np.median(win)), int(j - i + 1), ok
    )
