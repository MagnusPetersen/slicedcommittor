"""Reaction-coordinate resolution and the ``at=`` selector for rate profiles.

The diffusion, density, and flux quantities are all 1D profiles along a
reaction coordinate. By default that coordinate is the committor q(x) (levels
in [0, 1]); a per-sample collective-variable array can be supplied instead via
``coordinate=`` for diffusion and density. Flux is committor-specific (TPT).

Note on the ``at=`` grammar (shared by every quantity and rate):

* ``at=None``      -> the full :class:`Profile` along the coordinate.
* ``at=<scalar>``  -> the value at that single level (interpolated).
* ``at=(lo, hi)``  -> a range reduction. The reduction DIFFERS by quantity:
  density / diffusion average the profile over the range (:func:`value_at`),
  whereas the reactive flux and the rates take the bin-free plateau flux
  ``(Σ contrib)/(hi-lo)`` (``_plateau_flux``), which is the rate, not a mean.
* ``at="auto"``    -> the flux-flatness plateau located by :func:`find_plateau`
  (rate formulas only).
"""

from typing import NamedTuple

import jax.numpy as jnp
import numpy as np

# Default lag ladder (in frames) for the Kramers-Moyal diffusion scan. Shared by
# quantities.py and formulas.py so the default lives in one place.
_LAG_CANDIDATES = (1, 2, 5, 10, 20, 50)


class Profile(NamedTuple):
    """A quantity profiled along the reaction coordinate.

    levels: (n_bins,) bin centres along the coordinate.
    values: (n_bins,) the quantity (D, π, or Φ) per bin (NaN where undefined).
    counts: (n_bins,) per-bin sample counts.
    name:   coordinate label ("committor" or "cv").
    """

    levels: np.ndarray
    values: np.ndarray
    counts: np.ndarray
    name: str = "committor"


def coordinate_levels(committor, points, coordinate):
    """Per-sample coordinate values and the coordinate domain span.

    coordinate=None  -> committor levels ``q(points)`` clipped to [0, 1], span (0, 1).
    coordinate=array -> per-sample CV values (length == len(points)), span (min, max).
    """
    if coordinate is None:
        lv = np.asarray(committor(jnp.asarray(points)), dtype=np.float64)
        lv = np.clip(lv, 0.0, 1.0)
        return lv, (0.0, 1.0), "committor"
    cv = np.asarray(coordinate, dtype=np.float64).reshape(-1)
    n = int(np.asarray(points).shape[0])
    if cv.shape[0] != n:
        raise ValueError(f"coordinate length {cv.shape[0]} != number of points {n}")
    return cv, (float(np.min(cv)), float(np.max(cv))), "cv"


def _is_range(at) -> bool:
    """Whether ``at`` is a ``(lo, hi)`` range selector (vs a scalar/array level).

    A 2-tuple means a range; a length-2 array is NOT a range (it is two query
    levels), which keeps the selector grammar unambiguous across the rates API.
    """
    return isinstance(at, tuple) and len(at) == 2


def value_at(profile: Profile, at):
    """Sample a :class:`Profile` at point level(s) or average over a range.

    ``at`` is a scalar level (interpolated), a 1D array of levels (interpolated
    elementwise), or a ``(lo, hi)`` tuple (mean over the covered finite bins).
    """
    centers = np.asarray(profile.levels)
    values = np.asarray(profile.values)
    good = np.isfinite(values)
    if not np.any(good):
        raise ValueError("profile has no finite values to sample")
    if _is_range(at):
        lo, hi = float(at[0]), float(at[1])
        mask = good & (centers >= lo) & (centers <= hi)
        return float(np.mean(values[mask])) if np.any(mask) else float("nan")
    arr = np.atleast_1d(np.asarray(at, dtype=np.float64))
    out = np.interp(arr, centers[good], values[good])
    return float(out[0]) if np.ndim(at) == 0 else out


class PlateauWindow(NamedTuple):
    """The auto-located flux plateau (see :func:`find_plateau`).

    lo, hi:    window bounds along the committor coordinate.
    flatness:  relative spread σ/|μ| of the flux over the window's valid bins
               (≲ 0.2 ⇒ well-conserved / trustworthy).
    value:     median flux over the window (the plateau estimate).
    n_valid:   number of valid bins inside the window.
    ok:        whether the window met the flatness tolerance (False ⇒ fell back
               to the global minimum-flatness window; treat the rate as
               indicative, not converged).
    """

    lo: float
    hi: float
    flatness: float
    value: float
    n_valid: int
    ok: bool


def find_plateau(profile: Profile, *, min_width=0.2, tol=0.25, min_bins=3, min_count=1, margin=0.05):
    """Locate the flattest flux window -- the automatic reaction-rate plateau.

    The reactive flux Φ(c) (or any iso-committor flux profile, e.g. ``D_q·π``)
    is constant in ``c`` for the true committor by current conservation; for an
    approximate committor it is flat on a saddle plateau near c≈0.5 and inflates
    in the basins where committor error dominates. Rather than fix the window at
    ``[0.2, 0.8]`` (sc_simon / the main-repo ``dirichlet_stratified`` diagnostic),
    this scans candidate windows ``[lo, hi] ⊆ [margin, 1−margin]`` of width
    ≥ ``min_width`` and returns the **widest** whose relative flatness
    ``σ/|μ|`` over its valid (finite, positive, well-counted) bins is ≤ ``tol``
    (ties broken toward the flattest). If none meets ``tol`` it returns the
    global minimum-flatness window with ``ok=False``.

    Args:
        profile: a :class:`Profile` of the flux along the committor in ``[0, 1]``.
        min_width: minimum committor-range width of an admissible window.
        tol: flatness threshold ``σ/|μ|`` for an "acceptable" plateau.
        min_bins: minimum valid bins required inside a window.
        min_count: per-bin sample-count floor for a bin to count as valid.
        margin: basin guard -- only consider bins with ``margin ≤ c ≤ 1−margin``.

    Returns:
        :class:`PlateauWindow`.
    """
    centers = np.asarray(profile.levels, dtype=np.float64)
    values = np.asarray(profile.values, dtype=np.float64)
    counts = (
        np.asarray(profile.counts, dtype=np.float64)
        if profile.counts is not None
        else np.ones_like(centers)
    )
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
    # Prefix sums for O(1) per-window mean / variance.
    p1 = np.concatenate(([0.0], np.cumsum(vy)))
    p2 = np.concatenate(([0.0], np.cumsum(vy * vy)))

    best_ok = None      # (width, -flatness, i, j) for windows meeting tol
    best_any = None     # (flatness, -width, i, j) global minimum-flatness fallback
    for i in range(n):
        for j in range(i + min_bins - 1, n):
            width = vc[j] - vc[i]
            if width < min_width:
                continue
            cnt = j - i + 1
            s1 = p1[j + 1] - p1[i]
            s2 = p2[j + 1] - p2[i]
            mean = s1 / cnt
            var = max(s2 / cnt - mean * mean, 0.0)
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
    lo, hi = float(vc[i]), float(vc[j])
    win = vy[i : j + 1]
    mean = float(np.mean(win))
    flatness = float(np.std(win) / max(abs(mean), 1e-30))
    return PlateauWindow(lo, hi, flatness, float(np.median(win)), int(j - i + 1), ok)
