"""Committor-free baselines, for the comparison row.

:func:`pmf_kramers_rate` is the overdamped Kramers estimate on a physical
coordinate: harmonic fits to the wells and the barrier of ``F(s) = -log pi(s)``
and the diffusion at the barrier. It is a labelled baseline, not an estimator
of this library. It needs a coordinate with interior wells, a barrier well
above ``kT``, and a diffusive coordinate at the barrier, and it knows nothing
about the committor.
"""

import numpy as np

from ._coordinate import Profile, value_at


def _local_curvature(centers, F, target, window):
    """``(s*, F(s*), F''(s*))`` of the extremum of ``F`` in ``window`` (quadratic fit)."""
    lo, hi = window
    band = (centers >= lo) & (centers <= hi) & np.isfinite(F)
    if not np.any(band):
        return np.nan, np.nan, np.nan
    cb, Fb = centers[band], F[band]
    i = int(np.argmin(Fb)) if target == "min" else int(np.argmax(Fb))
    s_star, F_star = float(cb[i]), float(Fb[i])
    lo2, hi2 = max(0, i - 3), min(cb.shape[0], i + 4)
    if hi2 - lo2 < 3:
        return s_star, F_star, np.nan
    coef = np.polyfit(cb[lo2:hi2], Fb[lo2:hi2], 2)
    return s_star, F_star, float(2.0 * coef[0])


def pmf_kramers_rate(pi: Profile, D: Profile) -> dict:
    """Overdamped Kramers rate on a physical coordinate ``s``::

        k = D(s*) / (2 pi) sqrt(|F''(s_well)| |F''(s*)|) exp(-(F(s*) - F(s_well)))

    with ``F = -log pi`` in units of ``kT`` and ``s*`` the barrier. The A well is the minimum of ``F``
    on the low-``s`` 40% of the range, the B well on the high-``s`` 40%, the
    barrier the maximum on the central 60%; curvatures come from quadratic fits
    over seven bins.

    Args:
        pi: the density along ``s`` (:func:`density` with ``span=``).
        D: the diffusion along ``s`` (:func:`hummer_diffusion` or
            :func:`diffusion_profile`), read at the barrier by interpolation.

    Returns:
        dict ``{k_AB, k_BA, s_barrier, delta_F_AB, delta_F_BA, D_barrier}``;
        NaN rates when a well or the barrier has no usable curvature.
    """
    centers = np.asarray(pi.levels, dtype=np.float64)
    p = np.asarray(pi.values, dtype=np.float64)
    F = -np.log(np.where(p > 0, p, np.nan))
    lo, hi = float(centers[0]), float(centers[-1])
    span = hi - lo
    s_bar, F_bar, Fdd_bar = _local_curvature(centers, F, "max", (lo + 0.2 * span, lo + 0.8 * span))
    _, F_A, Fdd_A = _local_curvature(centers, F, "min", (lo, lo + 0.4 * span))
    _, F_B, Fdd_B = _local_curvature(centers, F, "min", (lo + 0.6 * span, hi))
    D_bar = value_at(D, s_bar) if np.isfinite(s_bar) else float("nan")

    def k(Fdd_well, dF):
        if not (
            np.isfinite(D_bar) and D_bar > 0 and np.isfinite(Fdd_well) and np.isfinite(Fdd_bar)
        ):
            return float("nan")
        curv = abs(Fdd_well) * abs(Fdd_bar)
        return (
            float(D_bar / (2.0 * np.pi) * np.sqrt(curv) * np.exp(-dF)) if curv > 0 else float("nan")
        )

    return {
        "k_AB": k(Fdd_A, F_bar - F_A),
        "k_BA": k(Fdd_B, F_bar - F_B),
        "s_barrier": float(s_bar),
        "delta_F_AB": float(F_bar - F_A),
        "delta_F_BA": float(F_bar - F_B),
        "D_barrier": float(D_bar),
    }
