"""The rate from the pair ``{D_q(q), pi(q)}``.

Projected onto the committor coordinate the dynamics is a 1D diffusion with
density ``pi(q)`` and diffusion ``D_q(q)``, and its reactive flux
``nu_R(q) = D_q(q) pi(q)`` is constant in ``q`` for the exact committor
(current conservation). The rate constants are ``k_AB = nu_R / rho_A`` and
``k_BA = nu_R / rho_B``. A fitted committor gives a profile that is not flat,
and a ``reduction`` picks the number:

* ``"arithmetic"``: ``nu_R = int_0^1 D_q pi dq``. By the co-area identity this
  is the Dirichlet form ``<D |grad q|^2>_rho``, the variational upper bound on
  the true rate. ``pi``-weighted, so the basins dominate it.
* ``"harmonic"``: ``k_AB = 1 / int_0^1 M_A(q) / (D_q pi) dq`` with
  ``M_A(q) = int_0^q pi``, the exact mean first-passage time of the projected
  1D diffusion. Bottleneck-dominated; exposed to the basin tails.
* ``"plateau"``: the median of ``nu_R`` over a band, ``(0.3, 0.7)`` by default
  or ``"auto"`` for the flattest band (:func:`find_plateau`). A robust location
  estimate of the constant, read away from the basins where committor error
  inflates the flux. The published reduction.
* ``"local"``: ``nu_R(q*)`` at one level, ``0.5`` by default: the
  Berezhkovskii-Szabo single-surface reading.

All coincide for the exact committor. Every result also carries the
arithmetic and harmonic values and the ``flatness`` of the flux over the
band: their spread is the committor-quality diagnostic. Note that the
harmonic value is not a bound; for an approximate committor it can lie on
either side of the arithmetic one.
"""

import warnings

import jax.numpy as jnp
import numpy as np

from ._coordinate import Profile, _check_band, find_plateau, flux_flatness
from .diffusion import diffusion_profile
from .quantities import basin_populations, density

_REDUCTIONS = ("plateau", "arithmetic", "harmonic", "local")
_DEFAULT_BAND = (0.3, 0.7)
_DEFAULT_Q_STAR = 0.5


def _on_grid(D_q: Profile, centers):
    """``D_q`` interpolated (finite positive nodes only) onto the density's grid."""
    lv = np.asarray(D_q.levels, dtype=np.float64)
    dv = np.asarray(D_q.values, dtype=np.float64)
    good = np.isfinite(dv) & (dv > 0)
    if not np.any(good):
        return np.full(centers.shape, np.nan)
    return np.interp(centers, lv[good], dv[good])


def _mfpt(centers, pi, Dq, window):
    """Mean first-passage times ``int M(q) / (D_q pi) dq`` of the projected 1D diffusion.

    The density is a histogram, so bin ``i`` carries the mass ``pi_i h`` and
    the cumulative mass at its centre is the mass of the bins below plus half
    its own. With that quadrature the discrete identity ``int M_A dq = rho_A``
    holds exactly, and a constant flux gives ``1 / MFPT_AB = nu_R / rho_A`` to
    rounding. ``window`` restricts the integration range to ``(lo, hi)``.
    """
    h = float(centers[1] - centers[0])
    mass = pi * h
    M_A = np.cumsum(mass) - 0.5 * mass
    M_B = np.cumsum(mass[::-1])[::-1] - 0.5 * mass
    good = (pi > 0) & np.isfinite(Dq) & (Dq > 0)
    nu = np.where(good, Dq * pi, 1.0)
    integ_AB = np.where(good, M_A / nu, 0.0)
    integ_BA = np.where(good, M_B / nu, 0.0)
    if window is None:
        mask = np.ones_like(centers, dtype=bool)
    else:
        lo, hi = _check_band(window)
        mask = (centers >= lo) & (centers <= hi)
    return float(np.sum(integ_AB[mask]) * h), float(np.sum(integ_BA[mask]) * h)


def _inv(m):
    return 1.0 / m if (m > 0 and np.isfinite(m)) else float("nan")


def _own_parameters(reduction, band, q_star, window):
    """Each reduction accepts only its own parameter; a foreign one is an error, not a no-op."""
    given = {"band": band, "q_star": q_star, "window": window}
    owner = {"band": "plateau", "q_star": "local", "window": "harmonic"}
    for name, value in given.items():
        if value is not None and owner[name] != reduction:
            raise TypeError(
                f"{name}= belongs to reduction={owner[name]!r}, not {reduction!r}; "
                f"it would be ignored"
            )


def rate_from_profiles(
    pi: Profile,
    D_q: Profile,
    rho_A,
    rho_B,
    *,
    reduction="plateau",
    band=None,
    q_star=None,
    window=None,
):
    """Reduce the committor-coordinate flux ``nu_R(q) = D_q(q) pi(q)`` to a rate.

    Args:
        pi: the committor density on ``[0, 1]`` (:func:`density`); its grid is
            the grid of the reduction.
        D_q: the committor-coordinate diffusion, on any grid (interpolated
            between its finite positive bins, constant beyond them).
        rho_A, rho_B: the populations (:func:`basin_populations`).
        reduction: ``"plateau"`` (default), ``"arithmetic"``, ``"harmonic"`` or
            ``"local"``, see the module docstring.
        band: plateau only. ``(lo, hi)``, default ``(0.3, 0.7)``, or ``"auto"``
            for :func:`find_plateau` (warns when nothing is flat).
        q_star: local only. The level, default ``0.5``.
        window: harmonic only. ``(lo, hi)`` integration range, default ``[0, 1]``.

    Returns:
        dict with ``nu_R`` (NaN for the harmonic reduction, which yields the
        rate constants directly), ``rho_A``, ``rho_B``, ``k_AB``, ``k_BA``,
        ``reduction``, ``band`` (the band the flatness is read on),
        ``flatness`` (:func:`flux_flatness` over it), ``nu`` (the flux
        :class:`Profile`), the diagnostics ``k_AB_arithmetic``,
        ``k_BA_arithmetic``, ``k_AB_harmonic``, ``k_BA_harmonic``, and the
        reduction's own keys: ``plateau_ok`` (plateau), ``q_star`` (local),
        ``mfpt_AB``/``mfpt_BA`` (harmonic).
    """
    if reduction not in _REDUCTIONS:
        raise ValueError(f"unknown reduction={reduction!r}; use one of {_REDUCTIONS}")
    _own_parameters(reduction, band, q_star, window)
    rho_A, rho_B = float(rho_A), float(rho_B)

    centers = np.asarray(pi.levels, dtype=np.float64)
    pi_v = np.where(np.isfinite(pi.values), np.asarray(pi.values, dtype=np.float64), 0.0)
    Dq = _on_grid(D_q, centers)
    nu = Dq * pi_v
    nu_profile = Profile(centers, nu, np.asarray(pi.counts))
    h = float(centers[1] - centers[0])
    positive = np.isfinite(nu) & (nu > 0)

    # the two ends, always: the variational bound and the exact projected MFPT
    nu_arith = float(np.sum(np.where(positive, nu, 0.0)) * h)
    mfpt_AB, mfpt_BA = _mfpt(centers, pi_v, Dq, None)
    out = {
        "reduction": reduction,
        "rho_A": rho_A,
        "rho_B": rho_B,
        "k_AB_arithmetic": nu_arith / max(rho_A, 1e-30),
        "k_BA_arithmetic": nu_arith / max(rho_B, 1e-30),
        "k_AB_harmonic": _inv(mfpt_AB),
        "k_BA_harmonic": _inv(mfpt_BA),
        "nu": nu_profile,
    }
    flat_band = _DEFAULT_BAND

    if reduction == "harmonic":
        if window is not None:
            mfpt_AB, mfpt_BA = _mfpt(centers, pi_v, Dq, window)
        out.update(nu_R=float("nan"), k_AB=_inv(mfpt_AB), k_BA=_inv(mfpt_BA))
        out.update(mfpt_AB=mfpt_AB, mfpt_BA=mfpt_BA)
    else:
        if reduction == "arithmetic":
            nu_R = nu_arith
        elif reduction == "plateau":
            if band is None:
                band = _DEFAULT_BAND
            if isinstance(band, str) and band == "auto":
                pw = find_plateau(nu_profile)
                if not pw.ok:
                    warnings.warn(
                        f"find_plateau: no band of width >= 0.2 is flat to tolerance "
                        f"(flattest has sigma/mu = {pw.flatness:.3f}); the rate is read "
                        "on it anyway and is indicative, not converged.",
                        UserWarning,
                        stacklevel=2,
                    )
                lo, hi, ok = pw.lo, pw.hi, pw.ok
            else:
                lo, hi = _check_band(band)
                ok = True
            sel = positive & (centers >= lo) & (centers <= hi)
            nu_R = float(np.median(nu[sel])) if np.any(sel) else float("nan")
            flat_band = (lo, hi)
            out.update(plateau_ok=bool(ok))
        else:  # local
            q_star = _DEFAULT_Q_STAR if q_star is None else float(q_star)
            nu_R = (
                float(np.interp(q_star, centers[positive], nu[positive]))
                if int(positive.sum()) >= 2
                else float("nan")
            )
            out.update(q_star=q_star)
        out.update(nu_R=nu_R, k_AB=nu_R / max(rho_A, 1e-30), k_BA=nu_R / max(rho_B, 1e-30))

    out.update(band=flat_band, flatness=flux_flatness(nu_profile, flat_band))
    return out


def committor_rate(
    committor,
    samples,
    *,
    D_q=None,
    trajectory=None,
    dt=None,
    lag=None,
    window_ids=None,
    sample_weights=None,
    in_A=None,
    in_B=None,
    n_bins=200,
    reduction="plateau",
    band=None,
    q_star=None,
    window=None,
):
    """The reaction rate of a fitted committor, in one call.

    Builds the pair ``{D_q, pi}`` and reduces it with
    :func:`rate_from_profiles`. The committor-coordinate diffusion comes from
    exactly one of two routes:

    * ``D_q=``: a profile you constructed, typically the Jacobian map
      :func:`committor_diffusion_from_cv` of a diffusion measured along the
      biased coordinate (the paper's route), or the same map with
      ``cv_grad_sq=1`` for an assumed configurational ``D0``.
    * ``trajectory=`` with ``dt=`` and ``lag=`` (and ``window_ids=`` for
      umbrella data): the Kramers-Moyal estimate on the committor coordinate
      itself, :func:`diffusion_profile` of ``q(trajectory)``. This needs a
      diffusive regime on the committor at that lag; check with
      :func:`lag_scan`.

    ``samples`` is the static ensemble (with ``sample_weights`` for biased
    data) that supplies the density and the populations; ``in_A``/``in_B``
    snap the populations to the basins. The remaining arguments are those of
    :func:`rate_from_profiles`.
    """
    if (D_q is None) == (trajectory is None):
        raise ValueError(
            "pass exactly one of D_q= (a committor-coordinate diffusion profile) or "
            "trajectory= (with dt= and lag=)"
        )
    if D_q is not None and (dt is not None or lag is not None or window_ids is not None):
        raise ValueError("dt=, lag= and window_ids= belong to the trajectory route, not to D_q=")
    if trajectory is not None and (dt is None or lag is None):
        raise ValueError("the trajectory route needs dt= and an explicit lag=")

    qx = np.asarray(committor(jnp.asarray(samples)), dtype=np.float64)
    pi = density(qx, sample_weights=sample_weights, n_bins=n_bins)
    if D_q is None:
        qt = np.asarray(committor(jnp.asarray(trajectory)), dtype=np.float64)
        D_q = diffusion_profile(qt, dt=dt, lag=lag, window_ids=window_ids, n_bins=n_bins)
    rho_A, rho_B = basin_populations(qx, sample_weights=sample_weights, in_A=in_A, in_B=in_B)
    return rate_from_profiles(
        pi, D_q, rho_A, rho_B, reduction=reduction, band=band, q_star=q_star, window=window
    )
