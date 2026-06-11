"""Reaction-rate formulas built on the quantity primitives.

Each takes the callable committor ``q`` (+ data) and returns a dict with
``nu_R`` (reactive flux), basin populations ``rho_A``, ``rho_B``, and the rate
constants ``k_AB = nu_R / rho_A``, ``k_BA = nu_R / rho_B``:

* :func:`dirichlet_rate`         -- variational Dirichlet form ⟨D|∇q̄|²⟩_π.
* :func:`tpt_rate`               -- flux through an iso-committor surface.
* :func:`berezhkovskii_szabo_rate` -- committor-coordinate MFPT / local D(q*)π(q*)
  (a thin named alias for :func:`committor_rate`).
* :func:`committor_rate`         -- unified {D_q, π} reductions, the coordinate-
  invariant sibling (harmonic/local reproduce Berezhkovskii-Szabo exactly).
* :func:`kramers_rate`           -- overdamped Kramers barrier-crossing estimate.

Populations use ρ_B = E_π[q̄], ρ_A = 1 − ρ_B, snapping to 0/1 on basin masks
when ``in_A`` / ``in_B`` are supplied.
"""

import jax.numpy as jnp
import numpy as np

from ._coordinate import _LAG_CANDIDATES, Profile, _is_range, find_plateau, value_at
from ._numerics import cumulative_trapezoid, normalize_weights
from .quantities import (
    _coarea_contributions,
    _coarea_profile,
    _density_and_diffusion,
    _plateau_flux,
)


def _populations(committor, samples, sample_weights=None, in_A=None, in_B=None):
    """ρ_A = E_π[1 − q̄], ρ_B = E_π[q̄], with optional 0/1 snapping on basins."""
    q = np.clip(np.asarray(committor(jnp.asarray(samples)), dtype=np.float64), 0.0, 1.0)
    if in_A is not None:
        q = np.where(np.asarray(in_A, dtype=bool), 0.0, q)
    if in_B is not None:
        q = np.where(np.asarray(in_B, dtype=bool), 1.0, q)
    W = normalize_weights(sample_weights, q.shape[0])
    rho_B = float(np.sum(W * q))
    return 1.0 - rho_B, rho_B


def _rate_dict(rho_A, rho_B, *, nu_R=None, k_AB=None, k_BA=None, **extra):
    """Assemble the standard rate dict ``{nu_R, rho_A, rho_B, k_AB, k_BA, ...}``.

    Two entry paths share one builder. Flux form: pass ``nu_R`` and the rate
    constants are derived as ``k = nu_R / ρ``. MFPT form: pass ``k_AB`` / ``k_BA``
    directly (the caller already computed ``k = 1/MFPT`` with its own guard) and
    ``nu_R`` is reported as NaN.
    """
    if nu_R is not None:
        nu = float(nu_R)
        k_AB = nu / max(rho_A, 1e-30)
        k_BA = nu / max(rho_B, 1e-30)
    else:
        nu = float("nan")
    out = {
        "nu_R": nu,
        "rho_A": float(rho_A),
        "rho_B": float(rho_B),
        "k_AB": float(k_AB) if k_AB is not None else float("nan"),
        "k_BA": float(k_BA) if k_BA is not None else float("nan"),
    }
    out.update(extra)
    return out


def _inv_mfpt(m):
    """``k = 1/MFPT`` with the standard finite-positive guard (else NaN)."""
    return 1.0 / m if m > 0 and np.isfinite(m) else float("nan")


def _plateau_report(centers, Phi, counts, lo, hi):
    """Flat plateau-report keys shared by the co-area-flux rates.

    Returns ``{plateau, plateau_flatness}`` where the flatness σ/|μ| is measured
    on the BINNED Φ(c) profile over the window. The reported rate itself is the
    bin-free plateau-MC flux, computed separately by the caller.
    """
    sel = (centers >= lo) & (centers <= hi) & (counts > 0)
    vals = Phi[sel]
    flat = float(np.std(vals) / max(abs(np.mean(vals)), 1e-30)) if vals.size else float("nan")
    return {"plateau": (lo, hi), "plateau_flatness": flat}


# =============================================================================
# Committor-coordinate machinery shared by Berezhkovskii-Szabo and committor_rate
# =============================================================================


def _committor_profiles(
    committor,
    samples,
    trajectory,
    *,
    dt,
    n_bins,
    sample_weights,
    window_ids,
    coordinate,
    traj_coordinate,
    lag,
    lag_candidates,
    min_count,
    diffusion_method,
):
    """The coordinate-invariant pair ``{D_q(q), π(q)}`` on one common grid.

    π(q) is the equilibrium density along the committor (from the static
    ensemble, MBAR-aware); D_q(q) is the committor-coordinate diffusion (from the
    trajectory, Kramers-Moyal or Hummer), interpolated onto π's grid. Both are
    functions of the committor VALUES q(x) only -- dimensionless and identical
    in any feature space -- so every reduction built on them is coordinate-
    invariant (no feature-space gradient, no length-scale D).
    """
    pi_prof, D_prof = _density_and_diffusion(
        committor,
        samples,
        trajectory,
        dt=dt,
        n_bins=n_bins,
        sample_weights=sample_weights,
        window_ids=window_ids,
        coordinate=coordinate,
        traj_coordinate=traj_coordinate,
        lag=lag,
        lag_candidates=lag_candidates,
        min_count=min_count,
        diffusion_method=diffusion_method,
    )
    centers = np.asarray(pi_prof.levels, dtype=np.float64)
    pi = np.where(np.isfinite(pi_prof.values), pi_prof.values, 0.0)
    goodD = np.isfinite(D_prof.values) & (D_prof.values > 0)
    if np.any(goodD):
        Dq = np.interp(centers, np.asarray(D_prof.levels)[goodD], np.asarray(D_prof.values)[goodD])
    else:
        Dq = np.full_like(centers, np.nan)
    return centers, pi, Dq, pi_prof, D_prof


def _mfpt_from_profiles(centers, pi, Dq, at):
    """Committor-coordinate MFPTs ``∫ M(q')/(D_q π) dq'`` (A->B and B->A)."""
    D_safe = np.where(np.isfinite(Dq) & (Dq > 0), Dq, 0.0)
    dq = float(centers[1] - centers[0])
    M_A = cumulative_trapezoid(pi, dx=dq, initial=0.0)
    M_B = cumulative_trapezoid(pi[::-1], dx=dq, initial=0.0)[::-1]
    good = (pi > 0) & (D_safe > 0)
    integ_AB = np.zeros_like(centers)
    integ_BA = np.zeros_like(centers)
    integ_AB[good] = M_A[good] / (D_safe[good] * pi[good])
    integ_BA[good] = M_B[good] / (D_safe[good] * pi[good])
    if at is None:
        mask = np.ones_like(centers, dtype=bool)
    elif _is_range(at):
        mask = (centers >= float(at[0])) & (centers <= float(at[1]))
    else:
        raise ValueError("MFPT/harmonic reduction needs at=None or (lo, hi)")
    return float(np.sum(integ_AB[mask]) * dq), float(np.sum(integ_BA[mask]) * dq)


def dirichlet_rate(
    committor,
    samples,
    *,
    D,
    at=None,
    sample_weights=None,
    in_A=None,
    in_B=None,
    n_bins=200,
):
    """Variational Dirichlet-form rate ν_R = ⟨D |∇q̄|²⟩_π.

    Args:
        committor: callable ``q(x)``.
        samples: ``(N, dim)`` static ensemble.
        D: scalar, callable ``level -> D``, or a diffusion :class:`Profile`.
        at: ``None`` -> volume integral D⟨|∇q̄|²⟩_π; ``(lo, hi)`` -> bin-free
            co-area plateau flux over that committor range (requires ``lo < hi``);
            ``"auto"`` -> the flux-flatness plateau located by
            :func:`find_plateau`; scalar -> co-area flux density Φ(q*) at the point.
        sample_weights, in_A, in_B: weighting and basin masks for populations.
        n_bins: resolution of the Φ(c) diagnostic profile (the plateau flux
            itself is bin-free).

    Returns:
        dict ``{nu_R, rho_A, rho_B, k_AB, k_BA, kind, ...}``. The ``kind`` key
        ("auto"/"full"/"range"/"point") labels which form was returned; range and
        auto add flat ``plateau``/``plateau_flatness`` (auto also ``plateau_auto``/
        ``plateau_ok``), matching :func:`tpt_rate`.
    """
    levels, contrib = _coarea_contributions(committor, samples, D, sample_weights)
    rho_A, rho_B = _populations(committor, samples, sample_weights, in_A, in_B)
    if at is None:
        nu_R = float(np.sum(contrib))  # volume integral D⟨|∇q̄|²⟩_π
        return _rate_dict(rho_A, rho_B, nu_R=nu_R, kind="full")
    # The auto / range / point forms all read the Φ(c) diagnostic profile; build
    # it once here rather than once per branch.
    centers, Phi, counts = _coarea_profile(levels, contrib, n_bins)
    if at == "auto":
        pw = find_plateau(Profile(levels=centers, values=Phi, counts=counts))
        nu_R = _plateau_flux(levels, contrib, pw.lo, pw.hi)
        # Use the plateau's own flatness (consistent with the window find_plateau
        # selected) instead of recomputing on a differently-filtered subset.
        report = {"plateau": (pw.lo, pw.hi), "plateau_flatness": pw.flatness}
        return _rate_dict(
            rho_A, rho_B, nu_R=nu_R, kind="auto", plateau_auto=True, plateau_ok=pw.ok, **report
        )
    if _is_range(at):
        lo, hi = float(at[0]), float(at[1])
        nu_R = _plateau_flux(levels, contrib, lo, hi)
        report = _plateau_report(centers, Phi, counts, lo, hi)
        return _rate_dict(rho_A, rho_B, nu_R=nu_R, kind="range", **report)
    good = counts > 0
    nu_R = float(np.interp(float(at), centers[good], Phi[good])) if np.any(good) else float("nan")
    return _rate_dict(rho_A, rho_B, nu_R=nu_R, kind="point", q_star=float(at))


def tpt_rate(
    committor,
    samples,
    *,
    D,
    at=(0.3, 0.7),
    sample_weights=None,
    in_A=None,
    in_B=None,
    n_bins=200,
):
    """Transition-path-theory rate from the reactive flux through a surface.

    Args:
        committor, samples, D: see :func:`reactive_flux`. Pass a
            :class:`~sliced_committor.rates.quantities.BridgeD` (from
            :func:`saddle_bridge_D`) as ``D`` for the calibrated geometric rate.
        at: ``(lo, hi)`` plateau (default, requires ``lo < hi``) -> bin-free
            plateau-MC flux plus a flatness diagnostic; ``"auto"`` -> the
            flux-flatness plateau located by :func:`find_plateau` (adds
            ``plateau_auto``/``plateau_ok``); scalar -> Φ(q*) at a single surface.
        sample_weights, in_A, in_B: populations weighting/masks.
        n_bins: resolution of the Φ(c) diagnostic profile (the plateau flux
            itself is bin-free).

    Returns:
        dict ``{nu_R, rho_A, rho_B, k_AB, k_BA, ...}``.
    """
    # Compute the co-area contributions once and derive both the Φ(c) profile
    # and the plateau flux from it (avoids two reactive_flux passes / autodiff).
    levels, contrib = _coarea_contributions(committor, samples, D, sample_weights)
    centers, Phi, counts = _coarea_profile(levels, contrib, n_bins)
    rho_A, rho_B = _populations(committor, samples, sample_weights, in_A, in_B)
    extra = {}
    if at == "auto":
        pw = find_plateau(Profile(levels=centers, values=Phi, counts=counts))
        lo, hi = pw.lo, pw.hi
        nu_R = _plateau_flux(levels, contrib, lo, hi)
        # Plateau's own flatness, consistent with the window find_plateau picked.
        extra = {
            "plateau": (lo, hi),
            "plateau_flatness": pw.flatness,
            "plateau_auto": True,
            "plateau_ok": pw.ok,
        }
    elif _is_range(at):
        lo, hi = float(at[0]), float(at[1])
        nu_R = _plateau_flux(levels, contrib, lo, hi)
        extra.update(_plateau_report(centers, Phi, counts, lo, hi))
    else:
        prof = Profile(levels=centers, values=Phi, counts=counts, name="committor")
        nu_R = value_at(prof, at)
    return _rate_dict(rho_A, rho_B, nu_R=nu_R, **extra)


def berezhkovskii_szabo_rate(
    committor,
    samples,
    trajectory,
    *,
    dt,
    at=0.5,
    mode="local",
    sample_weights=None,
    window_ids=None,
    in_A=None,
    in_B=None,
    n_bins=200,
    lag=None,
    lag_candidates=_LAG_CANDIDATES,
    min_count=5,
    coordinate=None,
    traj_coordinate=None,
    diffusion_method="kramers_moyal",
):
    """Berezhkovskii-Szabo rate on the committor coordinate.

    The committor-coordinate MFPT formula of Berezhkovskii & Szabo
    (first author A. Berezhkovskii): position-dependent diffusion D(q) from the
    trajectory and equilibrium density π(q) from the ensemble.

    This is a thin named alias for :func:`committor_rate`: ``mode="local"`` maps
    to ``reduction="local"`` (ν_R = D(q*)·π(q*) at the level ``at``) and
    ``mode="mfpt"`` to ``reduction="harmonic"`` (k = 1/MFPT, ``at`` is ``None``
    for the full [0,1] or a ``(lo, hi)`` window). Kept as a recognizable entry
    point; see :func:`committor_rate` for the full reduction set.

    Args:
        committor: callable ``q(x)``.
        samples: ``(N, dim)`` static ensemble (for π and populations).
        trajectory: ``(T, dim)`` time-ordered frames (for D).
        dt: trajectory frame spacing.
        mode: ``"local"`` or ``"mfpt"`` (see above).
        (remaining args forwarded verbatim to :func:`committor_rate`.)

    Returns:
        dict with ``k_AB``, ``k_BA`` (and ``nu_R`` in local mode / ``mfpt_*`` in
        mfpt mode), plus ``rho_A``, ``rho_B`` and ``mode`` (back-compat label).
    """
    reduction = {"local": "local", "mfpt": "harmonic"}.get(mode)
    if reduction is None:
        raise ValueError(f"unknown mode={mode!r}; use 'local' or 'mfpt'")
    out = committor_rate(
        committor,
        samples,
        trajectory,
        dt=dt,
        reduction=reduction,
        at=at,
        sample_weights=sample_weights,
        window_ids=window_ids,
        in_A=in_A,
        in_B=in_B,
        n_bins=n_bins,
        lag=lag,
        lag_candidates=lag_candidates,
        min_count=min_count,
        coordinate=coordinate,
        traj_coordinate=traj_coordinate,
        diffusion_method=diffusion_method,
    )
    out["mode"] = mode
    return out


def _local_curvature(centers, F, target, window):
    """Quadratic-fit second derivative of F near its extremum in ``window``.

    Returns ``(level_star, F_star, F_dd)``. ``target`` is "min" or "max".
    """
    lo, hi = window
    band = (centers >= lo) & (centers <= hi) & np.isfinite(F)
    if not np.any(band):
        return np.nan, np.nan, np.nan
    cb, Fb = centers[band], F[band]
    i = int(np.argmin(Fb)) if target == "min" else int(np.argmax(Fb))
    level_star, F_star = float(cb[i]), float(Fb[i])
    lo2, hi2 = max(0, i - 3), min(cb.shape[0], i + 4)
    if hi2 - lo2 < 3:
        return level_star, F_star, np.nan
    coef = np.polyfit(cb[lo2:hi2], Fb[lo2:hi2], 2)
    return level_star, F_star, float(2.0 * coef[0])


def kramers_rate(
    committor,
    samples,
    trajectory,
    *,
    dt,
    sample_weights=None,
    in_A=None,
    in_B=None,
    n_bins=200,
    coordinate=None,
    traj_coordinate=None,
    lag=None,
    lag_candidates=_LAG_CANDIDATES,
    window_ids=None,
    min_count=5,
    diffusion_method="kramers_moyal",
):
    """Overdamped Kramers barrier-crossing rate estimate.

    Uses the free energy ``F = -ln π`` (β=1 library convention) along the
    committor (or a CV) and the diffusion D at the barrier:

        k = (D(q‡) / 2π) · sqrt(|F''_well| · |F''_‡|) · exp(-(F‡ - F_well)).

    A basic harmonic estimate; for the rigorous diffusive result prefer
    :func:`berezhkovskii_szabo_rate`. ``k_AB`` uses the A-side well (low-level),
    ``k_BA`` the B-side well. Most meaningful on a physical reaction coordinate
    with interior free-energy wells -- pass ``coordinate=`` (ensemble) and
    ``traj_coordinate=`` (trajectory); along the bare committor the wells sit at
    the q=0/1 edges and the harmonic fit is only a rough proxy.

    Returns:
        dict with ``k_AB``, ``k_BA``, ``rho_A``, ``rho_B``, ``nu_R`` (= k_AB·ρ_A),
        and barrier diagnostics.
    """
    pi_prof, D_prof = _density_and_diffusion(
        committor,
        samples,
        trajectory,
        dt=dt,
        n_bins=n_bins,
        sample_weights=sample_weights,
        window_ids=window_ids,
        coordinate=coordinate,
        traj_coordinate=traj_coordinate,
        lag=lag,
        lag_candidates=lag_candidates,
        min_count=min_count,
        diffusion_method=diffusion_method,
    )
    centers = pi_prof.levels
    pi = pi_prof.values
    F = -np.log(np.where(pi > 0, pi, np.nan))
    span = (float(centers[0]), float(centers[-1]))
    rng = span[1] - span[0]

    q_bar, F_barrier, Fdd_barrier = _local_curvature(
        centers, F, "max", (span[0] + 0.2 * rng, span[0] + 0.8 * rng)
    )
    _, F_wA, Fdd_wA = _local_curvature(centers, F, "min", (span[0], span[0] + 0.4 * rng))
    _, F_wB, Fdd_wB = _local_curvature(centers, F, "min", (span[0] + 0.6 * rng, span[1]))
    D_barrier = value_at(D_prof, q_bar) if np.isfinite(q_bar) else float("nan")

    def _k(F_well, Fdd_well, dF):
        if not (np.isfinite(D_barrier) and np.isfinite(Fdd_well) and np.isfinite(Fdd_barrier)):
            return float("nan")
        # Use curvature magnitudes: along the committor the "wells" can sit at
        # the q=0/1 edges where a one-sided fit may give the wrong sign. Kramers
        # is best applied on a physical CV (pass coordinate=) with interior wells.
        curv = abs(Fdd_well) * abs(Fdd_barrier)
        if curv <= 0 or D_barrier <= 0:
            return float("nan")
        return float(D_barrier / (2.0 * np.pi) * np.sqrt(curv) * np.exp(-dF))

    k_AB = _k(F_wA, Fdd_wA, F_barrier - F_wA)
    k_BA = _k(F_wB, Fdd_wB, F_barrier - F_wB)
    rho_A, rho_B = _populations(committor, samples, sample_weights, in_A, in_B)
    nu_R = k_AB * rho_A if np.isfinite(k_AB) else float("nan")
    return {
        "nu_R": float(nu_R),
        "rho_A": float(rho_A),
        "rho_B": float(rho_B),
        "k_AB": float(k_AB),
        "k_BA": float(k_BA),
        "q_barrier": float(q_bar),
        "delta_F_AB": float(F_barrier - F_wA),
        "delta_F_BA": float(F_barrier - F_wB),
        "D_barrier": float(D_barrier),
    }


def committor_rate(
    committor,
    samples,
    trajectory,
    *,
    dt,
    reduction="harmonic",
    at=None,
    sample_weights=None,
    window_ids=None,
    in_A=None,
    in_B=None,
    n_bins=200,
    lag=None,
    lag_candidates=_LAG_CANDIDATES,
    min_count=5,
    coordinate=None,
    traj_coordinate=None,
    diffusion_method="kramers_moyal",
):
    """Unified committor-coordinate reaction rate from the pair ``{D_q(q), π(q)}``.

    Every committor-coordinate rate is a functional of the same coordinate-
    INVARIANT pair -- the committor-coordinate diffusion ``D_q(q)`` (from the
    trajectory) and the committor density ``π(q)`` (from the ensemble) -- with
    the local reactive flux ``ν_R(q) = D_q(q)·π(q)``. The ``reduction`` selects
    the functional (for the exact committor ``D_q·π`` is constant and all agree;
    the spread on an approximate committor is a quality diagnostic):

    * ``"arithmetic"`` / ``"dirichlet"`` -- ``ν_R = ∫₀¹ D_q π dq = ⟨D_q⟩_π``.
      The variational Dirichlet form; equals the feature-space ``⟨D|∇q̄|²⟩_ρ``
      EXACTLY but needs no gradient and no length-scale D, so it is coordinate-
      invariant (cartesian, torsions, TICA, ...). π-weighted, so basin-heavy.
    * ``"plateau"`` / ``"tpt"`` -- ``ν_R = median_{q∈[lo,hi]} D_q π`` (default
      ``at=(0.3, 0.7)``). Transition-region flux, robust to basin noise.
    * ``"local"`` -- ``ν_R = D_q(q*)·π(q*)`` at ``at`` (default 0.5).
    * ``"harmonic"`` / ``"mfpt"`` (default) -- ``k = 1/∫₀¹ dq/(D_q π)``; the
      exact 1-D Smoluchowski rate, bottleneck-dominated, the most physical.

    This is the committor-coordinate sibling of :func:`dirichlet_rate` /
    :func:`tpt_rate` (which take a length-scale ``D`` and the feature-space
    ``|∇q̄|²`` -- use those when a TRUSTED configurational D is known a priori).

    Args:
        committor, samples, trajectory, dt: as in :func:`berezhkovskii_szabo_rate`.
        reduction: which functional of ``{D_q, π}`` (see above).
        at: reduction-specific -- ``(lo, hi)`` plateau / MFPT window, ``"auto"``
            for the flux-flatness plateau (``reduction="plateau"`` only, via
            :func:`find_plateau`), or a scalar level for ``"local"`` (defaults:
            plateau ``(0.3, 0.7)``, local ``0.5``, arithmetic/harmonic full
            ``[0, 1]``).
        diffusion_method: ``"kramers_moyal"`` or ``"hummer"`` (lag-robust;
            needs ``window_ids``).
        (remaining args forwarded to :func:`density` / :func:`diffusion_coefficient`.)

    Returns:
        dict ``{nu_R, rho_A, rho_B, k_AB, k_BA, reduction, ...}`` (``nu_R`` is NaN
        for the harmonic reduction, which yields ``mfpt_AB/BA`` + ``k`` directly).
    """
    centers, pi, Dq, pi_prof, D_prof = _committor_profiles(
        committor,
        samples,
        trajectory,
        dt=dt,
        n_bins=n_bins,
        sample_weights=sample_weights,
        window_ids=window_ids,
        coordinate=coordinate,
        traj_coordinate=traj_coordinate,
        lag=lag,
        lag_candidates=lag_candidates,
        min_count=min_count,
        diffusion_method=diffusion_method,
    )
    rho_A, rho_B = _populations(committor, samples, sample_weights, in_A, in_B)
    red = reduction.lower()

    if red in ("harmonic", "mfpt"):
        mfpt_AB, mfpt_BA = _mfpt_from_profiles(centers, pi, Dq, at)
        return _rate_dict(
            rho_A,
            rho_B,
            k_AB=_inv_mfpt(mfpt_AB),
            k_BA=_inv_mfpt(mfpt_BA),
            mfpt_AB=mfpt_AB,
            mfpt_BA=mfpt_BA,
            reduction="harmonic",
        )

    flux = Dq * pi  # ν_R(q) = D_q(q) π(q)
    dq = float(centers[1] - centers[0])
    finite = np.isfinite(flux)

    if red in ("arithmetic", "dirichlet"):
        nu_R = float(np.sum(np.where(finite, flux, 0.0)) * dq)  # ∫ D_q π dq
        return _rate_dict(rho_A, rho_B, nu_R=nu_R, reduction="arithmetic")

    if red in ("plateau", "tpt"):
        extra = {}
        if at == "auto":
            pw = find_plateau(Profile(levels=centers, values=flux, counts=pi_prof.counts))
            lo, hi = pw.lo, pw.hi
            extra = {"plateau_auto": True, "plateau_flatness": pw.flatness, "plateau_ok": pw.ok}
        else:
            lo, hi = (0.3, 0.7) if at is None else (float(at[0]), float(at[1]))
        sel = finite & (centers >= lo) & (centers <= hi) & (pi > 0)
        nu_R = float(np.median(flux[sel])) if np.any(sel) else float("nan")
        if at != "auto":
            # Explicit-range flatness (auto already reports the plateau's own),
            # so plateau_flatness is present for every plateau call -- matching
            # dirichlet_rate / tpt_rate.
            vals = flux[sel]
            extra["plateau_flatness"] = (
                float(np.std(vals) / max(abs(np.mean(vals)), 1e-30)) if vals.size else float("nan")
            )
        return _rate_dict(rho_A, rho_B, nu_R=nu_R, reduction="plateau", plateau=(lo, hi), **extra)

    if red == "local":
        q_star = 0.5 if at is None else float(at)
        D_at = value_at(D_prof, q_star)
        pi_at = value_at(pi_prof, q_star)
        nu_R = D_at * pi_at
        return _rate_dict(
            rho_A,
            rho_B,
            nu_R=nu_R,
            reduction="local",
            D_at_q_star=float(D_at),
            pi_at_q_star=float(pi_at),
            q_star=q_star,
        )

    raise ValueError(
        f"unknown reduction={reduction!r}; use 'arithmetic'/'dirichlet', "
        f"'plateau'/'tpt', 'local', or 'harmonic'/'mfpt'"
    )
