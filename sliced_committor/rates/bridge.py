"""Principled CV->committor diffusion bridge (v2): position-dependent, regressed, with UQ.

Additive, opt-in extensions to the single-scalar :func:`mapped_committor_diffusion`
that sharpen the change of variables ``D_q(q) = D_s * <|grad q|^2>_q / <|grad s|^2>``
along four axes, validated on the known-D0 analytic toys (see
``experiments/bridge_v2.py`` for the validation ladder). Nothing here changes the
behaviour of the existing bridge; every function is new.

First principles
----------------
Both diffusivities are the SAME configuration-space tensor ``D(x)`` projected onto
different scalar coordinates (Markovian projection / co-area identity):

    D_eff(z) = < (grad z)^T D(x) (grad z) >_{z(x)=z}     (conditional iso-z average)

so measuring ``D_s(s)`` on the confined (Markovian) umbrella CV constrains ``D_q(q)``
on the committor coordinate (which has no diffusive plateau and cannot be measured
directly). The modes are a hierarchy of structural ansaetze on ``D(x)``:

    scalar   : D(x)=D0*M0          -- one scalar D0        (mapped_committor_diffusion)
    field    : D(x)=D0(x)*M0       -- position-dependent    (mapped_committor_diffusion_field)
    reparam  : q=f(s) deterministic-> D_q(q)=D_s(s)(dq/ds)^2 (mapped_committor_diffusion_reparam)

Key findings baked into the design:
  * The projected Hummer ``D_s`` (Var/tau_int, memory-integrated) is the RATE-relevant
    diffusion; it equals the isotropic configurational D0 only when the CV is a good
    reaction coordinate. A short-lag/Bayesian propagator recovers the LOCAL D instead,
    which overestimates the rate for a sub-optimal CV -- so the Bayesian estimator is
    exposed as a CV-memory DIAGNOSTIC (local/effective ratio), not the rate driver.
  * The reactive flux nu_R(q)=D_q(q)*pi(q) is constant in q for the EXACT committor
    (TPT); its coefficient of variation is a committor-quality diagnostic, and the
    reductions bracket ``[harmonic, arithmetic]`` (bottleneck / Dirichlet upper bound)
    is a committor-quality-aware uncertainty band.

Reuses: Profile, value_at, density, reactive_flux, cumulative_trapezoid,
_estimate_D_hummer, _estimate_D_q_stratified, _mfpt_from_profiles, mapped_committor_diffusion.
"""

from __future__ import annotations

from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from ._coordinate import Profile, value_at
from .quantities import _estimate_D_hummer, density, mapped_committor_diffusion, reactive_flux

_TINY = 1e-30


# ===========================================================================
# Gradients (autodiff)
# ===========================================================================
def committor_values_and_grad_sq(committor: Callable, samples) -> tuple[np.ndarray, np.ndarray]:
    """Per-sample committor value q_n (clipped to [0,1]) and |grad q_n|^2 via autodiff."""
    sj = jnp.asarray(samples)
    vals, grads = jax.vmap(jax.value_and_grad(committor))(sj)
    q = np.clip(np.asarray(vals, dtype=np.float64), 0.0, 1.0)
    gsq = np.asarray(jnp.sum(grads**2, axis=-1), dtype=np.float64)
    return q, gsq


def cv_values_and_grad_sq(cv_fn: Callable, samples, *, metric=None):
    """Per-sample CV value and metric-weighted |grad s|^2 via autodiff.

    Usable only when ``s`` is a differentiable function of the committor's feature
    space (e.g. a toy where s=x[...,0]); for a CV that is not (chignolin's native-
    contact Q vs a dihedral committor) use :func:`cv_feature_grad_sq_linear` or the
    reparametrisation bridge, which needs no feature-space gradient of s.
    """
    sj = jnp.asarray(samples)
    vals, grads = jax.vmap(jax.value_and_grad(cv_fn))(sj)
    g2 = grads**2
    if metric is not None:
        g2 = g2 * jnp.asarray(metric)[None, :]
    return np.asarray(vals, dtype=np.float64), np.asarray(jnp.sum(g2, axis=-1), dtype=np.float64)


def cv_feature_grad_sq_linear(cv_values, features, weights) -> float:
    """Ridge linear-response <|grad s|^2> = |a|^2 with s ~ a.x (the library default)."""
    X = np.asarray(features, dtype=np.float64)
    s = np.asarray(cv_values, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    w = w / w.sum() if w.sum() > 0 else np.full(s.shape[0], 1.0 / s.shape[0])
    xbar = (w[:, None] * X).sum(axis=0)
    Xc = X - xbar
    sc = s - float((w * s).sum())
    d = Xc.shape[1]
    A = Xc.T @ (w[:, None] * Xc)
    lam = 1e-6 * (float(np.trace(A)) / max(d, 1) + 1e-30)
    coef = np.linalg.solve(A + lam * np.eye(d), Xc.T @ (w * sc))
    return float(coef @ coef)


# ===========================================================================
# Conditional-expectation regression  E[y | x]  (landscape construction)
# ===========================================================================
class Regression(NamedTuple):
    value: np.ndarray
    slope: np.ndarray
    stderr: np.ndarray
    neff: np.ndarray


def _silverman_bandwidth(x, w):
    mu = float(np.average(x, weights=w))
    sd = np.sqrt(max(float(np.average((x - mu) ** 2, weights=w)), 1e-12))
    n_eff = (w.sum() ** 2) / max((w * w).sum(), _TINY)
    return float(max(0.9 * sd * n_eff ** (-0.2), 1e-3))


def conditional_mean(
    x,
    y,
    weights,
    query,
    *,
    method="local_linear",
    bandwidth=None,
    n_knots=25,
    degree=3,
    lam=None,
    min_neff=8.0,
) -> Regression:
    """MBAR-weighted conditional mean E[y|x] at ``query`` (Axis 3 landscape estimator).

    method: "hist" | "kernel" (Nadaraya-Watson) | "local_linear" (LOESS deg-1,
    boundary-bias-corrected, default; also returns the local slope) | "pspline"
    (penalized B-spline, GCV lambda; needs scipy). Replaces the ratio-of-histograms
    Phi/pi, which is unstable in the low-density barrier.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    w = np.ones_like(x) if weights is None else np.asarray(weights, dtype=np.float64)
    q = np.atleast_1d(np.asarray(query, dtype=np.float64))

    if method == "pspline":
        return _pspline_regression(x, y, w, q, n_knots=n_knots, degree=degree, lam=lam)

    if bandwidth is None:
        bandwidth = _silverman_bandwidth(x, w)
    h = float(bandwidth)
    val = np.full(q.shape, np.nan)
    slope = np.full(q.shape, np.nan)
    serr = np.full(q.shape, np.nan)
    neff = np.zeros(q.shape)

    if method == "hist":
        n = q.shape[0]
        edges = np.linspace(0.0, 1.0, n + 1)
        idx = np.clip(np.digitize(np.clip(x, 0.0, 1.0 - 1e-12), edges) - 1, 0, n - 1)
        for i in range(n):
            m = idx == i
            if not np.any(m):
                continue
            wi = w[m]
            sw = wi.sum()
            if sw <= 0:
                continue
            neff[i] = (sw * sw) / max((wi * wi).sum(), _TINY)
            val[i] = float(np.average(y[m], weights=wi))
            serr[i] = float(
                np.sqrt(np.average((y[m] - val[i]) ** 2, weights=wi) / max(neff[i], 1.0))
            )
        return Regression(val, slope, serr, neff)

    for i, x0 in enumerate(q):
        u = (x - x0) / h
        k = np.exp(-0.5 * u * u) * w
        sk = k.sum()
        if sk <= 0:
            continue
        ne = (sk * sk) / max((k * k).sum(), _TINY)
        neff[i] = ne
        if method == "kernel" or ne < min_neff:
            val[i] = float((k * y).sum() / sk)
        elif method == "local_linear":
            dx = x - x0
            Sw = sk
            Swx = float((k * dx).sum())
            Swxx = float((k * dx * dx).sum())
            Swy = float((k * y).sum())
            Swxy = float((k * dx * y).sum())
            det = Sw * Swxx - Swx * Swx
            if abs(det) <= _TINY * (Sw * Swxx + _TINY):
                val[i] = Swy / Sw
                slope[i] = 0.0
            else:
                val[i] = (Swxx * Swy - Swx * Swxy) / det
                slope[i] = (Sw * Swxy - Swx * Swy) / det
        else:
            raise ValueError(f"unknown method {method!r}")
        var = float((k * (y - val[i]) ** 2).sum() / sk)
        serr[i] = float(np.sqrt(var / max(ne, 1.0)))
    return Regression(val, slope, serr, neff)


def _pspline_regression(x, y, w, query, *, n_knots, degree, lam, order=2) -> Regression:
    from scipy.interpolate import BSpline

    lo, hi = 0.0, 1.0
    t = np.r_[[lo] * degree, np.linspace(lo, hi, n_knots), [hi] * degree]
    B = BSpline.design_matrix(np.clip(x, lo, hi), t, degree).toarray()
    Bq = BSpline.design_matrix(np.clip(query, lo, hi), t, degree).toarray()
    m = B.shape[1]
    D = np.eye(m)
    for _ in range(order):
        D = np.diff(D, axis=0)
    P = D.T @ D
    W = w / max(w.sum(), _TINY)
    A0 = (B.T * W[None, :]) @ B
    rhs = (B.T * W[None, :]) @ y
    if lam is None:
        n = x.shape[0]
        best, best_c = np.inf, None
        for lam_try in np.logspace(-6, 3, 24):
            c = np.linalg.solve(A0 + lam_try * P, rhs)
            tr = float(np.trace(np.linalg.solve(A0 + lam_try * P, A0)))
            rss = float(np.sum(W * (y - B @ c) ** 2)) * n
            gcv = rss / max((1.0 - tr / n) ** 2, 1e-12)
            if gcv < best:
                best, best_c = gcv, c
        c = best_c
    else:
        c = np.linalg.solve(A0 + float(lam) * P, rhs)
    spl = BSpline(t, c, degree)
    return Regression(
        np.asarray(Bq @ c),
        np.asarray(spl.derivative()(np.clip(query, lo, hi))),
        np.full(query.shape, np.nan),
        np.full(query.shape, np.nan),
    )


# ===========================================================================
# CV-space D_s(s) profiles + diagnostics
# ===========================================================================
def hummer_Ds_profile(s_values, window_ids, dt, *, min_count=5):
    """Per-window Hummer D_s(s) profile (memory-integrated, rate-relevant). Returns (s, D_s)."""
    prof = _estimate_D_hummer(
        np.asarray(s_values, float),
        np.asarray(window_ids).reshape(-1),
        float(dt),
        int(min_count),
        "cv",
    )
    sc = np.asarray(prof.levels, float)
    dv = np.asarray(prof.values, float)
    o = np.argsort(sc)
    return sc[o], dv[o]


def Ds_interpolator(s_centers, D_s):
    """Linear interpolator of a D_s(s) profile over finite positive points."""
    good = np.isfinite(s_centers) & np.isfinite(D_s) & (D_s > 0)
    sc = np.asarray(s_centers)[good]
    dv = np.asarray(D_s)[good]

    def fn(s_query):
        return np.interp(np.asarray(s_query, float), sc, dv, left=dv[0], right=dv[-1])

    return fn


def smooth_Ds_profile(s_centers, D_s, query, *, counts=None, method="local_linear", bandwidth=None):
    """Smooth the sparse per-window D_s(s) into a profile at ``query`` (log-space regression)."""
    s_centers = np.asarray(s_centers, float)
    D_s = np.asarray(D_s, float)
    good = np.isfinite(s_centers) & np.isfinite(D_s) & (D_s > 0)
    w = None if counts is None else np.asarray(counts, float)[good]
    reg = conditional_mean(
        s_centers[good],
        np.log(D_s[good]),
        w,
        np.asarray(query, float),
        method=method,
        bandwidth=bandwidth,
    )
    return np.exp(reg.value)


# ===========================================================================
# Mapping modes -> Profile of D_q(q)
# ===========================================================================
def _profile(grid, values, counts=None):
    if counts is None:
        counts = np.full(np.asarray(grid).shape, 100, dtype=np.int64)
    return Profile(
        levels=np.asarray(grid, float),
        values=np.asarray(values, float),
        counts=np.asarray(counts),
        name="committor",
    )


def committor_grad_profile(
    committor, samples, *, sample_weights=None, n_bins=200, method="local_linear", bandwidth=None
):
    """E[|grad q|^2 | q] on the [0,1] grid by regression (Axis 3; replaces Phi/pi)."""
    qx, gsq = committor_values_and_grad_sq(committor, samples)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    grid = 0.5 * (edges[:-1] + edges[1:])
    reg = conditional_mean(qx, gsq, sample_weights, grid, method=method, bandwidth=bandwidth)
    return _profile(grid, reg.value), reg


def mapped_committor_diffusion_field(
    committor,
    samples,
    *,
    D_s_profile,
    cv_grad_sq,
    s_values,
    sample_weights=None,
    n_bins=200,
    method="local_linear",
    bandwidth=None,
    at=None,
):
    """Position-dependent-D0 bridge: D_q(q) = E[ D0(s(x)) * |grad q|^2 | q ].

    D0(s) = D_s(s)/cv_grad_sq, with ``D_s_profile`` a ``(s_centers, D_s)`` pair (e.g.
    :func:`hummer_Ds_profile`) and ``cv_grad_sq`` a scalar <|grad s|^2> or callable
    g_s(s). Uses the joint (s,q) sample distribution. Reduces to the scalar bridge
    when D_s and g_s are constant; removes the barrier-median broadcast bias.
    """
    s_c, D_s = D_s_profile
    Ds_fn = Ds_interpolator(s_c, D_s)
    s_values = np.asarray(s_values, float)
    Ds_n = np.asarray(Ds_fn(s_values), float)
    gs_n = cv_grad_sq(s_values) if callable(cv_grad_sq) else float(cv_grad_sq)
    D0_n = Ds_n / np.asarray(gs_n, float)
    qx, gsq = committor_values_and_grad_sq(committor, samples)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    grid = 0.5 * (edges[:-1] + edges[1:])
    reg = conditional_mean(qx, D0_n * gsq, sample_weights, grid, method=method, bandwidth=bandwidth)
    prof = _profile(grid, reg.value)
    return prof if at is None else value_at(prof, at)


def mapped_committor_diffusion_reparam(
    committor,
    samples,
    *,
    D_s_profile,
    s_values,
    sample_weights=None,
    n_bins=200,
    bandwidth=None,
    at=None,
):
    """Deterministic-reparametrisation bridge: D_q(q) = D_s(s(q)) * (dq/ds)^2.

    Uses the empirical monotone map s(q)=E[s|q] and slope ds/dq from a local-linear
    regression -- NO feature-space gradient of s needed (the clean route when s is not
    a differentiable function of the committor's feature space). Valid when q(s) is
    ~monotone; the weighted R^2(q,s) is returned for gating (use >~0.9).
    """
    s_c, D_s = D_s_profile
    Ds_fn = Ds_interpolator(s_c, D_s)
    qx, _ = committor_values_and_grad_sq(committor, samples)
    s_values = np.asarray(s_values, float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    grid = 0.5 * (edges[:-1] + edges[1:])
    reg = conditional_mean(
        qx, s_values, sample_weights, grid, method="local_linear", bandwidth=bandwidth
    )
    D_q = np.asarray(Ds_fn(reg.value), float) / np.maximum(reg.slope**2, _TINY)
    D_q = np.where(np.isfinite(D_q) & (reg.slope**2 > 1e-12), D_q, np.nan)
    prof = _profile(grid, D_q)
    r2 = _weighted_r2(qx, s_values, sample_weights)
    if at is None:
        return prof, {"r2_monotone": r2}
    return value_at(prof, at), {"r2_monotone": r2}


def _weighted_r2(x, y, w):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    w = np.ones_like(x) if w is None else np.asarray(w, float)
    w = w / max(w.sum(), _TINY)
    xb = float((w * x).sum())
    yb = float((w * y).sum())
    cov = float((w * (x - xb) * (y - yb)).sum())
    vx = float((w * (x - xb) ** 2).sum())
    vy = float((w * (y - yb) ** 2).sum())
    return float(cov * cov / max(vx * vy, _TINY))


# ===========================================================================
# Reductions, committor-quality bracket, constant-flux MLE
# ===========================================================================
def committor_populations(q_values, sample_weights=None):
    """Committor-weighted populations rho_A=<1-q>, rho_B=<q> (the TPT rate normaliser)."""
    q = np.clip(np.asarray(q_values, float), 0.0, 1.0)
    w = np.ones_like(q) if sample_weights is None else np.asarray(sample_weights, float)
    w = w / max(w.sum(), _TINY)
    return float(np.sum(w * (1.0 - q))), float(np.sum(w * q))


def flux_reductions(grid, pi, D_q, rho_A, rho_B, *, band=(0.3, 0.7), stderr_logflux=None):
    """All reductions of nu(q)=D_q*pi as RATES + committor-quality bracket/MLE.

    ``pi`` is the normalised committor density (int pi dq=1); rho_A/rho_B from
    :func:`committor_populations`. Returns arithmetic (Dirichlet upper bound),
    plateau-median, local, constant-flux MLE (each with k_AB=nu/rho_A, k_BA=nu/rho_B),
    the harmonic BS-MFPT (k=1/MFPT), the flux-constancy CV, and the [harmonic,
    arithmetic] bracket.
    """
    from .formulas import _mfpt_from_profiles

    grid = np.asarray(grid, float)
    pi = np.asarray(pi, float)
    D_q = np.asarray(D_q, float)
    dq = float(grid[1] - grid[0])
    nu = D_q * pi
    finite = np.isfinite(nu) & (nu > 0) & (pi > 0) & np.isfinite(D_q) & (D_q > 0)

    def to_k(v):
        return {
            "nu": float(v),
            "k_AB": float(v) / max(rho_A, _TINY),
            "k_BA": float(v) / max(rho_B, _TINY),
        }

    nu_arith = float(np.sum(np.where(finite, nu, 0.0)) * dq)
    lo, hi = band
    sel = finite & (grid >= lo) & (grid <= hi)
    nu_plateau = float(np.median(nu[sel])) if np.any(sel) else float("nan")
    gf, nf = grid[finite], nu[finite]
    nu_local = float(np.interp(0.5, gf, nf)) if gf.size >= 2 else float("nan")
    if np.any(sel):
        lognu = np.log(nu[sel])
        if stderr_logflux is not None:
            v = np.asarray(stderr_logflux, float)[sel]
            prec = 1.0 / np.maximum(v * v, _TINY)
        else:
            prec = np.ones(int(sel.sum()))
        nu_mle = float(np.exp(np.sum(prec * lognu) / max(prec.sum(), _TINY)))
        cv = float(np.std(nu[sel]) / max(abs(np.mean(nu[sel])), _TINY))
    else:
        nu_mle, cv = float("nan"), float("nan")

    mfpt_AB, mfpt_BA = _mfpt_from_profiles(grid, pi, np.where(finite, D_q, np.nan), None)
    k_harm_AB = 1.0 / max(mfpt_AB, _TINY)
    k_harm_BA = 1.0 / max(mfpt_BA, _TINY)
    return {
        "arithmetic": to_k(nu_arith),
        "plateau_median": to_k(nu_plateau),
        "local": to_k(nu_local),
        "const_flux_mle": to_k(nu_mle),
        "harmonic": {"k_AB": k_harm_AB, "k_BA": k_harm_BA, "mfpt_AB": mfpt_AB, "mfpt_BA": mfpt_BA},
        "flux_cv": cv,
        "bracket_k_AB": (k_harm_AB, nu_arith / max(rho_A, _TINY)),
        "nu_profile": nu,
    }


def constancy_reconstruction(grid, pi, nu_R):
    """Flux-constancy-constrained landscape D_q^cons(q) = nu_R / pi(q)."""
    pi = np.asarray(pi, float)
    return _profile(grid, np.where(pi > 0, float(nu_R) / np.maximum(pi, _TINY), np.nan))


# ===========================================================================
# Uncertainty: window block bootstrap of the barrier D_s
# ===========================================================================
def bootstrap_barrier_Ds(
    qx,
    cv_values,
    sample_weights,
    window_ids,
    dt,
    *,
    band=(0.3, 0.7),
    n_boot=1000,
    seed=0,
    min_count=5,
):
    """Window block bootstrap of the barrier-band Hummer D_s (statistical rate CI).

    The scalar bridge rate is linear in D_s, so k_boot/k_hat = D_s_boot/D_s_hat.
    Returns (D_hat, boot_samples, rel_ci_lo, rel_ci_hi).
    """
    from ..workflows.committor_rates import _hummer_cv_barrier_scalar

    qx = np.asarray(qx, float)
    cvp = np.asarray(cv_values, float)
    w = np.asarray(sample_weights, float)
    wid = np.asarray(window_ids).reshape(-1)
    uw = np.unique(wid)
    idx_by_w = {k: np.where(wid == k)[0] for k in uw}
    D_hat = _hummer_cv_barrier_scalar(qx, cvp, w, wid, dt, min_count=min_count)
    rng = np.random.default_rng(seed)
    K = uw.shape[0]
    boots = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.integers(0, K, size=K)
        idx = np.concatenate([idx_by_w[uw[p]] for p in pick])
        wl = np.concatenate([np.full(idx_by_w[uw[p]].shape[0], j) for j, p in enumerate(pick)])
        boots[b] = _hummer_cv_barrier_scalar(qx[idx], cvp[idx], w[idx], wl, dt, min_count=min_count)
    boots = boots[np.isfinite(boots)]
    lo, hi = np.percentile(boots, [2.5, 97.5]) / max(D_hat, _TINY)
    return float(D_hat), boots, float(lo), float(hi)


# ===========================================================================
# Bayesian Smoluchowski (F, D) diagnostic  (LOCAL D; a CV-memory cross-check)
# ===========================================================================
def bayesian_smoluchowski_diffusion(
    s_values,
    window_ids,
    window_centers,
    window_kappa,
    beta,
    dt,
    *,
    n_bins=25,
    lag=5,
    max_iter=300,
    smooth_logD=0.5,
    smooth_F=0.1,
):
    """ML Smoluchowski-propagator (F(s), D(s)) inference (Hummer 2005), a LOCAL-D diagnostic.

    Fits each biased window's lag-``lag`` transition matrix as expm(dt*lag*R_w) with a
    shared D(s) and per-window F_w = F + 0.5*beta*k_w*(s-c_w)^2. Returns
    (edge_centers, D_s, info). NOTE: this recovers the LOCAL short-lag diffusion, which
    for a sub-optimal CV differs from the memory-integrated Hummer Var/tau_int value that
    the rate needs -- use the ratio as a CV-memory diagnostic, not as the rate D_s.
    """
    from jax.scipy.linalg import expm
    from scipy.optimize import minimize

    s = np.asarray(s_values, float)
    wid = np.asarray(window_ids).reshape(-1)
    c_w = np.asarray(window_centers, float).reshape(-1)
    k_w = np.asarray(window_kappa, float).reshape(-1)
    L = int(lag)
    tau = float(dt) * L
    lo, hi = float(np.min(s)), float(np.max(s))
    edges = np.linspace(lo, hi, n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    dx = float(centers[1] - centers[0])
    uw = np.unique(wid)
    counts = np.zeros((len(uw), n_bins, n_bins))
    for wi, wv in enumerate(uw):
        xi = s[wid == wv]
        if xi.shape[0] <= L:
            continue
        bi = np.clip(np.digitize(xi, edges) - 1, 0, n_bins - 1)
        np.add.at(counts[wi], (bi[L:], bi[:-L]), 1.0)
    counts_j = jnp.asarray(counts)
    c_w_j = jnp.asarray(np.resize(c_w, len(uw)))
    k_w_j = jnp.asarray(np.resize(k_w, len(uw)))
    cen_j = jnp.asarray(centers)
    hist = np.maximum(
        np.bincount(np.clip(np.digitize(s, edges) - 1, 0, n_bins - 1), minlength=n_bins), 1.0
    )
    F0 = -np.log(hist / hist.sum())
    F0 -= F0.min()
    theta0 = np.concatenate([F0, np.full(n_bins - 1, np.log(max(np.var(s) / max(L, 1), 1e-6)))])

    def build_R(F_w, logD):
        D_edge = jnp.exp(logD)
        dF = F_w[1:] - F_w[:-1]
        up = D_edge / dx**2 * jnp.exp(-0.5 * dF)
        dn = D_edge / dx**2 * jnp.exp(0.5 * dF)
        R = jnp.zeros((n_bins, n_bins))
        idx = jnp.arange(n_bins - 1)
        R = R.at[idx + 1, idx].set(up)
        R = R.at[idx, idx + 1].set(dn)
        return R - jnp.diag(jnp.sum(R, axis=0))

    def neg_ll(theta):
        F = theta[:n_bins]
        logD = theta[n_bins:]

        def one(wi):
            F_w = F + 0.5 * beta * k_w_j[wi] * (cen_j - c_w_j[wi]) ** 2
            T = jnp.clip(expm(tau * build_R(F_w, logD)), 1e-300, 1.0)
            return jnp.sum(counts_j[wi] * jnp.log(T))

        ll = jnp.sum(jax.vmap(one)(jnp.arange(counts_j.shape[0])))
        pen = smooth_logD * jnp.sum(jnp.diff(logD, 2) ** 2) + smooth_F * jnp.sum(
            jnp.diff(F, 2) ** 2
        )
        return -ll + pen

    vg = jax.jit(jax.value_and_grad(neg_ll))

    def f(theta):
        v, g = vg(jnp.asarray(theta))
        return float(v), np.asarray(g, float)

    res = minimize(f, theta0, jac=True, method="L-BFGS-B", options={"maxiter": int(max_iter)})
    D_edge = np.exp(np.asarray(res.x[n_bins:]))
    return (
        0.5 * (centers[:-1] + centers[1:]),
        D_edge,
        {"success": bool(res.success), "nll": float(res.fun)},
    )
