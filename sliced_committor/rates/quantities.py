"""Static-ensemble quantities, and the two routes to a committor-space diffusion.

* :func:`density`: the equilibrium density of a coordinate, reweightable.
* :func:`basin_populations`: ``rho_A = E[1 - q]``, ``rho_B = E[q]``.
* :func:`committor_grad_sq`: the iso-committor mean squared gradient
  ``<grad q^T M grad q | q>`` by the co-area identity, one autodiff pass.
* :func:`committor_diffusion_from_cv`: the Jacobian map of a diffusion measured
  along a collective variable into committor space, the paper's route.
* :func:`linear_response_grad_sq`: the collective variable's mean squared
  gradient that the map divides by, with its caveat.
* :func:`committor_diffusion_from_cv_reparam`: the same map through the
  empirical monotone relation ``s(q)``, needing no gradient of ``s``.

Only :func:`committor_grad_sq` takes the committor function (it differentiates
it); everything else takes per-sample values.
"""

import jax
import jax.numpy as jnp
import numpy as np
from scipy.stats import rankdata

from ._coordinate import Profile
from ._numerics import density_histogram, normalize_weights

_TINY = 1e-30


# ---------------------------------------------------------------------------
# density and populations
# ---------------------------------------------------------------------------
def density(coordinate, *, sample_weights=None, n_bins=200, span=(0.0, 1.0)) -> Profile:
    """Equilibrium density of a coordinate, integrating to one over ``span``.

    Args:
        coordinate: ``(N,)`` per-sample values, ``q(samples)`` for the committor
            (values are clipped to ``span``).
        sample_weights: ``(N,)`` MBAR/WHAM weights reweighting to the target
            ensemble; the profile's ``counts`` stay unweighted.
        n_bins, span: the histogram; ``span`` defaults to the committor's
            ``[0, 1]``, pass ``(s.min(), s.max())`` for a collective variable.
    """
    lo, hi = float(span[0]), float(span[1])
    x = np.clip(np.asarray(coordinate, dtype=np.float64).reshape(-1), lo, hi)
    edges = np.linspace(lo, hi, n_bins + 1)
    pi, counts = density_histogram(x, edges, sample_weights)
    return Profile(0.5 * (edges[:-1] + edges[1:]), pi, counts)


def basin_populations(q_values, *, sample_weights=None, in_A=None, in_B=None):
    """``(rho_A, rho_B)`` with ``rho_B = E[q]`` and ``rho_A = 1 - rho_B``.

    The committor-weighted populations are the normaliser of the rate:
    ``k_AB = nu_R / rho_A``. Basin masks, when given, snap ``q`` to 0 on A and
    1 on B first, so a committor that does not reach its boundary values does
    not leak population.
    """
    q = np.clip(np.asarray(q_values, dtype=np.float64).reshape(-1), 0.0, 1.0)
    if in_A is not None:
        q = np.where(np.asarray(in_A, dtype=bool).reshape(-1), 0.0, q)
    if in_B is not None:
        q = np.where(np.asarray(in_B, dtype=bool).reshape(-1), 1.0, q)
    W = normalize_weights(sample_weights, q.shape[0])
    rho_B = float(np.sum(W * q))
    return 1.0 - rho_B, rho_B


# ---------------------------------------------------------------------------
# the iso-committor mean squared gradient (co-area)
# ---------------------------------------------------------------------------
def _grad_sq(grads, metric):
    """``|grad q|^2`` or the metric-weighted ``grad q^T M grad q``.

    ``metric=None`` returns the original expression verbatim, not an
    equivalent one: routing it through an einsum with an identity changes the
    float64 reduction order, and the published tolerances would pass while the
    plateau flux silently drifted.
    """
    if metric is None:
        return jnp.sum(grads**2, axis=-1)
    M = jnp.asarray(metric)
    if M.ndim == 1:
        return jnp.sum((grads**2) * M[None, :], axis=-1)
    if M.ndim == 2:
        return jnp.einsum("nd,de,ne->n", grads, M, grads)
    raise ValueError(f"metric must be None, (d,) diagonal or (d, d); got ndim {M.ndim}")


def committor_grad_sq(committor, samples, *, sample_weights=None, n_bins=200, metric=None):
    """The iso-committor mean squared gradient ``<grad q^T M grad q>_q`` on ``[0, 1]``.

    By the co-area formula the Dirichlet form splits into iso-committor
    surfaces, ``<|grad q|^2>_pi = int_0^1 pi(q) <|grad q|^2>_q dq``, so the
    profile is each bin's weighted sum of ``|grad q(x_n)|^2`` divided by its
    density. One vmapped value-and-gradient pass over the samples supplies both
    the levels and the gradients; the density is binned from the same levels.

    ``metric`` is None (the identity, the published ``M0 = I``), a ``(d,)``
    diagonal, or a ``(d, d)`` tensor, replacing ``|grad q|^2`` by
    ``grad q^T M grad q``. Whatever is passed here must also be used for the
    collective-variable denominator in :func:`committor_diffusion_from_cv`: the
    map is only bias-free when both mean squared gradients live in one metric.
    """
    vals, grads = jax.vmap(jax.value_and_grad(committor))(jnp.asarray(samples))
    g = np.asarray(_grad_sq(grads, metric), dtype=np.float64)
    levels = np.clip(np.asarray(vals, dtype=np.float64), 0.0, 1.0)
    W = normalize_weights(sample_weights, levels.shape[0])
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_idx = np.clip(np.digitize(np.clip(levels, 0.0, 1.0 - 1e-12), edges) - 1, 0, n_bins - 1)
    Phi = np.bincount(bin_idx, weights=W * g, minlength=n_bins) * n_bins
    pi, counts = density_histogram(levels, edges, sample_weights)
    values = np.where(pi > 0, Phi / np.where(pi > 0, pi, 1.0), np.nan)
    return Profile(0.5 * (edges[:-1] + edges[1:]), values, counts)


# ---------------------------------------------------------------------------
# the CV -> committor map
# ---------------------------------------------------------------------------
def committor_diffusion_from_cv(grad_sq_profile, *, D_s, cv_grad_sq) -> Profile:
    """Map a diffusion measured along a collective variable into committor space.

    The diffusion is honestly measurable along the coordinate the sampling
    biases (``s``, the umbrella variable; :func:`pooled_acf_diffusion`), while
    the rate needs it along the committor, where nothing confines the dynamics
    at the barrier. Both scalars are one configurational diffusion tensor
    contracted along two gradients. With ``D = D0 M0`` (one scale, ``M0 = I``
    in the published setting) they read ``D_s = D0 <|grad s|^2>`` and
    ``D_q(q) = D0 <|grad q|^2>_q``, and eliminating ``D0`` gives the map::

        D_q(q) = D_s <|grad q|^2>_q / <|grad s|^2>

    The single scalar ``D0 = D_s / <|grad s|^2>`` carries all the physical
    time; the shape of ``D_q`` is geometry, from :func:`committor_grad_sq`.
    Because the map is exactly linear in the numerator and inverse-linear in
    the denominator, both must be taken in the same feature space and metric.

    Args:
        grad_sq_profile: :func:`committor_grad_sq` of the fitted committor.
        D_s: the diffusion along ``s``, in ``(units of s)^2 / time``.
        cv_grad_sq: ``<|grad s|^2>`` in the feature space and metric of the
            numerator (:func:`linear_response_grad_sq`), or ``1.0`` when ``s``
            is itself a feature, or when a known configurational ``D0`` is
            assumed and passed as ``D_s``.
    """
    D_s = float(D_s)
    g_s = float(cv_grad_sq)
    if not (np.isfinite(D_s) and D_s > 0):
        raise ValueError(f"D_s must be finite and positive; got {D_s!r}")
    if not (np.isfinite(g_s) and g_s > 0):
        raise ValueError(f"cv_grad_sq must be finite and positive; got {cv_grad_sq!r}")
    D0 = D_s / g_s
    return Profile(
        np.asarray(grad_sq_profile.levels, dtype=np.float64),
        D0 * np.asarray(grad_sq_profile.values, dtype=np.float64),
        grad_sq_profile.counts,
    )


def linear_response_grad_sq(cv_values, features, sample_weights=None, *, metric=None) -> float:
    """``<|grad s|^2>`` of a collective variable by linear response, ``a^T M a``.

    Fits ``s ~ a . x + b`` by weighted, lightly ridged least squares and reads
    the gradient off the coefficient. It is exact when ``s`` is a linear
    function of the features (``s`` itself a feature gives 1). It is the
    denominator the published rates use, and its caveat matters: ``a`` is a
    best-linear-predictor coefficient, not a pointwise gradient, so the value
    depends on the fitting window rather than on the field. On the pooled
    chignolin data (630,063 frames, 86 torsion features) the global fit gives
    0.222 while per-umbrella-window fits give 0.0066, a 33x swing that a
    genuine pointwise average cannot produce, and the rate is linear in it.
    When ``s = f(x)`` is differentiable, differentiate it instead; when it is
    not, :func:`committor_diffusion_from_cv_reparam` needs no gradient of
    ``s`` at all.

    ``metric`` (None, ``(d,)`` or ``(d, d)``) must match the one used in
    :func:`committor_grad_sq`.
    """
    X = np.asarray(features, dtype=np.float64)
    s = np.asarray(cv_values, dtype=np.float64).reshape(-1)
    w = normalize_weights(sample_weights, s.shape[0])
    xbar = (w[:, None] * X).sum(axis=0)
    Xc = X - xbar
    sc = s - float((w * s).sum())
    d = Xc.shape[1]
    A = Xc.T @ (w[:, None] * Xc)
    lam = 1e-6 * (float(np.trace(A)) / max(d, 1) + _TINY)
    coef = np.linalg.solve(A + lam * np.eye(d), Xc.T @ (w * sc))
    if metric is None:
        return float(coef @ coef)
    M = np.asarray(metric, dtype=np.float64)
    return float(coef @ (M * coef if M.ndim == 1 else M @ coef))


# ---------------------------------------------------------------------------
# the reparametrisation route: no gradient of s
# ---------------------------------------------------------------------------
def _silverman_bandwidth(x, w):
    mu = float(np.average(x, weights=w))
    sd = np.sqrt(max(float(np.average((x - mu) ** 2, weights=w)), 1e-12))
    n_eff = (w.sum() ** 2) / max((w * w).sum(), _TINY)
    return float(max(0.9 * sd * n_eff ** (-0.2), 1e-3))


def _local_linear(x, y, w, query, bandwidth, *, min_neff=8.0):
    """Gaussian-kernel local-linear regression of ``y`` on ``x``: value and slope at ``query``."""
    h = _silverman_bandwidth(x, w) if bandwidth is None else float(bandwidth)
    value = np.full(query.shape, np.nan)
    slope = np.full(query.shape, np.nan)
    for i, x0 in enumerate(query):
        u = (x - x0) / h
        k = np.exp(-0.5 * u * u) * w
        sk = k.sum()
        if sk <= 0:
            continue
        n_eff = (sk * sk) / max((k * k).sum(), _TINY)
        if n_eff < min_neff:  # too few effective points for a slope: kernel mean only
            value[i] = float((k * y).sum() / sk)
            continue
        dx = x - x0
        Swx = float((k * dx).sum())
        Swxx = float((k * dx * dx).sum())
        Swy = float((k * y).sum())
        Swxy = float((k * dx * y).sum())
        det = sk * Swxx - Swx * Swx
        if abs(det) <= _TINY * (sk * Swxx + _TINY):
            value[i], slope[i] = Swy / sk, 0.0
        else:
            value[i] = (Swxx * Swy - Swx * Swxy) / det
            slope[i] = (sk * Swxy - Swx * Swy) / det
    return value, slope


def _weighted_r2(x, y, w):
    xb = float((w * x).sum())
    yb = float((w * y).sum())
    cov = float((w * (x - xb) * (y - yb)).sum())
    vx = float((w * (x - xb) ** 2).sum())
    vy = float((w * (y - yb) ** 2).sum())
    return float(cov * cov / max(vx * vy, _TINY))


def committor_diffusion_from_cv_reparam(
    q_values, s_values, D_s, *, sample_weights=None, n_bins=200, bandwidth=None, r2_min=0.9
) -> Profile:
    """``D_q(q) = D_s(s(q)) (dq/ds)^2`` through the empirical relation ``s(q)``.

    When the committor is (close to) a monotone function of the collective
    variable, the change of variables needs only the map ``s(q) = E[s | q]``
    and its slope, which a local-linear regression supplies. No gradient of
    ``s`` in feature space is formed, so this route is free of the
    linear-response caveat of :func:`linear_response_grad_sq`. Its own
    assumption is monotonicity, gated by the weighted squared Spearman rank
    correlation between ``q`` and ``s`` (exactly 1 for any monotone relation,
    whatever its shape): the function raises when it is below ``r2_min``.

    Args:
        q_values, s_values: ``(N,)`` per-sample committor and CV values.
        D_s: the diffusion along ``s``, a scalar or a :class:`Profile` along
            ``s`` (:func:`hummer_diffusion`), interpolated at ``s(q)``.
        sample_weights: ``(N,)`` MBAR/WHAM weights.
        n_bins: committor grid of the returned profile.
        bandwidth: kernel bandwidth in ``q`` (Silverman's rule when None).
        r2_min: the monotonicity gate, on the squared rank correlation.
    """
    q = np.clip(np.asarray(q_values, dtype=np.float64).reshape(-1), 0.0, 1.0)
    s = np.asarray(s_values, dtype=np.float64).reshape(-1)
    if s.shape[0] != q.shape[0]:
        raise ValueError(f"s_values length {s.shape[0]} != q_values length {q.shape[0]}")
    w = normalize_weights(sample_weights, q.shape[0])
    rho2 = _weighted_r2(rankdata(q), rankdata(s), w)
    if rho2 < r2_min:
        raise ValueError(
            f"q and s are not monotonically related (weighted Spearman rho^2 = {rho2:.3f} "
            f"< r2_min = {r2_min}); the reparametrisation D_q = D_s (dq/ds)^2 needs a "
            "monotone s(q). Use committor_diffusion_from_cv instead."
        )
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    grid = 0.5 * (edges[:-1] + edges[1:])
    s_of_q, ds_dq = _local_linear(q, s, w, grid, bandwidth)
    if isinstance(D_s, Profile):
        lv = np.asarray(D_s.levels, dtype=np.float64)
        dv = np.asarray(D_s.values, dtype=np.float64)
        good = np.isfinite(lv) & np.isfinite(dv) & (dv > 0)
        if not np.any(good):
            raise ValueError("D_s profile has no finite positive values")
        Ds_at = np.interp(s_of_q, lv[good], dv[good])
    else:
        Ds_at = np.full(grid.shape, float(D_s))
    slope_sq = ds_dq**2
    ok = np.isfinite(slope_sq) & (slope_sq > 1e-12)
    Dq = np.where(ok, Ds_at / np.maximum(slope_sq, _TINY), np.nan)
    counts, _ = np.histogram(q, bins=edges)
    return Profile(grid, Dq, counts)
