"""Quantity primitives along the committor (or a collective variable):

* :func:`diffusion_coefficient` -- position-dependent diffusion D(level) from a
  trajectory via the drift-corrected Kramers-Moyal estimator.
* :func:`density` -- equilibrium density π(level) from a static ensemble.
* :func:`reactive_flux` -- TPT reactive flux Φ(q*) through an iso-committor
  surface, from the committor gradient.

Each takes the callable committor ``q`` (from ``build_committor`` /
``fit_committor``) plus its data, and returns a :class:`Profile` (when
``at=None``) or the value(s) at the requested committor level(s) / range.
Diffusion and density accept ``coordinate=`` (per-sample CV values) to profile
along an arbitrary collective variable; flux is committor-specific.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from ._coordinate import _LAG_CANDIDATES, Profile, _is_range, coordinate_levels, value_at
from ._numerics import density_histogram, normalize_weights

# =============================================================================
# Density
# =============================================================================


def density(committor, samples, *, at=None, sample_weights=None, n_bins=200, coordinate=None):
    """Equilibrium density π along the committor (or a CV).

    Args:
        committor: callable ``q(x)`` from :func:`build_committor`/:func:`fit_committor`.
        samples: ``(N, dim)`` static equilibrium ensemble.
        at: ``None`` returns the full :class:`Profile`; a scalar/array returns
            π at that committor level(s); a ``(lo, hi)`` tuple returns the mean.
        sample_weights: ``(N,)`` optional MBAR weights (reweights π to target).
        n_bins: histogram resolution along the coordinate.
        coordinate: ``None`` (committor) or a ``(N,)`` per-sample CV array.

    Returns:
        :class:`Profile` if ``at is None``, else a float / array.
    """
    levels, span, name = coordinate_levels(committor, samples, coordinate)
    edges = np.linspace(span[0], span[1], n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    pi, counts = density_histogram(levels, edges, sample_weights)
    prof = Profile(levels=centers, values=pi, counts=counts, name=name)
    return prof if at is None else value_at(prof, at)


# =============================================================================
# Diffusion (drift-corrected Kramers-Moyal)
# =============================================================================


def _estimate_D_q_lumped(levels_t, bin_idx, lag, dt, n_bins, min_count):
    """Drift-corrected KM D(level) at one lag; single contiguous trajectory."""
    T = levels_t.shape[0]
    if lag >= T:
        return np.array([]), np.array([])
    dq = levels_t[lag:] - levels_t[:-lag]
    bins = bin_idx[:-lag]
    counts = np.bincount(bins, minlength=n_bins)
    counts_safe = np.maximum(counts, 1)
    mean_dq = np.bincount(bins, weights=dq, minlength=n_bins) / counts_safe
    mean_dq2 = np.bincount(bins, weights=dq * dq, minlength=n_bins) / counts_safe
    D_values = (mean_dq2 - mean_dq**2) / (2.0 * lag * dt)
    return np.where(counts >= min_count, D_values, np.nan), counts


def _estimate_D_q_stratified(levels_t, bin_idx, window_ids, lag, dt, n_bins, min_count):
    """Window-stratified drift-corrected KM D(level) (umbrella-sampling data).

    Pairs are kept only within a single window; per-(window, bin) variances are
    pooled with a Bessel correction, removing inter-window drift contamination.
    """
    T = levels_t.shape[0]
    if lag >= T:
        return np.array([]), np.array([])
    wid = np.asarray(window_ids).reshape(-1)
    if wid.shape[0] != T:
        raise ValueError(f"window_ids length {wid.shape[0]} != trajectory length {T}")
    same = wid[:-lag] == wid[lag:]
    if not np.any(same):
        return np.full(n_bins, np.nan), np.zeros(n_bins, dtype=np.int64)
    dq = (levels_t[lag:] - levels_t[:-lag])[same]
    bins = bin_idx[:-lag][same]
    wins = wid[:-lag][same]
    _, win_compact = np.unique(wins, return_inverse=True)
    K = int(win_compact.max()) + 1
    flat = win_compact * n_bins + bins
    n_kb = np.bincount(flat, minlength=K * n_bins)
    sum_dq = np.bincount(flat, weights=dq, minlength=K * n_bins)
    sum_dq2 = np.bincount(flat, weights=dq * dq, minlength=K * n_bins)
    n_safe = np.maximum(n_kb, 1)
    mean_kb = sum_dq / n_safe
    pop_var = np.where(n_kb >= 1, sum_dq2 / n_safe - mean_kb**2, 0.0)
    multi = n_kb >= 2
    num = np.where(multi, n_kb * pop_var, 0.0).reshape(K, n_bins)
    den = np.where(multi, n_kb - 1, 0).reshape(K, n_bins)
    total = n_kb.reshape(K, n_bins).sum(axis=0)
    pool_num = num.sum(axis=0)
    pool_den = den.sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        pooled = np.where(pool_den > 0, pool_num / np.maximum(pool_den, 1), np.nan)
    D_values = pooled / (2.0 * lag * dt)
    D_values = np.where(total >= min_count, D_values, np.nan)
    return D_values, total.astype(np.int64)


def _tau_int_geyer_frames(x):
    """Integrated autocorrelation time of ``x`` in FRAMES via Geyer's
    initial-positive-sequence estimator.

    Returns ``tau = 0.5 + sum of positive autocorrelation PAIRS`` (the
    trapezoidal ``int rho(t) dt`` with the ``rho_0 = 1`` endpoint counted at
    half weight), so the Hummer diffusion ``D = var(x) / (tau * dt)`` recovers
    ``var / int_0^inf rho dt`` for a locally Ornstein-Uhlenbeck coordinate.
    Pairing consecutive lags and stopping at the first non-positive pair tames
    the noise tail that makes a single-lag estimate fragile.
    """
    x = np.asarray(x, dtype=np.float64)
    n = x.shape[0]
    x = x - x.mean()
    c0 = float(np.dot(x, x) / n)
    if not np.isfinite(c0) or c0 <= 0.0:
        return 0.5
    max_lag = min(n // 4, 1000)
    if max_lag < 1:
        return 0.5
    rho = np.array(
        [float(np.dot(x[: n - k], x[k:]) / ((n - k) * c0)) for k in range(1, max_lag + 1)]
    )
    s = 0.0
    for k in range(0, rho.shape[0] - 1, 2):
        pair = rho[k] + rho[k + 1]
        if not np.isfinite(pair) or pair <= 0.0:
            break
        s += pair
    return max(0.5 + s, 0.5)


def _estimate_D_hummer(levels_t, window_ids, dt, min_count, name):
    """Hummer/Kramers per-window diffusion ``D_k = var(level_k) / (tau_int_k dt)``.

    For a coordinate that is locally CONFINED within each window (umbrella
    sampling, or short restrained/swarm segments), the equilibrium variance and
    the integrated autocorrelation time give the effective diffusion directly --
    a LAG-ROBUST estimate that needs no lag scan. One D per window, placed at the
    window's mean coordinate value, so the result is a (coarse) D(level) profile
    consumable by the same machinery as the Kramers-Moyal estimator.
    """
    wid = np.asarray(window_ids).reshape(-1)
    levels_t = np.asarray(levels_t, dtype=np.float64)
    centers, D_vals, counts = [], [], []
    for k in np.unique(wid):
        x = levels_t[wid == k]
        if x.shape[0] < max(int(min_count), 4):
            continue
        var = float(np.var(x))
        if not np.isfinite(var) or var <= 0.0:
            continue
        D = var / (_tau_int_geyer_frames(x) * dt)
        if not np.isfinite(D) or D <= 0.0:
            continue
        centers.append(float(np.mean(x)))
        D_vals.append(D)
        counts.append(x.shape[0])
    if not centers:
        raise RuntimeError(
            "diffusion_coefficient(method='hummer'): no window had enough confined "
            "samples for a var/tau_int estimate (raise min_count windows or check window_ids)"
        )
    order = np.argsort(centers)
    return Profile(
        levels=np.asarray(centers)[order],
        values=np.asarray(D_vals)[order],
        counts=np.asarray(counts, dtype=np.int64)[order],
        name=name,
    )


def _barrier_band_median(D_values, counts, centers, span):
    """Count-weighted median of D over the central 60% of the coordinate span."""
    lo = span[0] + 0.2 * (span[1] - span[0])
    hi = span[0] + 0.8 * (span[1] - span[0])
    mask = (centers >= lo) & (centers <= hi) & np.isfinite(D_values) & (counts > 0)
    if not np.any(mask):
        finite = np.isfinite(D_values)
        return float(np.nanmedian(D_values[finite])) if np.any(finite) else float("nan")
    vals = D_values[mask]
    w = counts[mask].astype(np.float64)
    order = np.argsort(vals)
    cum = np.cumsum(w[order])
    if cum[-1] <= 0:
        return float(np.nanmedian(vals))
    idx = min(int(np.searchsorted(cum, 0.5 * cum[-1])), vals.shape[0] - 1)
    return float(vals[order][idx])


def _pick_D_peak(summaries):
    arr = np.asarray(summaries, dtype=np.float64)
    finite = np.isfinite(arr) & (arr > 0)
    if not np.any(finite):
        return 0
    return int(np.argmax(np.where(finite, arr, -np.inf)))


def diffusion_coefficient(
    committor,
    trajectory,
    *,
    dt,
    at=None,
    coordinate=None,
    lag=None,
    lag_candidates=_LAG_CANDIDATES,
    window_ids=None,
    n_bins=200,
    min_count=5,
    method="kramers_moyal",
):
    """Position-dependent diffusion D along the committor (or a CV).

    Args:
        committor: callable ``q(x)``.
        trajectory: ``(T, dim)`` TIME-ORDERED frames at spacing ``dt``.
        dt: time between consecutive frames.
        at: ``None`` -> :class:`Profile`; scalar/array -> D at level(s);
            ``(lo, hi)`` -> mean over the range.
        coordinate: ``None`` (committor) or a ``(T,)`` per-frame CV array.
        lag: fixed lag (in frames); when ``None``, scan ``lag_candidates`` and
            pick the lag whose barrier-band D is largest (the diffusive peak).
            (``method="kramers_moyal"``.)
        window_ids: ``(T,)`` integer per-frame labels. For ``"kramers_moyal"`` ->
            window-stratified KM (umbrella sampling); ``None`` -> single unbiased
            trajectory. REQUIRED for ``"hummer"``.
        n_bins: coordinate resolution (``"kramers_moyal"``). min_count: bins
            (KM) / windows (Hummer) below this -> dropped.
        method: ``"kramers_moyal"`` (default) -- drift-corrected
            ``Var(dlevel_lag)/(2 lag dt)`` per coordinate bin, lag-selected; or
            ``"hummer"`` -- Kramers/Hummer 2005 ``Var(level)/(tau_int dt)`` per
            WINDOW (lag-robust; integrates the autocorrelation via Geyer instead
            of picking a lag). ``"hummer"`` assumes the coordinate is locally
            confined within each window and so requires ``window_ids``;
            ``lag*`` are ignored. It returns one D per window placed at the
            window's mean coordinate value.

    Returns:
        :class:`Profile` if ``at is None``, else a float / array.
    """
    traj = np.asarray(trajectory)
    levels_t, span, name = coordinate_levels(committor, traj, coordinate)
    wid = None if window_ids is None else np.asarray(window_ids).reshape(-1)
    T = levels_t.shape[0]

    if method == "hummer":
        if wid is None:
            raise ValueError(
                "diffusion_coefficient(method='hummer') requires window_ids: the "
                "Var/tau_int estimator assumes the coordinate is locally confined "
                "within each window (umbrella sampling or short restrained segments)."
            )
        if wid.shape[0] != T:
            raise ValueError(f"window_ids length {wid.shape[0]} != trajectory length {T}")
        prof = _estimate_D_hummer(levels_t, wid, dt, min_count, name)
        return prof if at is None else value_at(prof, at)
    if method != "kramers_moyal":
        raise ValueError(f"unknown method={method!r}; use 'kramers_moyal' or 'hummer'")

    edges = np.linspace(span[0], span[1], n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    bin_idx = np.clip(np.digitize(levels_t, edges) - 1, 0, n_bins - 1)
    candidates = (int(lag),) if lag is not None else tuple(lag_candidates)

    per_lag, counts_per_lag, summaries = [], [], []
    for L in candidates:
        if L >= T:
            continue
        if wid is not None:
            D_v, cts = _estimate_D_q_stratified(levels_t, bin_idx, wid, L, dt, n_bins, min_count)
        else:
            D_v, cts = _estimate_D_q_lumped(levels_t, bin_idx, L, dt, n_bins, min_count)
        if D_v.size == 0:
            continue
        per_lag.append((L, D_v))
        counts_per_lag.append(cts)
        summaries.append(_barrier_band_median(D_v, cts, centers, span))

    if not per_lag:
        raise RuntimeError("trajectory too short for any Kramers-Moyal lag")

    # With an explicit lag there is a single candidate, so the picker returns 0.
    sel = _pick_D_peak(summaries)
    _, D_best = per_lag[sel]
    prof = Profile(levels=centers, values=D_best, counts=counts_per_lag[sel], name=name)
    return prof if at is None else value_at(prof, at)


def _density_and_diffusion(
    committor,
    samples,
    trajectory,
    *,
    dt,
    n_bins=200,
    sample_weights=None,
    window_ids=None,
    coordinate=None,
    traj_coordinate=None,
    lag=None,
    lag_candidates=_LAG_CANDIDATES,
    min_count=5,
    diffusion_method="kramers_moyal",
):
    """The committor density π(q) (from the ensemble) and diffusion D_q(q) (from
    the trajectory) on the same grid resolution.

    These are the two profiles every committor-coordinate rate is built from.
    Plain forwarding to :func:`density` / :func:`diffusion_coefficient`, factored
    out so the call sites (``_committor_profiles``, :func:`kramers_rate`,
    :func:`saddle_bridge_D`) cannot drift apart.
    """
    pi_prof = density(
        committor,
        samples,
        at=None,
        sample_weights=sample_weights,
        n_bins=n_bins,
        coordinate=coordinate,
    )
    D_prof = diffusion_coefficient(
        committor,
        trajectory,
        dt=dt,
        at=None,
        coordinate=traj_coordinate,
        lag=lag,
        lag_candidates=lag_candidates,
        window_ids=window_ids,
        n_bins=n_bins,
        min_count=min_count,
        method=diffusion_method,
    )
    return pi_prof, D_prof


# =============================================================================
# Reactive flux (TPT, committor-specific)
# =============================================================================


def _resolve_D(D, levels):
    """Resolve ``D`` (scalar / callable / :class:`Profile`) to per-level values."""
    levels = np.asarray(levels, dtype=np.float64)
    if isinstance(D, Profile):
        return np.atleast_1d(value_at(D, levels))
    if callable(D):
        return np.asarray(D(levels), dtype=np.float64)
    return np.full(levels.shape[0], float(D))


def _coarea_contributions(committor, samples, D, sample_weights):
    """Per-sample co-area contributions ``W_n · D(q_n) · |∇q̄(x_n)|²`` and levels.

    One vmapped ``value_and_grad`` pass yields both the committor value and its
    gradient at each sample (instead of a separate value pass + gradient pass).
    """
    samples_j = jnp.asarray(samples)
    vals, grads = jax.vmap(jax.value_and_grad(committor))(samples_j)
    g = np.asarray(jnp.sum(grads**2, axis=-1), dtype=np.float64)
    levels = np.clip(np.asarray(vals, dtype=np.float64), 0.0, 1.0)
    W = normalize_weights(sample_weights, levels.shape[0])
    D_n = _resolve_D(D, levels)
    return levels, W * D_n * g


def _coarea_profile(levels, contrib, n_bins):
    """Bin per-sample co-area contributions into the Φ(c) profile on [0, 1].

    Returns ``(centers, Phi, counts)`` where ``Phi_k = (Σ_{n∈bin k} contrib_n)/Δc``.
    Only the diagnostic profile is binned; the plateau rate itself is bin-free
    (see :func:`_plateau_flux`).
    """
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    bin_idx = np.clip(np.digitize(np.clip(levels, 0.0, 1.0 - 1e-12), edges) - 1, 0, n_bins - 1)
    Phi = np.bincount(bin_idx, weights=contrib, minlength=n_bins) * n_bins
    counts = np.bincount(bin_idx, minlength=n_bins)
    return centers, Phi, counts


def _plateau_flux(levels, contrib, lo, hi):
    """Bin-free plateau-MC flux: ``(Σ_{lo≤q≤hi} contrib) / (hi − lo)``."""
    if hi <= lo:
        raise ValueError(
            f"plateau range must satisfy lo < hi; got lo={lo}, hi={hi}. A "
            "zero-width range has no well-defined flux (pass at=(lo, hi) with lo < hi)."
        )
    mask = (levels >= lo) & (levels <= hi)
    return float(np.sum(contrib[mask]) / (hi - lo))


def reactive_flux(committor, samples, *, D, at=None, sample_weights=None, n_bins=200):
    """TPT reactive flux Φ through iso-committor surfaces.

    ``Φ(q*) = ∫_{q̄=q*} π D |∇q̄| dS`` is constant in q* for the true committor
    and equals the reaction rate ν_R; for an approximate committor it is flat on
    a saddle plateau near q*≈0.5 and inflates in the basins.

    Args:
        committor: callable ``q(x)``.
        samples: ``(N, dim)`` static ensemble.
        D: scalar, callable ``level -> D``, or a diffusion :class:`Profile`.
        at: ``None`` -> :class:`Profile` of Φ(c); scalar -> Φ(q*); ``(lo, hi)``
            -> bin-free plateau Monte-Carlo flux over the range (requires ``lo < hi``).
        sample_weights: ``(N,)`` optional MBAR weights.
        n_bins: number of iso-committor bins for the Φ(c) profile (the ``(lo, hi)``
            plateau flux itself is bin-free).

    Returns:
        :class:`Profile` if ``at is None``, else a float.
    """
    levels, contrib = _coarea_contributions(committor, samples, D, sample_weights)
    centers, Phi, counts = _coarea_profile(levels, contrib, n_bins)
    prof = Profile(levels=centers, values=Phi, counts=counts, name="committor")
    if at is None:
        return prof
    if _is_range(at):
        return _plateau_flux(levels, contrib, float(at[0]), float(at[1]))
    return value_at(prof, at)


# =============================================================================
# Saddle bridge: calibrated configurational D for the feature-space rate
# =============================================================================


class BridgeD(NamedTuple):
    """Calibrated scalar configurational diffusion from the iso-committor identity.

    The feature-space flux density at level ``q*`` is ``D·⟨|∇q̄|²⟩_{q*}·π(q*)``
    while the committor-coordinate flux density is ``D_q(q*)·π(q*)``. Matching
    them fixes the scalar ``D`` so the (geometric) feature-space estimator
    reproduces the (dynamical) committor-coordinate rate:

        D = D_q(q*) / ⟨|∇q̄|²⟩_{q*}          (mode="saddle_local")
          = ⟨D_q⟩_π / ⟨|∇q̄|²⟩_π             (mode="volume_average")

    This is the principled choice of the length-scale ``D`` for
    :func:`dirichlet_rate` / :func:`tpt_rate` / :func:`reactive_flux`: a raw
    local-atomic D mismatches the committor's coordinate units (it over- or
    under-counts the flux), whereas this ``D`` is calibrated to the trajectory's
    own committor-coordinate diffusion. The object is **callable** (returns the
    constant ``D`` for any level) so it drops straight into the ``D=`` argument;
    its calibration intermediates are kept as fields.
    """

    D: float  # the calibrated scalar configurational diffusion
    D_q_ref: float  # D_q(q*) (saddle_local) or ⟨D_q⟩_π (volume_average)
    g_ref: float  # ⟨|∇q̄|²⟩_{q*} (saddle_local) or ⟨|∇q̄|²⟩_π (volume_average)
    q_star: float
    mode: str

    def __call__(self, levels):
        return np.full(np.shape(levels), self.D, dtype=np.float64)


def saddle_bridge_D(
    committor,
    samples,
    trajectory,
    *,
    dt,
    q_star=0.5,
    mode="saddle_local",
    band=0.1,
    sample_weights=None,
    window_ids=None,
    n_bins=200,
    coordinate=None,
    traj_coordinate=None,
    lag=None,
    lag_candidates=_LAG_CANDIDATES,
    min_count=5,
    diffusion_method="kramers_moyal",
):
    """Calibrated configurational scalar D bridging the two rate families.

    Combines the committor-coordinate diffusion ``D_q(q)`` (from the trajectory)
    with the feature-space gradient geometry ``⟨|∇q̄|²⟩(q) = Φ_{D=1}(q)/π(q)``
    (from the static ensemble) into a single scalar ``D`` (see
    :class:`BridgeD`). Feed the result into :func:`tpt_rate` / :func:`dirichlet_rate`
    as ``D=`` to obtain the q-stratified-plateau ("geometric") rate calibrated so
    it reproduces the committor-coordinate value -- i.e. the principled version of
    the q-stratified flux estimator (sc_simon / main-repo ``dirichlet_stratified``),
    which otherwise depends on a hand-supplied scalar D.

    Args:
        committor, samples, trajectory, dt: as in :func:`berezhkovskii_szabo_rate`.
        q_star: calibration level for ``mode="saddle_local"`` (default 0.5).
        mode: ``"saddle_local"`` (single level ``q*``) or ``"volume_average"``.
        band: half-width of the averaging window around ``q*`` (saddle_local).
        diffusion_method: ``"kramers_moyal"`` or ``"hummer"`` (needs ``window_ids``).
        (remaining args forwarded to :func:`density` / :func:`diffusion_coefficient`
        / :func:`reactive_flux`; the gradient profile uses the same ``n_bins`` so
        its grid aligns with ``π``.)

    Returns:
        :class:`BridgeD` (callable scalar D + calibration diagnostics).
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
    # ⟨|∇q̄|²⟩(q) = Φ_{D=1}(q) / π(q): the iso-q-conditional mean squared gradient
    # (the co-area sum divided by the density on the SAME [0,1] grid).
    Phi1 = reactive_flux(
        committor, samples, D=1.0, at=None, sample_weights=sample_weights, n_bins=n_bins
    )
    pi = np.asarray(pi_prof.values, dtype=np.float64)
    g_of_q = np.where(
        pi > 0, np.asarray(Phi1.values, dtype=np.float64) / np.where(pi > 0, pi, 1.0), np.nan
    )
    g_prof = Profile(
        levels=np.asarray(pi_prof.levels, dtype=np.float64),
        values=g_of_q,
        counts=pi_prof.counts,
        name="committor",
    )

    if mode == "saddle_local":
        lo, hi = max(0.0, q_star - band), min(1.0, q_star + band)
        D_q_ref = value_at(D_prof, (lo, hi))
        if not np.isfinite(D_q_ref):
            D_q_ref = value_at(D_prof, q_star)
        g_ref = value_at(g_prof, (lo, hi))
        if not np.isfinite(g_ref):
            g_ref = value_at(g_prof, q_star)
        if not (np.isfinite(D_q_ref) and np.isfinite(g_ref) and g_ref > 0):
            raise RuntimeError(
                f"saddle_bridge_D(saddle_local@{q_star}): no finite positive "
                f"D_q/⟨|∇q̄|²⟩ around q*={q_star} (got D_q={D_q_ref}, g={g_ref})"
            )
        return BridgeD(float(D_q_ref / g_ref), float(D_q_ref), float(g_ref), float(q_star), mode)

    if mode == "volume_average":
        centers = np.asarray(pi_prof.levels, dtype=np.float64)
        dq = float(centers[1] - centers[0])
        D_lev = np.asarray(D_prof.levels, dtype=np.float64)
        D_val = np.asarray(D_prof.values, dtype=np.float64)
        goodD = np.isfinite(D_val) & (D_val > 0)
        if not np.any(goodD):
            raise RuntimeError("saddle_bridge_D(volume_average): no valid D_q bins")
        Dq_on_pi = np.interp(centers, D_lev[goodD], D_val[goodD])
        valid = np.isfinite(Dq_on_pi) & (Dq_on_pi > 0) & (pi > 0)
        pi_sum = float(np.sum(pi[valid]) * dq)
        if pi_sum <= 0:
            raise RuntimeError("saddle_bridge_D(volume_average): π integrates to ≤ 0")
        D_q_pi_avg = float(np.sum(Dq_on_pi[valid] * pi[valid]) * dq / pi_sum)
        # ⟨|∇q̄|²⟩_π = ∫ Φ_{D=1}(c) dc (the co-area identity), bin-free in counts.
        g_pi_avg = float(np.sum(np.asarray(Phi1.values, dtype=np.float64)) / n_bins)
        if not (np.isfinite(g_pi_avg) and g_pi_avg > 0):
            raise RuntimeError("saddle_bridge_D(volume_average): ⟨|∇q̄|²⟩_π ≤ 0")
        return BridgeD(float(D_q_pi_avg / g_pi_avg), D_q_pi_avg, g_pi_avg, float(q_star), mode)

    raise ValueError(f"unknown mode={mode!r}; use 'saddle_local' or 'volume_average'")
