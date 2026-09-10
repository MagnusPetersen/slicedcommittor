"""Diffusion along a coordinate, measured on a time-ordered trajectory.

Three estimators of the diffusion coefficient of a coordinate ``x(t)``:

* :func:`diffusion_profile`: the drift-corrected Kramers-Moyal (mean squared
  displacement) estimator ``Var(x[t + lag] - x[t]) / (2 lag dt)`` per bin of
  the coordinate, stratified by umbrella window when ``window_ids`` are given.
  It presumes a diffusive regime at ``lag``; :func:`lag_scan` shows whether the
  coordinate has one.
* :func:`hummer_diffusion`: Hummer's ``Var(x) / (tau_int dt)`` per umbrella
  window, for a coordinate confined by a restraint. There is no lag to choose;
  the integrated autocorrelation time takes its place.
* :func:`pooled_acf_diffusion`: the paper's estimator. The normalised
  autocorrelation functions of the selected windows are averaged BEFORE
  integrating, which removes the noise of a single window's ``tau_int`` (the
  per-window values span a factor of 92 across the chignolin windows). Windows
  are selected by their mean coordinate, never by the committor, so the
  committor cannot pick the windows that set the diffusion that sets the rate.

All three take the coordinate's per-frame VALUES, ``q(trajectory)`` or a
collective variable: diffusion is a property of a time series, not of the
committor function. ``run_ids`` mark independent contiguous runs, and no
estimator reads across a join: an autocorrelation is only defined within one
run (splicing the replicates of a window end to end reads the joins as slow
correlation, ``tau_int`` inflated by up to 77% on chignolin), and a
displacement pair is only a displacement within one run.
"""

from typing import NamedTuple

import numpy as np

from .._runs import segment_ids
from ._coordinate import Profile, _check_band
from ._numerics import bin_grid, per_frame


# ---------------------------------------------------------------------------
# autocorrelation machinery
# ---------------------------------------------------------------------------
def _acf(x, max_lag):
    """Normalised autocorrelation at lags ``1..max_lag`` (zeros when ``x`` is constant)."""
    x = np.asarray(x, dtype=np.float64)
    x = x - x.mean()
    n = x.size
    c0 = float(np.dot(x, x) / n)
    if not np.isfinite(c0) or c0 <= 0.0:
        return np.zeros(max_lag)
    return np.array(
        [float(np.dot(x[: n - k], x[k:]) / ((n - k) * c0)) for k in range(1, max_lag + 1)]
    )


def _geyer_tau(rho):
    """Integrated autocorrelation time in frames: ``0.5 + `` the initial positive sequence.

    Consecutive lags are paired and the sum stops at the first non-positive
    pair (Geyer), which tames the noise tail that makes a single-lag estimate
    fragile. The ``rho_0 = 1`` endpoint counts at half weight, so
    ``D = Var / (tau dt)`` recovers ``Var / int_0^inf rho dt`` for a locally
    Ornstein-Uhlenbeck coordinate.
    """
    s = 0.0
    for k in range(0, rho.size - 1, 2):
        pair = rho[k] + rho[k + 1]
        if not np.isfinite(pair) or pair <= 0.0:
            break
        s += pair
    return max(0.5 + s, 0.5)


def _tau_int(x):
    max_lag = min(x.size // 4, 1000)
    if max_lag < 1:
        return 0.5
    return _geyer_tau(_acf(x, max_lag))


def _hummer(x, dt):
    """``Var(x) / (tau_int dt)`` for one contiguous run; NaN when undefined."""
    var = float(np.var(x))
    if not np.isfinite(var) or var <= 0.0:
        return float("nan")
    D = var / (_tau_int(x) * dt)
    return D if (np.isfinite(D) and D > 0.0) else float("nan")


def _runs(n, window_ids, run_ids):
    """``(window labels or None, run index per frame)``: the runs are the
    contiguous stretches over which window and replicate are both constant."""
    wid = per_frame("window_ids", window_ids, n)
    rid = per_frame("run_ids", run_ids, n)
    labels = [ids for ids in (wid, rid) if ids is not None] or [np.zeros(n, np.int64)]
    return wid, segment_ids(*labels)


def _windows(coordinate, window_ids, run_ids):
    """Yield ``(window mask, [index array per contiguous run])`` per window."""
    x = np.asarray(coordinate, dtype=np.float64).reshape(-1)
    wid, seg = _runs(x.shape[0], window_ids, run_ids)
    for k in np.unique(wid):
        m = wid == k
        yield m, [np.flatnonzero(seg == s) for s in np.unique(seg[m])]


# ---------------------------------------------------------------------------
# Kramers-Moyal / mean squared displacement
# ---------------------------------------------------------------------------
def _km_lumped(dx, bins, lag, dt, n_bins, min_count):
    """Drift-corrected Kramers-Moyal ``D`` per bin of the displacements, one population."""
    counts = np.bincount(bins, minlength=n_bins)
    safe = np.maximum(counts, 1)
    mean_dx = np.bincount(bins, weights=dx, minlength=n_bins) / safe
    mean_dx2 = np.bincount(bins, weights=dx * dx, minlength=n_bins) / safe
    D = (mean_dx2 - mean_dx**2) / (2.0 * lag * dt)
    return np.where(counts >= min_count, D, np.nan), counts


def _km_stratified(dx, bins, window_of_pair, lag, dt, n_bins, min_count):
    """Window-stratified drift-corrected Kramers-Moyal ``D`` per bin.

    The per-(window, bin) variances of the displacements are pooled with a
    Bessel correction, which removes the drift the umbrella restraints induce
    between windows.
    """
    if dx.size == 0:
        return np.full(n_bins, np.nan), np.zeros(n_bins, dtype=np.int64)
    _, win = np.unique(window_of_pair, return_inverse=True)
    K = int(win.max()) + 1
    flat = win * n_bins + bins
    n_kb = np.bincount(flat, minlength=K * n_bins)
    sum_dx = np.bincount(flat, weights=dx, minlength=K * n_bins)
    sum_dx2 = np.bincount(flat, weights=dx * dx, minlength=K * n_bins)
    safe = np.maximum(n_kb, 1)
    mean_kb = sum_dx / safe
    pop_var = np.where(n_kb >= 1, sum_dx2 / safe - mean_kb**2, 0.0)
    multi = n_kb >= 2
    num = np.where(multi, n_kb * pop_var, 0.0).reshape(K, n_bins).sum(axis=0)
    den = np.where(multi, n_kb - 1, 0).reshape(K, n_bins).sum(axis=0)
    total = n_kb.reshape(K, n_bins).sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        pooled = np.where(den > 0, num / np.maximum(den, 1), np.nan)
    D = pooled / (2.0 * lag * dt)
    return np.where(total >= min_count, D, np.nan), total.astype(np.int64)


def diffusion_profile(
    coordinate,
    *,
    dt,
    lag,
    window_ids=None,
    run_ids=None,
    n_bins=200,
    span=(0.0, 1.0),
    min_count=5,
) -> Profile:
    """Kramers-Moyal diffusion ``D(x)`` of a coordinate, per bin, at one explicit lag.

    ``D = Var(x[t + lag] - x[t]) / (2 lag dt)`` over the pairs starting in each
    bin, drift-corrected by subtracting the mean displacement. Pairs never
    cross a run; with ``window_ids`` (umbrella sampling) they never cross a
    window either and the per-window variances are pooled, see
    :func:`_km_stratified`.

    ``lag`` is explicit on purpose. On a coordinate with a diffusive regime
    the estimate is flat across lags and any lag in the plateau will do; on one
    without (the committor of a slow folder, whose displacement saturates
    within one storage interval) there is no ``D`` to estimate, and a rule that
    picks the lag with the largest ``D`` would return the short-time bounce.
    Run :func:`lag_scan` and choose from the plateau.

    Args:
        coordinate: ``(T,)`` per-frame values of a TIME-ORDERED trajectory,
            e.g. ``q(trajectory)`` or a collective variable.
        dt: time between consecutive frames.
        lag: displacement lag in frames (``>= 1``).
        window_ids: ``(T,)`` per-frame umbrella window labels, or None.
        run_ids: ``(T,)`` labels of independent contiguous runs, or None.
        n_bins, span: the bins along the coordinate; ``span`` defaults to the
            committor's ``[0, 1]``.
        min_count: bins with fewer pairs than this are NaN.
    """
    x = np.asarray(coordinate, dtype=np.float64).reshape(-1)
    lag = int(lag)
    if lag < 1:
        raise ValueError(f"lag must be a positive number of frames; got {lag}")
    if lag >= x.shape[0]:
        raise ValueError(f"lag {lag} is not shorter than the trajectory ({x.shape[0]} frames)")
    wid, seg = _runs(x.shape[0], window_ids, run_ids)
    keep = seg[:-lag] == seg[lag:]  # both ends of a pair in one run
    edges, centers = bin_grid(n_bins, span)
    bin_idx = np.clip(np.digitize(x, edges) - 1, 0, n_bins - 1)
    dx = (x[lag:] - x[:-lag])[keep]
    bins = bin_idx[:-lag][keep]
    if wid is None:
        D, counts = _km_lumped(dx, bins, lag, dt, n_bins, min_count)
    else:
        D, counts = _km_stratified(dx, bins, wid[:-lag][keep], lag, dt, n_bins, min_count)
    return Profile(centers, D, counts)


def _weighted_median(values, weights):
    order = np.argsort(values)
    cum = np.cumsum(weights[order])
    if cum[-1] <= 0:
        return float(np.median(values))
    idx = min(int(np.searchsorted(cum, 0.5 * cum[-1])), values.shape[0] - 1)
    return float(values[order][idx])


def lag_scan(
    coordinate,
    *,
    dt,
    lags=(1, 2, 5, 10, 20, 50),
    window_ids=None,
    run_ids=None,
    n_bins=200,
    span=(0.0, 1.0),
    band=(0.2, 0.8),
    min_count=5,
) -> Profile:
    """The Kramers-Moyal ``D`` of a coordinate against the lag: the diffusivity check.

    Returns a profile whose ``levels`` are the lags (in frames), whose ``values``
    are the count-weighted median of :func:`diffusion_profile` over ``band`` at
    each lag, and whose ``counts`` are the pairs behind each. A diffusive
    coordinate gives a plateau across lags; the committor of a slow system
    usually does not (its displacement saturates within a few frames), and then
    no lag is right. Read ``D`` off the plateau, never off the maximum.
    """
    values, counts = [], []
    lo, hi = _check_band(band)
    for L in lags:
        prof = diffusion_profile(
            coordinate,
            dt=dt,
            lag=L,
            window_ids=window_ids,
            run_ids=run_ids,
            n_bins=n_bins,
            span=span,
            min_count=min_count,
        )
        sel = (
            (prof.levels >= lo) & (prof.levels <= hi) & np.isfinite(prof.values) & (prof.counts > 0)
        )
        values.append(
            _weighted_median(prof.values[sel], prof.counts[sel].astype(np.float64))
            if np.any(sel)
            else float("nan")
        )
        counts.append(int(prof.counts[sel].sum()))
    return Profile(np.asarray(lags, dtype=np.float64), np.asarray(values), np.asarray(counts))


# ---------------------------------------------------------------------------
# Hummer: variance over integrated autocorrelation time, per window
# ---------------------------------------------------------------------------
def hummer_diffusion(coordinate, window_ids, *, dt, run_ids=None, min_count=5) -> Profile:
    """Hummer's ``Var(x) / (tau_int dt)`` per umbrella window.

    For a coordinate confined by the window's restraint the equilibrium
    variance and the integrated autocorrelation time give the diffusion
    directly, with no lag to choose. One value per window, placed at the
    window's mean coordinate, so the result is a coarse ``D(x)`` profile that
    :func:`rate_from_profiles` interpolates like any other. With ``run_ids``
    each contiguous run of a window is estimated on its own and the runs are
    averaged weighted by length.

    Raises RuntimeError when no window has a usable run (fewer than
    ``max(min_count, 4)`` frames, or zero variance).
    """
    x_all = np.asarray(coordinate, dtype=np.float64).reshape(-1)
    levels, values, counts = [], [], []
    for mask, runs in _windows(x_all, window_ids, run_ids):
        Ds, ns = [], []
        for idx in runs:
            x = x_all[idx]
            if x.shape[0] < max(int(min_count), 4):
                continue
            D = _hummer(x, dt)
            if np.isfinite(D):
                Ds.append(D)
                ns.append(float(x.shape[0]))
        if not Ds:
            continue
        levels.append(float(np.mean(x_all[mask])))
        values.append(Ds[0] if len(Ds) == 1 else float(np.average(Ds, weights=ns)))
        counts.append(int(mask.sum()))
    if not levels:
        raise RuntimeError(
            "hummer_diffusion: no window has a run with enough frames and a positive "
            "variance (check window_ids / run_ids, or lower min_count)"
        )
    order = np.argsort(levels)
    return Profile(
        np.asarray(levels)[order],
        np.asarray(values)[order],
        np.asarray(counts, dtype=np.int64)[order],
    )


class PooledDiffusion(NamedTuple):
    """The pooled-autocorrelation diffusion of :func:`pooled_acf_diffusion`.

    Attributes:
        D: ``Var / (tau_int dt)``, variance and ACF pooled over the runs.
        tau_int: integrated autocorrelation time of the pooled ACF, in frames.
        n_windows: windows that contributed at least one run.
        n_runs: contiguous runs pooled.
        ci: ``(lo, hi)`` 95% run-bootstrap interval on ``D``, or None.
    """

    D: float
    tau_int: float
    n_windows: int
    n_runs: int
    ci: tuple | None


def pooled_acf_diffusion(
    coordinate,
    window_ids,
    *,
    dt,
    run_ids=None,
    window_band=None,
    max_lag=1000,
    n_boot=0,
    seed=0,
) -> PooledDiffusion:
    """Hummer diffusion with the autocorrelation function pooled across windows.

    ``Var / tau_int`` needs a contiguous time series, so its unit of estimation
    is the umbrella window. Statistically equivalent windows share an ACF, and
    the factor-of-several scatter between per-window estimates lives almost
    entirely in the noisy tail of a single window's ``tau_int``. Averaging the
    normalised ACFs over the windows (weighted by length) before integrating
    removes it; the variances are pooled the same way.

    Args:
        coordinate: ``(T,)`` per-frame values of the biased coordinate.
        window_ids: ``(T,)`` per-frame window labels.
        dt: time between consecutive frames.
        run_ids: ``(T,)`` labels of independent contiguous runs (replicates),
            or None when every window is one run.
        window_band: ``(lo, hi)``; only windows whose MEAN coordinate lies in it
            contribute (the paper's committor-free barrier selection). None
            pools every window.
        max_lag: longest lag of the ACF, in frames (each run is also capped at
            a quarter of its length).
        n_boot: run-bootstrap draws for the 95% interval on ``D`` (0 skips it).
        seed: bootstrap seed.

    Raises ValueError when no run is long enough (16 frames) for an ACF.
    """
    x_all = np.asarray(coordinate, dtype=np.float64).reshape(-1)
    band = None if window_band is None else _check_band(window_band)
    rhos, var, n = [], [], []
    n_windows = 0
    for mask, runs in _windows(x_all, window_ids, run_ids):
        if band is not None and not (band[0] <= float(np.mean(x_all[mask])) <= band[1]):
            continue
        contributed = False
        for idx in runs:
            x = x_all[idx]
            ml = int(min(x.shape[0] // 4, max_lag))
            if ml < 4:
                continue
            rhos.append(_acf(x, ml))
            var.append(float(np.var(x)))
            n.append(float(x.shape[0]))
            contributed = True
        n_windows += int(contributed)
    if not rhos:
        raise ValueError("pooled_acf_diffusion: no window run is long enough for an ACF")
    L = min(r.size for r in rhos)
    R = np.array([r[:L] for r in rhos])
    var = np.asarray(var)
    n = np.asarray(n)
    tau = _geyer_tau(np.average(R, axis=0, weights=n))
    D = float(np.average(var, weights=n) / (tau * dt))
    ci = None
    if n_boot:
        rng = np.random.default_rng(seed)
        draws = np.empty(int(n_boot))
        for b in range(int(n_boot)):
            i = rng.choice(R.shape[0], size=R.shape[0], replace=True)
            tau_b = _geyer_tau(np.average(R[i], axis=0, weights=n[i]))
            draws[b] = np.average(var[i], weights=n[i]) / (tau_b * dt)
        lo, hi = np.percentile(draws, [2.5, 97.5])
        ci = (float(lo), float(hi))
    return PooledDiffusion(D, float(tau), n_windows, int(R.shape[0]), ci)
