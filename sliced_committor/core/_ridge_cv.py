"""Calibration-free ridge selection by the held-out Dirichlet cap.

The EBMC solve ``min_w w'Gw  s.t.  (b-a)'w = 1`` is, structurally, the global
minimum variance portfolio (Markowitz) and the MVDR/Capon beamformer; the
Tikhonov ridge is their "diagonal loading".  Two facts from that literature set
the design here:

* The in-sample objective is biased LOW.  For Wishart ``G``,
  ``E[1/(d'G^-1 d)] = (1 - p/n)/(d'S^-1 d)``, so the fit scores better than the
  truth on its own objective.  This is why ``E_bar`` can sit below ``nu_AB`` and
  why it can never signal "stop" -- it is monotone in both M and the ridge.
* Frobenius-optimal shrinkage (Ledoit-Wolf) targets the wrong loss and
  under-shrinks for quadratic optimisation.  Measured here on the 2D benchmark it
  reintroduces the M-degradation outright (2.5x over M = 64..2048 at N = 1e5),
  while cross-validating the out-of-sample variance does not.

So the criterion is the held-out cap

    cap_out(r) = (w' G_test w) / ((b_test - a_test)' w)^2,   w fitted on the rest,

which is the quantity the variational principle bounds below by ``nu_AB`` for
every trial function.  Both factors are out-of-sample, so it is comparable across
M, solvers and hyper-parameters -- unlike the reported energy, which is only
comparable at fixed fidelity.

The ridge is the BARE ARGMIN of that curve.  A 1-SE tie-break (take the largest
ridge within one standard error of the best) was shipped first, on the grounds
that under-shrinking is the failure mode that brings back the M-degradation and
that the paired-SE version reproduced the argmin on all 22 benchmark
configurations.  That justification did not survive contact with the paper's own
2D panel, where the cap curve is flat relative to its fold noise (0.2% variation
across two decades against a 2.6% fold SE) and the tie-break walks three grid
points too far:

    cv, paired 1-SE   eta 9.7e-2   panel RMSE 0.00770   +15.1% over optimum
    cv, level 1-SE    eta 1.5e0    panel RMSE 0.01823   +172%
    cv, BARE ARGMIN   eta 1.2e-2   panel RMSE 0.00678   +1.3%

The tie-break is still available (``one_se=True`` or ``"paired"``) but it is no
longer the default: measured, it costs 14 percentage points where it bites and
gains nothing anywhere it was tested.

WHERE THIS WINS.  Against each configuration's own oracle ridge, over 22
configurations (2D Wolfe-Quapp at three N and two binning settings, AIB9, villin;
M = 64..2048), split by whether the 1D histograms are adequately resolved:

    adequately binned (n_min=10, quantile -- every molecular system here):
        this rule     median 1.004   worst 1.075
        auto_lambda   median 1.016   worst 1.517
        Ledoit-Wolf   median 1.016   worst 1.425

    2D with n_min=1, equal_width (basis noise dominates):
        this rule     median 1.142   worst 1.904
        auto_lambda   median 1.014   worst 1.496

So this is the better rule wherever the 1D slice histograms are resolved, and in
the under-binned regime neither rule is reliable -- there the fix is the binning,
not the ridge.  (``auto_lambda``'s constant was itself fitted on a set that
included those under-binned configurations, so its edge there is partly
in-sample.)

COST.  About 2x a plain ``tikhonov='auto'`` fit, measured
(``experiments/prof_ridge_cv.py``; 2D, N = 1e5, M = 256: 5.9s against 2.9s warm).
Three parts, and the first two are each about one extra pass over the data:

* K fold Grams.  These partition the samples, so all K together cost the same
  ``O(N M^2)`` as assembling one -- 1.6x one full assembly, measured, the excess
  being the per-fold gather and up to three XLA recompilations for the differing
  fold shapes.
* K fold basin-moment rows, 1.3x one ``compute_basin_moments``.  The
  interpolation is shared across folds (see ``fold_basin_moments``); doing it per
  fold instead, as this originally did, made this stage alone 78% of the total.
* K eigendecompositions (``K M^3/3``, about a second at M = 2048) plus the grid
  sweep, which is a diagonal rescale per grid point.  Under 2% of the total at
  M = 256, and the only part that grows faster than linearly in M.

Peak memory is K ``(M, M)`` blocks plus one ``(M, N/K)`` gather.

An earlier version of this note claimed the extra work was "K
eigendecompositions" alone.  That omitted both data passes and understated the
cost by 30x.

See ``docs/ridge_rule.md`` for the benchmark this rule was selected by.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_N_FOLDS = 5
DEFAULT_N_RIDGE = 41
# Span of the absolute-ridge grid, in units of the anchor M * mean(diag G).  The
# oracle ridge sits near 1e-5 * anchor on 2D at N = 1e5 and near 1e-4 at
# N = 25000, so twelve decades brackets it with a wide margin at both ends; the
# selector reports when it lands on an edge rather than silently truncating.
RIDGE_LO, RIDGE_HI = 1e-10, 1e2


def make_folds(N: int, n_folds: int, *, contiguous: bool = True, strata=None):
    """Fold label per sample.

    ``contiguous=True`` (the default) assigns consecutive BLOCKS.  MD frames are
    serially correlated, so a random permutation puts near-duplicates on both
    sides of the split and every held-out score comes out optimistic.  Pass
    ``contiguous=False`` only for genuinely i.i.d. draws.

    ``strata`` (an integer label per sample, e.g. 0 = A, 1 = B, 2 = transition)
    blocks WITHIN each stratum.  Frames adjacent in a stratum are still adjacent
    in time, so the anti-leakage property survives, but every fold is guaranteed
    samples of every basin.  Without it a trajectory that visits A before B --
    or any basin-sorted array -- puts all of A in the early folds, and the
    held-out cap is undefined on folds that hold no A or no B.
    """
    if not contiguous:
        return np.random.default_rng(0).permutation(N) % n_folds
    if strata is None:
        return (np.arange(N) * n_folds) // N
    strata = np.asarray(strata)
    out = np.empty(N, np.int64)
    for s in np.unique(strata):
        idx = np.flatnonzero(strata == s)  # already in time order
        out[idx] = (np.arange(len(idx)) * n_folds) // max(len(idx), 1)
    return out


def fold_gram_blocks(F, W, cos_matrix, fold_of, n_folds):
    """Per-fold Gram blocks and their weight sums, at the cost of ONE assembly.

    The folds partition the samples, so ``sum_k O(M^2 N_k) = O(M^2 N)``.  This
    only holds if each block is taken as a COLUMN SUBSET of ``F`` rather than by
    reweighting the full matrix -- zeroing weights leaves the matmul the same
    size and would cost K times as much.  Peak extra memory is one ``(M, N/K)``
    block, not a second copy of ``F``.

    Each block is normalised by its own weight sum, so the caller can form any
    train/test combination as a weighted mean.
    """
    import jax.numpy as jnp

    from .gram import _assemble_gram_matrix

    fold_of = np.asarray(fold_of)
    G_folds, w_folds = [], []
    for k in range(n_folds):
        idx = np.flatnonzero(fold_of == k)
        if idx.size == 0:
            raise ValueError(f"fold_gram_blocks: fold {k} is empty.")
        lo, hi = int(idx[0]), int(idx[-1]) + 1
        # Slice when the fold is one contiguous run, gather otherwise.  The gather
        # is the NORMAL path, not the exception: the cv caller always stratifies
        # by basin, and a stratified fold is one block per stratum, so it is
        # contiguous *within* each stratum and never globally.  Measured on the 2D
        # panel, 0 of 5 folds take the slice path.  Cost is still O(N M^2) overall
        # because the folds partition the samples; the gather adds one (M, N/K)
        # block per fold.
        Fk = F[:, lo:hi] if hi - lo == idx.size else F[:, jnp.asarray(idx)]
        Wk = W[lo:hi] if hi - lo == idx.size else W[jnp.asarray(idx)]
        tot = float(jnp.sum(Wk))
        if tot <= 0:
            raise ValueError(f"fold_gram_blocks: fold {k} carries zero weight.")
        G_folds.append(np.asarray(_assemble_gram_matrix(Fk, Wk / tot, cos_matrix), np.float64))
        w_folds.append(tot)
        del Fk, Wk
    return np.stack(G_folds), np.asarray(w_folds, np.float64)


def fold_basin_moments(ctx, fold_of, n_folds, batch_size: int = 512):
    """Per-fold basin moments ``(a_folds, b_folds, wA_folds, wB_folds)``.

    Mirrors ``_bmc.compute_basin_moments`` -- same chunking over directions, same
    masked interpolation -- but accumulates one row per fold.  Unweighted counts,
    matching the moments the solver is constrained against.

    The interpolation is done ONCE per direction batch and reused for all ``2K``
    accumulations.  It used to be redone for each of them, which was the single
    largest cost in ``tikhonov='cv'`` (78% of it, measured; see
    ``experiments/prof_ridge_cv.py``).  That was pure waste: the mask in
    ``_interpolate_q_at_samples_masked`` is applied *after* the interpolation, so
    every one of the ``2K`` calls evaluated the same ``(batch, N)`` committor
    field and then threw away all but one fold's worth of it.

    The rewrite is deliberately bitwise-preserving rather than merely equivalent.
    ``where(in_X & fold_k, q, 0)`` and ``where(fold_k, where(in_X, q, 0), 0)``
    are the same array elementwise, and the reduction is still a single
    ``jnp.sum`` along the same axis of the same shape, so the summation order is
    unchanged.  That matters here: the held-out cap curve can be flat to a
    fraction of a percent, so a last-bit change in a moment could move the
    selected ridge by a grid point and shift published numbers.
    """
    import jax.numpy as jnp

    from .weights import _interpolate_q_at_samples_masked

    in_A = np.asarray(ctx.in_A, bool)
    in_B = np.asarray(ctx.in_B, bool)
    M = ctx.projected_samples.shape[0]
    fold_of = np.asarray(fold_of)
    in_A_j, in_B_j = jnp.asarray(in_A), jnp.asarray(in_B)
    fold_j = [jnp.asarray(fold_of == k) for k in range(n_folds)]
    wA = np.array([float((in_A & (fold_of == k)).sum()) for k in range(n_folds)])
    wB = np.array([float((in_B & (fold_of == k)).sum()) for k in range(n_folds)])
    if (wA <= 0).any() or (wB <= 0).any():
        raise ValueError(
            "fold_basin_moments: some fold contains no samples of a basin. "
            "Reduce n_folds, or pass contiguous=False if the samples are i.i.d."
        )

    a = np.zeros((n_folds, M))
    b = np.zeros((n_folds, M))
    for start in range(0, M, batch_size):
        end = min(start + batch_size, M)
        s_grid = ctx.slice_coords[start:end]
        q_grid = ctx.committors_1d[start:end]
        ps = ctx.projected_samples[start:end]
        # Once per batch, per basin.  Masking by the whole basin here (rather
        # than not at all) keeps each of the K reductions below a masked view of
        # exactly the array the old code built for that fold.
        qA = _interpolate_q_at_samples_masked(s_grid, q_grid, ps, in_A_j)
        qB = _interpolate_q_at_samples_masked(s_grid, q_grid, ps, in_B_j)
        for k in range(n_folds):
            a[k, start:end] = np.asarray(jnp.sum(jnp.where(fold_j[k], qA, 0.0), axis=1)) / wA[k]
            b[k, start:end] = np.asarray(jnp.sum(jnp.where(fold_j[k], qB, 0.0), axis=1)) / wB[k]
        del qA, qB
    return a, b, wA, wB


def _solve(L, V, dv, r):
    """EBMC weights at absolute ridge ``r`` from a precomputed eigendecomposition."""
    den = L + r
    s = float(dv @ (dv / den))
    if not np.isfinite(s) or s <= 0:
        return None
    return (V @ (dv / den)) / s


def select_ridge_cv(
    G_folds,
    w_folds,
    a_folds,
    b_folds,
    wA_folds,
    wB_folds,
    valid_mask,
    *,
    n_ridge: int = DEFAULT_N_RIDGE,
    one_se=False,
):
    """Absolute ridge chosen by the held-out cap.

    Args:
        G_folds: ``(K, M, M)`` Gram of each fold, each normalised by that fold's
            own sample-weight sum.
        w_folds: ``(K,)`` sample-weight sum per fold (``n_k/N`` for uniform
            weights).  Train Grams are the weight-averaged complement, which is
            exact because ``G`` is a weighted MEAN over samples.
        a_folds, b_folds: ``(K, M)`` basin moments per fold, each normalised by
            that fold's own basin weight.
        wA_folds, wB_folds: ``(K,)`` basin weight sums per fold.
        valid_mask: ``(M,)`` bool; invalid directions are removed from the solve.
        one_se: ``True`` for the 1-SE rule on the cap LEVEL, ``"paired"`` for the
            1-SE rule on the paired difference from the argmin (a much tighter,
            and better-calibrated, tolerance), ``False`` for the bare argmin.

    Returns:
        dict with ``ridge`` (absolute), ``anchor``, ``curve``, ``se``, ``idx``,
        ``at_edge`` and ``n_folds``.
    """
    G_folds = np.asarray(G_folds, np.float64)
    w_folds = np.asarray(w_folds, np.float64)
    a_folds = np.asarray(a_folds, np.float64)
    b_folds = np.asarray(b_folds, np.float64)
    wA_folds = np.asarray(wA_folds, np.float64)
    wB_folds = np.asarray(wB_folds, np.float64)
    keep = np.asarray(valid_mask, bool)
    K = G_folds.shape[0]
    if keep.sum() < 2:
        raise ValueError("select_ridge_cv: fewer than two valid directions.")

    idx = np.flatnonzero(keep)
    Gf = G_folds[:, idx][:, :, idx]
    af, bf = a_folds[:, idx], b_folds[:, idx]

    G_all = np.tensordot(w_folds, Gf, axes=(0, 0)) / max(w_folds.sum(), 1e-300)
    # M_valid, not M: the solve runs on the valid sub-block, and the M-law
    # (ridge proportional to the number of directions actually in the trial
    # space) is what makes this anchor roughly M-independent.  Only the grid
    # CENTRE depends on it -- the span is twelve decades -- but keeping it
    # consistent means the reported r/anchor is comparable across runs.
    anchor = len(idx) * float(np.mean(np.diag(G_all)))
    if not np.isfinite(anchor) or anchor <= 0:
        raise ValueError("select_ridge_cv: non-positive Gram anchor.")
    grid = np.geomspace(RIDGE_LO * anchor, RIDGE_HI * anchor, n_ridge)

    caps = np.full((K, n_ridge), np.nan)
    for k in range(K):
        other = [j for j in range(K) if j != k]
        wo = w_folds[other]
        Gtr = np.tensordot(wo, Gf[other], axes=(0, 0)) / max(wo.sum(), 1e-300)
        atr = (wA_folds[other] @ af[other]) / max(wA_folds[other].sum(), 1e-300)
        btr = (wB_folds[other] @ bf[other]) / max(wB_folds[other].sum(), 1e-300)
        if wA_folds[k] <= 0 or wB_folds[k] <= 0:
            continue  # fold holds no samples of some basin
        L, V = np.linalg.eigh(0.5 * (Gtr + Gtr.T))
        L = np.maximum(L, 0.0)
        dv = V.T @ (btr - atr)
        dte = bf[k] - af[k]
        for i, r in enumerate(grid):
            w = _solve(L, V, dv, r)
            if w is None:
                continue
            gap = float(dte @ w)
            if abs(gap) > 1e-30:
                caps[k, i] = float(w @ Gf[k] @ w) / gap**2

    with np.errstate(invalid="ignore"):
        mean = np.nanmean(caps, axis=0)
        n_ok = np.sum(np.isfinite(caps), axis=0)
        se = np.nanstd(caps, axis=0, ddof=1) / np.sqrt(np.maximum(n_ok, 1))
    if not np.isfinite(mean).any():
        raise ValueError("select_ridge_cv: the held-out cap is undefined at every ridge.")

    scored = np.where(np.isfinite(mean), mean, np.inf)
    i_best = int(np.argmin(scored))
    i_sel = i_best

    # PAIRED standard error, referenced to the argmin.  The fold-to-fold scatter
    # in the cap LEVEL is a common offset -- each fold estimates nu_AB slightly
    # differently -- so a tolerance built on it overstates the uncertainty on a
    # ridge-to-ridge DIFFERENCE by about an order of magnitude and walks a long
    # way up a flat curve (2D, equal-width bins: 2.0-2.3x the oracle, against
    # 1.08-1.16x for the bare argmin).  The paired SE is the uncertainty in the
    # comparison actually being made.
    with np.errstate(invalid="ignore"):
        d = caps - caps[:, i_best : i_best + 1]
        se_paired = np.nanstd(d, axis=0, ddof=1) / np.sqrt(np.maximum(n_ok, 1))

    tol = {"paired": se_paired, True: se}.get(one_se)
    if tol is not None and np.isfinite(tol[i_best]):
        ok = np.flatnonzero(
            (scored - mean[i_best] <= np.where(np.isfinite(tol), tol, -np.inf))
            & (np.arange(n_ridge) >= i_best)
        )
        if len(ok):
            i_sel = int(ok.max())

    at_edge = i_sel in (0, n_ridge - 1)
    if at_edge:
        logger.warning(
            "select_ridge_cv: selected ridge is at the edge of the search grid "
            "(r/anchor = %.2e); the optimum may lie outside [%g, %g] * anchor.",
            grid[i_sel] / anchor,
            RIDGE_LO,
            RIDGE_HI,
        )
    return dict(
        ridge=float(grid[i_sel]),
        anchor=float(anchor),
        idx=int(i_sel),
        idx_argmin=int(i_best),
        at_edge=bool(at_edge),
        grid=grid,
        curve=mean,
        se=se,
        se_paired=se_paired,
        n_folds=int(K),
    )
