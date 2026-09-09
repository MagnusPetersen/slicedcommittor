"""The weight solve: basin-moment-constrained minimisation of the Dirichlet form.

The recombined committor is ``q(x) = c + sum_j w_j q_j(theta_j . x)``. Its
Dirichlet energy is the quadratic form ``w^T G w`` on the slice Gram matrix
``G_jk = (theta_j^T Mbar theta_k) <q_j' q_k'>_rho``, and the boundary
conditions enter as the two basin-moment constraints ``<q>_A = 0``,
``<q>_B = 1``. Eliminating ``c = -a^T w`` leaves one linear constraint
``(b - a)^T w = 1`` (``a_j = <q_j>_A``, ``b_j = <q_j>_B``) and the closed form

    R   := (b - a)^T G^-1 (b - a)            (the moment gap)
    w   = G^-1 (b - a) / R,     c = -a^T w,    min w^T G w = 1 / R.

``G`` is overcomplete in practice (``M > dim``) and ill-conditioned, so
``G^-1`` is a regularised solve. Two rules and a number are accepted for
``tikhonov``:

* ``'halfset_eigen'`` (default): the half-set spectral filter of
  :mod:`sliced_committor.core._halfset`: two half-Grams from interleaved
  contiguous basin-stratified folds, per-eigenband SSNR, Wiener damping, and
  the solve on the filtered Gram with zero additional ridge. Tuning-free;
  frames must be time-ordered.
* ``'auto'``: the closed-form ridge ``eta = 1/sqrt(N_eff)`` times
  ``median diag G``; no extra pass over the data.
* a float: an absolute ridge in the units of ``G``.

The solve needs float64 (``jax_enable_x64``): ``cond(G)`` reaches 1e12. The
``(M, M)`` linear algebra is numpy/scipy; JAX carries the ``(M, N)`` assembly.
"""

import logging
from itertools import pairwise
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from . import _halfset as hs
from ._moments import compute_basin_moments
from .gram import _assemble_gram_matrix, _compute_derivative_matrix, cos_matrix_from_metric

logger = logging.getLogger(__name__)

TIKHONOV_RULES = ("halfset_eigen", "auto")
# Below this ``R / max(A, B)`` the moment gap has collapsed in the G^-1 norm:
# no direction distinguishes A from B at the moment level.
COND_THRESHOLD = 1e-6


class RepresentationError(RuntimeError):
    """The slice basis cannot represent the boundary conditions.

    Raised when the moment gap ``R = (b - a)^T G^-1 (b - a)`` collapses relative
    to ``max(a^T G^-1 a, b^T G^-1 b)``: every slice has effectively identical
    basin moments, so no projection direction separates A from B. The remedy
    is more or differently aligned directions, not a smaller ridge.
    """


class Weights(NamedTuple):
    """The solved weights and what the solve knows about them.

    Fields:
        w: ``(M,)`` slice weights (zero on invalid directions).
        c: the global bias ``-a^T w``.
        dirichlet_energy: ``w^T G w`` on the UNregularised Gram: the variational
            objective of the committor actually built. Lower is closer to the
            true committor, so it ranks fits of the same data without a reference.
        moment_gap: ``R``; ``1/R`` is the energy under the regularised solve.
        cond: ``R / max(A, B)``, the representation gate.
        ridge: the absolute ridge added to ``G`` (0 for the half-set filter).
        tikhonov: the rule (or number) that was requested.
        heldout_cap: the out-of-sample Dirichlet cap when requested
            (``cap``, ``se``, ``per_fold``, ``gap``); None otherwise.
        diagnostics: secondary quantities: the basin moments ``a``, ``b``, the
            constraint checks ``mu_A_check`` (~0) and ``mu_B_check`` (~1),
            ``N_eff``, ``cond_chol``, ``n_negative_weights``,
            ``off_diagonal_magnitude``, and for the half-set filter ``G_reg``,
            ``band_ssnr``, ``lam_max``, ``lam_min_reg``.
    """

    w: jnp.ndarray
    c: float
    dirichlet_energy: float
    moment_gap: float
    cond: float
    ridge: float
    tikhonov: str | float
    heldout_cap: dict | None
    diagnostics: dict


def _require_x64():
    if not jax.config.read("jax_enable_x64"):
        raise ValueError(
            "the weight solve requires jax_enable_x64=True: add "
            "jax.config.update('jax_enable_x64', True) before fitting (cond(G) ~ 1e12)."
        )


def _effective_n(W):
    """``N_eff = (sum W)^2 / sum W^2``."""
    W = jnp.asarray(W)
    Z, Z2 = jnp.sum(W), jnp.sum(W**2)
    return float(Z**2 / Z2) if float(Z2) > 0 else float(W.shape[0])


def _ridged_gram(G, valid_mask, ridge):
    """``G`` with invalid directions replaced by decoupled identity rows, plus ``ridge I``."""
    G = np.asarray(G, np.float64)
    valid = np.asarray(valid_mask, bool)
    G_masked = G * np.outer(valid, valid) + np.diag((~valid).astype(np.float64))
    return G_masked + ridge * np.eye(valid.size)


def _solve(G_reg, a, b, valid_mask):
    """The closed form on the valid block of a regularised Gram.

    ``w = G_reg^-1 (b - a) / R`` with ``R = (b - a)^T G_reg^-1 (b - a)`` and
    ``c = -a^T w``; scipy Cholesky with a least-squares fallback. Invalid
    directions get ``w = 0``.
    """
    from scipy.linalg import cho_factor, cho_solve

    valid = np.asarray(valid_mask, bool)
    iv = np.flatnonzero(valid)
    G_v = np.asarray(G_reg, np.float64)[np.ix_(iv, iv)]
    a_np = np.asarray(a, np.float64)
    b_np = np.asarray(b, np.float64)
    av, bv = a_np[iv], b_np[iv]
    delta = bv - av
    cond_chol = float("inf")
    try:
        cf = cho_factor(G_v, lower=True)
        w_dual = cho_solve(cf, delta)
        Ginv_a = cho_solve(cf, av)
        Ginv_b = cho_solve(cf, bv)
        L_abs = np.abs(np.diag(cf[0]))
        cond_chol = float((L_abs.max() / max(L_abs.min(), 1e-30)) ** 2)
    except np.linalg.LinAlgError:
        w_dual = np.linalg.lstsq(G_v, delta, rcond=None)[0]
        Ginv_a = np.linalg.lstsq(G_v, av, rcond=None)[0]
        Ginv_b = np.linalg.lstsq(G_v, bv, rcond=None)[0]
    A_s = float(av @ Ginv_a)
    B_s = float(bv @ Ginv_b)
    C_s = float(av @ Ginv_b)
    R = float(delta @ w_dual)
    w_v = w_dual / R if abs(R) > 1e-300 else w_dual
    w = np.zeros(valid.size)
    w[iv] = w_v
    return dict(
        w=jnp.asarray(w),
        c=float(-(av @ w_v)),
        R=R,
        A=A_s,
        B=B_s,
        C=C_s,
        cond=R / max(A_s, B_s, 1e-30),
        cond_chol=cond_chol,
    )


def _gate(out, raise_on_degenerate):
    cond = float(out["cond"])
    if raise_on_degenerate and cond < COND_THRESHOLD:
        raise RepresentationError(
            f"cond = R / max(A, B) = {cond:.3g} < {COND_THRESHOLD:.0e}: the moment gap (b - a) "
            "collapses in the G^-1 norm; the slice basis cannot distinguish A from B at the "
            "moment level. Add more or differently aligned directions "
            "(raise_on_degenerate=False overrides)."
        )
    return cond


def _assemble(result, W):
    """The derivative matrix ``F``, the cosine matrix and the Gram ``G = cos * (F sqrt W)(F sqrt W)^T``."""
    F = _compute_derivative_matrix(
        result.slice_coords, result.committors_1d, result.projected_samples
    )
    cos_matrix = cos_matrix_from_metric(result.directions, result.feature_metric)
    F = F.astype(jnp.float64)
    W = jnp.asarray(W, dtype=jnp.float64)
    G = _assemble_gram_matrix(F, W, cos_matrix)
    return F, W, cos_matrix, G


def _sample_weights(result, counts=None):
    """The sample weights the Gram averages over: uniform ``1/N`` or the stored
    ``sample_weights`` verbatim (they should sum to 1; the weights ``w`` are
    invariant to their scale, only the reported energy is not). A bootstrap
    replicate multiplies in its frame multiplicities and is rescaled back to
    the original total so its energy stays comparable."""
    N = result.projected_samples.shape[1]
    W = (
        jnp.ones(N) / N
        if result.sample_weights is None
        else jnp.asarray(result.sample_weights, jnp.float64)
    )
    if counts is not None:
        Wc = W * jnp.asarray(counts, jnp.float64)
        W = Wc * (jnp.sum(W) / jnp.sum(Wc))
    return W


def solve_weights(
    result, *, tikhonov="halfset_eigen", heldout_cap=False, raise_on_degenerate=True, counts=None
):
    """Solve the slice weights of a :class:`~sliced_committor.SlicedCommittorResult`.

    Args:
        result: the slice basis (with ``projected_samples`` and basin labels).
        tikhonov: ``'halfset_eigen'`` (default) | ``'auto'`` | a float, an
            absolute ridge in the units of ``G``. See the module docstring.
        heldout_cap: with the half-set filter, also read the held-out
            Dirichlet cap ``w^T G_k w / ((b_k - a_k) . w)^2`` on each fold,
            fitted on the other folds through this same solve. The cap bounds
            the reaction flux for whatever trial space produced it and is read
            out of sample, so it ranks direction sets and sampler settings
            without a reference committor; ``gap`` (held-out minus in-sample
            cap) is the overfitting diagnostic. Off by default (costs the
            per-fold moments and K small solves).
        raise_on_degenerate: raise :class:`RepresentationError` when the
            moment gap collapses (``cond < 1e-6``) instead of returning weights.
        counts: ``(N,)`` frame multiplicities of a bootstrap replicate
            (:func:`bootstrap_weights`); None means every frame once.

    Returns:
        :class:`Weights`.
    """
    _require_x64()
    if isinstance(tikhonov, str):
        if tikhonov not in TIKHONOV_RULES:
            raise ValueError(
                f"tikhonov must be one of {TIKHONOV_RULES} or a float (absolute ridge); got {tikhonov!r}"
            )
        rule = tikhonov
    else:
        rule = float(tikhonov)
        if not rule >= 0.0:
            raise ValueError(f"an absolute ridge must be >= 0; got {tikhonov!r}")
    if heldout_cap and rule != "halfset_eigen":
        raise ValueError(
            "heldout_cap=True needs tikhonov='halfset_eigen': the cap is read through the half-set solve."
        )
    if result.projected_samples is None:
        raise ValueError("solve_weights needs result.projected_samples.")

    valid_mask = jnp.asarray(result.valid_mask, bool)
    n_valid = int(np.asarray(valid_mask).sum())
    W = _sample_weights(result, counts)
    F, W, cos_matrix, G = _assemble(result, W)
    a, b = compute_basin_moments(result, counts=counts)
    diagnostics = {"a": a, "b": b, "N_eff": _effective_n(W)}
    cap_info = None

    if rule == "halfset_eigen" and n_valid < 2:
        # With one valid direction the constraint fixes w outright; nothing to filter.
        logger.warning(
            "tikhonov='halfset_eigen': only %d valid direction(s); falling back to 'auto'.", n_valid
        )
        rule = "auto"
        heldout_cap = False

    if rule == "halfset_eigen":
        strata = hs.basin_strata(result.in_A, result.in_B)
        fold_of = hs.make_folds(int(W.shape[0]), hs.N_FOLDS, strata=strata)
        G_folds, w_folds = hs.fold_gram_blocks(F, W, cos_matrix, fold_of, hs.N_FOLDS)
        del F
        if heldout_cap:
            a_folds, b_folds, wA, wB = hs.fold_basin_moments(result, fold_of, hs.N_FOLDS)
            cap_info = _heldout_cap(G_folds, w_folds, a_folds, b_folds, wA, wB, valid_mask)
        G1, G2 = hs.halfset_grams(G_folds, w_folds)
        del G_folds
        G_reg, info = hs.regularized_gram(G1, G2, valid_mask)
        out = _solve(G_reg, a, b, valid_mask)
        ridge = 0.0
        diagnostics.update(
            G_reg=jnp.asarray(G_reg),
            band_ssnr=[float(s) for s in info["band_ssnr"]],
            lam_max=float(np.max(info["lam"])),
            lam_min_reg=float(np.min(info["lam_reg"])),
            n_folds=hs.N_FOLDS,
            n_bands=hs.N_BANDS,
        )
    else:
        del F
        if rule == "auto":
            diag = np.asarray(jnp.diag(G))[np.asarray(valid_mask)]
            med = float(np.median(diag)) if diag.size else 1.0
            med = med if np.isfinite(med) and med > 0 else 1.0
            ridge = max(1e-12, 1.0 / diagnostics["N_eff"] ** 0.5) * med
        else:
            ridge = rule
        G_reg = _ridged_gram(G, valid_mask, ridge)
        out = _solve(G_reg, a, b, valid_mask)
    cond = _gate(out, raise_on_degenerate)

    w = out["w"]
    c = float(out["c"])
    w_np = np.asarray(w, np.float64)
    G_np = np.asarray(G, np.float64)
    a_np, b_np = np.asarray(a, np.float64), np.asarray(b, np.float64)
    valid_np = np.asarray(valid_mask)
    G_diag = np.diag(G_np)
    denom = np.sqrt(np.maximum(np.outer(G_diag, G_diag), 1e-30))
    offdiag = ~np.eye(w_np.size, dtype=bool) & np.outer(valid_np, valid_np)
    diagnostics.update(
        A=float(out["A"]),
        B=float(out["B"]),
        C=float(out["C"]),
        cond_chol=float(out["cond_chol"]),
        mu_A_check=float(a_np @ w_np + c),
        mu_B_check=float(b_np @ w_np + c),
        constraint_residual=float(abs((b_np - a_np) @ w_np - 1.0)),
        sum_w=float(w_np.sum()),
        n_negative_weights=int(((w_np < 0) & valid_np).sum()),
        off_diagonal_magnitude=float(np.abs(G_np / denom)[offdiag].mean())
        if offdiag.any()
        else 0.0,
    )
    return Weights(
        w=w,
        c=c,
        dirichlet_energy=float(w_np @ G_np @ w_np),
        moment_gap=float(out["R"]),
        cond=cond,
        ridge=ridge,
        tikhonov=tikhonov,
        heldout_cap=cap_info,
        diagnostics=diagnostics,
    )


def _heldout_cap(G_folds, w_folds, a_folds, b_folds, wA, wB, valid_mask):
    """The Dirichlet cap read out of sample through the half-set solve.

    For each fold ``k`` the other folds are the training set: their even and
    odd halves give the half-Grams the filter reads the noise off, the solve
    runs on the filtered training Gram, and the cap
    ``w^T G_k w / ((b_k - a_k) . w)^2`` is evaluated on the fold held out.
    ``per_fold_train`` is the same cap read on the training Gram, so
    ``gap = mean(held-out - train)`` is the overfitting diagnostic.
    """
    G_folds = np.asarray(G_folds, np.float64)
    w_folds = np.asarray(w_folds, np.float64)
    a_folds = np.asarray(a_folds, np.float64)
    b_folds = np.asarray(b_folds, np.float64)
    wA = np.asarray(wA, np.float64)
    wB = np.asarray(wB, np.float64)
    K = G_folds.shape[0]
    caps = np.full(K, np.nan)
    caps_train = np.full(K, np.nan)
    for k in range(K):
        other = np.array([j for j in range(K) if j != k])
        if (
            wA[k] <= 0
            or wB[k] <= 0
            or wA[other].sum() <= 0
            or wB[other].sum() <= 0
            or other.size < 2
        ):
            continue
        wo = w_folds[other]
        Gtr = np.tensordot(wo, G_folds[other], axes=(0, 0)) / max(wo.sum(), 1e-300)
        atr = (wA[other] @ a_folds[other]) / max(wA[other].sum(), 1e-300)
        btr = (wB[other] @ b_folds[other]) / max(wB[other].sum(), 1e-300)
        G1, G2 = hs.halfset_grams(G_folds, w_folds, folds=other)
        try:
            G_reg, _ = hs.regularized_gram(G1, G2, valid_mask)
            out = _solve(G_reg, atr, btr, valid_mask)
        except (np.linalg.LinAlgError, ValueError):
            continue
        w = np.asarray(out["w"], np.float64)
        if not np.all(np.isfinite(w)):
            continue
        gap_k = float((b_folds[k] - a_folds[k]) @ w)
        gap_tr = float((btr - atr) @ w)
        if abs(gap_k) > 1e-30 and abs(gap_tr) > 1e-30:
            caps[k] = float(w @ G_folds[k] @ w) / gap_k**2
            caps_train[k] = float(w @ Gtr @ w) / gap_tr**2
    n_ok = int(np.isfinite(caps).sum())
    if n_ok == 0:
        raise ValueError("heldout cap: undefined on every fold.")
    with np.errstate(invalid="ignore"):
        mean = float(np.nanmean(caps))
        se = float(np.nanstd(caps, ddof=1) / np.sqrt(n_ok)) if n_ok > 1 else float("nan")
        gap = float(np.nanmean(caps - caps_train))
    return dict(
        cap=mean,
        se=se,
        per_fold=caps,
        per_fold_train=caps_train,
        gap=gap,
        n_folds=int(K),
        n_ok=n_ok,
    )


class Bootstrap(NamedTuple):
    """Block-bootstrap replicates of a weight solve (one row per replicate).

    ``se`` of any quantity is the standard deviation over the rows. Variance
    only: when barrier undersampling dominates the error the spread is far too
    small (it cannot see bias), and ``block_len`` must reach the integrated
    autocorrelation time of the frames or the spread is optimistic.
    """

    w: np.ndarray
    c: np.ndarray
    moment_gap: np.ndarray
    dirichlet_energy: np.ndarray
    q: np.ndarray | None
    n_ok: int


def bootstrap_weights(
    result, weights, *, n_boot=40, block_len, run_ids=None, seed=0, points=None, min_basin_frames=5
):
    """Block bootstrap over frames with the 1D basis held fixed.

    Contiguous blocks of ``block_len`` frames are resampled with replacement
    (never across ``run_ids`` boundaries, so independent trajectories are never
    spliced), the resampled frame multiplicities reweight the Gram and the
    basin moments, and :func:`solve_weights` is re-run with the same
    ``tikhonov`` rule the fit used. ``points`` (optional, ``(P, dim)``) adds the
    replicate committor values there.
    """
    from .committor import build_committor

    rng = np.random.default_rng(seed)
    N = int(result.projected_samples.shape[1])
    runs = np.zeros(N, np.int64) if run_ids is None else np.asarray(run_ids)
    starts = np.concatenate(
        [np.arange(lo, hi, block_len) for lo, hi in _run_segments(runs)]
        if N
        else [np.zeros(0, np.int64)]
    )
    ends = np.minimum(
        starts + block_len,
        np.concatenate(
            [
                np.repeat(hi, ((hi - lo) + block_len - 1) // block_len)
                for lo, hi in _run_segments(runs)
            ]
        ),
    )
    in_A = np.asarray(result.in_A, bool)
    in_B = np.asarray(result.in_B, bool)
    ws, cs, gaps, energies, qs = [], [], [], [], []
    for _ in range(n_boot):
        pick = rng.choice(starts.size, size=starts.size, replace=True)
        counts = np.zeros(N, np.float64)
        for p in pick:
            counts[starts[p] : ends[p]] += 1.0
        if (counts[in_A] > 0).sum() < min_basin_frames or (
            counts[in_B] > 0
        ).sum() < min_basin_frames:
            continue
        try:
            rep = solve_weights(
                result, tikhonov=weights.tikhonov, raise_on_degenerate=False, counts=counts
            )
        except (ValueError, np.linalg.LinAlgError):
            continue
        ws.append(np.asarray(rep.w, np.float64))
        cs.append(rep.c)
        gaps.append(rep.moment_gap)
        energies.append(rep.dirichlet_energy)
        if points is not None:
            qs.append(np.asarray(build_committor(result, rep)(jnp.asarray(points)), np.float64))
    return Bootstrap(
        w=np.asarray(ws),
        c=np.asarray(cs),
        moment_gap=np.asarray(gaps),
        dirichlet_energy=np.asarray(energies),
        q=np.asarray(qs) if points is not None else None,
        n_ok=len(ws),
    )


def _run_segments(runs):
    """``(lo, hi)`` index ranges of the contiguous runs in ``run_ids``."""
    runs = np.asarray(runs)
    if runs.size == 0:
        return []
    change = np.flatnonzero(runs[1:] != runs[:-1]) + 1
    bounds = np.concatenate([[0], change, [runs.size]])
    return list(pairwise(bounds))
