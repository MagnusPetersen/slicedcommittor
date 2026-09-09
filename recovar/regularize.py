"""Halfset (FSC-style) Wiener regularization of the Gram matrix (RECOVAR sec. 1).

The transfer: split the data in two halves, assemble a Gram matrix from each,
and read the sampling noise off their DISAGREEMENT, band by band, exactly as
cryoEM FSC reads the noise off two half-map reconstructions. The measured
per-band SSNR then sets a Wiener ridge with no tuned constant. Benchmarked in
``_reference/REPORT.md`` sec. 1: tuning-free, and it beats a ridge tuned
against the truth on the hard (ill-conditioned) cases, with the flux ratio
staying 0.98-1.02 where the default ridge gives 0.70-0.92.

This module also carries the two convenience entry points that produce the
halfset Grams (from a ``recovar`` basis, or from arrays of an existing
``sliced_committor`` lib fit) and the closed-form solve on an
already-regularized G.
"""
import numpy as np
from scipy.linalg import cho_factor, cho_solve, eigh


def _ssnr_from_corr(c):
    """FSC -> SSNR of the combined (full) dataset: SSNR = 2c/(1-c)."""
    c = np.clip(c, 0.0, 0.999999)
    return 2.0 * c / (1.0 - c)


def halfset_eigen_regularize(G1, G2, n_bands=12):
    """Wiener ridge in eigen-bands of Gbar (the pragmatic shell index).

    G_reg = V diag(lam_k (1 + 1/SSNR_k)) V',  so the *inverse* is damped by
    the Wiener factor SSNR/(1+SSNR) band by band. PSD by construction.

    The SSNR of band k is read from the correlation of the band's rows of
    V'G1V against V'G2V (the two halfset Grams expressed in the eigenbasis of
    their average), via the FSC identity SSNR = 2c/(1-c).

    Validated lesson (``_reference/REPORT.md`` sec. 1): the sampling noise of
    G is NOT diagonal in any a-priori slice-property basis; it is diagonal in
    the EIGENBASIS of G. The literal RECOVAR analogue (shell pairs indexed by
    each slice's own profile bandwidth; ``halfset_shell_regularize`` in the
    prototype) is refuted and deliberately not ported: its measured shell
    SSNRs come out at 1e3-1e5, so the Wiener factor is ~1 and no shrinkage
    happens, and its two SSNR estimators disagree in that grouping
    (corr -0.02..0.37) -- which is itself the signal that the grouping is
    wrong.

    Returns (G_reg, info) with info = dict(band_ssnr, lam, lam_reg).
    """
    G1 = np.asarray(G1, dtype=np.float64)
    G2 = np.asarray(G2, dtype=np.float64)
    M = G1.shape[0]
    Gbar = 0.5 * (G1 + G2)
    lam, V = eigh(Gbar)
    order = np.argsort(-lam)
    lam, V = lam[order], V[:, order]
    A1 = V.T @ G1 @ V
    A2 = V.T @ G2 @ V

    edges = np.linspace(0, M, n_bands + 1).astype(int)
    ssnr = np.zeros(M)
    band_ssnr = np.zeros(n_bands)
    for k in range(n_bands):
        sl = slice(edges[k], edges[k + 1])
        v1 = A1[sl, :].ravel()
        v2 = A2[sl, :].ravel()
        if v1.size < 3 or v1.std() == 0 or v2.std() == 0:
            s = 1e6
        else:
            s = _ssnr_from_corr(np.corrcoef(v1, v2)[0, 1])
        band_ssnr[k] = s
        ssnr[sl] = s
    lam_reg = np.maximum(lam, 0.0) * (1.0 + 1.0 / np.maximum(ssnr, 1e-12))
    floor = 1e-12 * max(lam_reg.max(), 1e-300)
    lam_reg = np.maximum(lam_reg, floor)
    G_reg = (V * lam_reg) @ V.T
    return G_reg, dict(band_ssnr=band_ssnr, lam=lam, lam_reg=lam_reg)


def solve_with_G(G_reg, a, b):
    """Closed-form dual solve on an ALREADY-regularized Gram matrix.

        w_dual = G_reg^{-1} (b - a),   M_gap = (b - a)' w_dual,
        w = w_dual / M_gap,            c = -a' w,     nu_hat = 1 / M_gap.

    Counterpart of ``assemble.solve_dual(G, delta, G_is_regularised=True)``
    that also carries the affine offset c, so the caller can evaluate
    qbar = c + sum_j w_j q_j directly from (G_reg, a, b) -- the intended use
    after ``halfset_eigen_regularize``, whose output must not be re-ridged.
    Cholesky solve with an lstsq fallback for a (numerically) singular G_reg.

    Returns dict(w, c, M_gap, nu_hat, w_dual), all float64.
    """
    G_reg = np.asarray(G_reg, dtype=np.float64)
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    delta = b - a
    try:
        cf = cho_factor(G_reg, lower=True)
        w_dual = cho_solve(cf, delta)
    except np.linalg.LinAlgError:
        w_dual = np.linalg.lstsq(G_reg, delta, rcond=None)[0]
    M_gap = float(delta @ w_dual)
    w = w_dual / M_gap if abs(M_gap) > 1e-300 else w_dual
    return dict(w=w, c=float(-a @ w), M_gap=M_gap,
                nu_hat=1.0 / M_gap if abs(M_gap) > 1e-300 else np.inf,
                w_dual=w_dual)


def halfset_grams_recovar(basis, X, in_A, in_B, split, sample_weights=None):
    """Halfset Grams via ``recovar.assemble`` on the two halves of ``split``.

    ``split`` is a pair (half 1, half 2) of index arrays or boolean masks, as
    produced by ``recovar.splits`` (``iid_split`` for i.i.d. draws,
    ``block_split`` for time-correlated frames -- on MD data the block split
    is load-bearing, see ``_reference/REPORT.md`` sec. 10). Sample weights,
    when given, are sliced per half; each assembly renormalises internally.

    One convenience wrapper used by the IDS fitter and the experiments: the
    two half Grams feed ``halfset_eigen_regularize`` and the full-data
    moments feed ``solve_with_G``.

    Returns (G1, G2, Gfull, a, b) with Gfull/a/b assembled from the FULL
    data; all float64.
    """
    # Local import: keeps this module importable (and its pure functions
    # testable) without the sibling assembly module.
    from .assemble import assemble

    i1, i2 = split
    i1 = np.asarray(i1)
    i2 = np.asarray(i2)
    if i1.dtype == bool:
        i1 = np.flatnonzero(i1)
    if i2.dtype == bool:
        i2 = np.flatnonzero(i2)
    w = None if sample_weights is None else np.asarray(sample_weights, dtype=np.float64)
    G1, _, _ = assemble(basis, X[i1], in_A[i1], in_B[i1],
                        sample_weights=None if w is None else w[i1])
    G2, _, _ = assemble(basis, X[i2], in_A[i2], in_B[i2],
                        sample_weights=None if w is None else w[i2])
    Gfull, a, b = assemble(basis, X, in_A, in_B, sample_weights=w)
    return G1, G2, Gfull, a, b


def halfset_grams_from_arrays(F, W, cos_matrix, in_A, in_B, n_folds=10):
    """Halfset Grams for a fit made by the ``sliced_committor`` lib pipeline.

    Builds ``n_folds`` contiguous, basin-stratified folds with
    ``sliced_committor.core._ridge_cv.make_folds`` (strata 0=A, 1=B,
    2=transition, blocked WITHIN each stratum) and recombines the per-fold
    Gram blocks of ``fold_gram_blocks`` into two interleaved halves:

        G1 = weighted mean of the even-fold blocks,
        G2 = weighted mean of the odd-fold blocks.

    Each fold block is normalised by its own weight sum exactly as
    ``fold_gram_blocks`` normalises it, so the weighted-mean recombination is
    exact: G is a weighted MEAN over samples, and the folds partition them.
    Interleaving even/odd folds alternates contiguous blocks between the
    halves (the ``block_split`` pattern), which is what makes the halfset
    disagreement honest on serially correlated frames.

    FRAMES MUST BE TIME-ORDERED. The contiguous stratified folds assume
    serial order within each stratum; on a shuffled frame array near-duplicate
    frames land in both halves and the measured SSNR is optimistic (on
    correlated data a random split does not merely understate the noise, it
    can invert the diagnostic's sign -- ``_reference/REPORT.md`` sec. 10).

    Args:
        F: (M, N) per-frame slice-derivative matrix (the Gram feature of the
            lib pipeline), float32 or float64.
        W: (N,) per-frame sample weights (need not be normalised), or None
            for uniform.
        cos_matrix: (M, M) = directions @ directions.T, float64.
        in_A, in_B: (N,) boolean basin membership.
        n_folds: number of folds; an even value gives balanced halves.

    Returns (G1, G2), float64 (M, M).
    """
    # Local import: the lib pipeline (and its jax import chain) is only
    # needed on this path.
    from sliced_committor.core._ridge_cv import fold_gram_blocks, make_folds

    if n_folds < 2:
        raise ValueError(
            f"halfset_grams_from_arrays: n_folds={n_folds} leaves one half "
            "empty (its Gram would be 0/0 = NaN); need n_folds >= 2.")
    in_A = np.asarray(in_A, bool)
    in_B = np.asarray(in_B, bool)
    N = in_A.shape[0]
    W = np.full(N, 1.0 / N) if W is None else np.asarray(W, dtype=np.float64)
    strata = np.where(in_A, 0, np.where(in_B, 1, 2))
    fold_of = make_folds(N, n_folds, contiguous=True, strata=strata)
    G_folds, w_folds = fold_gram_blocks(F, W, cos_matrix, fold_of, n_folds)
    even = np.arange(0, n_folds, 2)
    odd = np.arange(1, n_folds, 2)
    G1 = np.tensordot(w_folds[even], G_folds[even], axes=(0, 0)) / w_folds[even].sum()
    G2 = np.tensordot(w_folds[odd], G_folds[odd], axes=(0, 0)) / w_folds[odd].sum()
    return G1, G2
