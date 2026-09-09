"""Half-set methods: contiguous folds, half-Gram pairs and the eigenband filter.

The Gram matrix ``G`` of the slice basis is a sample average, and the weight
solve inverts it, so its sampling noise has to be controlled. The half-set
idea (the cryo-EM FSC construction carried over to the committor problem):
deal the frames into two halves, assemble a Gram matrix from each, and read
the noise off their DISAGREEMENT band by band in the eigenbasis of their
average. Each eigenvalue is then inflated by ``1 + 1/SSNR``, which damps the
inverse by the Wiener factor ``SNR/(1 + SNR)``. No tuned constant anywhere.

Two facts are load-bearing and encoded here rather than left to the caller:

* Molecular-dynamics frames are serially correlated, so the halves must be
  built from CONTIGUOUS blocks, dealt alternately; a random frame split puts
  near-duplicates on both sides and reports the noise as smaller than it is.
  ``make_folds`` blocks within each basin stratum so every fold holds samples
  of both basins.
* The sampling noise of ``G`` is diagonal in the eigenbasis of ``G`` and not
  in any a-priori slice-property basis; that is why the bands are eigenbands.

The paper calls this the half-set spectral filter. It is the default
regularisation of :func:`sliced_committor.solve_weights`
(``tikhonov='halfset_eigen'``).
"""

import jax.numpy as jnp
import numpy as np

from ._moments import interpolate_masked
from .gram import _assemble_gram_matrix

N_FOLDS = 10
N_BANDS = 12


def make_folds(N: int, n_folds: int, *, strata=None):
    """Fold label per sample: consecutive blocks, blocked within each stratum.

    Frames adjacent in a stratum are still adjacent in time, so the
    anti-leakage property of contiguous blocks survives the stratification,
    and every fold is guaranteed samples of every basin.
    """
    if strata is None:
        return (np.arange(N) * n_folds) // N
    strata = np.asarray(strata)
    out = np.empty(N, np.int64)
    for s in np.unique(strata):
        idx = np.flatnonzero(strata == s)
        out[idx] = (np.arange(len(idx)) * n_folds) // max(len(idx), 1)
    return out


def basin_strata(in_A, in_B):
    """0 = A, 1 = B, 2 = transition; the stratification every fold uses."""
    in_A = np.asarray(in_A, bool)
    in_B = np.asarray(in_B, bool)
    return np.where(in_A, 0, np.where(in_B, 1, 2))


def fold_gram_blocks(F, W, cos_matrix, fold_of, n_folds):
    """Per-fold Gram blocks ``G_k`` and their weight sums, at the cost of ONE assembly.

    Each block is normalised by its own weight sum, so any train/test
    combination is a weighted mean of blocks. The folds partition the samples,
    so the total cost stays ``O(M^2 N)``.
    """
    fold_of = np.asarray(fold_of)
    G_folds, w_folds = [], []
    for k in range(n_folds):
        idx = np.flatnonzero(fold_of == k)
        if idx.size == 0:
            raise ValueError(f"fold_gram_blocks: fold {k} is empty.")
        lo, hi = int(idx[0]), int(idx[-1]) + 1
        Fk = F[:, lo:hi] if hi - lo == idx.size else F[:, jnp.asarray(idx)]
        Wk = W[lo:hi] if hi - lo == idx.size else W[jnp.asarray(idx)]
        tot = float(jnp.sum(Wk))
        if tot <= 0:
            raise ValueError(f"fold_gram_blocks: fold {k} carries zero weight.")
        G_folds.append(np.asarray(_assemble_gram_matrix(Fk, Wk / tot, cos_matrix), np.float64))
        w_folds.append(tot)
        del Fk, Wk
    return np.stack(G_folds), np.asarray(w_folds, np.float64)


def fold_basin_moments(result, fold_of, n_folds, batch_size: int = 512):
    """Per-fold basin moments ``(a_folds, b_folds, wA_folds, wB_folds)``.

    Unweighted counts, matching :func:`~sliced_committor.core._moments.compute_basin_moments`.
    The interpolation is done once per direction batch and reused for every
    fold; the reductions are bitwise the same as the single-fold ones.
    """
    in_A = np.asarray(result.in_A, bool)
    in_B = np.asarray(result.in_B, bool)
    M = result.projected_samples.shape[0]
    fold_of = np.asarray(fold_of)
    in_A_j, in_B_j = jnp.asarray(in_A), jnp.asarray(in_B)
    fold_j = [jnp.asarray(fold_of == k) for k in range(n_folds)]
    wA = np.array([float((in_A & (fold_of == k)).sum()) for k in range(n_folds)])
    wB = np.array([float((in_B & (fold_of == k)).sum()) for k in range(n_folds)])
    if (wA <= 0).any() or (wB <= 0).any():
        raise ValueError(
            "fold_basin_moments: some fold contains no samples of a basin; reduce n_folds."
        )
    a = np.zeros((n_folds, M))
    b = np.zeros((n_folds, M))
    for start in range(0, M, batch_size):
        end = min(start + batch_size, M)
        s_grid = result.slice_coords[start:end]
        q_grid = result.committors_1d[start:end]
        ps = result.projected_samples[start:end]
        qA = interpolate_masked(s_grid, q_grid, ps, in_A_j)
        qB = interpolate_masked(s_grid, q_grid, ps, in_B_j)
        for k in range(n_folds):
            a[k, start:end] = np.asarray(jnp.sum(jnp.where(fold_j[k], qA, 0.0), axis=1)) / wA[k]
            b[k, start:end] = np.asarray(jnp.sum(jnp.where(fold_j[k], qB, 0.0), axis=1)) / wB[k]
        del qA, qB
    return a, b, wA, wB


def halfset_grams(G_folds, w_folds, folds=None):
    """The two half-Grams: even folds dealt to one half, odd folds to the other.

    Each half is the weight-weighted mean of its fold blocks, so it IS the Gram
    matrix of that half of the frames. FRAMES MUST BE TIME-ORDERED within each
    stratum: on shuffled frames the two halves are not independent and the
    measured SSNR is optimistic.
    """
    G_folds = np.asarray(G_folds, np.float64)
    w_folds = np.asarray(w_folds, np.float64)
    folds = np.arange(G_folds.shape[0]) if folds is None else np.asarray(folds)
    if folds.size < 2:
        raise ValueError("halfset_grams: at least two folds are needed to form two halves.")
    even, odd = folds[0::2], folds[1::2]
    G1 = np.tensordot(w_folds[even], G_folds[even], axes=(0, 0)) / max(w_folds[even].sum(), 1e-300)
    G2 = np.tensordot(w_folds[odd], G_folds[odd], axes=(0, 0)) / max(w_folds[odd].sum(), 1e-300)
    return G1, G2


def _ssnr_from_corr(c):
    """FSC to SSNR of the combined dataset: ``SSNR = 2c / (1 - c)``."""
    c = np.clip(c, 0.0, 0.999999)
    return 2.0 * c / (1.0 - c)


def halfset_eigen_regularize(G1, G2, n_bands=N_BANDS):
    """Wiener-filtered Gram from a half-Gram pair, in the eigenbands of their mean.

    ``lam_reg = max(lam, 0) (1 + 1/SSNR_k)`` band by band, floored at
    ``1e-12 max(lam_reg)`` so the result is strictly positive definite.
    Returns ``(G_reg, info)`` with ``info = dict(band_ssnr, lam, lam_reg)``.
    """
    from scipy.linalg import eigh

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


def regularized_gram(G1, G2, valid_mask, n_bands=N_BANDS):
    """The filtered Gram on the full direction set: the valid block is
    regularised, invalid directions get decoupled identity rows.

    The restriction happens BEFORE the eigendecomposition: the zero rows of a
    masked direction would otherwise poison the low eigenbands' SSNR.
    """
    valid = np.asarray(valid_mask, bool)
    iv = np.flatnonzero(valid)
    M = valid.size
    G_reg_v, info = halfset_eigen_regularize(
        np.asarray(G1, np.float64)[np.ix_(iv, iv)],
        np.asarray(G2, np.float64)[np.ix_(iv, iv)],
        n_bands=n_bands,
    )
    G_reg = np.eye(M)
    G_reg[np.ix_(iv, iv)] = G_reg_v
    return G_reg, info
