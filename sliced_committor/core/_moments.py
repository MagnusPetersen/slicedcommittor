"""Basin-conditional moments of the slice committors.

    a_j = <q_j(theta_j . x)>_A,    b_j = <q_j(theta_j . x)>_B

are the two numbers per slice that the weight solve is constrained against:
the recombined committor has to average to 0 over basin A and to 1 over basin
B. They are UNWEIGHTED basin means (the moments the published solve is
constrained against); ``counts`` lets a bootstrap resample the frames, and
``fold_of`` adds the per-fold moments the held-out cap reads. Everything is
read off ONE pass over the slice committors at the samples, streamed in
direction batches by :func:`slice_values`.
"""

import logging
from functools import partial
from typing import NamedTuple

import jax.numpy as jnp
import numpy as np
from jax import jit, lax, vmap

from .solver import _interp_committor

logger = logging.getLogger(__name__)

BATCH = 512  # directions per block: bounds every (batch, N) transient


@partial(jit, static_argnames=("size",))
def _interpolate(slice_coords, committors_1d, projected_samples, start, size):
    """``(size, N)`` slice committors of directions ``start:start + size``, clipped to ``[0, 1]``.

    The projections are sliced inside the jit, so the block itself is the
    only transient.
    """
    ps = lax.dynamic_slice_in_dim(projected_samples, start, size, axis=0)
    return vmap(_interp_committor)(slice_coords, committors_1d, ps)


def slice_values(result, batch_size: int = BATCH):
    """Yield the blocks ``Q[j, n] = q_j(theta_j . x_n)``, ``(batch, N)``, in direction order.

    A generator, so a single solve never holds more than one block; a
    bootstrap materialises the tuple once and reuses it.
    """
    M = result.projected_samples.shape[0]
    for start in range(0, M, batch_size):
        size = min(batch_size, M - start)
        yield _interpolate(
            result.slice_coords[start : start + size],
            result.committors_1d[start : start + size],
            result.projected_samples,
            start,
            size,
        )


@jit
def _masked_row_sums(Q, mask):
    """``sum_n Q[j, n]`` over the frames where ``mask`` is True."""
    return jnp.sum(jnp.where(mask, Q, 0.0), axis=1)


class FoldMoments(NamedTuple):
    """The unweighted basin moments of every slice per fold, ``(K, M)``, and the basin counts per fold, ``(K,)``."""

    a: np.ndarray
    b: np.ndarray
    wA: np.ndarray
    wB: np.ndarray


class BasinMoments(NamedTuple):
    """``a``, ``b`` of every slice and, when folds were requested, per fold."""

    a: jnp.ndarray
    b: jnp.ndarray
    folds: FoldMoments | None = None


def basin_moments(result, chunks, *, counts=None, fold_of=None) -> BasinMoments:
    """The basin means of every slice committor, read off the streamed ``chunks``.

    Args:
        result: a :class:`~sliced_committor.SlicedCommittorResult`.
        chunks: the ``(batch, N)`` blocks of :func:`slice_values`, in order.
        counts: optional ``(N,)`` frame multiplicities (a bootstrap replicate);
            None means every frame once.
        fold_of: optional ``(N,)`` fold labels; adds the unweighted per-fold
            moments with the same reductions as the single-fold ones. The two
            options are exclusive (the cap is never read on a replicate).
    """
    in_A = jnp.asarray(result.in_A)
    in_B = jnp.asarray(result.in_B)
    n_A_int = int(jnp.sum(in_A))
    n_B_int = int(jnp.sum(in_B))
    if n_A_int < 50 or n_B_int < 50:
        logger.warning(
            "basin moments: few basin samples (n_A=%d, n_B=%d); the moments will be noisy.",
            n_A_int,
            n_B_int,
        )
    if counts is None:
        w_A = in_A.astype(jnp.float64)
        w_B = in_B.astype(jnp.float64)
    else:
        counts = jnp.asarray(counts, dtype=jnp.float64)
        w_A = counts * in_A
        w_B = counts * in_B
    n_A = jnp.maximum(jnp.sum(w_A), 1.0)
    n_B = jnp.maximum(jnp.sum(w_B), 1.0)

    fold_masks = None
    if fold_of is not None:
        if counts is not None:
            raise ValueError("basin_moments: the per-fold moments are unweighted (no counts).")
        fold_of = np.asarray(fold_of)
        n_folds = int(fold_of.max()) + 1
        in_A_np, in_B_np = np.asarray(in_A, bool), np.asarray(in_B, bool)
        wA = np.bincount(fold_of[in_A_np], minlength=n_folds).astype(np.float64)
        wB = np.bincount(fold_of[in_B_np], minlength=n_folds).astype(np.float64)
        if (wA <= 0).any() or (wB <= 0).any():
            raise ValueError(
                "basin_moments: some fold contains no samples of a basin; reduce n_folds."
            )
        fold_masks = [
            (jnp.asarray((fold_of == k) & in_A_np), jnp.asarray((fold_of == k) & in_B_np))
            for k in range(n_folds)
        ]
        M = result.projected_samples.shape[0]
        a_f, b_f = np.zeros((n_folds, M)), np.zeros((n_folds, M))

    a_chunks, b_chunks = [], []
    start = 0
    for Q in chunks:
        end = start + Q.shape[0]
        if counts is None:
            a_chunks.append(_masked_row_sums(Q, in_A) / n_A)
            b_chunks.append(_masked_row_sums(Q, in_B) / n_B)
        else:
            a_chunks.append((Q @ w_A) / n_A)
            b_chunks.append((Q @ w_B) / n_B)
        if fold_masks is not None:
            for k, (mask_A, mask_B) in enumerate(fold_masks):
                a_f[k, start:end] = np.asarray(_masked_row_sums(Q, mask_A)) / wA[k]
                b_f[k, start:end] = np.asarray(_masked_row_sums(Q, mask_B)) / wB[k]
        start = end
    folds = None if fold_masks is None else FoldMoments(a_f, b_f, wA, wB)
    return BasinMoments(jnp.concatenate(a_chunks), jnp.concatenate(b_chunks), folds)
