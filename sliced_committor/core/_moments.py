"""Basin-conditional moments of the slice committors.

    a_j = <q_j(theta_j . x)>_A,    b_j = <q_j(theta_j . x)>_B

are the two numbers per slice that the weight solve is constrained against:
the recombined committor has to average to 0 over basin A and to 1 over basin
B. They are UNWEIGHTED basin means (the moments the published solve is
constrained against); ``counts`` lets a bootstrap resample the frames.
"""

import logging

import jax.numpy as jnp
from jax import jit, vmap

from .solver import _interp_1d_at_samples

logger = logging.getLogger(__name__)


@jit
def interpolate_masked(slice_coords, committors_1d, projected_samples, mask):
    """``(M, N)`` slice committors at the samples, zero where ``mask`` is False."""

    def _interp_single(s_grid, q_grid, s_proj):
        q = jnp.clip(_interp_1d_at_samples(s_grid, q_grid, s_proj), 0.0, 1.0)
        return jnp.where(mask, q, 0.0)

    return vmap(_interp_single)(slice_coords, committors_1d, projected_samples)


def compute_basin_moments(result, counts=None, batch_size: int = 512):
    """``(a, b)``: the basin means of every slice committor.

    Args:
        result: a :class:`~sliced_committor.SlicedCommittorResult`.
        counts: optional ``(N,)`` frame multiplicities (a bootstrap replicate);
            None means every frame once.
        batch_size: directions per chunk, bounding the ``(batch, N)`` buffer.
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

    M = result.projected_samples.shape[0]
    a_chunks, b_chunks = [], []
    for start in range(0, M, batch_size):
        end = min(start + batch_size, M)
        s_grid = result.slice_coords[start:end]
        q_grid = result.committors_1d[start:end]
        ps = result.projected_samples[start:end]
        if counts is None:
            q_A = interpolate_masked(s_grid, q_grid, ps, in_A)
            q_B = interpolate_masked(s_grid, q_grid, ps, in_B)
            a_chunks.append(jnp.sum(q_A, axis=1) / n_A)
            b_chunks.append(jnp.sum(q_B, axis=1) / n_B)
        else:
            q_A = interpolate_masked(s_grid, q_grid, ps, in_A)
            q_B = interpolate_masked(s_grid, q_grid, ps, in_B)
            a_chunks.append((q_A @ w_A) / n_A)
            b_chunks.append((q_B @ w_B) / n_B)
        del q_A, q_B
    return jnp.concatenate(a_chunks), jnp.concatenate(b_chunks)
