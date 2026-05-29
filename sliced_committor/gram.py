"""Shared Gram-assembly primitives and diagnostics for the sliced committor.

Both the full-Gram simplex solver (:func:`sliced_committor.weights.full_gram_weights`)
and the basin-moment-constrained solver
(:func:`sliced_committor.basin_moment_weights`) share a single derivative
pipeline, Gram assembler, and post-solve diagnostic block. Functions here
duck-type ``ctx`` as anything with ``slice_coords``, ``committors_1d``, and
``free_energies`` attributes so this module stays import-light (no circular
dependency on the full ``WeightingContext`` definition in ``solver.py``).
"""

import jax.numpy as jnp
from jax import jit, vmap


@jit
def _compute_derivative_matrix(ctx, projected_samples):
    """Piecewise-constant derivative matrix F where F[j,n] = dq_j/ds(θ_j · x_n).

    Uses the piecewise-constant slope of the linearly interpolated committor
    (the exact derivative of the piecewise-linear interpolant used in
    evaluate_committor), rather than central-diff + re-interpolation.

    Precomputes slopes as (M, n_bins-1) and gathers via take_along_axis
    to avoid per-sample division inside the vmap.

    Args:
        ctx: WeightingContext with slice_coords and committors_1d.
        projected_samples: (M, N) pre-computed projections θ_j · x_n.

    Returns:
        F: (M, N) array of committor derivatives at sample positions.
    """
    # Precompute piecewise-constant slopes: (M, n_bins-1)
    ds = jnp.diff(ctx.slice_coords, axis=1)
    dq = jnp.diff(ctx.committors_1d, axis=1)
    slopes = dq / jnp.maximum(ds, 1e-10)

    n_bins = ctx.slice_coords.shape[1]

    def _find_bins(s_grid, s_proj):
        idx = jnp.searchsorted(s_grid, s_proj, side="right") - 1
        return jnp.clip(idx, 0, n_bins - 2)

    bin_indices = vmap(_find_bins)(ctx.slice_coords, projected_samples)  # (M, N)
    return jnp.take_along_axis(slopes, bin_indices, axis=1)  # (M, N)


@jit
def _assemble_gram_matrix(F, W, cos_matrix):
    """Assemble the cross-Dirichlet Gram matrix.

    G_jk = cos_matrix_jk × Σ_n W_n F_jn F_kn

    Args:
        F: (M, N) derivative matrix. May be float32 or float64; the
            (M, M) inner product is computed in F's dtype and upcast to
            ``cos_matrix.dtype`` before the cosine factor is applied.
        W: (N,) sample weights (normalised), same dtype as F.
        cos_matrix: (M, M) = directions @ directions.T.

    Returns:
        G: (M, M) symmetric positive semi-definite Gram matrix in
        ``cos_matrix.dtype``.
    """
    F_scaled = F * jnp.sqrt(W)[None, :]  # (M, N)
    gram_inner = F_scaled @ F_scaled.T  # (M, M) in F's dtype
    return cos_matrix * gram_inner.astype(cos_matrix.dtype)


def compute_shared_gram_diagnostics(result, G, valid_mask, ctx=None):
    """Append shared Gram diagnostics to ``result`` in place.

    Computes the diagnostics common to every Gram-based weight solver:

      * ``n_negative_weights``, ``negative_weight_mass``: counts and total
        signed mass of strictly-negative entries in ``result['w']``.
      * ``off_diagonal_magnitude``: mean of ``|G_jk| / √(G_jj G_kk)`` over
        the valid off-diagonal block. Near zero, the diagonal approximation
        is essentially exact; near one, slices are highly correlated.
      * ``diagonal_sanity``: per-slice ``|G_jj - D_matched_j| / D_matched_j``,
        with ``D_matched_j = ∫ ρ_j (dq_j/ds)² ds`` computed against the
        unit-integral density and the same piecewise-constant slopes as ``G``.

    When ``ctx.free_energies`` is None, only the first two diagnostics are
    populated; ``diagonal_sanity`` is filled with NaN.

    Args:
        result: dict to mutate (must contain ``'w'``).
        G: (M, M) Gram matrix.
        valid_mask: (M,) bool.
        ctx: optional duck-typed context with ``free_energies``,
            ``slice_coords``, ``committors_1d``.

    Returns:
        ``(D_matched_norm, valid_m)`` arrays for callers that want to
        derive further diagnostics (e.g. ``R_M_ratio`` in the full-Gram
        solver). Both are None when ``ctx`` lacks free energies.
    """
    M = G.shape[0]
    valid = valid_mask
    w = result["w"]

    neg_mask = (w < 0) & valid
    result["n_negative_weights"] = int(jnp.sum(neg_mask))
    result["negative_weight_mass"] = float(jnp.sum(jnp.where(neg_mask, w, 0.0)))

    G_diag = jnp.diag(G)
    denom = jnp.sqrt(jnp.maximum(G_diag[:, None] * G_diag[None, :], 1e-30))
    corr = jnp.abs(G / denom)
    mask_offdiag = ~jnp.eye(M, dtype=bool) & valid[:, None] & valid[None, :]
    n_offdiag = jnp.sum(mask_offdiag)
    result["off_diagonal_magnitude"] = float(
        jnp.where(
            n_offdiag > 0,
            jnp.sum(corr * mask_offdiag) / n_offdiag,
            0.0,
        )
    )

    if ctx is None or ctx.free_energies is None:
        result["diagonal_sanity"] = jnp.full(M, jnp.nan)
        return None, None

    F = ctx.free_energies
    s = ctx.slice_coords
    F_min = jnp.min(F, axis=1, keepdims=True)
    rho = jnp.exp(-(F - F_min))
    ds = jnp.diff(s, axis=1)
    rho_mid = 0.5 * (rho[:, :-1] + rho[:, 1:])
    Z_rel = jnp.sum(rho_mid * ds, axis=1)
    dq = jnp.diff(ctx.committors_1d, axis=1)
    slopes = dq / jnp.maximum(ds, 1e-30)
    D_matched = jnp.sum(slopes**2 * rho_mid * ds, axis=1)
    D_matched_norm = jnp.where(Z_rel > 1e-30, D_matched / Z_rel, D_matched)
    valid_m = valid & jnp.isfinite(D_matched_norm) & (D_matched_norm > 1e-30)
    result["diagonal_sanity"] = jnp.where(
        valid_m,
        jnp.abs(G_diag - D_matched_norm) / D_matched_norm,
        jnp.nan,
    )
    return D_matched_norm, valid_m
