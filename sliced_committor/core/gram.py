"""Shared Gram-assembly primitives and diagnostics for the sliced committor.

Both the full-Gram simplex solver (:func:`sliced_committor.full_gram_weights`)
and the basin-moment-constrained solver
(:func:`sliced_committor.compute_basin_moment_weights`) share a single derivative
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
        cos_matrix: (M, M) of theta_j^T D theta_k. Equals
            ``directions @ directions.T`` under the default D = I; see
            :func:`cos_matrix_from_metric` for the anisotropic case.

    Returns:
        G: (M, M) symmetric positive semi-definite Gram matrix in
        ``cos_matrix.dtype``.
    """
    F_scaled = F * jnp.sqrt(W)[None, :]  # (M, N)
    gram_inner = F_scaled @ F_scaled.T  # (M, M) in F's dtype
    return cos_matrix * gram_inner.astype(cos_matrix.dtype)


def compute_shared_gram_diagnostics(result, G, valid_mask, ctx=None, metric_diag=None):
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
        metric_diag: optional (M,) ``d_j = theta_j^T Mbar theta_j``. Required for
            a metric-carrying fit: ``diag(G)`` then carries ``d_j`` while
            ``D_matched`` is a metric-free integral, so without it
            ``diagonal_sanity`` reports a spurious ``|d_j - 1|`` and the
            ``R_M_ratio`` derived from it inverts. None means d_j = 1.

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
    if metric_diag is not None:
        # diag(G)_j = d_j * INT rho (dq/ds)^2, so the reference must carry d_j too.
        D_matched_norm = D_matched_norm * jnp.asarray(metric_diag, dtype=D_matched_norm.dtype)
    valid_m = valid & jnp.isfinite(D_matched_norm) & (D_matched_norm > 1e-30)
    result["diagonal_sanity"] = jnp.where(
        valid_m,
        jnp.abs(G_diag - D_matched_norm) / D_matched_norm,
        jnp.nan,
    )
    return D_matched_norm, valid_m


# ===========================================================================
# FEATURE-SPACE METRIC HOOK
# ===========================================================================
#
# The Dirichlet form in feature space is  INT rho (grad q)^T D (grad q),  which
# for the sliced ansatz enters ONLY through theta_j^T D theta_k. Supplying a
# constant feature-space tensor Mbar therefore reduces to replacing the cosine
# matrix; the 1D slice profiles, the directions, the binning and the basin
# moments are all untouched. See :mod:`sliced_committor.core.metric` for how
# Mbar = <J M0 J^T> is built.
#
# feature_metric=None reproduces  directions @ directions.T  verbatim.
# ===========================================================================


def validate_feature_metric(feature_metric, dim, *, cond_warn=1e4, psd_rtol=1e-10):
    """Symmetrise, PSD-project and range-check a feature-space metric.

    Args:
        feature_metric: ``(dim, dim)`` array, or None (the identity default).
        dim: expected feature dimension.
        cond_warn: warn above this condition number. An anisotropic ``diag(G)``
            is matched against a SCALAR Tikhonov anchor, so a badly conditioned
            metric degrades the ridge without raising.
        psd_rtol: tolerance for the negative-eigenvalue check.

    Returns:
        The validated ``(dim, dim)`` array, or None.

    Raises:
        ValueError: on wrong shape, non-finite entries, gross asymmetry, or a
            negative eigenvalue below ``-psd_rtol * max(lam_max, 1)``.
    """
    if feature_metric is None:
        return None
    M = jnp.asarray(feature_metric)
    # Never downcast: _assemble_gram_matrix returns in cos_matrix.dtype, so a
    # float32 metric would silently demote the entire Gram and downstream solve.
    M = M.astype(jnp.promote_types(M.dtype, jnp.zeros(0).dtype))
    if M.ndim != 2 or M.shape[0] != M.shape[1]:
        raise ValueError(f"feature_metric must be square (dim, dim); got shape {tuple(M.shape)}")
    if int(M.shape[0]) != int(dim):
        raise ValueError(
            f"feature_metric has dim {int(M.shape[0])} but samples have dim {int(dim)}"
        )
    if not bool(jnp.all(jnp.isfinite(M))):
        raise ValueError("feature_metric contains non-finite entries (NaN or Inf).")
    nrm = float(jnp.linalg.norm(M))
    if nrm <= 0.0:
        raise ValueError("feature_metric is all zeros; it must be positive semi-definite.")
    asym = float(jnp.linalg.norm(M - M.T)) / nrm
    if asym > 1e-8:
        raise ValueError(
            f"feature_metric is not symmetric (||M - M^T||/||M|| = {asym:.3e}). "
            "A diffusion tensor is symmetric by construction; check the assembly."
        )
    M = 0.5 * (M + M.T)
    lam = jnp.linalg.eigvalsh(M)
    lam_min, lam_max = float(lam[0]), float(lam[-1])
    if lam_min < -psd_rtol * max(lam_max, 1.0):
        raise ValueError(
            f"feature_metric is not positive semi-definite (min eigenvalue {lam_min:.3e} "
            f"vs max {lam_max:.3e}). A diffusion tensor must be PSD."
        )
    if lam_max <= 0.0:
        raise ValueError("feature_metric has no positive eigenvalue.")
    cond = lam_max / lam_min if lam_min > 0 else float("inf")
    if cond > float(cond_warn):
        import warnings as _warnings

        _warnings.warn(
            f"feature_metric is ill-conditioned (cond = {cond:.3e} > {float(cond_warn):.1e}). "
            "diag(G) then spans that range while the Tikhonov ridge is a single scalar "
            "anchored on median(diag G); prefer tikhonov='halfset_eigen' or 'cv', which "
            "re-select from the data, and consider shrink_metric().",
            UserWarning,
            stacklevel=2,
        )
    return M


def metric_factor(feature_metric):
    """``(d, d)`` factor ``S`` with ``S S^T = Mbar``; None passes through.

    Uses ``eigh``, not ``cholesky``: a legitimate pulled-back metric can be
    rank-deficient (a sin/cos pull-back has per-frame rank d/2 exactly, and
    pair-distance maps are rank-deficient even on average), and ``cholesky``
    fails on singular input while ``eigh`` yields the PSD clip for free.

    ``feature_metric`` exactly equal to the identity short-circuits, so
    ``feature_metric=eye(d)`` is bit-identical to ``feature_metric=None``.
    """
    if feature_metric is None:
        return None
    M = jnp.asarray(feature_metric)
    d = int(M.shape[0])
    if bool(jnp.array_equal(M, jnp.eye(d, dtype=M.dtype))):
        return jnp.eye(d, dtype=M.dtype)
    lam, V = jnp.linalg.eigh(0.5 * (M + M.T))
    return V * jnp.sqrt(jnp.maximum(lam, 0.0))[None, :]


def cos_matrix_from_metric(directions, feature_metric=None):
    """``(M, M)`` matrix of ``theta_j^T Mbar theta_k``.

    Assembled as ``(Theta S)(Theta S)^T`` rather than ``Theta Mbar Theta^T``.
    The factored form is PSD in floating point for ANY ``Theta``, so by the Schur
    product theorem the Hadamard product in :func:`_assemble_gram_matrix` stays
    PSD and the Cholesky inside the weight solvers cannot fail on a marginally
    indefinite input. The cost is a rounding error of the Gram matmul.

    ``feature_metric=None`` returns ``directions @ directions.T`` verbatim.
    """
    Theta = jnp.asarray(directions)
    if feature_metric is None:
        return Theta @ Theta.T
    Y = Theta @ metric_factor(feature_metric).astype(Theta.dtype)
    return Y @ Y.T


def _is_no_op_metric(feature_metric) -> bool:
    """True for None or an exact identity -- the cases that must change nothing."""
    if feature_metric is None:
        return True
    M = jnp.asarray(feature_metric)
    return bool(jnp.array_equal(M, jnp.eye(int(M.shape[0]), dtype=M.dtype)))


def metric_diagonal(directions, feature_metric=None):
    """``(M,)`` per-slice ``d_j = theta_j^T Mbar theta_j``; None for a no-op metric.

    Returning None rather than ``||theta_j||^2`` is deliberate on two counts. The
    diagonal weighting has always implicitly assumed unit directions, so folding
    ``||theta_j||^2`` in unconditionally would change results for any caller
    passing non-unit directions. And ``diag(Theta Theta^T)`` is 1 only to rounding,
    so returning it for ``feature_metric=eye(d)`` would break the promise that the
    identity metric is bit-identical to no metric at all.
    """
    if _is_no_op_metric(feature_metric):
        return None
    return jnp.diag(cos_matrix_from_metric(directions, feature_metric))


def resolve_cos_matrix(ctx):
    """``ctx.cos_matrix`` if cached, else built from ``ctx.directions`` + metric.

    Every Gram solver funnels through this so a metric can never be silently
    dropped by a hand-built context.
    """
    cached = getattr(ctx, "cos_matrix", None)
    if cached is not None:
        return cached
    return cos_matrix_from_metric(ctx.directions, getattr(ctx, "feature_metric", None))


def resolve_metric_diagonal(ctx):
    """``(M,)`` ``d_j`` for a metric-carrying context, else None.

    Derived from the (cached) cosine matrix so it is exactly the factor that
    multiplies ``diag(G)``, with no second eigendecomposition.
    """
    if _is_no_op_metric(getattr(ctx, "feature_metric", None)):
        return None
    return jnp.diag(resolve_cos_matrix(ctx))
