"""The slice Gram matrix and the feature-space metric it carries.

``G_jk = (theta_j^T Mbar theta_k) sum_n W_n q_j'(theta_j . x_n) q_k'(theta_k . x_n)``

is the Dirichlet form of the slice basis: the cosine matrix of the directions
in the feature-space metric ``Mbar`` (identity by default), Hadamard-multiplied
by the sample average of the slice derivatives. The derivative of each slice
committor is the piecewise-constant slope of its piecewise-linear interpolant,
i.e. exactly the derivative of the function :func:`sliced_committor.build_committor`
evaluates.
"""

import warnings

import jax.numpy as jnp
from jax import jit, vmap


@jit
def _compute_derivative_matrix(slice_coords, committors_1d, projected_samples):
    """``F[j, n] = dq_j/ds`` at ``theta_j . x_n`` (piecewise-constant slopes, gathered)."""
    ds = jnp.diff(slice_coords, axis=1)
    dq = jnp.diff(committors_1d, axis=1)
    slopes = dq / jnp.maximum(ds, 1e-10)
    n_bins = slice_coords.shape[1]

    def _find_bins(s_grid, s_proj):
        idx = jnp.searchsorted(s_grid, s_proj, side="right", method="scan_unrolled") - 1
        return jnp.clip(idx, 0, n_bins - 2)

    bin_indices = vmap(_find_bins)(slice_coords, projected_samples)
    return jnp.take_along_axis(slopes, bin_indices, axis=1)


@jit
def _assemble_gram_matrix(F, W, cos_matrix):
    """``G = cos_matrix * (F sqrt(W)) (F sqrt(W))^T``, in ``cos_matrix``'s dtype."""
    F_scaled = F * jnp.sqrt(W)[None, :]
    gram_inner = F_scaled @ F_scaled.T
    return cos_matrix * gram_inner.astype(cos_matrix.dtype)


# ---------------------------------------------------------------------------
# Feature-space metric: enters only through theta_j^T Mbar theta_k.
# ---------------------------------------------------------------------------


_PSD_RTOL = 1e-10  # eigenvalues below -rtol * lam_max are not rounding
_COND_WARN = 1e4


def validate_feature_metric(feature_metric, dim):
    """Symmetrise, PSD-project and range-check a ``(dim, dim)`` metric; None passes through."""
    if feature_metric is None:
        return None
    M = jnp.asarray(feature_metric)
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
    if lam_min < -_PSD_RTOL * max(lam_max, 1.0):
        raise ValueError(
            f"feature_metric is not positive semi-definite (min eigenvalue {lam_min:.3e} "
            f"vs max {lam_max:.3e}). A diffusion tensor must be PSD."
        )
    if lam_max <= 0.0:
        raise ValueError("feature_metric has no positive eigenvalue.")
    cond = lam_max / lam_min if lam_min > 0 else float("inf")
    if cond > _COND_WARN:
        warnings.warn(
            f"feature_metric is ill-conditioned (cond = {cond:.3e} > {_COND_WARN:.1e}); "
            "diag(G) then spans that range. The half-set filter (the default tikhonov) "
            "regularises band by band and copes; a scalar ridge does not.",
            UserWarning,
            stacklevel=2,
        )
    return M


def metric_factor(feature_metric):
    """``(d, d)`` factor ``S`` with ``S S^T = Mbar``; None passes through.

    ``eigh`` rather than ``cholesky``: a pulled-back metric can be
    rank-deficient (a sin/cos pull-back has per-frame rank d/2). An exact
    identity short-circuits so ``feature_metric=eye(d)`` is bit-identical to None.
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
    """``(M, M)`` matrix of ``theta_j^T Mbar theta_k``, assembled as ``(Theta S)(Theta S)^T``
    so it is PSD in floating point; None gives ``directions @ directions.T`` verbatim."""
    Theta = jnp.asarray(directions)
    if feature_metric is None:
        return Theta @ Theta.T
    Y = Theta @ metric_factor(feature_metric).astype(Theta.dtype)
    return Y @ Y.T
