"""The slice basis: 1D committors along random directions.

``compute_sliced_committor`` projects the samples onto ``M`` unit directions,
estimates a 1D free energy along each, solves the 1D reaction-diffusion (RD)
committor problem on that slice, and returns the whole basis as a
:class:`SlicedCommittorResult`. The weights that recombine the slices are
solved separately (:func:`sliced_committor.solve_weights`) and the callable
committor is assembled by :func:`sliced_committor.build_committor`.

Numerics that are load-bearing for the published results and must not drift:

* one projection per direction, shared by the free-energy histogram, the
  boundary detection and the RD solve;
* the adaptive density floor ``n_min / (N * ds)`` on the 1D histograms;
* the RD committor with soft absorption ``kappa`` inside the basins, solved by
  the Thomas algorithm; no inversion is ever applied because the absorption
  terms fix the orientation;
* the fused per-slice diagnostics ``log_dirichlet`` and ``boundary_errors``
  computed inside the same vmap.

beta is fixed at 1: it cancels in every downstream quantity, so
``free_energies`` stores ``-log rho`` directly.
"""

from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import jit, lax, random, vmap

from .directions import DirectionSamplingConfig, _sample_directions, directions_uniform
from .gram import validate_feature_metric

# Samples above this size use a subsample for the quantile edges (the exact
# sort is O(N log N) per direction; the approximate one histograms all N
# samples into edges estimated from a random subsample of this size).
_QUANTILE_SUBSAMPLE_ABOVE = 20000
_QUANTILE_SUBSAMPLE_SIZE = 10000

_BINNING_METHODS = ("quantile", "equal_width")


class SlicedCommittorResult(NamedTuple):
    """The slice basis: per-direction 1D committors and their diagnostics.

    Fields:
        directions: ``(M, dim)`` unit vectors.
        slice_coords: ``(M, n_bins)`` bin centres along each slice.
        free_energies: ``(M, n_bins)`` ``-log rho`` along each slice (beta = 1).
        committors_1d: ``(M, n_bins)`` the 1D RD committor on each slice.
        boundary_indices: ``(M, 2)`` grid indices of the inner basin edges.
        valid_mask: ``(M,)`` False where the 1D solve produced a non-finite
            committor. Slices whose basins overlap in projection are NOT
            masked: that is deliberate, the weight solve zeroes them through
            the moment gap ``b_j - a_j``. Callers may exclude a direction by
            hand with ``result._replace(valid_mask=...)``.
        in_A, in_B: ``(N,)`` basin labels of the input samples.
        projected_samples: ``(M, N)`` the projections ``theta_j . x_n``.
        log_dirichlet: ``(M,)`` ``log INT rho (dq/ds)^2`` per slice.
        boundary_errors: ``(M,)`` equilibrium boundary error
            ``eps_j = <q_j>_A + <1 - q_j>_B`` per slice.
        sample_weights: ``(N,)`` reweighting (e.g. MBAR), or None for uniform.
        axis: ``(dim,)`` the LDA axis when ``direction_sampling.mode == 'lda'``.
        lda_info: LDA diagnostics, or None.
        feature_metric: ``(dim, dim)`` feature-space metric, or None for the
            identity.
    """

    directions: jnp.ndarray
    slice_coords: jnp.ndarray
    free_energies: jnp.ndarray
    committors_1d: jnp.ndarray
    boundary_indices: jnp.ndarray
    valid_mask: jnp.ndarray
    in_A: jnp.ndarray
    in_B: jnp.ndarray
    projected_samples: jnp.ndarray
    log_dirichlet: jnp.ndarray
    boundary_errors: jnp.ndarray
    sample_weights: jnp.ndarray | None = None
    axis: jnp.ndarray | None = None
    lda_info: dict | None = None
    feature_metric: jnp.ndarray | None = None

    def summary(self) -> str:
        """Human-readable diagnostic summary (format not stable; do not parse)."""
        M = int(self.directions.shape[0])
        n_bins = int(self.slice_coords.shape[1])
        dim = int(self.directions.shape[1])
        N = int(self.in_A.shape[0])
        n_A = int(jnp.sum(self.in_A))
        n_B = int(jnp.sum(self.in_B))
        valid = jnp.asarray(self.valid_mask, dtype=bool)
        n_valid = int(jnp.sum(valid))
        lines = [
            "SlicedCommittorResult",
            "---------------------",
            f"  directions      : M = {M}, dim = {dim}",
            f"  bins per slice  : {n_bins}",
            f"  samples         : N = {N}  (|A| = {n_A}, |B| = {n_B})",
            f"  valid slices    : {n_valid}/{M}",
        ]
        if n_valid > 0:
            eps = jnp.asarray(self.boundary_errors)[valid]
            lines.append(
                f"  mean eps (eq.)  : {float(jnp.mean(eps)):.4f}  "
                f"(min {float(jnp.min(eps)):.4f}, max {float(jnp.max(eps)):.4f})"
            )
            log_D = jnp.asarray(self.log_dirichlet)[valid]
            lines.append(f"  mean log D[q]   : {float(jnp.mean(log_D)):.3f}")
        return "\n".join(lines)


# =============================================================================
# 1D INTERPOLATION
# =============================================================================


@jit
def _interp_1d_at_samples(s_grid, y_grid, s_samples):
    """Linear interpolation of grid values at sample positions (non-uniform grids)."""
    n = s_grid.shape[0]
    idx = jnp.searchsorted(s_grid, s_samples, side="right") - 1
    idx = jnp.clip(idx, 0, n - 2)
    ds = s_grid[idx + 1] - s_grid[idx]
    frac = (s_samples - s_grid[idx]) / jnp.maximum(ds, 1e-10)
    frac = jnp.clip(frac, 0.0, 1.0)
    return (1 - frac) * y_grid[idx] + frac * y_grid[idx + 1]


@jit
def _interp_committor_at_samples(s_grid, q_grid, s_samples):
    """Interpolate q(s) at sample positions, clamped to the grid end values."""
    q_interp = _interp_1d_at_samples(s_grid, q_grid, s_samples)
    q_interp = jnp.clip(q_interp, 0.0, 1.0)
    s_min, s_max = s_grid[0], s_grid[-1]
    q_interp = jnp.where(s_samples < s_min, q_grid[0], q_interp)
    q_interp = jnp.where(s_samples > s_max, q_grid[-1], q_interp)
    return q_interp


@jit
def _qall_onthefly(directions, s_coords, q_1d, points_flat):
    """``(M, P)`` matrix of slice committors ``q_j(theta_j . x_p)``."""

    def per_dir(theta, s_grid, q_grid):
        return _interp_1d_at_samples(s_grid, q_grid, points_flat @ theta)

    return vmap(per_dir, in_axes=(0, 0, 0))(directions, s_coords, q_1d)


# =============================================================================
# 1D FREE ENERGY WITH THE ADAPTIVE DENSITY FLOOR
# =============================================================================


@partial(
    jit,
    static_argnames=(
        "n_bins",
        "density_floor",
        "binning_method",
        "n_min",
        "use_subsample_quantile",
    ),
)
def compute_1d_free_energy(
    s_projected,
    n_bins=200,
    density_floor=1e-6,
    binning_method="quantile",
    n_min=10,
    s_subsample=None,
    use_subsample_quantile=False,
):
    """``-log rho(s)`` along one slice from the projected samples.

    Bin centres and free energies (shifted so the minimum is 0). ``quantile``
    bins hold equal counts, ``equal_width`` bins are uniform in ``s``. Every
    bin density is floored at ``max(n_min / (N ds), density_floor)`` so that
    tail bins with a handful of samples cannot produce arbitrarily high free
    energies.
    """
    s = s_projected
    s_min, s_max = jnp.min(s), jnp.max(s)
    span = jnp.maximum(s_max - s_min, 1e-8)

    if binning_method == "quantile" and use_subsample_quantile:
        s_sorted = jnp.sort(s_subsample)
        K = s_subsample.shape[0]
        edge_indices = jnp.linspace(0, K - 1, n_bins + 1).astype(jnp.int32)
        bin_edges = s_sorted[edge_indices]
        s_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        ds_per_bin = jnp.maximum(bin_edges[1:] - bin_edges[:-1], 1e-10)
        N_total = s.shape[0]
        bin_idx = jnp.clip(jnp.searchsorted(bin_edges[1:-1], s), 0, n_bins - 1)
        counts = jnp.zeros(n_bins).at[bin_idx].add(jnp.ones(N_total))
        N = jnp.float64(N_total)
        effective_floor = jnp.maximum(n_min / (N * ds_per_bin), density_floor)
        density = jnp.maximum(counts / (N * ds_per_bin), effective_floor)
    elif binning_method == "quantile":
        s_sorted = jnp.sort(s)
        N_total = s.shape[0]
        edge_indices = jnp.linspace(0, N_total - 1, n_bins + 1).astype(jnp.int32)
        bin_edges = s_sorted[edge_indices]
        s_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        ds_per_bin = jnp.maximum(bin_edges[1:] - bin_edges[:-1], 1e-10)
        counts_per_bin = N_total / n_bins
        N = jnp.float64(N_total)
        effective_floor = jnp.maximum(n_min / (N * ds_per_bin), density_floor)
        density = jnp.maximum(counts_per_bin / (N * ds_per_bin), effective_floor)
    else:
        bin_edges = jnp.linspace(s_min, s_max, n_bins + 1)
        s_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        idx = jnp.clip(jnp.floor((s - s_min) / span * n_bins).astype(jnp.int32), 0, n_bins - 1)
        counts = jnp.bincount(idx, length=n_bins)
        N = jnp.maximum(jnp.sum(counts), 1.0)
        bin_width = span / n_bins
        effective_floor = jnp.maximum(n_min / (N * bin_width), density_floor)
        density = jnp.maximum(counts / (N * bin_width), effective_floor)

    F_values = -jnp.log(density)
    return s_centers, F_values - jnp.min(F_values)


# =============================================================================
# INNER BASIN EDGES
# =============================================================================


def _inner_edge_index(s_sorted, quantile, high):
    """Index of the ``quantile`` inner edge in a sorted state projection."""
    n = s_sorted.shape[0]
    if high:
        return jnp.clip(jnp.floor(quantile * (n - 1)).astype(jnp.int32), 0, n - 1)
    return jnp.clip(jnp.ceil((1.0 - quantile) * (n - 1)).astype(jnp.int32), 0, n - 1)


def find_boundary_indices(s_values, s_A_sorted, s_B_sorted, boundary_quantile=1.0):
    """Grid indices of the inner basin edges along one slice.

    The basin facing edges are the extreme projections (``boundary_quantile
    = 1``) or the ``boundary_quantile`` quantiles of the sorted state
    projections (``< 1`` shrinks the edges toward the basin centres, which
    mitigates the halo of a basin projected from many nuisance dimensions).
    Returns ``(a_idx, b_idx)`` with ``a_idx <= b_idx``; the interval is always
    defined, overlapping basins simply give a narrow one.
    """
    A_is_left = jnp.mean(s_A_sorted) < jnp.mean(s_B_sorted)
    s_A_extreme = jnp.where(A_is_left, s_A_sorted[-1], s_A_sorted[0])
    s_B_extreme = jnp.where(A_is_left, s_B_sorted[0], s_B_sorted[-1])

    s_A_quantile = jnp.where(
        A_is_left,
        s_A_sorted[_inner_edge_index(s_A_sorted, boundary_quantile, True)],
        s_A_sorted[_inner_edge_index(s_A_sorted, boundary_quantile, False)],
    )
    s_B_quantile = jnp.where(
        A_is_left,
        s_B_sorted[_inner_edge_index(s_B_sorted, boundary_quantile, False)],
        s_B_sorted[_inner_edge_index(s_B_sorted, boundary_quantile, True)],
    )
    use_quantile = boundary_quantile < 1.0
    s_A_inner = jnp.where(use_quantile, s_A_quantile, s_A_extreme)
    s_B_inner = jnp.where(use_quantile, s_B_quantile, s_B_extreme)
    s_left = jnp.minimum(s_A_inner, s_B_inner)
    s_right = jnp.maximum(s_A_inner, s_B_inner)

    def _nearest_idx(s_grid, s_val):
        n = s_grid.shape[0]
        idx = jnp.clip(jnp.searchsorted(s_grid, s_val, side="right"), 1, n - 1)
        return jnp.where(
            jnp.abs(s_grid[idx - 1] - s_val) <= jnp.abs(s_grid[idx] - s_val), idx - 1, idx
        )

    a_idx = _nearest_idx(s_values, s_left)
    b_idx = _nearest_idx(s_values, s_right)
    return jnp.minimum(a_idx, b_idx), jnp.maximum(a_idx, b_idx)


# =============================================================================
# 1D REACTION-DIFFUSION COMMITTOR
# =============================================================================


def _absorption_keep_mask(s_grid, s_sorted, is_left, quantile):
    """True where a basin's absorption density is retained (inside its inner-edge quantile)."""
    n = s_sorted.shape[0]
    idx = jnp.where(
        is_left,
        jnp.clip(jnp.floor(quantile * (n - 1)).astype(jnp.int32), 0, n - 1),
        jnp.clip(jnp.ceil((1.0 - quantile) * (n - 1)).astype(jnp.int32), 0, n - 1),
    )
    cutoff = s_sorted[idx]
    return jnp.where(is_left, s_grid <= cutoff, s_grid >= cutoff)


def compute_1d_rd_committor(
    s_values,
    F_values,
    s_A_proj,
    s_B_proj,
    *,
    rd_kappa,
    ds_arr,
    absorption_quantile,
    s_A_sorted,
    s_B_sorted,
):
    """The 1D reaction-diffusion committor on one slice (Thomas algorithm).

    Minimises ``INT rho (dq/ds)^2 + kappa INT rho_A q^2 + kappa INT rho_B (1-q)^2``
    with Neumann ends, i.e. solves ``d/ds[rho dq/ds] = kappa [rho_A q - rho_B (1-q)]``.
    Distributing the absorption through the basin bulks fixes the orientation
    ``q -> 0`` in A and ``q -> 1`` in B without any inversion step.
    ``absorption_quantile < 1`` zeroes ``rho_A``, ``rho_B`` past the inner-edge
    quantile of the state projections (the halo tail).
    """
    n = s_values.shape[0]
    rho = jnp.exp(-(F_values - jnp.min(F_values)))
    s_min, s_max = s_values[0], s_values[-1]
    span = s_max - s_min + 1e-10

    def _state_density(s_proj_state):
        idx = jnp.clip(jnp.floor((s_proj_state - s_min) / span * n).astype(jnp.int32), 0, n - 1)
        n_state = s_proj_state.shape[0]
        hist = jnp.zeros(n).at[idx].add(jnp.ones(n_state))
        total = jnp.maximum(jnp.float32(n_state), 1e-10)
        return hist / (total * ds_arr + 1e-30)

    rho_A = _state_density(s_A_proj)
    rho_B = _state_density(s_B_proj)

    if absorption_quantile < 1.0:
        A_is_left = jnp.mean(s_A_proj) < jnp.mean(s_B_proj)
        rho_A = jnp.where(
            _absorption_keep_mask(s_values, s_A_sorted, A_is_left, absorption_quantile), rho_A, 0.0
        )
        rho_B = jnp.where(
            _absorption_keep_mask(s_values, s_B_sorted, ~A_is_left, absorption_quantile), rho_B, 0.0
        )

    # Tridiagonal system of the finite-difference discretisation.
    rho_half_plus = 0.5 * (rho[:-1] + rho[1:])
    ds2 = ds_arr**2
    sub_interior = rho_half_plus[:-1] / ds2[1 : n - 1]
    sup_interior = rho_half_plus[1:] / ds2[1 : n - 1]
    c0 = rho_half_plus[0] / ds2[0]
    an = rho_half_plus[-1] / ds2[n - 1]
    kappa_rhoAB = rd_kappa * (rho_A + rho_B)
    sub = jnp.concatenate([jnp.zeros(1), sub_interior, an[None]])
    sup = jnp.concatenate([c0[None], sup_interior, jnp.zeros(1)])
    diag = -(sub + sup) - kappa_rhoAB
    rhs = -rd_kappa * rho_B

    # Thomas algorithm with scalar carries.
    b0 = jnp.where(jnp.abs(diag[0]) > 1e-30, diag[0], 1e-30)
    cp0 = sup[0] / b0
    dp0 = rhs[0] / b0

    def _forward_step(carry, xs):
        cp_prev, dp_prev = carry
        sub_i, diag_i, sup_i, rhs_i = xs
        w_denom = diag_i - sub_i * cp_prev
        w_denom_safe = jnp.where(jnp.abs(w_denom) > 1e-30, w_denom, 1e-30)
        cp_i = sup_i / w_denom_safe
        dp_i = (rhs_i - sub_i * dp_prev) / w_denom_safe
        return (cp_i, dp_i), (cp_i, dp_i)

    _, (cp_rest, dp_rest) = lax.scan(
        _forward_step, (cp0, dp0), (sub[1:], diag[1:], sup[1:], rhs[1:])
    )
    cp = jnp.concatenate([cp0[None], cp_rest])
    dp = jnp.concatenate([dp0[None], dp_rest])

    def _back_step(q_next, xs):
        dp_i, cp_i = xs
        q_i = dp_i - cp_i * q_next
        return q_i, q_i

    _, q_reversed = lax.scan(_back_step, dp[-1], (dp[:-1][::-1], cp[:-1][::-1]))
    q_rd = jnp.concatenate([q_reversed[::-1], dp[-1:]])
    return jnp.clip(q_rd, 0.0, 1.0)


# =============================================================================
# PER-SLICE DIRICHLET ENERGY
# =============================================================================


def _compute_dqds_grid(s_vals, q_vals):
    """``dq/ds`` on the grid: central differences inside, one-sided at the ends."""
    q_safe = jnp.where(jnp.isnan(q_vals), 0.0, q_vals)
    ds_central = jnp.maximum(s_vals[2:] - s_vals[:-2], 1e-30)
    grad_interior = (q_safe[2:] - q_safe[:-2]) / ds_central
    ds_fwd = jnp.maximum(s_vals[1] - s_vals[0], 1e-30)
    ds_bwd = jnp.maximum(s_vals[-1] - s_vals[-2], 1e-30)
    return jnp.concatenate(
        [
            ((q_safe[1] - q_safe[0]) / ds_fwd)[None],
            grad_interior,
            ((q_safe[-1] - q_safe[-2]) / ds_bwd)[None],
        ]
    )


def _compute_log_dirichlet(s_vals, F_vals, q_vals, ds_arr):
    """``log INT rho (dq/ds)^2 ds`` over the whole slice (log-space trapezoid)."""
    n = s_vals.shape[0]
    grad = _compute_dqds_grid(s_vals, q_vals)
    log_rho = -(F_vals - jnp.min(F_vals))
    log_grad_sq = jnp.log(jnp.maximum(grad**2, 1e-30))
    idxs = jnp.arange(n)
    is_endpoint = (idxs == 0) | (idxs == n - 1)
    log_trap_weight = jnp.where(is_endpoint, jnp.log(0.5), 0.0)
    log_ds_arr = jnp.log(jnp.maximum(ds_arr, 1e-30))
    return jax.scipy.special.logsumexp(log_rho + log_grad_sq + log_trap_weight + log_ds_arr)


# =============================================================================
# THE SLICE BASIS
# =============================================================================


def compute_sliced_committor(
    samples,
    *,
    in_A,
    in_B,
    n_directions: int = 256,
    n_bins: int = 200,
    seed: int = 42,
    binning_method: str = "quantile",
    density_floor: float = 1e-6,
    n_min: int = 10,
    rd_kappa: float = 1e12,
    boundary_quantile: float = 1.0,
    direction_batch_size: int | None = None,
    sample_weights=None,
    directions=None,
    direction_sampling: DirectionSamplingConfig | None = None,
    feature_metric=None,
) -> SlicedCommittorResult:
    """Build the slice basis: one 1D RD committor per direction.

    Args:
        samples: ``(N, dim)`` configurations.
        in_A, in_B: ``(N,)`` bool basin labels (disjoint, each non-empty).
        n_directions: number of directions ``M`` (ignored when ``directions``
            is given).
        n_bins: histogram resolution of each slice.
        seed: PRNG seed for the direction draw.
        binning_method: ``'quantile'`` (equal-count bins) or ``'equal_width'``.
        density_floor: static floor on the 1D density.
        n_min: adaptive floor ``n_min / (N ds)`` on the 1D density.
        rd_kappa: absorption strength of the RD problem (large = hard
            absorption inside the basins).
        boundary_quantile: inner basin edges at this quantile of the state
            projections (1 = the extremes). Below 1 the absorption density is
            truncated past the same quantile, which suppresses the halo that
            many nuisance dimensions inflate around a projected basin.
        direction_batch_size: process the directions in batches of this size
            to bound the ``(batch, N)`` working memory; None = one vmap.
        sample_weights: ``(N,)`` reweighting (e.g. MBAR); None = uniform.
        directions: ``(M, dim)`` pre-supplied directions; overrides the draw.
        direction_sampling: :class:`DirectionSamplingConfig` for the LDA cone
            (``mode='lda'``); None or ``mode='uniform'`` draws uniformly.
        feature_metric: ``(dim, dim)`` feature-space metric ``Mbar`` so the
            Dirichlet form is ``INT rho grad q^T Mbar grad q``; None = identity.
            Enters the weight solve only (the 1D profiles are unchanged). Build
            it with :func:`sliced_committor.sincos_pullback_metric`.
    """
    samples = jnp.asarray(samples)
    if samples.ndim != 2:
        raise ValueError(f"samples must be 2-D (N, dim); got shape {samples.shape}")
    n_samples_total, dim = int(samples.shape[0]), int(samples.shape[1])
    if n_samples_total == 0:
        raise ValueError("samples is empty; cannot compute a sliced committor on zero points.")
    if not bool(jnp.all(jnp.isfinite(samples))):
        n_bad = int(jnp.sum(~jnp.isfinite(samples)))
        raise ValueError(
            f"samples contains {n_bad} non-finite entry/entries (NaN or Inf). "
            "Filter or impute before calling compute_sliced_committor."
        )
    in_A = jnp.asarray(in_A, dtype=bool)
    in_B = jnp.asarray(in_B, dtype=bool)
    if in_A.shape != (n_samples_total,) or in_B.shape != (n_samples_total,):
        raise ValueError(
            f"in_A / in_B must be (N,) bool arrays matching samples; got "
            f"in_A.shape={in_A.shape}, in_B.shape={in_B.shape}, N={n_samples_total}"
        )
    n_A = int(jnp.sum(in_A))
    n_B = int(jnp.sum(in_B))
    if n_A == 0:
        raise ValueError("in_A has no True entries; basin A is empty.")
    if n_B == 0:
        raise ValueError("in_B has no True entries; basin B is empty.")
    if bool(jnp.any(in_A & in_B)):
        n_overlap = int(jnp.sum(in_A & in_B))
        raise ValueError(
            f"in_A and in_B must be disjoint; {n_overlap} sample(s) are flagged in both basins."
        )
    if binning_method not in _BINNING_METHODS:
        raise ValueError(
            f"binning_method must be one of {_BINNING_METHODS}; got {binning_method!r}"
        )
    if not (0.0 < boundary_quantile <= 1.0):
        raise ValueError(f"boundary_quantile must be in (0, 1]; got {boundary_quantile}")
    if dim > n_samples_total:
        import warnings

        warnings.warn(
            f"samples.shape={samples.shape}: dim ({dim}) exceeds N ({n_samples_total}). "
            "Per-slice density estimation is fragile in this regime; consider "
            "lowering boundary_quantile (e.g. 0.95) to mitigate the halo artifact.",
            UserWarning,
            stacklevel=2,
        )
    feature_metric = validate_feature_metric(feature_metric, dim)

    A_indices = jnp.nonzero(in_A, size=n_A)[0]
    B_indices = jnp.nonzero(in_B, size=n_B)[0]
    n_A_f = jnp.float32(n_A)
    n_B_f = jnp.float32(n_B)

    if sample_weights is None:
        W_A_norm = W_B_norm = sumWA = sumWB = None
    else:
        sample_weights = jnp.asarray(sample_weights)
        W_norm = sample_weights / jnp.maximum(jnp.sum(sample_weights), 1e-30)
        W_A_norm, W_B_norm = W_norm[A_indices], W_norm[B_indices]
        sumWA, sumWB = jnp.sum(W_A_norm), jnp.sum(W_B_norm)

    # Directions: pre-supplied > config > uniform.
    lda_info = None
    axis = None
    if directions is not None:
        directions = jnp.asarray(directions)
        if directions.shape[1] != dim:
            raise ValueError(
                f"directions shape {directions.shape} incompatible with samples.shape[1]={dim}."
            )
        n_directions = directions.shape[0]
        if direction_sampling is not None:
            import warnings

            warnings.warn(
                "Both `directions` and `direction_sampling` were supplied; ignoring "
                "`direction_sampling` because pre-supplied `directions` take precedence.",
                stacklevel=2,
            )
    else:
        key = random.PRNGKey(seed)
        if direction_sampling is None or direction_sampling.mode == "uniform":
            directions = directions_uniform(key, n_directions, dim)
        else:
            directions, lda_info = _sample_directions(
                key, n_directions, dim, samples, in_A, in_B, direction_sampling
            )
            axis = lda_info.get("axis", None)

    # Approximate quantile edges from a random subsample above the size threshold.
    N = n_samples_total
    use_subsample = binning_method == "quantile" and N > _QUANTILE_SUBSAMPLE_ABOVE
    if use_subsample:
        sub_key = random.PRNGKey(seed + 999)
        subsample_idx = random.choice(sub_key, N, shape=(_QUANTILE_SUBSAMPLE_SIZE,), replace=False)
    else:
        subsample_idx = None

    def _slice(theta):
        s_projected = samples @ theta
        s_sub = s_projected[subsample_idx] if use_subsample else None
        s_vals, F_vals = compute_1d_free_energy(
            s_projected,
            n_bins,
            density_floor=density_floor,
            binning_method=binning_method,
            n_min=n_min,
            s_subsample=s_sub,
            use_subsample_quantile=use_subsample,
        )
        ds_arr = jnp.diff(s_vals)
        ds_arr = jnp.concatenate([ds_arr, ds_arr[-1:]])
        s_A = s_projected[A_indices]
        s_B = s_projected[B_indices]
        s_A_sorted = jnp.sort(s_A)
        s_B_sorted = jnp.sort(s_B)
        a_idx, b_idx = find_boundary_indices(s_vals, s_A_sorted, s_B_sorted, boundary_quantile)
        q_rd = compute_1d_rd_committor(
            s_vals,
            F_vals,
            s_A,
            s_B,
            rd_kappa=rd_kappa,
            ds_arr=ds_arr,
            absorption_quantile=boundary_quantile,
            s_A_sorted=s_A_sorted,
            s_B_sorted=s_B_sorted,
        )
        is_valid = jnp.all(jnp.isfinite(q_rd))
        log_D = jnp.where(is_valid, _compute_log_dirichlet(s_vals, F_vals, q_rd, ds_arr), jnp.inf)
        q_rd_safe = jnp.where(jnp.isnan(q_rd), 0.0, q_rd)
        q_A = _interp_committor_at_samples(s_vals, q_rd_safe, s_A)
        q_B = _interp_committor_at_samples(s_vals, q_rd_safe, s_B)
        if W_A_norm is None:
            eps_A = jnp.sum(q_A) / jnp.maximum(n_A_f, 1.0)
            eps_B = jnp.sum(1.0 - q_B) / jnp.maximum(n_B_f, 1.0)
        else:
            eps_A = jnp.sum(W_A_norm * q_A) / jnp.maximum(sumWA, 1e-30)
            eps_B = jnp.sum(W_B_norm * (1.0 - q_B)) / jnp.maximum(sumWB, 1e-30)
        return (
            s_vals,
            F_vals,
            q_rd,
            jnp.stack([a_idx, b_idx]),
            is_valid,
            log_D,
            eps_A + eps_B,
            s_projected,
        )

    batched = jit(vmap(_slice, in_axes=0))
    if direction_batch_size is None or direction_batch_size >= n_directions:
        out = batched(directions)
    else:
        parts = [
            batched(directions[start : start + direction_batch_size])
            for start in range(0, n_directions, direction_batch_size)
        ]
        out = tuple(jnp.concatenate([p[k] for p in parts], axis=0) for k in range(len(parts[0])))

    (
        slice_coords,
        free_energies,
        committors_1d,
        boundary_indices,
        valid_mask,
        log_dirichlet,
        eps,
        projected,
    ) = out
    return SlicedCommittorResult(
        directions=directions,
        slice_coords=slice_coords,
        free_energies=free_energies,
        committors_1d=committors_1d,
        boundary_indices=boundary_indices,
        valid_mask=valid_mask,
        in_A=in_A,
        in_B=in_B,
        projected_samples=projected,
        log_dirichlet=log_dirichlet,
        boundary_errors=eps,
        sample_weights=sample_weights,
        axis=axis,
        lda_info=lda_info,
        feature_metric=feature_metric,
    )
