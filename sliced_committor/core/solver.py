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

import warnings
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

    Attributes:
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
        log_dirichlet: ``(M,)`` the log Dirichlet energy ``INT rho (dq/ds)^2`` of
            each slice's discrete functional.
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
    idx = jnp.searchsorted(s_grid, s_samples, side="right", method="scan_unrolled") - 1
    idx = jnp.clip(idx, 0, n - 2)
    ds = s_grid[idx + 1] - s_grid[idx]
    frac = (s_samples - s_grid[idx]) / jnp.maximum(ds, 1e-10)
    frac = jnp.clip(frac, 0.0, 1.0)
    return (1 - frac) * y_grid[idx] + frac * y_grid[idx + 1]


def _interp_committor(s_grid, q_grid, s_samples):
    """A slice committor at sample positions, clipped to ``[0, 1]``."""
    return jnp.clip(_interp_1d_at_samples(s_grid, q_grid, s_samples), 0.0, 1.0)


class _Basis1D(NamedTuple):
    """What evaluating the committor needs from a fit: the directions, the
    slice grids, the NaN-free 1D committors and the validity mask. A callable
    built on it holds none of the fit's samples."""

    directions: jnp.ndarray
    slice_coords: jnp.ndarray
    committors_1d: jnp.ndarray
    valid_mask: jnp.ndarray


def _basis_1d(result) -> _Basis1D:
    return _Basis1D(
        result.directions,
        result.slice_coords,
        jnp.where(jnp.isnan(result.committors_1d), 0.0, result.committors_1d),
        jnp.asarray(result.valid_mask),
    )


@jit
def _qall_onthefly(directions, s_coords, q_1d, points_flat):
    """``(M, P)`` matrix of slice committors ``q_j(theta_j . x_p)``."""

    def per_dir(theta, s_grid, q_grid):
        return _interp_1d_at_samples(s_grid, q_grid, points_flat @ theta)

    return vmap(per_dir, in_axes=(0, 0, 0))(directions, s_coords, q_1d)


def _slice_matrix(basis: _Basis1D, points_flat):
    """``(M, P)`` slice committors at the points."""
    return _qall_onthefly(basis.directions, basis.slice_coords, basis.committors_1d, points_flat)


@partial(jit, static_argnames=("original_shape",))
def _combine(q_all, w, c, valid_mask, original_shape):
    """``c + sum_j w_j clip(q_j(x))`` over the valid slices."""
    w_eff = w * valid_mask.astype(w.dtype)
    q_clip = jnp.clip(q_all, 0.0, 1.0)
    return (c + jnp.sum(w_eff[:, None] * q_clip, axis=0)).reshape(original_shape)


# =============================================================================
# 1D FREE ENERGY WITH THE ADAPTIVE DENSITY FLOOR
# =============================================================================


def _weighted_bin_density(bin_idx, weights, ds, n_bins, n_min, density_floor, nominal_count=None):
    """The density of a weighted ensemble on fixed bins, floored like the unweighted one.

    A bin's mass is the summed weight of its samples. Bins with fewer than
    ``n_min`` samples are floored at ``n_min`` samples of their own mean weight
    (an empty bin borrows the mean weight of the nearest non-empty bin), so
    uniform weights ``1/N`` reproduce the unweighted floor ``n_min / (N ds)``.
    ``nominal_count`` is the exact-quantile convention: every bin is taken to
    hold that many samples, now at its own mean weight.
    """
    counts = jnp.zeros(n_bins).at[bin_idx].add(1.0)
    mass = jnp.zeros(n_bins).at[bin_idx].add(weights)
    ar = jnp.arange(n_bins)
    nonempty = counts > 0
    left = lax.cummax(jnp.where(nonempty, ar, -1))
    right = lax.cummin(jnp.where(nonempty, ar, n_bins), reverse=True)
    use_left = (left >= 0) & ((right >= n_bins) | (ar - left <= right - ar))
    nearest = jnp.clip(jnp.where(use_left, left, right), 0, n_bins - 1)
    mean_w = jnp.where(nonempty, mass / jnp.maximum(counts, 1.0), 0.0)
    mean_w = jnp.where(nonempty, mean_w, mean_w[nearest])
    if nominal_count is not None:
        mass = nominal_count * mean_w
    return jnp.maximum(jnp.maximum(mass, n_min * mean_w) / ds, density_floor)


@partial(
    jit,
    static_argnames=(
        "n_bins",
        "density_floor",
        "binning_method",
        "n_min",
        "use_subsample_quantile",
        "return_edges",
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
    return_edges=False,
    weights=None,
):
    """``-log rho(s)`` along one slice from the projected samples.

    Bin centres and free energies (shifted so the minimum is 0), and with
    ``return_edges`` also the ``n_bins + 1`` bin edges. ``quantile`` bins
    hold equal counts, ``equal_width`` bins are uniform in ``s``. Every bin
    density is floored at ``max(n_min / (N ds), density_floor)`` so that tail
    bins with a handful of samples cannot produce arbitrarily high free
    energies. With ``weights`` (``(N,)``, summing to one) the density is the
    weighted histogram on the same bins, floored at ``n_min`` samples of each
    bin's mean weight, which uniform weights reduce to the unweighted case.
    """
    s = s_projected
    s_min, s_max = jnp.min(s), jnp.max(s)
    span = jnp.maximum(s_max - s_min, 1e-8)

    if binning_method == "quantile":
        # Edges at the quantiles of the subsample (large N) or of all samples;
        # exact quantile bins hold N / n_bins samples each by construction.
        src = s_subsample if use_subsample_quantile else s
        s_sorted = jnp.sort(src)
        edge_indices = jnp.linspace(0, src.shape[0] - 1, n_bins + 1).astype(jnp.int32)
        bin_edges = s_sorted[edge_indices]
        s_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        ds_per_bin = jnp.maximum(bin_edges[1:] - bin_edges[:-1], 1e-10)
        N_total = s.shape[0]
        if weights is not None:
            bin_idx = jnp.clip(jnp.searchsorted(bin_edges[1:-1], s), 0, n_bins - 1)
            density = _weighted_bin_density(
                bin_idx,
                weights,
                ds_per_bin,
                n_bins,
                n_min,
                density_floor,
                nominal_count=None if use_subsample_quantile else N_total / n_bins,
            )
        else:
            if use_subsample_quantile:
                bin_idx = jnp.clip(jnp.searchsorted(bin_edges[1:-1], s), 0, n_bins - 1)
                counts = jnp.zeros(n_bins).at[bin_idx].add(jnp.ones(N_total))
            else:
                counts = N_total / n_bins
            N = jnp.float64(N_total)
            effective_floor = jnp.maximum(n_min / (N * ds_per_bin), density_floor)
            density = jnp.maximum(counts / (N * ds_per_bin), effective_floor)
    else:
        bin_edges = jnp.linspace(s_min, s_max, n_bins + 1)
        s_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        idx = jnp.clip(jnp.floor((s - s_min) / span * n_bins).astype(jnp.int32), 0, n_bins - 1)
        bin_width = span / n_bins
        if weights is not None:
            density = _weighted_bin_density(
                idx, weights, jnp.full(n_bins, bin_width), n_bins, n_min, density_floor
            )
        else:
            counts = jnp.bincount(idx, length=n_bins)
            N = jnp.maximum(jnp.sum(counts), 1.0)
            effective_floor = jnp.maximum(n_min / (N * bin_width), density_floor)
            density = jnp.maximum(counts / (N * bin_width), effective_floor)

    F_values = -jnp.log(density)
    F_values = F_values - jnp.min(F_values)
    if return_edges:
        return s_centers, F_values, bin_edges
    return s_centers, F_values


# =============================================================================
# INNER BASIN EDGES
# =============================================================================


def _inner_edge_index(s_sorted, quantile, high):
    """Index of the ``quantile`` inner edge in a sorted state projection.

    ``high`` (traced bool) selects the upper edge, else the lower one; at
    ``quantile = 1`` the edge is the extreme sample.
    """
    n = s_sorted.shape[0]
    upper = jnp.floor(quantile * (n - 1)).astype(jnp.int32)
    lower = jnp.ceil((1.0 - quantile) * (n - 1)).astype(jnp.int32)
    return jnp.clip(jnp.where(high, upper, lower), 0, n - 1)


def find_boundary_indices(s_values, s_A_sorted, s_B_sorted, boundary_quantile=1.0):
    """Grid indices of the inner basin edges along one slice.

    The basin-facing edges are the ``boundary_quantile`` quantiles of the
    sorted state projections: the extreme projections at 1, and edges shrunk
    toward the basin centres below 1 (which mitigates the halo of a basin
    projected from many nuisance dimensions).
    Returns ``(a_idx, b_idx)`` with ``a_idx <= b_idx``; the interval is always
    defined, overlapping basins simply give a narrow one.
    """
    A_is_left = jnp.mean(s_A_sorted) < jnp.mean(s_B_sorted)
    s_A_inner = s_A_sorted[_inner_edge_index(s_A_sorted, boundary_quantile, A_is_left)]
    s_B_inner = s_B_sorted[_inner_edge_index(s_B_sorted, boundary_quantile, ~A_is_left)]
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
    cutoff = s_sorted[_inner_edge_index(s_sorted, quantile, is_left)]
    return jnp.where(is_left, s_grid <= cutoff, s_grid >= cutoff)


def compute_1d_rd_committor(
    s_values,
    F_values,
    s_A_proj,
    s_B_proj,
    *,
    bin_edges,
    rd_kappa,
    boundary_quantile,
    s_A_sorted,
    s_B_sorted,
    W_A=None,
    W_B=None,
    stiffness=None,
):
    """The 1D reaction-diffusion committor on one slice (Thomas algorithm).

    Minimises ``INT rho (dq/ds)^2 + kappa INT rho_A q^2 + kappa INT rho_B (1-q)^2``
    with Neumann ends, i.e. solves ``d/ds[rho dq/ds] = kappa [rho_A q - rho_B (1-q)]``.
    Distributing the absorption through the basin bulks fixes the orientation
    ``q -> 0`` in A and ``q -> 1`` in B without any inversion step.
    ``boundary_quantile < 1`` zeroes ``rho_A``, ``rho_B`` past the inner-edge
    quantile of the state projections (the halo tail).

    The linear system is the stationarity condition of the discrete functional
    on the grid of bin centres ``s_values``: node ``i`` carries the state
    masses of bin ``i`` (each state's projections binned with the same
    ``bin_edges`` as the density), and nodes ``i`` and ``i + 1`` are coupled
    by the conductance ``rho_{i+1/2} / (s_{i+1} - s_i)``. The scheme is
    conservative on any grid, equal-count bins included: summing its rows
    gives ``INT rho_A q = INT rho_B (1 - q)`` over the absorbing masses. With
    ``boundary_quantile = 1`` those are all of each state's samples, so the
    slice fidelity ``INT rho_B q - INT rho_A q`` lies in ``[0, 1]``; below 1 the
    balance leaves out the state mass past the cut. ``W_A``, ``W_B`` weight
    the state masses (a biased ensemble's equilibrium weights).

    ``stiffness`` replaces the absolute ``rd_kappa`` by the dimensionless
    ``kappa~ = kappa p_max sigma^2`` of the slice (``p_max`` and ``sigma`` the
    peak and the spread of its probability density), so that the absorbing
    wall penetrates about ``sigma sqrt(P_state / kappa~)`` into a state on
    every slice alike. ``stiffness = 0`` is the ``kappa -> 0`` limit in closed
    form: the flux through interface ``i + 1/2`` is ``C_A(i) - C_B(i)``, the
    difference of the state CDFs, i.e. ``q' ~ (C_A - C_B) / rho``, the
    maximiser of the slice's own quotient
    ``(INT rho_B q - INT rho_A q)^2 / INT rho q'^2``, rescaled to ``[0, 1]``.
    """
    n = s_values.shape[0]
    rho = jnp.exp(-(F_values - jnp.min(F_values)))
    inner_edges = bin_edges[1:-1]

    def _state_mass(s_proj_state, W_state):
        idx = jnp.clip(jnp.searchsorted(inner_edges, s_proj_state), 0, n - 1)
        if W_state is None:
            hist = jnp.zeros(n).at[idx].add(1.0)
            return hist / max(s_proj_state.shape[0], 1)
        hist = jnp.zeros(n).at[idx].add(W_state)
        return hist / jnp.maximum(jnp.sum(W_state), 1e-300)

    m_A = _state_mass(s_A_proj, W_A)
    m_B = _state_mass(s_B_proj, W_B)

    if boundary_quantile < 1.0:
        A_is_left = jnp.mean(s_A_proj) < jnp.mean(s_B_proj)
        m_A = jnp.where(
            _absorption_keep_mask(s_values, s_A_sorted, A_is_left, boundary_quantile), m_A, 0.0
        )
        m_B = jnp.where(
            _absorption_keep_mask(s_values, s_B_sorted, ~A_is_left, boundary_quantile), m_B, 0.0
        )

    rho_half = 0.5 * (rho[:-1] + rho[1:])
    conductance = rho_half / jnp.maximum(jnp.diff(s_values), 1e-10)

    if stiffness == 0.0:
        mA = m_A / jnp.maximum(jnp.sum(m_A), 1e-300)
        mB = m_B / jnp.maximum(jnp.sum(m_B), 1e-300)
        q = jnp.concatenate([jnp.zeros(1), jnp.cumsum(jnp.cumsum(mA - mB)[:-1] / conductance)])
        lo, span = jnp.min(q), jnp.max(q) - jnp.min(q)
        return jnp.where(span > 0, (q - lo) / jnp.where(span > 0, span, 1.0), 0.5)
    if stiffness is None:
        kappa = rd_kappa
    else:
        h = jnp.maximum(bin_edges[1:] - bin_edges[:-1], 1e-12)
        p = rho / jnp.sum(rho * h)
        mean = jnp.sum(p * h * s_values)
        var = jnp.maximum(jnp.sum(p * h * (s_values - mean) ** 2), 1e-300)
        kappa = stiffness / (jnp.max(p) * var)

    # Tridiagonal system: row i is sub_i q_{i-1} + diag_i q_i + sup_i q_{i+1} = rhs_i.
    sub = jnp.concatenate([jnp.zeros(1), conductance])
    sup = jnp.concatenate([conductance, jnp.zeros(1)])
    diag = -(sub + sup) - kappa * (m_A + m_B)
    rhs = -kappa * m_B

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


def _compute_log_dirichlet(s_vals, F_vals, q_vals):
    """``log sum_i c_{i+1/2} (q_{i+1} - q_i)^2``: the Dirichlet energy ``INT rho (dq/ds)^2``
    of the slice's discrete functional, with the conductances
    ``c_{i+1/2} = rho_{i+1/2} / (s_{i+1} - s_i)`` of :func:`compute_1d_rd_committor`."""
    q = jnp.where(jnp.isnan(q_vals), 0.0, q_vals)
    rho = jnp.exp(-(F_vals - jnp.min(F_vals)))
    log_c = jnp.log(0.5 * (rho[:-1] + rho[1:])) - jnp.log(jnp.maximum(jnp.diff(s_vals), 1e-10))
    log_dq2 = jnp.log(jnp.maximum(jnp.diff(q) ** 2, 1e-300))
    return jax.scipy.special.logsumexp(log_c + log_dq2)


# =============================================================================
# THE SLICE BASIS
# =============================================================================


class _SliceInputs(NamedTuple):
    """The arrays every slice reads: the samples, the basin index sets, the
    sample weights with the basin subsets and their sums (None for uniform
    weights), and the quantile subsample index (None below the size threshold)."""

    samples: jnp.ndarray
    A_indices: jnp.ndarray
    B_indices: jnp.ndarray
    W: jnp.ndarray | None
    W_A: jnp.ndarray | None
    W_B: jnp.ndarray | None
    sumWA: jnp.ndarray | None
    sumWB: jnp.ndarray | None
    subsample_idx: jnp.ndarray | None


class _SliceSettings(NamedTuple):
    """The static settings of the 1D problem (hashable: one compile per value)."""

    n_bins: int
    binning_method: str
    density_floor: float
    n_min: int
    rd_kappa: float
    boundary_quantile: float
    use_subsample: bool
    rd_stiffness: float | None


@partial(jit, static_argnames=("settings",))
def _slices(directions, inputs: _SliceInputs, settings: _SliceSettings):
    """The 1D problem on every direction: projection, free energy, RD committor,
    inner edges and the fused per-slice diagnostics, vmapped over ``directions``.
    Compiled once per shape and setting; the arrays enter as arguments."""
    n_A_f = jnp.float32(inputs.A_indices.shape[0])
    n_B_f = jnp.float32(inputs.B_indices.shape[0])

    def _slice(theta):
        s_projected = inputs.samples @ theta
        s_sub = s_projected[inputs.subsample_idx] if settings.use_subsample else None
        s_vals, F_vals, edges = compute_1d_free_energy(
            s_projected,
            settings.n_bins,
            density_floor=settings.density_floor,
            binning_method=settings.binning_method,
            n_min=settings.n_min,
            s_subsample=s_sub,
            use_subsample_quantile=settings.use_subsample,
            return_edges=True,
            weights=inputs.W,
        )
        s_A = s_projected[inputs.A_indices]
        s_B = s_projected[inputs.B_indices]
        s_A_sorted = jnp.sort(s_A)
        s_B_sorted = jnp.sort(s_B)
        a_idx, b_idx = find_boundary_indices(
            s_vals, s_A_sorted, s_B_sorted, settings.boundary_quantile
        )
        q_rd = compute_1d_rd_committor(
            s_vals,
            F_vals,
            s_A,
            s_B,
            bin_edges=edges,
            rd_kappa=settings.rd_kappa,
            boundary_quantile=settings.boundary_quantile,
            s_A_sorted=s_A_sorted,
            s_B_sorted=s_B_sorted,
            W_A=inputs.W_A,
            W_B=inputs.W_B,
            stiffness=settings.rd_stiffness,
        )
        is_valid = jnp.all(jnp.isfinite(q_rd))
        log_D = jnp.where(is_valid, _compute_log_dirichlet(s_vals, F_vals, q_rd), jnp.inf)
        q_rd_safe = jnp.where(jnp.isnan(q_rd), 0.0, q_rd)
        q_A = _interp_committor(s_vals, q_rd_safe, s_A)
        q_B = _interp_committor(s_vals, q_rd_safe, s_B)
        if inputs.W_A is None:
            eps_A = jnp.sum(q_A) / n_A_f
            eps_B = jnp.sum(1.0 - q_B) / n_B_f
        else:
            eps_A = jnp.sum(inputs.W_A * q_A) / jnp.maximum(inputs.sumWA, 1e-30)
            eps_B = jnp.sum(inputs.W_B * (1.0 - q_B)) / jnp.maximum(inputs.sumWB, 1e-30)
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

    return vmap(_slice)(directions)


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
    rd_stiffness: float | None = None,
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
            absorption inside the basins), in the units of the slice's
            ``rho / max(rho)``; ignored when ``rd_stiffness`` is given.
        rd_stiffness: the absorption as the dimensionless ``kappa~ = kappa
            p_max sigma^2`` of each slice (``p_max``, ``sigma``: peak and
            spread of its probability density), which means the same on every
            slice and system; ``0`` is the ``kappa -> 0`` limit in closed form,
            the profile ``q' ~ (C_A - C_B) / rho`` that maximises the slice's
            own fidelity-to-energy quotient. None (default) uses ``rd_kappa``.
            A soft wall gives a better-shaped basis but lets the profile vary
            inside the states, which biases the basin-moment fidelity the
            weights are solved against; see the settings page before using it.
        boundary_quantile: inner basin edges at this quantile of the state
            projections (1 = the extremes). Below 1 the absorption density is
            truncated past the same quantile, which suppresses the halo that
            many nuisance dimensions inflate around a projected basin.
        direction_batch_size: process the directions in batches of this size
            to bound the ``(batch, N)`` working memory; None = one vmap.
        sample_weights: ``(N,)`` reweighting (e.g. MBAR) summing to one; None
            = uniform. Each slice's density and state masses are the weighted
            histograms, and the weight solve averages its Gram with them; the
            basin moments stay plain averages over the labelled frames. The
            weights are used verbatim, so the reported Dirichlet energies of
            fits are comparable only on that scale.
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
    if rd_stiffness is not None:
        rd_stiffness = float(rd_stiffness)
        if not (rd_stiffness >= 0.0 and rd_stiffness < float("inf")):
            raise ValueError(f"rd_stiffness must be finite and >= 0; got {rd_stiffness}")
    if dim > n_samples_total:
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

    if sample_weights is None:
        W_A = W_B = sumWA = sumWB = None
    else:
        sample_weights = jnp.asarray(sample_weights, jnp.float64)
        if sample_weights.shape != (n_samples_total,):
            raise ValueError(
                f"sample_weights must be (N,); got {sample_weights.shape}, N={n_samples_total}"
            )
        total = float(jnp.sum(sample_weights))
        if not abs(total - 1.0) <= 1e-6:
            raise ValueError(
                f"sample_weights must sum to one (got {total!r}): normalise them first, the "
                "solve uses them verbatim."
            )
        W_A, W_B = sample_weights[A_indices], sample_weights[B_indices]
        sumWA, sumWB = jnp.sum(W_A), jnp.sum(W_B)

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

    inputs = _SliceInputs(
        samples, A_indices, B_indices, sample_weights, W_A, W_B, sumWA, sumWB, subsample_idx
    )
    settings = _SliceSettings(
        n_bins,
        binning_method,
        density_floor,
        n_min,
        rd_kappa,
        boundary_quantile,
        use_subsample,
        rd_stiffness,
    )
    if direction_batch_size is None or direction_batch_size >= n_directions:
        out = _slices(directions, inputs, settings)
    else:
        parts = [
            _slices(directions[start : start + direction_batch_size], inputs, settings)
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
