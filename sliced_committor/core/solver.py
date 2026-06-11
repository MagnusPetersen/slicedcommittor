"""
Sliced Committor Computation
=============================

Sliced committor pipeline with RD (reaction-diffusion) solver and
statistically motivated free energy truncation.

The adaptive density floor

    adaptive_floor = n_min / (N · Δs)

adapts to the sample size N and bin width Δs, preventing tail bins
with very few samples from producing artificially high free energies.

Key design choices:

1. **Single projection per direction**: ``compute_1d_free_energy`` accepts
   pre-projected values, so the expensive ``samples @ theta`` is done once
   inside ``_compute_direction_core`` and shared by free-energy estimation,
   boundary detection, and the RD solver.

2. **RD committor**: The RD committor (reaction-diffusion with soft
   absorption) is used for all 1D solves.

3. **Fused weight diagnostics**: log(D) and ε̂ are computed inside the
   main vmap as a byproduct of the 1D solve.

4. **Pre-extracted state indices**: Boundary error interpolation touches
   only |A| + |B| samples instead of all N.

5. **No O(M·N) storage by default**: The ``(M, N)`` projected sample
   array is only materialised when ``store_projected_samples=True``.

Pipeline:
    1. compute_sliced_committor  →  SlicedCommittorResult
    2. compute_weights_multi     →  {name: weights}
    3. evaluate_committor        →  q̄(x)
"""

from collections.abc import Callable
from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import jit, lax, random, vmap

from ._internal import sample_random_directions, signed_logsumexp, to_log_abs_sign
from .directions import DirectionSamplingConfig, sample_directions

# =============================================================================
# DATA STRUCTURES
# =============================================================================


class SlicedCommittorResult(NamedTuple):
    """Complete result from sliced committor computation."""

    directions: jnp.ndarray  # (M, dim)
    slice_coords: jnp.ndarray  # (M, n_bins)
    free_energies: jnp.ndarray  # (M, n_bins), equals -log ρ (β=1 convention)
    committors_1d: jnp.ndarray  # (M, n_bins)
    boundary_indices: jnp.ndarray  # (M, 2)  extreme-based boundaries
    valid_mask: jnp.ndarray  # (M,)
    in_A: jnp.ndarray  # (N,) bool, basin-A membership of input samples
    in_B: jnp.ndarray  # (N,) bool, basin-B membership of input samples
    projected_samples: jnp.ndarray | None = None  # (M, N)
    log_dirichlet: jnp.ndarray | None = None  # (M,) Dirichlet energy
    boundary_errors: jnp.ndarray | None = None  # (M,) boundary ε̂
    sample_weights: jnp.ndarray | None = None  # (N,) optional non-uniform weights
    axis: jnp.ndarray | None = None  # (dim,) LDA axis if direction_sampling.mode='lda'
    lda_info: dict | None = None  # LDA diagnostics dict

    def summary(self) -> str:
        """Human-readable diagnostic summary of the result.

        Returns a multi-line string covering the number of directions and
        bins, the valid-mask fraction (and why slices were dropped), mean
        cached boundary error if available, mean Dirichlet energy across
        valid slices, and whether projected samples are stored.

        The output is intended for ``print(result.summary())`` at the REPL
        or in tutorial notebooks. The format may change between minor
        versions; do not parse it.
        """
        M = int(self.directions.shape[0])
        n_bins = int(self.slice_coords.shape[1])
        dim = int(self.directions.shape[1])
        N = int(self.in_A.shape[0])
        n_A = int(jnp.sum(self.in_A))
        n_B = int(jnp.sum(self.in_B))

        valid = jnp.asarray(self.valid_mask, dtype=bool)
        n_valid = int(jnp.sum(valid))
        n_invalid = M - n_valid

        lines = [
            "SlicedCommittorResult",
            "---------------------",
            f"  directions      : M = {M}, dim = {dim}",
            f"  bins per slice  : {n_bins}",
            f"  samples         : N = {N}  (|A| = {n_A}, |B| = {n_B})",
            f"  valid slices    : {n_valid}/{M}" + (f"  ({n_invalid} masked)" if n_invalid else ""),
        ]

        if self.boundary_errors is not None and n_valid > 0:
            eps = jnp.asarray(self.boundary_errors)
            eps_valid = eps[valid]
            lines.append(
                f"  mean eps (eq.)  : {float(jnp.mean(eps_valid)):.4f}  "
                f"(min {float(jnp.min(eps_valid)):.4f}, "
                f"max {float(jnp.max(eps_valid)):.4f})"
            )

        if self.log_dirichlet is not None and n_valid > 0:
            log_D = jnp.asarray(self.log_dirichlet)
            log_D_valid = log_D[valid]
            lines.append(f"  mean log D[q]   : {float(jnp.mean(log_D_valid)):.3f}")

        lines.append(
            "  projected stored: "
            + ("yes (M, N) buffer" if self.projected_samples is not None else "no")
        )
        return "\n".join(lines)


def why_masked(result: "SlicedCommittorResult", slice_index: int) -> str:
    """Explain why a particular slice ended up with ``valid_mask=False``.

    Inspects the 1D committor row and projected basin extents to attribute
    the masking reason. Returns ``"slice is valid"`` for valid slices.

    Args:
        result: a :class:`SlicedCommittorResult`.
        slice_index: integer in ``[0, M)``.

    Returns:
        A one-line human-readable explanation. Possible outcomes:

        - ``"slice is valid"``
        - ``"slice contains NaN in committors_1d"``
        - ``"degenerate density (committor is flat across the slice)"``
        - ``"A and B project onto overlapping s-intervals; no separation"``
        - ``"unknown reason (slice masked but no diagnostic matches)"``
    """
    M = int(result.directions.shape[0])
    if not (0 <= slice_index < M):
        raise IndexError(f"slice_index={slice_index} out of range [0, {M}).")
    if bool(result.valid_mask[slice_index]):
        return "slice is valid"

    q_row = jnp.asarray(result.committors_1d[slice_index])
    if bool(jnp.any(jnp.isnan(q_row))):
        return "slice contains NaN in committors_1d"

    q_span = float(jnp.max(q_row) - jnp.min(q_row))
    if q_span < 1e-8:
        return "degenerate density (committor is flat across the slice)"

    if result.projected_samples is not None:
        s_proj = jnp.asarray(result.projected_samples[slice_index])
        s_A = s_proj[result.in_A]
        s_B = s_proj[result.in_B]
        if s_A.size > 0 and s_B.size > 0:
            a_lo, a_hi = float(jnp.min(s_A)), float(jnp.max(s_A))
            b_lo, b_hi = float(jnp.min(s_B)), float(jnp.max(s_B))
            overlap = (a_lo <= b_hi) and (b_lo <= a_hi)
            if overlap:
                return "A and B project onto overlapping s-intervals; no separation"

    return "unknown reason (slice masked but no diagnostic matches)"


def summarize_gram_diagnostics(result_dict: dict) -> str:
    """Format the diagnostic block from a Gram-based weight solver result.

    Accepts any dict returned by :func:`compute_full_gram_weights`,
    :func:`compute_basin_moment_weights`,
    :func:`compute_enriched_basin_moment_weights`, or
    :func:`compute_enriched_basin_moment_weights_power`. Missing fields are
    silently skipped, so the same helper works across solvers.

    Returns a multi-line summary string suitable for printing at the REPL.
    The format may change between minor versions; do not parse it.
    """
    if not isinstance(result_dict, dict):
        raise TypeError(
            "summarize_gram_diagnostics expects a dict (the return value of "
            f"a Gram-based solver), got {type(result_dict).__name__}."
        )

    lines: list[str] = ["Gram solver diagnostics", "-----------------------"]

    keys_in_order = [
        ("constraint", "constraint"),
        ("eta_used", "tikhonov used"),
        ("N_eff", "N_eff"),
        ("condition_number", "cond(G_reg)"),
        ("off_diagonal_magnitude", "mean |G_jk|/sqrt(G_jj G_kk)"),
        ("n_negative_weights", "# negative weights"),
        ("negative_weight_mass", "negative weight mass"),
        ("sum_w", "sum_w"),
        ("constraint_residual_A", "|a.w|"),
        ("constraint_residual_B", "|b.w - 1|"),
        ("cond_basin", "cond_basin"),
        ("cond_enriched", "cond_enriched"),
        ("improvement_over_bmc", "EBMC Dirichlet gain over BMC"),
        ("improvement_over_ebmc", "PESB gain over EBMC"),
        ("optimal_dirichlet_energy", "optimal D-energy"),
    ]

    for key, label in keys_in_order:
        if key not in result_dict:
            continue
        val = result_dict[key]
        if val is None:
            continue
        if isinstance(val, (int,)):
            lines.append(f"  {label:<32}: {val}")
        elif hasattr(val, "shape") and val.shape == ():
            lines.append(f"  {label:<32}: {float(val):.4g}")
        elif isinstance(val, float):
            lines.append(f"  {label:<32}: {val:.4g}")
        elif isinstance(val, str):
            lines.append(f"  {label:<32}: {val}")
        else:
            lines.append(f"  {label:<32}: {val}")
    return "\n".join(lines)


class WeightingContext(NamedTuple):
    """Bundle of per-slice arrays passed to every weight solver and ε estimator.

    Built via :func:`make_weighting_context` from a
    :class:`SlicedCommittorResult`. The first eight fields are always
    populated by ``make_weighting_context``; the remainder are either
    inherited from the result or filled in by the weight pipeline.

    Required fields:
        directions: (M, dim) unit vectors.
        slice_coords: (M, n_bins) per-slice 1D grid.
        free_energies: (M, n_bins) ``-log ρ`` along each slice (β=1 convention).
        committors_1d: (M, n_bins) per-slice 1D RD committor.
        boundary_indices: (M, 2) indices of the A/B boundary nodes in
            ``slice_coords``.
        valid_mask: (M,) bool, False where states A and B overlapped in
            projection or the slice was otherwise rejected.
        in_A, in_B: (N,) bool basin labels of the input samples.

    Optional fields (None unless explicitly populated):
        ds: (M,) per-direction bin spacing. Populated when constant-spacing
            grids are used.
        n_directions, n_bins: counts; informational.
        projected_samples: (M, N) only when ``store_projected_samples=True``
            in :func:`compute_sliced_committor`. Required by the full-Gram
            and BMC solvers.
        log_dirichlet: (M,) ``log(D_j[q_j])``; populated by the fused-diagnostic
            path inside :func:`compute_sliced_committor`.
        boundary_errors: (M,) cached equilibrium ε. Reused by
            :func:`full_gram_weights` when ``epsilon_fn`` is None.
        cos_matrix: (M, M) direction cosines; populated when slice
            correlations are requested.
        sample_weights: (N,) reweighting (e.g. MBAR). None means uniform 1/N.
    """

    directions: jnp.ndarray
    slice_coords: jnp.ndarray
    free_energies: jnp.ndarray
    committors_1d: jnp.ndarray
    boundary_indices: jnp.ndarray
    valid_mask: jnp.ndarray
    in_A: jnp.ndarray
    in_B: jnp.ndarray
    ds: jnp.ndarray = None
    n_directions: int = 0
    n_bins: int = 0
    projected_samples: jnp.ndarray | None = None
    log_dirichlet: jnp.ndarray | None = None
    boundary_errors: jnp.ndarray | None = None
    cos_matrix: jnp.ndarray | None = None
    sample_weights: jnp.ndarray | None = None


# =============================================================================
# 1D INTERPOLATION HELPERS (used by fused diagnostics and weights)
# =============================================================================


@jit
def _interp_1d_at_samples(
    s_grid: jnp.ndarray, y_grid: jnp.ndarray, s_samples: jnp.ndarray
) -> jnp.ndarray:
    """Linear interpolation of grid values at sample positions.

    Uses searchsorted to handle non-uniform grids (e.g. quantile binning).
    """
    n = s_grid.shape[0]
    idx = jnp.searchsorted(s_grid, s_samples, side="right") - 1
    idx = jnp.clip(idx, 0, n - 2)
    ds = s_grid[idx + 1] - s_grid[idx]
    frac = (s_samples - s_grid[idx]) / jnp.maximum(ds, 1e-10)
    frac = jnp.clip(frac, 0.0, 1.0)

    return (1 - frac) * y_grid[idx] + frac * y_grid[idx + 1]


@jit
def _interp_committor_at_samples(
    s_grid: jnp.ndarray, q_grid: jnp.ndarray, s_samples: jnp.ndarray
) -> jnp.ndarray:
    """
    Interpolate committor q(s) at sample projection positions.

    Clamps to grid boundary values for out-of-range samples, correctly
    handling inverted slices where q(s_min) = 1 and q(s_max) = 0.
    """
    q_interp = _interp_1d_at_samples(s_grid, q_grid, s_samples)
    q_interp = jnp.clip(q_interp, 0.0, 1.0)

    s_min, s_max = s_grid[0], s_grid[-1]
    q_interp = jnp.where(s_samples < s_min, q_grid[0], q_interp)
    q_interp = jnp.where(s_samples > s_max, q_grid[-1], q_interp)
    return q_interp


# =============================================================================
# 1D FREE ENERGY WITH STATISTICAL TRUNCATION
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
    s_projected: jnp.ndarray,
    beta: float,
    n_bins: int = 200,
    density_floor: float = 1e-6,
    binning_method: str = "quantile",
    n_min: int = 10,
    s_subsample: jnp.ndarray = None,
    use_subsample_quantile: bool = False,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Compute 1D free energy F_θ(s) from pre-projected sample values.

    Given projected values s_k = θ·x_k, estimate density via histogram
    and define F_θ(s) = −β⁻¹ log p_θ(s) + const.

    Args:
        s_projected: (N,) projected sample values (already on a normalised direction)
        beta: Inverse temperature
        n_bins: Number of histogram bins (static for JIT)
        density_floor: Minimum density value to prevent log(0) (static for JIT)
        binning_method: 'equal_width' (default) or 'quantile' (equal-count bins)
        n_min: Minimum count per bin for adaptive floor (static for JIT)
        s_subsample: (K,) projected subsample for approximate quantile edges
        use_subsample_quantile: If True, use s_subsample for quantile edges
            and histogram all N samples into those bins (static for JIT)

    Returns:
        s_centers: (n_bins,) bin centre coordinates
        F_values: (n_bins,) free energy values (shifted so min = 0)
    """
    s = s_projected

    s_min, s_max = jnp.min(s), jnp.max(s)
    span = jnp.maximum(s_max - s_min, 1e-8)

    if binning_method == "quantile" and use_subsample_quantile:
        # Approximate quantile binning: sort subsample for edges,
        # histogram all N samples for density. O(K log K + N log n_bins)
        # instead of O(N log N) for exact quantile.
        s_sorted = jnp.sort(s_subsample)
        K = s_subsample.shape[0]
        edge_indices = jnp.linspace(0, K - 1, n_bins + 1).astype(jnp.int32)
        bin_edges = s_sorted[edge_indices]
        s_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

        ds_per_bin = bin_edges[1:] - bin_edges[:-1]
        ds_per_bin = jnp.maximum(ds_per_bin, 1e-10)

        # Histogram all N samples into approximate quantile bins
        N_total = s.shape[0]
        bin_idx = jnp.searchsorted(bin_edges[1:-1], s)
        bin_idx = jnp.clip(bin_idx, 0, n_bins - 1)
        counts = jnp.zeros(n_bins).at[bin_idx].add(jnp.ones(N_total))
        N = jnp.float64(N_total)

        adaptive_floor = n_min / (N * ds_per_bin)
        effective_floor = jnp.maximum(adaptive_floor, density_floor)
        density = jnp.maximum(counts / (N * ds_per_bin), effective_floor)

        F_values = -(1.0 / beta) * jnp.log(density)
        F_values = F_values - jnp.min(F_values)

    elif binning_method == "quantile":
        # Exact quantile binning: equal-count bins. O(N log N)
        s_sorted = jnp.sort(s)
        N_total = s.shape[0]
        edge_indices = jnp.linspace(0, N_total - 1, n_bins + 1).astype(jnp.int32)
        bin_edges = s_sorted[edge_indices]
        s_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

        ds_per_bin = bin_edges[1:] - bin_edges[:-1]
        ds_per_bin = jnp.maximum(ds_per_bin, 1e-10)

        counts_per_bin = N_total / n_bins
        N = jnp.float64(N_total)

        # Adaptive density floor: n_min / (N * Δs) per bin
        adaptive_floor = n_min / (N * ds_per_bin)
        effective_floor = jnp.maximum(adaptive_floor, density_floor)

        density = counts_per_bin / (N * ds_per_bin)
        density = jnp.maximum(density, effective_floor)

        F_values = -(1.0 / beta) * jnp.log(density)
        F_values = F_values - jnp.min(F_values)
    else:
        # Standard equal-width binning
        bin_edges = jnp.linspace(s_min, s_max, n_bins + 1)
        s_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

        idx = jnp.floor((s - s_min) / span * n_bins).astype(jnp.int32)
        idx = jnp.clip(idx, 0, n_bins - 1)

        counts = jnp.bincount(idx, length=n_bins)
        N = jnp.maximum(jnp.sum(counts), 1.0)
        bin_width = span / n_bins

        # Adaptive density floor: n_min / (N * bin_width)
        adaptive_floor = n_min / (N * bin_width)
        effective_floor = jnp.maximum(adaptive_floor, density_floor)

        density = jnp.maximum(counts / (N * bin_width), effective_floor)
        F_values = -(1.0 / beta) * jnp.log(density)
        F_values = F_values - jnp.min(F_values)

    return s_centers, F_values


@partial(jit, static_argnames=("n_bins", "n_min"))
def compute_1d_free_energy_from_samples(
    samples: jnp.ndarray,
    beta: float,
    theta: jnp.ndarray,
    n_bins: int = 200,
    n_min: int = 10,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Compute 1D free energy F_θ(s) by projecting samples onto direction θ.

    Convenience wrapper around compute_1d_free_energy that does the
    projection internally.
    """
    theta_norm = theta / jnp.linalg.norm(theta)
    s_projected = samples @ theta_norm
    return compute_1d_free_energy(s_projected, beta, n_bins, n_min=n_min)


# =============================================================================
# EXTREME-BASED BOUNDARY DETECTION
# =============================================================================


def find_boundary_indices_from_samples(
    s_values: jnp.ndarray,
    s_projected: jnp.ndarray,
    in_A: jnp.ndarray,
    in_B: jnp.ndarray,
    s_A_proj: jnp.ndarray = None,
    s_B_proj: jnp.ndarray = None,
    boundary_quantile: float = 1.0,
    s_A_sorted: jnp.ndarray = None,
    s_B_sorted: jnp.ndarray = None,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Extreme-based boundaries from projected sample positions.

    For each direction θ:
    1. Orient by mean projections (A left, B right after possible swap).
    2. Compute inner extremes: A⁺ = max{θ·x : x ∈ A}, B⁻ = min{θ·x : x ∈ B}.
    3. Set a = min(A⁺, B⁻), b = max(A⁺, B⁻).

    This always gives a valid interval:
    - Separated (A⁺ < B⁻): transition region is the gap [A⁺, B⁻], ε̂ = 0.
    - Overlapping (A⁺ > B⁻): transition region is the overlap [B⁻, A⁺], ε̂ > 0.

    No direction is ever marked invalid due to state overlap.

    Args:
        s_values: (n_bins,) grid of 1D coordinates
        s_projected: (N,) projected sample values θ·x
        in_A: (N,) boolean, which samples are in state A
        in_B: (N,) boolean, which samples are in state B
        s_A_proj: (n_A,) pre-extracted projections of state A samples (optional)
        s_B_proj: (n_B,) pre-extracted projections of state B samples (optional)
        boundary_quantile: Quantile for inner-edge placement (default 1.0 = extreme).
            Only effective on the sparse path (when ``s_A_proj`` / ``s_B_proj`` are
            pre-supplied, as ``compute_sliced_committor`` always does); the dense
            fallback ignores it. Values < 1.0 shrink the boundary toward the state
            centroid, mitigating the halo artifact in high dimensions where d >> k.

    Returns:
        a_idx, b_idx, is_valid  (all JAX scalars)
    """
    if s_A_proj is not None and s_B_proj is not None:
        # Sparse path: operate on pre-extracted |A| and |B| arrays
        s_A_min, s_A_max = jnp.min(s_A_proj), jnp.max(s_A_proj)
        s_B_min, s_B_max = jnp.min(s_B_proj), jnp.max(s_B_proj)
        s_A_center = jnp.mean(s_A_proj)
        s_B_center = jnp.mean(s_B_proj)
        has_A = s_A_proj.shape[0] > 0
        has_B = s_B_proj.shape[0] > 0
    else:
        # Dense path: mask full N-length arrays
        s_A = jnp.where(in_A, s_projected, jnp.nan)
        s_B = jnp.where(in_B, s_projected, jnp.nan)
        s_A_min, s_A_max = jnp.nanmin(s_A), jnp.nanmax(s_A)
        s_B_min, s_B_max = jnp.nanmin(s_B), jnp.nanmax(s_B)
        s_A_center = jnp.nanmean(s_A)
        s_B_center = jnp.nanmean(s_B)
        has_A = jnp.any(in_A)
        has_B = jnp.any(in_B)

    A_is_left = s_A_center < s_B_center

    # Inner extremes (edges facing each other)
    # Default (boundary_quantile=1.0): use max/min as before.
    # boundary_quantile < 1.0: use sorted quantile to shrink boundaries,
    # mitigating extreme-value inflation from non-state dimensions.
    s_A_inner_extreme = jnp.where(A_is_left, s_A_max, s_A_min)
    s_B_inner_extreme = jnp.where(A_is_left, s_B_min, s_B_max)

    if s_A_proj is not None and s_B_proj is not None:
        if s_A_sorted is None:
            s_A_sorted = jnp.sort(s_A_proj)
        if s_B_sorted is None:
            s_B_sorted = jnp.sort(s_B_proj)
        n_A_local = s_A_proj.shape[0]
        n_B_local = s_B_proj.shape[0]

        # A inner edge: high quantile when A is left, low quantile when A is right
        idx_A_high = jnp.clip(
            jnp.floor(boundary_quantile * (n_A_local - 1)).astype(jnp.int32), 0, n_A_local - 1
        )
        idx_A_low = jnp.clip(
            jnp.ceil((1.0 - boundary_quantile) * (n_A_local - 1)).astype(jnp.int32),
            0,
            n_A_local - 1,
        )
        s_A_inner_quantile = jnp.where(A_is_left, s_A_sorted[idx_A_high], s_A_sorted[idx_A_low])

        # B inner edge: low quantile when A is left (B is right), high when A is right
        idx_B_low = jnp.clip(
            jnp.ceil((1.0 - boundary_quantile) * (n_B_local - 1)).astype(jnp.int32),
            0,
            n_B_local - 1,
        )
        idx_B_high = jnp.clip(
            jnp.floor(boundary_quantile * (n_B_local - 1)).astype(jnp.int32), 0, n_B_local - 1
        )
        s_B_inner_quantile = jnp.where(A_is_left, s_B_sorted[idx_B_low], s_B_sorted[idx_B_high])

        use_quantile = boundary_quantile < 1.0
        s_A_inner = jnp.where(use_quantile, s_A_inner_quantile, s_A_inner_extreme)
        s_B_inner = jnp.where(use_quantile, s_B_inner_quantile, s_B_inner_extreme)
    else:
        # Dense path: quantile not supported, fall back to extremes
        s_A_inner = s_A_inner_extreme
        s_B_inner = s_B_inner_extreme

    # Always-valid interval: a = min(A⁺, B⁻), b = max(A⁺, B⁻)
    s_left = jnp.minimum(s_A_inner, s_B_inner)
    s_right = jnp.maximum(s_A_inner, s_B_inner)

    # Map to grid indices (searchsorted + nearest-neighbor)
    def _nearest_idx(s_grid, s_val):
        n = s_grid.shape[0]
        idx = jnp.searchsorted(s_grid, s_val, side="right")
        idx = jnp.clip(idx, 1, n - 1)
        return jnp.where(
            jnp.abs(s_grid[idx - 1] - s_val) <= jnp.abs(s_grid[idx] - s_val),
            idx - 1,
            idx,
        )

    a_idx = _nearest_idx(s_values, s_left)
    b_idx = _nearest_idx(s_values, s_right)
    a_idx_final = jnp.minimum(a_idx, b_idx)
    b_idx_final = jnp.maximum(a_idx, b_idx)

    # Invalid only if missing state samples entirely
    mid = s_values.shape[0] // 2
    is_valid = has_A & has_B
    a_idx_final = jnp.where(is_valid, a_idx_final, mid)
    b_idx_final = jnp.where(is_valid, b_idx_final, mid)

    return a_idx_final, b_idx_final, is_valid


# =============================================================================
# 1D RD COMMITTOR (reaction-diffusion with soft absorption)
# =============================================================================


def _absorption_keep_mask(s_grid, s_sorted, is_left, quantile):
    """Boolean mask over ``s_grid``: True where the state's absorption
    density should be retained, False past the inner-edge quantile cutoff
    (the long tail toward the *other* state).

    Mirrors the inner-edge convention used for ``boundary_quantile``:
    the inner edge of a left state is its high quantile, of a right
    state its low quantile.
    """
    n = s_sorted.shape[0]
    idx = jnp.where(
        is_left,
        jnp.clip(jnp.floor(quantile * (n - 1)).astype(jnp.int32), 0, n - 1),
        jnp.clip(jnp.ceil((1.0 - quantile) * (n - 1)).astype(jnp.int32), 0, n - 1),
    )
    cutoff = s_sorted[idx]
    return jnp.where(is_left, s_grid <= cutoff, s_grid >= cutoff)


def compute_1d_rd_committor(
    s_values: jnp.ndarray,
    F_values: jnp.ndarray,
    beta: float,
    s_projected: jnp.ndarray,
    in_A: jnp.ndarray,
    in_B: jnp.ndarray,
    is_valid: jnp.ndarray,
    rd_kappa: float = 100.0,
    ds_arr: jnp.ndarray = None,
    s_A_proj: jnp.ndarray = None,
    s_B_proj: jnp.ndarray = None,
    absorption_quantile: float | None = None,
    s_A_sorted: jnp.ndarray = None,
    s_B_sorted: jnp.ndarray = None,
) -> jnp.ndarray:
    """
    1D Reaction-Diffusion committor via tridiagonal solve (Thomas algorithm).

    Minimises E_κ[q] = ∫ ρ(dq/ds)² ds + κ ∫ ρ_A q² ds + κ ∫ ρ_B (1-q)² ds

    Euler-Lagrange (strong form):
        d/ds[ρ dq/ds] = κ [ρ_A q − ρ_B (1 − q)]

    with Neumann BCs: dq/ds = 0 at domain boundaries.

    The RD formulation eliminates the flat-committor problem by distributing
    absorption throughout the state bulks rather than relying on hard
    boundaries at sparse extreme positions.

    Note: κ must be positive for a non-trivial solution.  With κ = 0
    and Neumann BCs, the equation d/ds[ρ dq/ds] = 0 has only the trivial
    q = const solution.

    Args:
        s_values: (n_bins,) grid coordinates
        F_values: (n_bins,) free energy (shifted so min = 0)
        beta: Inverse temperature
        s_projected: (N,) projected sample values (for density estimation)
        in_A: (N,) boolean, state A membership
        in_B: (N,) boolean, state B membership
        is_valid: Whether this direction is valid
        rd_kappa: Absorption strength κ (must be > 0)
        ds_arr: Pre-computed bin spacing (optional, computed if not provided)
        s_A_proj: (n_A,) pre-extracted projections of state A samples (optional)
        s_B_proj: (n_B,) pre-extracted projections of state B samples (optional)
        absorption_quantile: Halo mitigation. ``None`` or ``1.0`` → no
            truncation. Values < 1.0 zero out ρ_A and ρ_B beyond their
            inner-edge quantile, suppressing absorption in the tail
            inflated by non-state nuisance dimensions (d ≫ k regime).
            Requires ``s_A_proj`` and ``s_B_proj``.
        s_A_sorted, s_B_sorted: pre-sorted state projections (optional;
            sorted internally if needed by ``absorption_quantile``).

    Returns:
        q_rd: (n_bins,) RD committor values (NaN if invalid)
    """
    n = s_values.shape[0]

    # Per-bin spacing
    if ds_arr is None:
        ds_arr = jnp.diff(s_values)
        ds_arr = jnp.concatenate([ds_arr, ds_arr[-1:]])

    # Density ρ(s) = exp(−βF(s)), unnormalised
    F_min = jnp.min(F_values)
    rho = jnp.exp(-beta * (F_values - F_min))

    # State densities ρ_A(s), ρ_B(s) via histogram of projected state samples
    s_min, s_max = s_values[0], s_values[-1]
    span = s_max - s_min + 1e-10

    def _bin_state_density_sparse(s_proj_state):
        """Histogram of pre-extracted state samples onto grid."""
        idx = jnp.floor((s_proj_state - s_min) / span * n).astype(jnp.int32)
        idx = jnp.clip(idx, 0, n - 1)
        n_state = s_proj_state.shape[0]
        hist = jnp.zeros(n).at[idx].add(jnp.ones(n_state))
        total = jnp.maximum(jnp.float32(n_state), 1e-10)
        return hist / (total * ds_arr + 1e-30)

    if s_A_proj is not None and s_B_proj is not None:
        # Fast path: use pre-extracted state projections (avoids N-length scatter)
        rho_A = _bin_state_density_sparse(s_A_proj)
        rho_B = _bin_state_density_sparse(s_B_proj)
    else:
        # Fallback: full scatter with boolean mask
        def _bin_state_density(s_proj, in_state):
            idx = jnp.floor((s_proj - s_min) / span * n).astype(jnp.int32)
            idx = jnp.clip(idx, 0, n - 1)
            counts = jnp.where(in_state, 1.0, 0.0)
            hist = jnp.zeros(n).at[idx].add(counts)
            total = jnp.maximum(jnp.sum(hist), 1e-10)
            return hist / (total * ds_arr + 1e-30)

        rho_A = _bin_state_density(s_projected, in_A)
        rho_B = _bin_state_density(s_projected, in_B)

    # ── Halo mitigation: zero ρ_A, ρ_B past the inner-edge quantile of state
    # projections. Only meaningful with sparse state projections; skipped at
    # trace time when ``absorption_quantile`` is None or 1.0 so the no-op case
    # adds zero JAX overhead.
    if (
        absorption_quantile is not None
        and absorption_quantile < 1.0
        and s_A_proj is not None
        and s_B_proj is not None
    ):
        if s_A_sorted is None:
            s_A_sorted = jnp.sort(s_A_proj)
        if s_B_sorted is None:
            s_B_sorted = jnp.sort(s_B_proj)
        A_is_left = jnp.mean(s_A_proj) < jnp.mean(s_B_proj)
        rho_A = jnp.where(
            _absorption_keep_mask(s_values, s_A_sorted, A_is_left, absorption_quantile), rho_A, 0.0
        )
        rho_B = jnp.where(
            _absorption_keep_mask(s_values, s_B_sorted, ~A_is_left, absorption_quantile), rho_B, 0.0
        )

    # ── Build tridiagonal system ──
    #
    # FD discretisation of d/ds[ρ dq/ds] = κ [ρ_A q − ρ_B (1 − q)]
    #
    # Interior node i (second-order central):
    #   [ρ_{i+½} (q_{i+1} - q_i) - ρ_{i-½} (q_i - q_{i-1})] / ds²
    #   = κ [ρ_A_i q_i - ρ_B_i (1 - q_i)]
    #
    # Rearranging:  [diffusion] q = κ (ρ_A + ρ_B) q - κ ρ_B
    # So:  [diffusion - κ(ρ_A + ρ_B)] q = -κ ρ_B

    rho_half_plus = 0.5 * (rho[:-1] + rho[1:])  # (n-1,)
    ds2 = ds_arr**2  # (n,)

    # Direct construction of tridiagonal coefficients (no zeros+scatter)
    # Interior coefficients (i = 1 to n-2)
    sub_interior = rho_half_plus[:-1] / ds2[1 : n - 1]  # a_i = ρ_{i-½}/ds²
    sup_interior = rho_half_plus[1:] / ds2[1 : n - 1]  # c_i = ρ_{i+½}/ds²

    # Boundary coefficients (Neumann BCs: dq/ds = 0)
    c0 = rho_half_plus[0] / ds2[0]  # Node 0: connection to node 1
    an = rho_half_plus[-1] / ds2[n - 1]  # Node n-1: connection to node n-2

    kappa_rhoAB = rd_kappa * (rho_A + rho_B)  # (n,) absorption term

    # sub: [0, interior..., an]
    sub = jnp.concatenate([jnp.zeros(1), sub_interior, an[None]])
    # sup: [c0, interior..., 0]
    sup = jnp.concatenate([c0[None], sup_interior, jnp.zeros(1)])
    # diag: -(sub_i + sup_i) - κ(ρ_A + ρ_B)  for all nodes
    diag = -(sub + sup) - kappa_rhoAB
    # rhs: -κ ρ_B  (uniform formula for all nodes)
    rhs = -rd_kappa * rho_B

    # ── Thomas algorithm (tridiagonal solve), O(n) ──
    #
    # Scalar-carry scan: carry only (cp_prev, dp_prev) scalars,
    # collect outputs via scan's stacking mechanism.
    #
    # Forward sweep:
    #   cp[0] = c[0] / b[0],              dp[0] = rhs[0] / b[0]
    #   w     = b[i] - a[i] * cp[i-1]
    #   cp[i] = c[i] / w,                 dp[i] = (rhs[i] - a[i]*dp[i-1]) / w
    # Back substitution:
    #   q[n-1] = dp[n-1],                 q[i]  = dp[i] - cp[i]*q[i+1]

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

    # Back substitution (scalar carry)
    def _back_step(q_next, xs):
        dp_i, cp_i = xs
        q_i = dp_i - cp_i * q_next
        return q_i, q_i

    _, q_reversed = lax.scan(_back_step, dp[-1], (dp[:-1][::-1], cp[:-1][::-1]))
    q_rd = jnp.concatenate([q_reversed[::-1], dp[-1:]])

    # Clip to [0, 1]
    q_rd = jnp.clip(q_rd, 0.0, 1.0)

    # NOTE: No inversion applied for RD committor.
    # The absorption terms κ·ρ_A·q² and κ·ρ_B·(1-q)² inherently encode
    # q→0 at A and q→1 at B regardless of spatial ordering, so the RD
    # solution is always in the correct convention without inversion.

    return jnp.where(is_valid, q_rd, jnp.nan)


# =============================================================================
# GRADIENT-BASED DIRICHLET ENERGY (for RD committor)
# =============================================================================


def _compute_dqds_grid(s_vals: jnp.ndarray, q_vals: jnp.ndarray) -> jnp.ndarray:
    """Compute dq/ds on the 1D grid via central differences.

    Forward difference at left boundary, backward at right, central interior.
    NaN values in q are replaced with 0 before differencing.

    Returns:
        grad: (n_bins,) array of dq/ds values on the grid.
    """
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


def _compute_log_dirichlet_gradient(
    s_vals: jnp.ndarray,
    F_vals: jnp.ndarray,
    q_vals: jnp.ndarray,
    beta: float,
    is_valid: jnp.ndarray,
    ds_arr: jnp.ndarray = None,
) -> jnp.ndarray:
    """
    Compute log(D^RD) = log(∫ ρ (dq/ds)² ds) via numerical gradient.

    This is the correct Dirichlet energy for the RD committor.  The
    partition-function identity D = 1/Z does NOT hold for RD because
    q^RD satisfies an inhomogeneous ODE.

    Integration is over the full grid (not restricted to [a,b]) since
    the RD committor uses Neumann BCs and the gradient is naturally
    zero outside the transition region.
    """
    n = s_vals.shape[0]

    # Per-bin spacing
    if ds_arr is None:
        ds_arr = jnp.diff(s_vals)
        ds_arr = jnp.concatenate([ds_arr, ds_arr[-1:]])

    grad = _compute_dqds_grid(s_vals, q_vals)

    # Log-space integral: log(ρ · (dq/ds)²)
    F_min = jnp.min(F_vals)
    log_rho = -beta * (F_vals - F_min)
    grad_sq = grad**2
    log_grad_sq = jnp.log(jnp.maximum(grad_sq, 1e-30))
    log_integrand = log_rho + log_grad_sq

    # Trapezoidal integration with endpoint halving
    idxs = jnp.arange(n)
    is_endpoint = (idxs == 0) | (idxs == n - 1)
    log_trap_weight = jnp.where(is_endpoint, jnp.log(0.5), 0.0)
    log_ds_arr = jnp.log(jnp.maximum(ds_arr, 1e-30))

    log_D = jax.scipy.special.logsumexp(log_integrand + log_trap_weight + log_ds_arr)

    return jnp.where(is_valid, log_D, jnp.inf)


# =============================================================================
# MAIN COMPUTATION
# =============================================================================


def compute_sliced_committor(
    samples: jnp.ndarray,
    *,
    in_A: jnp.ndarray,
    in_B: jnp.ndarray,
    n_directions: int = 256,
    n_bins: int = 200,
    seed: int = 42,
    store_projected_samples: bool = True,
    density_floor: float = 1e-6,
    binning_method: str = "quantile",
    rd_kappa: float = 1e12,
    n_min: int = 10,
    quantile_subsample: int = None,
    direction_batch_size: int = None,
    boundary_quantile: float = 1.0,
    absorption_quantile: float | None = None,
    sample_weights: jnp.ndarray | None = None,
    directions: jnp.ndarray | None = None,
    direction_sampling: DirectionSamplingConfig | None = None,
) -> SlicedCommittorResult:
    """
    Compute sliced committor approximation from samples.

    Args:
        samples: (N, dim) samples
        in_A: (N,) bool, basin-A membership of each sample
        in_B: (N,) bool, basin-B membership of each sample
        n_directions: Number of projection directions M
        n_bins: Histogram resolution for 1D free energy
        seed: Random seed for direction sampling
        store_projected_samples: Store (M, N) projected samples
        density_floor: Minimum density for histogram bins
        binning_method: 'equal_width' or 'quantile'
        rd_kappa: RD absorption strength κ.
        quantile_subsample: If set and > 0, use a random subsample of this
            size for quantile edge estimation. Reduces O(N log N) sort to
            O(K log K + N log n_bins). When None (default), auto-enabled for
            N > 20000 (with binning_method='quantile') using a subsample of
            10000; pass 0 to force the exact full-N sort.
        direction_batch_size: Process directions in batches of this size to
            limit memory. If None (default), all directions are processed
            in a single vmap call. Set to e.g. 64 or 128 when N is large.
        boundary_quantile: Quantile for boundary placement (default 1.0
            = extreme). Values < 1.0 (e.g. 0.75) shrink boundaries toward
            state centroids, mitigating the halo artifact in high
            dimensions where d >> k.
        absorption_quantile: Quantile for RD absorption-density truncation
            (paired halo-mitigation knob). ``None`` (default) inherits
            ``boundary_quantile`` so a single setting tunes both. Pass an
            explicit float to decouple; ``1.0`` disables truncation.
        sample_weights: (N,) optional non-uniform weights (e.g. MBAR). When
            provided, free-energy estimation, basin densities, and ε
            diagnostics are reweighted. ``None`` ⇒ uniform 1/N.
        directions: (M, dim) pre-supplied directions, overriding random
            sampling. Useful for canonical-CV pipelines.
        direction_sampling: ``DirectionSamplingConfig`` for biased samplers
            (LDA / power-spherical mixture). Ignored if ``directions`` is
            also passed.

    Returns:
        SlicedCommittorResult

    Note:
        The method is β-invariant given fixed samples: an inverse-temperature
        argument would simply rescale the stored ``free_energies`` field
        without affecting any computed committor or weight. The library
        therefore stores ``free_energies = -log ρ`` (β=1 convention); multiply
        by ``1/β_physical`` to recover physical-units free energy.
    """
    key = random.PRNGKey(seed)
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
    if dim > n_samples_total:
        import warnings as _warnings

        _warnings.warn(
            f"samples.shape={samples.shape}: dim ({dim}) exceeds N ({n_samples_total}). "
            "Per-slice density estimation is fragile in this regime; consider "
            "lowering boundary_quantile (e.g. 0.95) to mitigate the halo artifact.",
            UserWarning,
            stacklevel=2,
        )
    # β cancels in every downstream output; fixing β=1 makes free_energies = -log ρ.
    beta = 1.0
    # Halo mitigation: absorption truncation defaults to the boundary quantile.
    if absorption_quantile is None:
        absorption_quantile = boundary_quantile

    # Pre-extract state sample indices for efficient boundary error. n_A / n_B
    # are reused from the validation block above; no second jnp.sum.
    A_indices = jnp.nonzero(in_A, size=n_A)[0]
    B_indices = jnp.nonzero(in_B, size=n_B)[0]
    n_A_f = jnp.float32(n_A)
    n_B_f = jnp.float32(n_B)

    # MBAR sample_weights: precompute state-restricted subsets once.
    # Membership is defined in ambient space so W_A, W_B are direction-invariant.
    if sample_weights is None:
        W_A_norm = None
        W_B_norm = None
        sumWA = None
        sumWB = None
    else:
        W_total = jnp.sum(sample_weights)
        W_norm = sample_weights / jnp.maximum(W_total, 1e-30)
        W_A_norm = W_norm[A_indices]
        W_B_norm = W_norm[B_indices]
        sumWA = jnp.sum(W_A_norm)
        sumWB = jnp.sum(W_B_norm)

    # Direction sampling. Three sources, in priority order:
    #   1. ``directions`` pre-supplied (CV-discovery pipelines: canonical
    #      Cartesian, sin/cos torsion lift, ...).
    #   2. ``direction_sampling`` config object (uniform or LDA-biased
    #      power-spherical + uniform mixture; see preprocessing.py).
    #   3. Default: uniform on the (dim-1)-sphere.
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
            import warnings as _warnings

            _warnings.warn(
                "Both `directions` and `direction_sampling` were supplied; "
                "ignoring `direction_sampling` because pre-supplied "
                "`directions` take precedence.",
                stacklevel=2,
            )
    else:
        key = random.PRNGKey(seed)
        if direction_sampling is None or direction_sampling.mode == "uniform":
            directions = sample_random_directions(key, n_directions, dim)
        else:
            directions, lda_info = sample_directions(
                key,
                n_directions,
                dim,
                samples,
                in_A,
                in_B,
                direction_sampling,
            )
            if lda_info is not None:
                # Single-axis modes (lda) carry 'axis'; geometric multi-axis
                # modes (pca / gcpca) don't, so this defaults to None and
                # the multi-axis info is still stored on the result.
                axis = lda_info.get("axis", None)

    # Prepare subsample for approximate quantile binning (large N optimization)
    N = samples.shape[0]
    if quantile_subsample is None:
        # Auto-enable for large N: subsample to 10000
        _subsample_size = 10000 if N > 20000 and binning_method == "quantile" else 0
    else:
        _subsample_size = quantile_subsample

    if _subsample_size > 0 and _subsample_size < N and binning_method == "quantile":
        sub_key = random.PRNGKey(seed + 999)  # Separate key for subsample
        subsample_idx = random.choice(sub_key, N, shape=(_subsample_size,), replace=False)
        _use_subsample = True
    else:
        subsample_idx = None
        _use_subsample = False

    # --- Per-direction computation, vmapped ---
    def _compute_direction_core(theta):
        s_projected = samples @ theta

        if _use_subsample:
            s_sub = s_projected[subsample_idx]
        else:
            s_sub = None

        s_vals, F_vals = compute_1d_free_energy(
            s_projected,
            beta,
            n_bins,
            density_floor=density_floor,
            binning_method=binning_method,
            n_min=n_min,
            s_subsample=s_sub,
            use_subsample_quantile=_use_subsample,
        )

        # Compute ds_arr once, shared by RD committor and Dirichlet gradient
        ds_arr = jnp.diff(s_vals)
        ds_arr = jnp.concatenate([ds_arr, ds_arr[-1:]])

        # Pre-extract state projections (shared by boundary detection and RD)
        s_A = s_projected[A_indices]
        s_B = s_projected[B_indices]

        # Sort state projections once (shared by boundary detection and RD absorption)
        s_A_sorted = jnp.sort(s_A)
        s_B_sorted = jnp.sort(s_B)

        a_idx, b_idx, is_valid = find_boundary_indices_from_samples(
            s_vals,
            s_projected,
            in_A,
            in_B,
            s_A_proj=s_A,
            s_B_proj=s_B,
            boundary_quantile=boundary_quantile,
            s_A_sorted=s_A_sorted,
            s_B_sorted=s_B_sorted,
        )

        # ── RD committor (tridiagonal solve) ──
        q_rd = compute_1d_rd_committor(
            s_vals,
            F_vals,
            beta,
            s_projected,
            in_A,
            in_B,
            is_valid,
            rd_kappa=rd_kappa,
            ds_arr=ds_arr,
            s_A_proj=s_A,
            s_B_proj=s_B,
            absorption_quantile=absorption_quantile,
            s_A_sorted=s_A_sorted,
            s_B_sorted=s_B_sorted,
        )

        # ── RD diagnostics ──
        log_D_rd = _compute_log_dirichlet_gradient(
            s_vals,
            F_vals,
            q_rd,
            beta,
            is_valid,
            ds_arr=ds_arr,
        )

        q_rd_safe = jnp.where(jnp.isnan(q_rd), 0.0, q_rd)
        q_A_rd = _interp_committor_at_samples(s_vals, q_rd_safe, s_A)
        q_B_rd = _interp_committor_at_samples(s_vals, q_rd_safe, s_B)
        if W_A_norm is None:
            eps_A_rd = jnp.sum(q_A_rd) / jnp.maximum(n_A_f, 1.0)
            eps_B_rd = jnp.sum(1.0 - q_B_rd) / jnp.maximum(n_B_f, 1.0)
        else:
            eps_A_rd = jnp.sum(W_A_norm * q_A_rd) / jnp.maximum(sumWA, 1e-30)
            eps_B_rd = jnp.sum(W_B_norm * (1.0 - q_B_rd)) / jnp.maximum(sumWB, 1e-30)
        epsilon_rd = eps_A_rd + eps_B_rd

        return (
            s_vals,
            F_vals,
            q_rd,
            jnp.stack([a_idx, b_idx]),
            is_valid,
            log_D_rd,
            epsilon_rd,
            s_projected,
        )

    if store_projected_samples:
        core_fn = _compute_direction_core
    else:

        def core_fn(theta):
            *core, _ = _compute_direction_core(theta)
            return tuple(core)

    # --- Execute (optionally batched to limit memory) ---
    if direction_batch_size is None or direction_batch_size >= n_directions:
        # Single vmap over all directions (original path)
        results_tuple = jit(vmap(core_fn, in_axes=0))(directions)
    else:
        # Process directions in batches to limit peak memory
        n_batches = (n_directions + direction_batch_size - 1) // direction_batch_size
        batch_results = []
        batched_fn = jit(vmap(core_fn, in_axes=0))
        for i in range(n_batches):
            start = i * direction_batch_size
            end = min(start + direction_batch_size, n_directions)
            batch_results.append(batched_fn(directions[start:end]))
        # Concatenate along the direction axis
        results_tuple = tuple(
            jnp.concatenate([b[k] for b in batch_results], axis=0)
            for k in range(len(batch_results[0]))
        )

    if store_projected_samples:
        (
            slice_coords,
            free_energies,
            committors_1d,
            boundary_indices,
            valid_mask,
            log_dirichlet,
            boundary_errors_arr,
            all_projected,
        ) = results_tuple
        projected_samples = all_projected
    else:
        (
            slice_coords,
            free_energies,
            committors_1d,
            boundary_indices,
            valid_mask,
            log_dirichlet,
            boundary_errors_arr,
        ) = results_tuple
        projected_samples = None

    return SlicedCommittorResult(
        directions=directions,
        slice_coords=slice_coords,
        free_energies=free_energies,
        committors_1d=committors_1d,
        boundary_indices=boundary_indices,
        valid_mask=valid_mask,
        in_A=in_A,
        in_B=in_B,
        projected_samples=projected_samples,
        log_dirichlet=log_dirichlet,
        boundary_errors=boundary_errors_arr,
        sample_weights=sample_weights,
        axis=axis,
        lda_info=lda_info,
    )


# =============================================================================
# WEIGHTING CONTEXT CONSTRUCTION
# =============================================================================


def make_weighting_context(result: SlicedCommittorResult) -> WeightingContext:
    """Create WeightingContext from SlicedCommittorResult."""
    ds = result.slice_coords[:, 1] - result.slice_coords[:, 0]  # (M,)
    cos_matrix = result.directions @ result.directions.T  # (M, M)

    return WeightingContext(
        directions=result.directions,
        slice_coords=result.slice_coords,
        free_energies=result.free_energies,
        committors_1d=result.committors_1d,
        boundary_indices=result.boundary_indices,
        valid_mask=result.valid_mask,
        in_A=result.in_A,
        in_B=result.in_B,
        ds=ds,
        n_directions=int(result.directions.shape[0]),
        n_bins=int(result.slice_coords.shape[1]),
        projected_samples=result.projected_samples,
        log_dirichlet=result.log_dirichlet,
        boundary_errors=result.boundary_errors,
        cos_matrix=cos_matrix,
        sample_weights=result.sample_weights,
    )


# =============================================================================
# COMMITTOR EVALUATION
# =============================================================================


def _normalize_log_space_weights(sign_w, log_abs_w, valid_mask):
    """Mask invalid directions (sign_w → 0) and divide by the signed sum.

    Mirrors the old linear-space ``jnp.where(w_sum > 0, w / w_sum, w)``:
    when the total is exactly zero the weights are returned unchanged,
    which produces q̄ ≡ 0 downstream.
    """
    sign_w = sign_w * valid_mask.astype(sign_w.dtype)
    sgn_Z, log_Z = signed_logsumexp(sign_w, log_abs_w)
    all_zero = jnp.isneginf(log_Z)
    safe_log_Z = jnp.where(all_zero, 0.0, log_Z)
    safe_sgn_Z = jnp.where(all_zero, 1.0, sgn_Z)
    sign_norm = sign_w * safe_sgn_Z
    log_abs_norm = log_abs_w - safe_log_Z
    return sign_norm, log_abs_norm


def _weighted_sum_over_directions(sign_w_norm, log_abs_w_norm, q_per_dir):
    """Given normalized log-space weights and q_j(x) values of shape
    (M, N_pts), accumulate Σ_j ŵ_j q_j(x) via signed logsumexp.
    """
    sign_q, log_abs_q = to_log_abs_sign(q_per_dir)
    sign_contrib = sign_w_norm[:, None] * sign_q  # (M, N_pts)
    log_abs_contrib = log_abs_w_norm[:, None] + log_abs_q  # (M, N_pts)
    sgn_sum, log_abs_sum = signed_logsumexp(sign_contrib, log_abs_contrib, axis=0)
    return sgn_sum * jnp.exp(log_abs_sum)


@jit
def _qall_stored(s_coords, q_1d, projected_samples):
    """Per-direction committor matrix from pre-computed projections.

    Returns (M, N) where N is the stored sample count.
    """
    return vmap(_interp_1d_at_samples, in_axes=(0, 0, 0))(s_coords, q_1d, projected_samples)


@jit
def _qall_onthefly(directions, s_coords, q_1d, points_flat):
    """Per-direction committor matrix, projecting ``points_flat`` on the fly.

    Returns (M, N_pts).
    """

    def per_dir(theta, s_grid, q_grid):
        return _interp_1d_at_samples(s_grid, q_grid, points_flat @ theta)

    return vmap(per_dir, in_axes=(0, 0, 0))(directions, s_coords, q_1d)


@partial(jit, static_argnames=("original_shape",))
def _evaluate_logspace_from_qall(q_all, sign_w, log_abs_w, valid_mask, original_shape):
    """Log-space normalised aggregation: q̄(x) = Σ_j ŵ_j q_j(x).

    Uses the signed-logsumexp trick to keep numerics stable when
    ``sign_w`` carries negative entries (full-Gram / BMC outputs).
    """
    sign_w_norm, log_abs_w_norm = _normalize_log_space_weights(sign_w, log_abs_w, valid_mask)
    q_sum = _weighted_sum_over_directions(sign_w_norm, log_abs_w_norm, q_all)
    return q_sum.reshape(original_shape)


@partial(jit, static_argnames=("original_shape",))
def _evaluate_centered_from_qall(q_all, w, c, q_bar, valid_mask, original_shape):
    """Centered-basis aggregation: q̂(x) = c + Σ_m w_m (q_m(x) - q_bar_m).

    Bypasses the sum-to-one normalisation used by the log-space evaluator;
    centered-basis weights are in absolute units rather than convex
    coefficients. Per-slice committors are clipped to ``[0, 1]`` before
    centring (matches the per-slice clip applied in the other paths).
    """
    w_eff = w * valid_mask.astype(w.dtype)
    q_clip = jnp.clip(q_all, 0.0, 1.0)
    contrib = w_eff[:, None] * (q_clip - q_bar[:, None])
    q_sum = c + jnp.sum(contrib, axis=0)
    return q_sum.reshape(original_shape)


@partial(jit, static_argnames=("original_shape",))
def _evaluate_powered_smoothstep_from_qall(
    q_all, w_by_power, n_values, c, valid_mask, original_shape
):
    """PESB-EBMC smoothstep aggregator.

        q̂(x) = c + Σ_{j,k} w_{j,k} · Ψ_{n_k}(clip(q_j(x))),
        Ψ_n(v) = v^n / (v^n + (1 − v)^n).

    Per-slice committors are clipped to ``[0, 1]`` before applying Ψ_n
    (basin endpoints and the half-committor level are invariant under
    Ψ_n for every n ≥ 1; clipping just contains finite-sample drift).
    """
    valid_f = valid_mask.astype(w_by_power.dtype)
    w_eff = w_by_power * valid_f[:, None]  # (M, P)
    n_arr = n_values.astype(w_by_power.dtype)
    q_clip = jnp.clip(q_all, 0.0, 1.0)  # (M, N_pts)

    Q_b = q_clip[:, None, :]  # (M, 1, N_pts)
    Qc = 1.0 - Q_b
    n_b = n_arr[None, :, None]  # (1, P, 1)
    denom = jnp.maximum(Q_b**n_b + Qc**n_b, 1e-30)
    Psi = (Q_b**n_b) / denom  # (M, P, N_pts)
    contrib = jnp.sum(w_eff[:, :, None] * Psi, axis=(0, 1))
    return (c + contrib).reshape(original_shape)


def _apply_boundary_conditions(q, in_A, in_B, original_shape, rescale_transition):
    """Snap query points in basin A → 0 and B → 1, optional transition rescale.

    Shared by :func:`evaluate_committor` and :func:`build_committor`'s closure so
    the two aggregators apply identical boundary handling. ``in_A`` / ``in_B`` may
    be ``None`` (treated as the empty mask) and are reshaped to ``original_shape``.
    When ``rescale_transition`` is set, the non-basin region is affine-rescaled to
    span ``[0, 1]`` before the snap (the weighted average of 1D committors can
    compress the range).
    """
    mask_A = (
        jnp.asarray(in_A).reshape(original_shape)
        if in_A is not None
        else jnp.zeros(original_shape, dtype=bool)
    )
    mask_B = (
        jnp.asarray(in_B).reshape(original_shape)
        if in_B is not None
        else jnp.zeros(original_shape, dtype=bool)
    )
    if rescale_transition:
        in_transition = ~mask_A & ~mask_B
        q_trans = jnp.where(in_transition, q, jnp.nan)
        q_min = jnp.nanmin(q_trans)
        span = jnp.maximum(jnp.nanmax(q_trans) - q_min, 1e-10)
        q = jnp.where(in_transition, (q - q_min) / span, q)
    if in_A is not None:
        q = jnp.where(mask_A, 0.0, q)
    if in_B is not None:
        q = jnp.where(mask_B, 1.0, q)
    return q


def evaluate_committor(
    result: SlicedCommittorResult,
    points: jnp.ndarray,
    weights: jnp.ndarray,
    batch_size: int = None,
    enforce_boundary_conditions: bool = True,
    in_A: jnp.ndarray | None = None,
    in_B: jnp.ndarray | None = None,
    rescale_transition: bool = False,
    use_stored_projections: bool = False,
    clip: bool = True,
) -> jnp.ndarray:
    """
    Evaluate weighted sliced committor at given points.

    Args:
        result: SlicedCommittorResult from compute_sliced_committor
        points: (..., dim) evaluation points
        weights: (M,) direction weights, OR a dict from a weight solver.
            Three dict shapes are accepted:
              - {'sign_w', 'log_abs_w'}    log-space signed weights
              - {'w'}                      raw (M,) weights
              - {'w', 'c', 'q_bar'}        centered-basis form for affine
                ansätze with an intercept:
                hat q(x) = c + sum_m w_m (q_m(x) - q_bar_m).
                The 'c' and 'q_bar' keys, when present, fold into a
                scalar offset added before clipping/clamping.
        batch_size: Process directions in batches to limit memory.
            Auto-selected if estimated memory > 1 GB.
        enforce_boundary_conditions: If True (default), set q = 0 for
            points in state A and q = 1 for points in state B.
        in_A: (N,) boolean override for state A membership
        in_B: (N,) boolean override for state B membership
        rescale_transition: If True, affine-rescale the transition region
            (outside both states) so that its min maps to 0 and max to 1.
            Corrects range compression from weighted averaging.
        use_stored_projections: If True and result.projected_samples is
            available, skip the M×dim×N projection matmul and use stored
            (M, N) projections directly. Only valid when evaluating at the
            same samples used in compute_sliced_committor.
        clip: If True (default), clip the per-point output to [0, 1] before
            BC clamping.

    Returns:
        Committor values with shape points.shape[:-1]
    """
    points = jnp.asarray(points)

    # Back-compat shim: accept either (a) a dict with 'sign_w' and 'log_abs_w'
    # fields (preferred, full-precision log-space path), (b) a dict carrying a
    # raw 'w' array, or (c) a raw array. (a) and (b) come from the weighting
    # functions; (c) is the historical call signature.
    # Affine-ansatz dicts additionally carry 'c' and 'q_bar' keys for the
    # centered basis: hat q(x) = c + sum_m w_m (q_m(x) - q_bar_m). Their
    # presence routes evaluation to the centered-basis path that bypasses
    # the sum-to-one normalization.
    is_centered_basis = False
    is_pesb_smoothstep = False
    centered_w = None
    centered_c = 0.0
    centered_q_bar = None
    pesb_w_by_power = None
    pesb_n_values = None
    if isinstance(weights, dict):
        if "sign_w" in weights and "log_abs_w" in weights:
            sign_w = jnp.asarray(weights["sign_w"])
            log_abs_w = jnp.asarray(weights["log_abs_w"])
        else:
            w_raw = jnp.asarray(weights["w"])
            sign_w, log_abs_w = to_log_abs_sign(w_raw)
        if "w_by_power" in weights and "n_values" in weights:
            is_pesb_smoothstep = True
            pesb_w_by_power = jnp.asarray(weights["w_by_power"])
            pesb_n_values = jnp.asarray(weights["n_values"])
            centered_c = float(weights.get("c", 0.0))
        elif "c" in weights or ("q_bar" in weights and weights["q_bar"] is not None):
            is_centered_basis = True
            centered_w = jnp.asarray(weights["w"])
            centered_c = float(weights.get("c", 0.0))
            qb = weights.get("q_bar", None)
            centered_q_bar = jnp.asarray(qb) if qb is not None else jnp.zeros_like(centered_w)
    else:
        w_raw = jnp.asarray(weights)
        sign_w, log_abs_w = to_log_abs_sign(w_raw)

    directions = result.directions
    s_coords = result.slice_coords

    q_1d_raw = result.committors_1d

    valid_mask = result.valid_mask.astype(sign_w.dtype)
    q_1d = jnp.where(jnp.isnan(q_1d_raw), 0.0, q_1d_raw)
    original_shape = points.shape[:-1]
    points_flat = points.reshape(-1, points.shape[-1])

    n_directions = directions.shape[0]

    # Auto batch size
    if batch_size is None:
        n_pts = points_flat.shape[0]
        estimated_mem = n_directions * n_pts * 4
        if estimated_mem > 1e9:
            batch_size = max(256, int(1e9 / (n_pts * 4)))

    def _qall():
        if use_stored_projections and result.projected_samples is not None:
            return _qall_stored(s_coords, q_1d, result.projected_samples)
        return _qall_onthefly(directions, s_coords, q_1d, points_flat)

    if is_pesb_smoothstep:
        # PESB-EBMC path: powered slice basis (Ψ_n family) with EBMC bias.
        q_all = _qall()
        q_result = _evaluate_powered_smoothstep_from_qall(
            q_all,
            pesb_w_by_power,
            pesb_n_values,
            centered_c,
            result.valid_mask,
            original_shape,
        )
        _legacy_dispatch_done = True
    elif is_centered_basis:
        # Affine-ansatz path: weights are in absolute units (not convex). Skip
        # the sum-to-one log-space normalization and the legacy batched
        # path; those mix poorly with the additive intercept.
        q_all = _qall()
        q_result = _evaluate_centered_from_qall(
            q_all,
            centered_w,
            centered_c,
            centered_q_bar,
            result.valid_mask,
            original_shape,
        )
        # Skip to post-processing (clip + BC clamp).
        # Use a sentinel to bypass the legacy if/elif chain below.
        _legacy_dispatch_done = True
    else:
        _legacy_dispatch_done = False

    if _legacy_dispatch_done:
        pass  # q_result already populated by the centered-basis or PESB path.
    elif batch_size is not None and batch_size < n_directions:
        # Batched non-adaptive path: running signed-logsumexp accumulator.
        # The normalised log-space weights are consumed only here; the unbatched
        # path below re-normalises internally in _evaluate_logspace_from_qall.
        sign_w_norm, log_abs_w_norm = _normalize_log_space_weights(sign_w, log_abs_w, valid_mask)
        n_pts = points_flat.shape[0]
        sgn_q = jnp.zeros(n_pts)
        log_q = jnp.full(n_pts, -jnp.inf)

        for start in range(0, n_directions, batch_size):
            end = min(start + batch_size, n_directions)

            def _single(theta, s_grid, q_grid):
                s = points_flat @ theta
                return _interp_1d_at_samples(s_grid, q_grid, s)

            q_batch = vmap(_single, in_axes=(0, 0, 0))(
                directions[start:end],
                s_coords[start:end],
                q_1d[start:end],
            )  # (batch, n_pts)
            sgn_w_b = sign_w_norm[start:end]
            log_w_b = log_abs_w_norm[start:end]
            sgn_b = sgn_w_b[:, None] * jnp.sign(q_batch)
            log_b = log_w_b[:, None] + jnp.where(
                jnp.abs(q_batch) > 0, jnp.log(jnp.abs(q_batch)), -jnp.inf
            )
            sgn_batch, log_batch = signed_logsumexp(sgn_b, log_b, axis=0)
            sgn_q, log_q = signed_logsumexp(
                jnp.stack([sgn_q, sgn_batch]), jnp.stack([log_q, log_batch]), axis=0
            )

        q_result = (sgn_q * jnp.exp(log_q)).reshape(original_shape)
    else:
        q_all = _qall()
        q_result = _evaluate_logspace_from_qall(
            q_all,
            sign_w,
            log_abs_w,
            valid_mask,
            original_shape,
        )

    # Clip to [0, 1]: with negative weights (full Gram solver) or with an
    # affine-ansatz intercept, the weighted average can slightly exceed the
    # range of individual 1D committors.
    if clip:
        q_result = jnp.clip(q_result, 0.0, 1.0)

    if enforce_boundary_conditions and (in_A is not None or in_B is not None):
        q_result = _apply_boundary_conditions(
            q_result, in_A, in_B, original_shape, rescale_transition
        )

    return q_result


# =============================================================================
# WEIGHT COMPUTATION AND RMSE
# =============================================================================


def compute_weights_multi(
    result: SlicedCommittorResult,
    weight_fns: list[Callable],
    samples: jnp.ndarray | None = None,
) -> dict[str, jnp.ndarray]:
    """
    Compute weights using multiple weighting functions.

    Args:
        result: SlicedCommittorResult
        weight_fns: List of weighting functions, each taking WeightingContext
        samples: Optional (N, dim) for on-the-fly projection if not stored

    Returns:
        {function_name: weights_array} dictionary
    """
    ctx = make_weighting_context(result)

    # Compute projections on-the-fly if needed
    if samples is not None and ctx.projected_samples is None:
        ctx = ctx._replace(projected_samples=result.directions @ samples.T)

    return {fn.__name__: fn(ctx) for fn in weight_fns}


def compute_full_gram_weights(
    result: SlicedCommittorResult,
    samples: jnp.ndarray,
    sample_weights: jnp.ndarray | None = None,
    tikhonov="auto",
    constraint: str = "sum",
    clamp_epsilon: bool = False,
    return_overlap: bool = False,
    gram_dtype: str = "float32",
    compute_condition_number: bool = False,
    *,
    epsilon_fn=None,
) -> dict:
    """Compute weights using the full cross-Dirichlet Gram matrix solver.

    Assembles the (M, M) cross-Dirichlet Gram matrix
    ``G_jk = <D̂ ∂_j q_j, ∂_k q_k>_ρ`` and minimises the Dirichlet-energy
    error functional ``E(w) = wᵀG w − 2 bᵀ w + const`` subject to either
    the simplex constraint ``Σw = 1`` (``constraint='sum'``, default) or
    the flux-normal constraint ``bᵀw = 1`` (``constraint='flux'``, closed
    form). The unknown amplitude ``D[q]`` is fixed by a self-consistency
    quadratic on top of a single Cholesky factor of the regularised Gram.

    Prefer this over :func:`corrected_dirichlet_inv_rd` when the inter-slice
    coupling matters (typically d ≫ 2 with biased directions, or when the
    diagonal sanity check ``|G_jj − D_j| / D_j`` rises above ~5%). The
    resulting weights may carry negative entries: these are legitimate
    off-diagonal coupling corrections, not pathologies, and no positive-part
    truncation is applied.

    Args:
        result: from compute_sliced_committor
        samples: (N, dim) the same samples used in compute_sliced_committor
        sample_weights: (N,) optional MBAR reweighting; if None, uniform 1/N
            (with a warning).
        tikhonov: Gram matrix regularisation. Float or ``'auto'`` (default,
            N_eff-adaptive: η = 1/√N_eff, clamped at 1e-12).
        constraint: ``'sum'`` (Σw=1, default) or ``'flux'`` (bᵀw=1).
        clamp_epsilon: if True, clamp the boundary-error ε into ``[0, 1]``
            before forming ``b``. Default False (negative/over-unity ε are
            left as-is, matching the diagonal estimator).
        return_overlap: if True, also assemble and return the (M, M) slice
            L² overlap matrix under key ``'M'``. Default False.
        gram_dtype: dtype for the dominant (M, N)×(N, M) inner product.
            Default ``'float32'`` (matmul in single precision, solve in
            double); pass ``'float64'`` for the legacy bit-equivalent path.
        compute_condition_number: if True, compute the exact SVD-based
            condition number diagnostic (O(M³) extra work). Default False
            (returns NaN).
        epsilon_fn: optional ε estimator override. Pass
            ``sliced_committor.weights.compute_epsilon_rms`` for the RMS variant
            (gram_rms). ``None`` (default) uses cached equilibrium ε.

    Returns:
        dict with keys:
            w: (M,) constrained-optimal weights (may contain negative entries)
            G: (M, M) Gram matrix
            u, v: decomposition vectors
            P, Q, beta_G, Delta_G, Dq_hat, disc: self-consistency scalars
            w_unconstrained: normalized-unconstrained weights
            condition_number (exact SVD when opted in, else NaN),
            off_diagonal_magnitude, R_M_ratio: diagnostics
            diagonal_sanity: (M,) |G_jj - D_j / Z_rel,j| / (D_j / Z_rel,j)
            constraint, eta_used, N_eff: provenance.
    """
    from .weights import full_gram_weights

    ctx = make_weighting_context(result)
    return full_gram_weights(
        ctx,
        samples,
        sample_weights,
        tikhonov,
        constraint=constraint,
        clamp_epsilon=clamp_epsilon,
        return_overlap=return_overlap,
        gram_dtype=gram_dtype,
        compute_condition_number=compute_condition_number,
        epsilon_fn=epsilon_fn,
    )


def compute_basin_moment_weights(
    result: SlicedCommittorResult,
    samples: jnp.ndarray,
    sample_weights: jnp.ndarray | None = None,
    tikhonov: float | str = "auto",
    raise_on_degenerate: bool = True,
    cond_basin_threshold: float = 1e-6,
    gram_dtype: str = "float64",
) -> dict:
    """Basin-Moment-Constrained variational weight solver.

    Sister of :func:`compute_full_gram_weights`. Replaces the simplex
    constraint ``Σw = 1`` with the two basin-conditional moment constraints

        μ_A[q̂] = 0,   μ_B[q̂] = 1

    which encode the committor's boundary values directly into the
    optimisation rather than relying on a post-hoc correction.
    Constants are infeasible (rank-1 constraint matrix); near-constant
    slices receive algebraic zero weight. The optimum is a closed-form 2×2
    KKT solve on top of a single Cholesky factor of the regularised Gram;
    no self-consistency loop and no amplitude estimate are required.

    Prefer over :func:`compute_full_gram_weights` when the basin labels
    are reliable and you want hyperparameter-free calibration baked into
    the weight solve. Output ``sum_w`` is reported but not constrained
    (a non-zero deviation from 1 is normal for BMC).

    **Requires float64.** Call ``jax.config.update('jax_enable_x64', True)``
    before constructing ``result``; otherwise this function raises
    ``RuntimeError``. The KKT solve is rank-2 and ill-conditions in
    single precision.

    Args:
        result: from compute_sliced_committor (must carry
            ``projected_samples``, i.e. compute
            with ``store_projected_samples=True``).
        samples: (N, dim) equilibrium samples used in compute_sliced_committor.
            Required only if ``result.projected_samples`` is None.
        sample_weights: (N,) optional MBAR weights; falls back to
            ``result.sample_weights``, then to uniform 1/N (with a warning).
        tikhonov: Gram matrix regularisation. Float or ``'auto'`` (default,
            N_eff-adaptive: η = 1/√N_eff, clamped at 1e-12).
        raise_on_degenerate: If True (default), raise
            :class:`bmc.BMCRepresentationError` when ``cond_basin`` falls
            below ``cond_basin_threshold`` (the projection basis cannot
            distinguish basins A and B).
        cond_basin_threshold: Threshold below which BMC is considered
            ill-posed. Default 1e-6.
        gram_dtype: dtype for the dominant ``(M, N)×(N, M)`` inner product.
            Default ``'float64'``.

    Returns:
        dict with keys:
            w: (M,) BMC-optimal weights (may contain negative entries,
               legitimate off-diagonal coupling corrections).
            a, b: (M,) basin-conditional moments.
            G: (M, M) cross-Dirichlet Gram matrix.
            A, B, C, det, lambda_a, lambda_b, cond_basin: KKT scalars.
            condition_number: Cholesky-based estimate of cond(G_reg).
            constraint_residual_A, constraint_residual_B: |aᵀw|,
                |bᵀw − 1| (must be ~machine epsilon).
            sum_w: Σw, reported but NOT constrained.
            n_negative_weights, negative_weight_mass,
            off_diagonal_magnitude, diagonal_sanity: shared Gram
                diagnostics.
            eta_used, N_eff: provenance.
    """
    from ._bmc import basin_moment_weights

    ctx = make_weighting_context(result)
    return basin_moment_weights(
        ctx,
        samples,
        sample_weights,
        tikhonov,
        raise_on_degenerate=raise_on_degenerate,
        cond_basin_threshold=cond_basin_threshold,
        gram_dtype=gram_dtype,
    )


def compute_enriched_basin_moment_weights(
    result: SlicedCommittorResult,
    samples: jnp.ndarray,
    sample_weights: jnp.ndarray | None = None,
    tikhonov: float | str = "auto",
    raise_on_degenerate: bool = True,
    cond_enriched_threshold: float = 1e-6,
    gram_dtype: str = "float64",
) -> dict:
    """Enriched Basin-Moment-Constrained (EBMC) variational weight solver.

    Strict extension of :func:`compute_basin_moment_weights`: adds a free
    global bias ``c`` to the ansatz, which collapses the two basin-moment
    constraints ``aᵀw = 0`` and ``bᵀw = 1`` into the single constraint
    ``(b − a)ᵀw = 1``. The closed-form solution is strictly lower in
    Dirichlet energy than vanilla BMC, shift-invariant in the per-slice
    1D committor calibration, and algebraically zero on flat slices
    (where ``a_j ≈ b_j``).

    Returned ansatz: ``q̄(x) = c + Σ_j w_j q_{θ_j}(θ_j·x)``. The result dict
    carries ``'c'`` and ``'q_bar'`` (zeros), so passing it to
    :func:`evaluate_committor` activates the centered-basis path
    automatically::

        ebmc = compute_enriched_basin_moment_weights(result, samples)
        q = evaluate_committor(result, points, ebmc)

    This is the library's recommended default solver: it strictly improves
    on vanilla BMC, has a closed form, and routes through
    :func:`evaluate_committor` with no extra plumbing.

    **Requires float64.** Call ``jax.config.update('jax_enable_x64', True)``
    before constructing ``result``.

    Args:
        result: from :func:`compute_sliced_committor` (must carry
            ``in_A``, ``in_B`` and ``projected_samples``; use
            ``store_projected_samples=True``).
        samples: ``(N, dim)`` equilibrium samples. Required only if
            ``result.projected_samples`` is None.
        sample_weights: ``(N,)`` optional MBAR weights; falls back to
            ``result.sample_weights``, then to uniform ``1/N``.
        tikhonov: Gram regularisation. Float or ``'auto'`` (default,
            N_eff-adaptive).
        raise_on_degenerate: if True (default), raise
            :class:`sliced_committor.EnrichedBMCRepresentationError` when
            ``cond_enriched`` falls below ``cond_enriched_threshold`` (the
            moment gap ``b − a`` is collapsing).
        cond_enriched_threshold: threshold on ``M_gap / max(A, B)``;
            default 1e-6.
        gram_dtype: dtype for the dominant ``(M, N)×(N, M)`` inner product.
            Default ``'float64'``.

    Returns:
        dict with ``w`` (M,), ``c`` (float), ``q_bar`` (M, zeros), basin
        moments ``a``, ``b``, the Gram matrix ``G``, the moment-gap
        scalars (``M_gap``, ``A``, ``B``, ``C``, ``Delta``,
        ``improvement_over_bmc``, ``cond_enriched``,
        ``optimal_dirichlet_energy``), and the shared Gram diagnostics
        (negative-weight stats, off-diagonal magnitude, diagonal sanity).
    """
    from ._bmc_enriched import enriched_basin_moment_weights

    ctx = make_weighting_context(result)
    return enriched_basin_moment_weights(
        ctx,
        samples,
        sample_weights,
        tikhonov,
        raise_on_degenerate=raise_on_degenerate,
        cond_enriched_threshold=cond_enriched_threshold,
        gram_dtype=gram_dtype,
    )


def compute_enriched_basin_moment_weights_power(
    result: SlicedCommittorResult,
    samples: jnp.ndarray,
    sample_weights: jnp.ndarray | None = None,
    P: int = 2,
    n_values: jnp.ndarray | None = None,
    tikhonov: float | str = "auto",
    raise_on_degenerate: bool = True,
    cond_enriched_threshold: float = 1e-6,
    gram_dtype: str = "float64",
    direction_batch_size: int = 512,
) -> dict:
    """Power-Enriched Slice Basis EBMC (PESB-EBMC) with the smoothstep basis.

    Enriches each per-ridge basis from ``q_{θ_j}`` to the smoothstep family

        Ψ_n(v) = v^n / (v^n + (1 − v)^n)    for n in ``n_values``

    while keeping the EBMC bias and basin-moment constraint structure. The
    ansatz becomes

        q̂(x) = c + Σ_{j=1..M} Σ_{k=1..P} w_{j,k} · Ψ_{n_k}(q_{θ_j}(θ_j·x)).

    Galerkin: the PESB trial space strictly contains the EBMC one (set
    ``w_{j,k} = 0`` for ``k ≥ 2`` to recover EBMC), so at the population
    level ``M_gap_PESB ≥ M_gap_EBMC``. The returned
    ``improvement_over_ebmc`` quantifies this ratio at finite sampling.

    At ``P=1`` (or ``n_values=[1.0]``) this reproduces
    :func:`compute_enriched_basin_moment_weights` on shared keys.

    The returned dict carries ``'c'``, ``'q_bar'``, ``'w_by_power'``, and
    ``'n_values'``, so passing it to :func:`evaluate_committor` activates
    the smoothstep-aggregator dispatch automatically.

    Only the smoothstep basis is shipped; the legacy monomial basis (``v^p``)
    is intentionally not exposed: it is asymmetric around v=0.5 and shows
    Hilbert-matrix-like within-block conditioning at large P.

    **Requires float64.**

    Args:
        result, samples, sample_weights, tikhonov, raise_on_degenerate,
        cond_enriched_threshold, gram_dtype: see
            :func:`compute_enriched_basin_moment_weights`.
        P: enrichment dimension per direction. Used when ``n_values`` is
            None (then ``n_values = linspace(1.0, P, P) = [1.0, 2.0]``
            for the default P=2). Practical ceiling P=3.
        n_values: explicit smoothstep exponents ``(P,)``, each ``≥ 1``.
            Overrides ``P`` when given.
        direction_batch_size: directions per JIT chunk during field and
            moment assembly; controls peak memory (~``batch · P · N``).

    Returns:
        dict with all EBMC keys plus ``P``, ``n_values``,
        ``w_by_power`` ``(M, P)``, ``power_weight_mass`` ``(P,)``,
        ``within_block_cond`` ``(M,)``, ``improvement_over_ebmc``.
        ``w``, ``a``, ``b``, ``q_bar`` have shape ``(M*P,)``
        (flattened row-major: index ``j*P + (k-1)``).
    """
    from ._bmc_enriched import enriched_basin_moment_weights_power

    ctx = make_weighting_context(result)
    return enriched_basin_moment_weights_power(
        ctx,
        samples,
        sample_weights,
        P=P,
        n_values=n_values,
        tikhonov=tikhonov,
        raise_on_degenerate=raise_on_degenerate,
        cond_enriched_threshold=cond_enriched_threshold,
        gram_dtype=gram_dtype,
        direction_batch_size=direction_batch_size,
    )
