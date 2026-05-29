"""
1D RD recalibration along the sliced committor coordinate.

Treats the sliced aggregate q̄(x) as a 1D reaction coordinate and solves
the 1D RD committor along it; replaces q̄(x) by f(q̄(x)) where f is that
solution. In the noise-free limit this inverts any monotone deformation
of the true committor, including the affine shrinkage residual seen on
WQ at d ≥ 40 (α̂ < 1). For α̂ > 1 (over-extension, e.g. curved channel)
it is approximately a no-op.

Status
------
This module is off-by-default. There is no automatic α-gate yet; the
caller decides whether to apply recalibration. Spec §7.4 describes a
future gate using a θ²-sepdist α estimator; that is deferred.

API
---
``compute_recalibration_curve(q_bar_samples, in_A, in_B, beta, ...)``
    Build the 1D RD curve along q̄.
``apply_recalibration(q_bar_new, s_centers, q_recal)``
    Linearly interpolate the curve at new q̄ values.
``evaluate_committor_with_recal(...)``
    Two-pass: build the curve at samples, apply at points.
"""

import logging
from typing import Optional, Tuple

import jax.numpy as jnp

from .._internal import resolve_basin_labels
from ..solver import (
    SlicedCommittorResult,
    _interp_1d_at_samples,
    compute_1d_free_energy,
    compute_1d_rd_committor,
    evaluate_committor,
)

logger = logging.getLogger(__name__)


def _weighted_free_energy_1d(
    s: jnp.ndarray,
    weights: jnp.ndarray,
    beta: float,
    n_bins: int,
    density_floor: float = 1e-6,
    n_min: int = 1,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Weighted equal-width 1D free energy.

    Mirrors ``compute_1d_free_energy(... binning_method='equal_width')`` but
    histograms ``s`` with per-sample ``weights`` (e.g. MBAR weights).
    """
    s_min = jnp.min(s)
    s_max = jnp.max(s)
    span = jnp.maximum(s_max - s_min, 1e-8)
    bin_edges = jnp.linspace(s_min, s_max, n_bins + 1)
    s_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    idx = jnp.floor((s - s_min) / span * n_bins).astype(jnp.int32)
    idx = jnp.clip(idx, 0, n_bins - 1)

    w_norm = weights / jnp.maximum(jnp.sum(weights), 1e-30)
    counts = jnp.zeros(n_bins).at[idx].add(w_norm)
    bin_width = span / n_bins
    adaptive_floor = n_min / jnp.maximum(weights.shape[0] * bin_width, 1e-30)
    effective_floor = jnp.maximum(adaptive_floor, density_floor)
    density = jnp.maximum(counts / bin_width, effective_floor)
    F_values = -(1.0 / beta) * jnp.log(density)
    F_values = F_values - jnp.min(F_values)
    return s_centers, F_values


def compute_recalibration_curve(
    q_bar_samples: jnp.ndarray,
    in_A: jnp.ndarray,
    in_B: jnp.ndarray,
    n_bins: int = 200,
    rd_kappa: float = 1e12,
    clip_to_unit: bool = True,
    binning_method: str = "equal_width",
    sample_weights: jnp.ndarray | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Fit a 1D RD committor along the sliced aggregate q̄.

    Args:
        q_bar_samples: (N,) sliced committor evaluated at samples
            (typically ``evaluate_committor(result, samples, weights)``).
        in_A, in_B: (N,) boolean basin membership masks for the same samples.
        n_bins: number of bins along q̄ for the 1D free energy.
        rd_kappa: RD absorption strength (strong by default; matches the
            project's standard for "hard" absorption).
        clip_to_unit: if True (default), clamp q_bar_samples to [0, 1]
            before fitting.
        binning_method: 'equal_width' (default) or 'quantile'. Equal-width
            is more robust when q̄ piles up near 0 or 1 (the usual case).
            Ignored when ``sample_weights`` is not None (always equal-width).
        sample_weights: (N,) optional non-uniform weights (e.g. MBAR). When
            provided, the 1D free energy along q̄ is weighted accordingly,
            and the histogram in ``compute_1d_rd_committor`` for ρ_A / ρ_B
            is built from MBAR-weighted basin samples by replication-style
            normalisation. None ⇒ uniform 1/N.

    Returns:
        s_centers: (n_bins,) bin centres along q̄.
        q_recal:   (n_bins,) recalibrated committor along q̄, monotone-ish.
    """
    if clip_to_unit:
        s = jnp.clip(q_bar_samples, 0.0, 1.0)
    else:
        s = q_bar_samples

    if sample_weights is not None:
        s_centers, F_values = _weighted_free_energy_1d(
            s,
            jnp.asarray(sample_weights),
            beta=1.0,
            n_bins=n_bins,
            density_floor=1e-6,
            n_min=1,
        )
    else:
        s_centers, F_values = compute_1d_free_energy(
            s,
            1.0,
            n_bins=n_bins,
            density_floor=1e-6,
            binning_method=binning_method,
            n_min=1,
        )

    # Inversion check: if q̄ is *larger* on average in A than in B, the
    # 1D RD solve along q̄ should treat the higher-q̄ end as state A.
    n_A_int = int(jnp.sum(in_A))
    n_B_int = int(jnp.sum(in_B))
    if n_A_int == 0 or n_B_int == 0:
        raise ValueError(
            f"compute_recalibration_curve: empty basin (n_A={n_A_int}, "
            f"n_B={n_B_int}); cannot fit recalibration curve."
        )
    A_idx = jnp.nonzero(in_A, size=n_A_int)[0]
    B_idx = jnp.nonzero(in_B, size=n_B_int)[0]
    s_A_proj = s[A_idx]
    s_B_proj = s[B_idx]
    needs_inversion = jnp.mean(s_A_proj) > jnp.mean(s_B_proj)

    q_recal = compute_1d_rd_committor(
        s_centers,
        F_values,
        1.0,
        s_projected=s,
        in_A=in_A,
        in_B=in_B,
        is_valid=jnp.array(True),
        needs_inversion=needs_inversion,
        rd_kappa=rd_kappa,
        s_A_proj=s_A_proj,
        s_B_proj=s_B_proj,
    )
    return s_centers, q_recal


def apply_recalibration(
    q_bar_new: jnp.ndarray,
    s_centers: jnp.ndarray,
    q_recal: jnp.ndarray,
    clip_to_unit: bool = True,
) -> jnp.ndarray:
    """Apply a fitted recalibration curve to new q̄ values.

    Args:
        q_bar_new: arbitrary-shaped q̄ values to recalibrate.
        s_centers: (n_bins,) curve coordinates from compute_recalibration_curve.
        q_recal:   (n_bins,) curve values.
        clip_to_unit: if True, clamp inputs to [0, 1] before interpolation.

    Returns:
        Same shape as ``q_bar_new``.
    """
    s = jnp.clip(q_bar_new, 0.0, 1.0) if clip_to_unit else q_bar_new
    flat = s.reshape(-1)
    out = _interp_1d_at_samples(s_centers, q_recal, flat)
    return out.reshape(s.shape)


def evaluate_committor_with_recal(
    result: SlicedCommittorResult,
    points: jnp.ndarray,
    weights,
    samples: jnp.ndarray,
    *,
    in_A_samples: jnp.ndarray | None = None,
    in_B_samples: jnp.ndarray | None = None,
    in_A_points: jnp.ndarray | None = None,
    in_B_points: jnp.ndarray | None = None,
    recal_n_bins: int = 200,
    recal_kappa: float = 1e12,
    enforce_boundary_conditions: bool = True,
    return_curve: bool = False,
    sample_weights: jnp.ndarray | None = None,
    **eval_kwargs,
):
    """Two-pass evaluation with 1D-RD recalibration along q̄.

    Pass 1: evaluate q̄ at ``samples``; fit the recalibration curve.
    Pass 2: evaluate q̄ at ``points``; apply the curve.
    Re-enforce basin BCs on the output if ``enforce_boundary_conditions``
    is True AND basin labels for ``points`` were supplied via
    ``in_A_points`` / ``in_B_points``.

    Args:
        result: from compute_sliced_committor.
        points: (..., dim) evaluation locations.
        weights: weight array or dict (anything ``evaluate_committor`` accepts).
        samples: (N, dim) same samples used in compute_sliced_committor;
            used to fit the recalibration curve.
        in_A_samples, in_B_samples: (N,) bool labels for the *samples*. If
            None, falls back to ``result.in_A`` / ``result.in_B``.
        in_A_points, in_B_points: bool labels for the *evaluation points*,
            used only when ``enforce_boundary_conditions=True``. If None,
            no boundary enforcement is applied at the output (the
            recalibration curve already pulls q̄ toward 0/1 near the
            basin samples it was fit on).
        recal_n_bins, recal_kappa: 1D RD solver knobs along q̄.
        enforce_boundary_conditions: if True (default), enforce q̂ = 0 / 1
            on output points labelled by ``in_A_points`` / ``in_B_points``.
        return_curve: if True, also return (s_centers, q_recal) for plotting.
        **eval_kwargs: forwarded to ``evaluate_committor``.

    Returns:
        q_out (and optionally the (s_centers, q_recal) tuple).
    """
    # eval_kwargs may include enforce_boundary_conditions=False to keep the
    # raw q̄ on samples; we never want to clobber that pre-recal pass with
    # BC clamping. Build a separate kwargs dict.
    sample_eval_kwargs = dict(eval_kwargs)
    sample_eval_kwargs.setdefault("enforce_boundary_conditions", True)
    if in_A_samples is not None:
        sample_eval_kwargs.setdefault("in_A", in_A_samples)
    if in_B_samples is not None:
        sample_eval_kwargs.setdefault("in_B", in_B_samples)

    q_bar_samples = evaluate_committor(result, samples, weights, **sample_eval_kwargs)

    in_A_samples, in_B_samples = resolve_basin_labels(
        result, in_A_samples, in_B_samples, "evaluate_committor_with_recal"
    )

    s_centers, q_recal = compute_recalibration_curve(
        q_bar_samples,
        in_A_samples,
        in_B_samples,
        n_bins=recal_n_bins,
        rd_kappa=recal_kappa,
        sample_weights=sample_weights,
    )

    point_eval_kwargs = dict(eval_kwargs)
    # We re-enforce BCs after the curve is applied, so suppress BC clamping
    # on the raw q̄ at points to avoid clamping below the recalibration
    # curve's mapping.
    point_eval_kwargs["enforce_boundary_conditions"] = False
    q_bar_points = evaluate_committor(result, points, weights, **point_eval_kwargs)

    q_out = apply_recalibration(q_bar_points, s_centers, q_recal)

    if enforce_boundary_conditions and (in_A_points is not None or in_B_points is not None):
        if in_A_points is not None:
            q_out = jnp.where(jnp.asarray(in_A_points).reshape(q_out.shape), 0.0, q_out)
        if in_B_points is not None:
            q_out = jnp.where(jnp.asarray(in_B_points).reshape(q_out.shape), 1.0, q_out)

    if return_curve:
        return q_out, (s_centers, q_recal)
    return q_out
