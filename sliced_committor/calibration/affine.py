"""Affine label-mean calibration for the sliced committor.

The sliced aggregator suffers basin-mean shrinkage in high dimensions:
the projected basin means drift inward from {0, 1}, and the slope of
q_hat vs. truth drops. Class labels at A and B samples carry the exact
boundary information q(A) = 0, q(B) = 1. This module turns those labels
into a unique, hyperparameter-free affine map that pins
E_A[q_hat] = 0 and E_B[q_hat] = 1 exactly (pre-clip).

Two modes:

- ``global``:    one scalar pair (q_bar_A, q_bar_B). Maps
                 q_hat_cal(x) = (q_hat_base(x) - q_bar_A) / (q_bar_B - q_bar_A).
- ``per_slice``: rescales each slice F_j so its A-mean is 0 and B-mean
                 is 1, then aggregates. Tighter on heterogeneous slice
                 spans; falls back to "drop slice" on degenerate slices.

Both modes algebraically reduce to a centered-basis weight dict
``{w, c, q_bar=0}`` that the existing ``evaluate_committor``
centered-basis path consumes directly. No changes to the evaluator are
required, and clip / boundary enforcement / cross-correlated correction
all compose automatically.

Caller idiom::

    from sliced_committor.calibration.affine import calibrate_weights_affine
    w_cal = calibrate_weights_affine(result, raw_w, mode="global")
    q = evaluate_committor(result, points, w_cal)
"""

from typing import Any, Dict, Optional, Union

import jax.numpy as jnp
import numpy as np
from jax import vmap

from .._internal import resolve_basin_labels
from ..solver import (
    SlicedCommittorResult,
    _interp_1d_at_samples,
)


def _basin_means(
    result: SlicedCommittorResult,
    samples: jnp.ndarray | None,
    in_A: jnp.ndarray | None,
    in_B: jnp.ndarray | None,
):
    """Per-slice basin means F_A[j], F_B[j] of the slice committor.

    Returns (F_A, F_B, valid_mask) as JAX arrays of shape (M,).
    """
    if result.projected_samples is not None:
        projected = jnp.asarray(result.projected_samples)  # (M, N)
    elif samples is not None:
        projected = (jnp.asarray(samples) @ result.directions.T).T  # (M, N)
    else:
        raise ValueError(
            "Affine calibration needs per-slice projected samples. Re-run "
            "compute_sliced_committor with store_projected_samples=True, "
            "or pass `samples=...` explicitly to calibrate_weights_affine."
        )

    in_A, in_B = resolve_basin_labels(result, in_A, in_B, "Affine calibration")

    in_A_b = jnp.asarray(in_A).astype(bool)
    in_B_b = jnp.asarray(in_B).astype(bool)
    n_A = jnp.maximum(jnp.sum(in_A_b.astype(jnp.float64)), 1.0)
    n_B = jnp.maximum(jnp.sum(in_B_b.astype(jnp.float64)), 1.0)

    q_1d = jnp.where(jnp.isnan(result.committors_1d), 0.0, result.committors_1d)
    s_coords = result.slice_coords  # (M, n_bins)

    def per_slice(s_grid, q_grid, s_proj):
        F = _interp_1d_at_samples(s_grid, q_grid, s_proj)
        F = jnp.clip(F, 0.0, 1.0)  # match the evaluator's per-slice clip
        F_A = jnp.sum(jnp.where(in_A_b, F, 0.0)) / n_A
        F_B = jnp.sum(jnp.where(in_B_b, F, 0.0)) / n_B
        return F_A, F_B

    F_A_arr, F_B_arr = vmap(per_slice, in_axes=(0, 0, 0))(s_coords, q_1d, projected)
    valid = jnp.asarray(result.valid_mask).astype(bool)
    return F_A_arr, F_B_arr, valid


def _coerce_weights(weights) -> jnp.ndarray:
    """Extract a raw (M,) weight array from any of the dict shapes
    accepted by ``evaluate_committor``."""
    if isinstance(weights, dict):
        if "w" in weights:
            return jnp.asarray(weights["w"])
        if "sign_w" in weights and "log_abs_w" in weights:
            return jnp.asarray(weights["sign_w"]) * jnp.exp(jnp.asarray(weights["log_abs_w"]))
        raise ValueError("weights dict must carry either 'w' or 'sign_w'+'log_abs_w'.")
    return jnp.asarray(weights)


def calibrate_weights_affine(
    result: SlicedCommittorResult,
    weights: jnp.ndarray | dict[str, Any],
    mode: str = "global",
    samples: jnp.ndarray | None = None,
    *,
    in_A: jnp.ndarray | None = None,
    in_B: jnp.ndarray | None = None,
    guard_rel: float = 0.01,
    guard_abs: float = 1e-6,
    return_diag: bool = False,
) -> dict[str, Any]:
    """Compute affine label-mean calibration as a centered-basis weight dict.

    Returns a dict ``{'w': w_cal, 'c': c_cal, 'q_bar': zeros(M)}`` consumable
    directly by ``evaluate_committor``. The resulting estimator satisfies
    E_A[q_hat] = 0 and E_B[q_hat] = 1 on the training samples to machine
    precision (pre-clip).

    Args:
        result: ``SlicedCommittorResult`` from ``compute_sliced_committor``.
            Must carry either ``projected_samples`` (set by
            ``store_projected_samples=True``) or have ``samples`` passed
            explicitly here.
        weights: ``(M,)`` weight array OR a dict from a weight solver
            (``{'w'}``, ``{'sign_w','log_abs_w'}``). Only the raw weight
            magnitudes are used; any existing ``'c'``/``'q_bar'`` are
            discarded; the calibration produces its own affine intercept.
        mode: ``'global'`` (default) or ``'per_slice'``. The two forms are
            equivalent when per-slice spans are homogeneous and differ at
            second order otherwise. ``'global'`` is unconditionally stable;
            ``'per_slice'`` is tighter but requires a stability guard on
            degenerate slices.
        samples: ``(N, dim)`` original training samples. Only needed when
            ``result.projected_samples`` is ``None``.
        in_A, in_B: ``(N,)`` boolean overrides for basin labels. Default
            is to read from ``the result``.
        guard_rel: per-slice guard threshold relative to the median of
            ``|Delta_j|``. A slice is dropped if
            ``|Delta_j| <= max(guard_rel * median, guard_abs)``. Used only
            in ``per_slice`` mode.
        guard_abs: absolute floor on ``|span|`` (global) and ``|Delta_j|``
            (per_slice). Pure numerical safety.
        return_diag: if True, the returned dict carries an additional
            ``'diag'`` key with basin means, span, ``n_good``, etc. The
            diagnostics are passive; they don't affect evaluation.

    Returns:
        dict with keys ``'w'`` (M,), ``'c'`` (float), ``'q_bar'`` (M,) all
        zeros, and optionally ``'diag'``. Pass directly to
        ``evaluate_committor(result, points, returned_dict)``.

    Notes:
        - The calibration corrects first-order shrinkage exactly: the
          slope alpha of ``q_hat_cal`` vs. truth is rescaled to 1 in
          expectation on the basin set. It does NOT invert non-linear
          monotone distortions; compose with
          ``recalibrate.evaluate_committor_with_recal`` for that, passing
          the calibrated dict in place of the raw weights.
        - Composes with ``enforce_boundary_conditions``, ``clip`` on the
          downstream ``evaluate_committor`` call with no extra plumbing.
    """
    if mode not in ("global", "per_slice"):
        raise ValueError(f"mode must be 'global' or 'per_slice', got {mode!r}")

    w_raw = _coerce_weights(weights)
    M = w_raw.shape[0]

    F_A, F_B, valid_mask = _basin_means(result, samples, in_A, in_B)
    Delta_j = F_B - F_A
    valid_f = valid_mask.astype(w_raw.dtype)
    zeros_M = jnp.zeros(M, dtype=w_raw.dtype)

    if mode == "global":
        w_eff = w_raw * valid_f
        bar_qA = jnp.sum(w_eff * F_A)
        bar_qB = jnp.sum(w_eff * F_B)
        span = bar_qB - bar_qA
        # Floor |span| at guard_abs without flipping sign; if span == 0,
        # default to +guard_abs (calibration is ill-defined anyway).
        span_sign = jnp.where(span >= 0, 1.0, -1.0)
        span_safe = jnp.where(jnp.abs(span) >= guard_abs, span, span_sign * guard_abs)
        w_cal = w_raw / span_safe
        c_cal = -bar_qA / span_safe

        if return_diag:
            valid_host = np.asarray(valid_mask).astype(bool)
            dj_host = np.asarray(Delta_j)
            if valid_host.any():
                med_span = float(np.median(np.abs(dj_host[valid_host])))
                slice_span_min = float(np.min(dj_host[valid_host]))
                slice_span_max = float(np.max(dj_host[valid_host]))
            else:
                med_span = 0.0
                slice_span_min = 0.0
                slice_span_max = 0.0
            diag = {
                "mode": "global",
                "bar_q_A": float(bar_qA),
                "bar_q_B": float(bar_qB),
                "span": float(span),
                "F_A": F_A,
                "F_B": F_B,
                "Delta_j": Delta_j,
                "median_span": med_span,
                "slice_span_min": slice_span_min,
                "slice_span_max": slice_span_max,
                "n_good": int(np.sum(valid_host)),
                "n_total": int(M),
            }
        else:
            diag = None

    else:  # per_slice
        # Host-side median over the valid slices (small array; called once).
        valid_host = np.asarray(valid_mask).astype(bool)
        dj_host = np.asarray(Delta_j)
        abs_dj_host = np.abs(dj_host)
        if valid_host.any():
            med = float(np.median(abs_dj_host[valid_host]))
        else:
            med = 0.0
        guard = float(max(guard_rel * med, guard_abs))

        is_good = jnp.asarray((abs_dj_host > guard) & valid_host)
        # Renormalize input weights over the surviving set so
        # E_B[q_hat] = sum_{good} w_j = 1 exactly when sum_{valid} w_j = 1.
        sum_w_good = jnp.sum(jnp.where(is_good, w_raw, 0.0))
        sum_w_good_safe = jnp.where(jnp.abs(sum_w_good) > 1e-30, sum_w_good, 1.0)
        w_renorm = jnp.where(is_good, w_raw / sum_w_good_safe, 0.0)

        Delta_safe = jnp.where(is_good, Delta_j, 1.0)
        w_cal = w_renorm / Delta_safe
        c_cal = -jnp.sum(w_cal * F_A)

        if return_diag:
            bar_qA_g = jnp.sum(w_renorm * F_A)
            bar_qB_g = jnp.sum(w_renorm * F_B)
            slice_span_min = float(np.min(dj_host[valid_host])) if valid_host.any() else 0.0
            slice_span_max = float(np.max(dj_host[valid_host])) if valid_host.any() else 0.0
            diag = {
                "mode": "per_slice",
                "bar_q_A": float(bar_qA_g),
                "bar_q_B": float(bar_qB_g),
                "span": float(bar_qB_g - bar_qA_g),
                "F_A": F_A,
                "F_B": F_B,
                "Delta_j": Delta_j,
                "median_span": med,
                "guard": guard,
                "slice_span_min": slice_span_min,
                "slice_span_max": slice_span_max,
                "n_good": int(np.sum(np.asarray(is_good))),
                "n_total": int(M),
            }
        else:
            diag = None

    out = {
        "w": w_cal,
        "c": float(c_cal),
        "q_bar": zeros_M,
    }
    if return_diag:
        out["diag"] = diag
    return out
