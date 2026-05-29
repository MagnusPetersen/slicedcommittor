"""ABC_v2: affine basin-condition correction for the sliced committor.

The legacy ``adaptive_boundary_correction`` (ABC v1) applies a per-query
Hájek ratio ``(q_raw - f_B(x)) / (1 - f_A(x) - f_B(x))``. It works for
isotropic direction sampling with simple positive weights, and fails under
structured sampling (TICA-EMA, biased) combined with non-trivial weights
(``full_gram`` with negatives, weight concentration on basin-shadow
slices). The cross-correlated GLS extension (CCC) was tried and removed.

ABC_v2 is a fresh attempt. Derivation: the committor satisfies
``E_A[q] = 0`` and ``E_B[q] = 1``. Take the smallest two-parameter family
that can meet those two constraints, a global affine ``q̂ = α·q_raw + β``,
and solve for ``α, β`` from the sample-mean equations. Unique solution:

    α = 1 / (μ_B - μ_A),   β = -μ_A / (μ_B - μ_A)

where ``μ_S = mean(q_raw(x) for x ∈ S)``. Hyperparameter-free, stable
under negative weights, identity when the raw estimator is already
calibrated, and composes cleanly with ABC v1 (``abc=True`` calibrates on
the ABC output, applying the affine to the spec's ``affine ∘ ABC`` chain).

Caller idiom::

    from sliced_committor.calibration.abc import compute_affine_calibration
    cal = compute_affine_calibration(result, weights)
    q = evaluate_committor(
        result, points, weights,
        affine_correction=True, affine_calibration=cal,
    )

Or, with auto-calibration from ``the result``::

    q = evaluate_committor(
        result, points, weights, affine_correction=True,
    )

Or, with ABC v1 composition (the spec's ``affine ∘ ABC``)::

    q = evaluate_committor(
        result, points, weights,
        adaptive_boundary_correction=True, affine_correction=True,
    )

NOTE on the relationship to ``sliced_committor.calibration.affine``: the
``calibrate_weights_affine`` function in that module implements the same
``abc=False`` math via a centered-basis weight dict ``{w, c, q_bar}`` that
the existing evaluator natively consumes. ABC_v2 ships as a separate
module specifically to support the ``abc=True`` composition path, which
the centered-basis trick cannot express (ABC v1's per-query Hájek
denominator is non-linear in the slice committors, so it cannot be
rebaked into the weight vector). The two modules coexist.
"""

import warnings
from typing import Any, Dict, NamedTuple, Optional, Union

import jax
import jax.numpy as jnp
import numpy as np

from .._internal import resolve_basin_labels
from ..solver import (
    SlicedCommittorResult,
    evaluate_committor,
)
from .affine import _coerce_weights


class AffineCalibration(NamedTuple):
    """Parameters of the ABC_v2 affine correction ``q̂ = α·q_base + β``.

    ``q_base`` is the raw estimator (``abc=False``, default) or the ABC v1
    output (``abc=True``). ``μ_A, μ_B`` are sample means of ``q_base`` on
    the calibration subsample. ``is_degenerate=True`` means the calibration
    fell back to identity (α=1, β=0) due to one of the §4.3 edge cases.
    """

    alpha: float
    beta: float
    mu_A: float
    mu_B: float
    n_A: int
    n_B: int
    is_degenerate: bool


def _identity() -> "AffineCalibration":
    return AffineCalibration(
        alpha=1.0,
        beta=0.0,
        mu_A=float("nan"),
        mu_B=float("nan"),
        n_A=0,
        n_B=0,
        is_degenerate=True,
    )


def compute_affine_calibration(
    result: SlicedCommittorResult,
    weights: jnp.ndarray | dict[str, Any],
    samples: jnp.ndarray | None = None,
    in_A: jnp.ndarray | None = None,
    in_B: jnp.ndarray | None = None,
    *,
    n_calib: int = 4096,
    seed: int = 0,
    abc: bool = False,
    min_transition_fraction: float = 0.05,
    tol_degenerate: float = 1e-6,
) -> AffineCalibration:
    """Fit ``(α, β)`` so that ``E_A[α·q_base + β] = 0`` and ``E_B[…] = 1``.

    Args:
        result: ``SlicedCommittorResult`` from ``compute_sliced_committor``.
        weights: ``(M,)`` weight array OR a solver dict carrying ``{'w'}``
            or ``{'sign_w','log_abs_w'}``. Centered-basis dicts
            (``{'w','c','q_bar'}``) are rejected; use
            ``sliced_committor.calibration.affine.calibrate_weights_affine`` for
            those (it implements the same math via the centered basis).
        samples, in_A, in_B: optional overrides. Default to
            ``the result.samples / in_A / in_B``.
        n_calib: calibration subsample size (default 4096). When the
            ``A ∪ B`` set has more than this many points, a uniform random
            subset is drawn for the ``evaluate_committor`` call that
            computes ``μ_A, μ_B``. Smaller calibration sets buy
            diminishing precision; 4096 puts the standard error on each
            mean well below the differences in α that matter.
        seed: RNG seed for the subsample selection.
        abc: if True, calibrate on the ABC v1 output (``affine ∘ ABC``);
            otherwise calibrate on the raw estimator. The internal
            ``DeprecationWarning`` from ABC v1 is suppressed since ABC_v2
            uses it as an implementation detail, not a user choice.
        min_transition_fraction: passed through to ABC v1 when ``abc=True``.
        tol_degenerate: absolute floor on ``|μ_B - μ_A|`` below which the
            calibration falls back to identity. Default 1e-6.

    Returns:
        ``AffineCalibration`` named tuple. ``is_degenerate=True`` flags
        any of the three §4.3 edge cases (too few basin samples,
        degenerate span, sign-inverted span); in all degenerate cases,
        ``α=1, β=0`` so ``apply_affine`` becomes a no-op.
    """
    # Reject centered-basis dicts; they need calibrate_weights_affine.
    if isinstance(weights, dict) and ("c" in weights or "q_bar" in weights):
        raise ValueError(
            "compute_affine_calibration expects raw weight arrays or "
            "{'w'}/{'sign_w','log_abs_w'} dicts, not centered-basis "
            "{'w','c','q_bar'} dicts. For centered-basis affine "
            "calibration use "
            "sliced_committor.calibration.calibrate_weights_affine "
            "instead."
        )

    if samples is None:
        raise ValueError(
            "compute_affine_calibration needs `samples` (the (N, dim) array "
            "passed to compute_sliced_committor)."
        )
    in_A, in_B = resolve_basin_labels(result, in_A, in_B, "compute_affine_calibration")

    samples = jnp.asarray(samples)
    in_A_b = np.asarray(in_A).astype(bool)
    in_B_b = np.asarray(in_B).astype(bool)
    n_A_total = int(in_A_b.sum())
    n_B_total = int(in_B_b.sum())

    # §4.3 edge: too few basin samples → identity + warning.
    if n_A_total < 10 or n_B_total < 10:
        warnings.warn(
            f"ABC_v2: too few basin samples (|A|={n_A_total}, "
            f"|B|={n_B_total}); returning identity calibration.",
            stacklevel=2,
        )
        return _identity()._replace(
            n_A=n_A_total,
            n_B=n_B_total,
        )

    # Calibration subsample over A ∪ B.
    basin_idx = np.flatnonzero(in_A_b | in_B_b)
    if basin_idx.size > n_calib:
        rng = np.random.default_rng(seed)
        sel = rng.choice(basin_idx, size=n_calib, replace=False)
    else:
        sel = basin_idx
    samp_cal = samples[jnp.asarray(sel)]
    in_A_cal = in_A_b[sel]
    in_B_cal = in_B_b[sel]

    # Coerce weights to a raw (M,) array so evaluate_committor's
    # non-centered path is exercised (matters when the caller passes a
    # {'sign_w','log_abs_w'} dict but we want the simplest path here).
    weights_arr = _coerce_weights(weights)

    # Evaluate the base estimator at the calibration samples. We need the
    # un-clamped output (no enforce_boundary_conditions) so the basin
    # means are non-trivial; clip=True matches the default eval path so
    # the affine applied at eval time sees the same pre-clip values.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        q_base = evaluate_committor(
            result,
            samp_cal,
            weights_arr,
            enforce_boundary_conditions=False,
            adaptive_boundary_correction=abc,
            min_transition_fraction=min_transition_fraction,
            clip=True,
        )
    q_base_np = np.asarray(q_base)

    n_A_cal = int(in_A_cal.sum())
    n_B_cal = int(in_B_cal.sum())
    mu_A = float(q_base_np[in_A_cal].mean()) if n_A_cal > 0 else float("nan")
    mu_B = float(q_base_np[in_B_cal].mean()) if n_B_cal > 0 else float("nan")

    denom = mu_B - mu_A

    # §4.3 edge: degenerate span → identity + warning.
    if not np.isfinite(denom) or abs(denom) < tol_degenerate:
        warnings.warn(
            f"ABC_v2: degenerate span μ_B - μ_A = {denom:.3e} "
            f"(μ_A={mu_A:.4f}, μ_B={mu_B:.4f}); raw estimator fails to "
            "distinguish basins. Returning identity calibration; check "
            "weights / direction sampling upstream.",
            stacklevel=2,
        )
        return AffineCalibration(
            alpha=1.0,
            beta=0.0,
            mu_A=mu_A,
            mu_B=mu_B,
            n_A=n_A_cal,
            n_B=n_B_cal,
            is_degenerate=True,
        )

    # §4.3 edge: sign-inverted span → identity + warning (label-swap bug).
    if denom < 0:
        warnings.warn(
            f"ABC_v2: μ_B ({mu_B:.4f}) < μ_A ({mu_A:.4f}). Raw estimator "
            "assigns higher values to A than B on average; typically a "
            "label-swap bug, not a calibration failure. Returning identity.",
            stacklevel=2,
        )
        return AffineCalibration(
            alpha=1.0,
            beta=0.0,
            mu_A=mu_A,
            mu_B=mu_B,
            n_A=n_A_cal,
            n_B=n_B_cal,
            is_degenerate=True,
        )

    alpha = 1.0 / denom
    beta = -mu_A / denom
    return AffineCalibration(
        alpha=alpha,
        beta=beta,
        mu_A=mu_A,
        mu_B=mu_B,
        n_A=n_A_cal,
        n_B=n_B_cal,
        is_degenerate=False,
    )


@jax.jit
def _apply_affine_clip(q_values: jnp.ndarray, alpha: float, beta: float) -> jnp.ndarray:
    return jnp.clip(alpha * q_values + beta, 0.0, 1.0)


@jax.jit
def _apply_affine_noclip(q_values: jnp.ndarray, alpha: float, beta: float) -> jnp.ndarray:
    return alpha * q_values + beta


def apply_affine(
    q_values: jnp.ndarray, alpha: float, beta: float, clip: bool = True
) -> jnp.ndarray:
    """Apply the affine map ``q̂ = α·q + β`` to a committor field.

    Args:
        q_values: arbitrary-shaped committor values to transform.
        alpha, beta: scalars from :class:`AffineCalibration`. The identity
            calibration (``α=1, β=0``) leaves ``q_values`` unchanged
            (modulo the optional ``[0, 1]`` clip).
        clip: if True (default), clamp the output to ``[0, 1]``. Pass
            ``False`` when you want to inspect the raw affine output
            (e.g., to flag points where the calibration pushes outside
            the unit interval).

    Returns:
        Same shape as ``q_values``.
    """
    if clip:
        return _apply_affine_clip(q_values, alpha, beta)
    return _apply_affine_noclip(q_values, alpha, beta)


def evaluate_committor_calibrated(
    result: SlicedCommittorResult,
    points: jnp.ndarray,
    weights: jnp.ndarray | dict[str, Any],
    samples: jnp.ndarray,
    *,
    in_A: jnp.ndarray | None = None,
    in_B: jnp.ndarray | None = None,
    abc: bool = False,
    **kwargs,
):
    """One-shot ABC_v2 calibration + evaluation.

    Fits ``(α, β)`` from ``samples`` (via :func:`compute_affine_calibration`)
    so that ``E_A[α·q + β] = 0`` and ``E_B[…] = 1`` on the calibration
    subsample, then evaluates the corrected committor at ``points``.

    Args:
        result: from :func:`compute_sliced_committor`.
        points: ``(..., dim)`` evaluation locations.
        weights: weight array or solver dict
            (``{'w'}`` / ``{'sign_w','log_abs_w'}``). Centered-basis dicts
            are rejected by the underlying calibration; use
            :func:`calibrate_weights_affine` for that path.
        samples: ``(N, dim)`` calibration samples (typically the same
            samples passed to ``compute_sliced_committor``).
        in_A, in_B: optional ``(N,)`` bool overrides. Default to
            ``result.in_A`` / ``result.in_B``.
        abc: if True, calibrate on top of ABC v1 (``affine ∘ ABC``);
            otherwise calibrate on the raw estimator.
        **kwargs: forwarded to :func:`evaluate_committor` (e.g.
            ``in_A``, ``in_B`` at the *evaluation* points for boundary
            enforcement, ``clip``, ``batch_size``).

    Returns:
        ``(q, AffineCalibration)`` where ``q`` has the shape of
        ``points[..., 0]`` and the calibration scalars are exposed for
        downstream inspection (e.g. logging ``α``, ``μ_A``, ``μ_B``,
        ``is_degenerate``).
    """
    cal = compute_affine_calibration(
        result,
        weights,
        samples=samples,
        in_A=in_A,
        in_B=in_B,
        abc=abc,
    )
    q = evaluate_committor(
        result,
        points,
        weights,
        adaptive_boundary_correction=abc,
        affine_correction=True,
        affine_calibration=cal,
        **kwargs,
    )
    return q, cal
