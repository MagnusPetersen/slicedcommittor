"""
Basin-Moment-Constrained (BMC) variational solver for the sliced committor.

Replaces the simplex constraint  1ᵀw = 1  used by ``full_gram_weights`` with
the two basin-conditional moment constraints

        μ_A[q̂] = 0,    μ_B[q̂] = 1,

where μ_X[q̂] = (1/|X|) Σ_{x ∈ X} q̂(x). The true committor satisfies these
exactly; constants are infeasible (the constraint matrix becomes rank-1);
slices that cannot distinguish basins (a_j ≈ b_j) receive algebraic zero
weight rather than soft suppression.

Closed-form solution
--------------------
Lagrangian: ℒ(w; λ_a, λ_b) = wᵀG w − 2 λ_a aᵀw − 2 λ_b (bᵀw − 1).
Stationarity gives w = G⁻¹(λ_a a + λ_b b). Define

    A = aᵀG⁻¹a,  B = bᵀG⁻¹b,  C = aᵀG⁻¹b,  Δ = A B − C².

Applying both constraints,

    λ_a = −C / Δ,   λ_b = A / Δ,
    w_BMC = (1/Δ) G⁻¹ (A b − C a).

Numerical requirements
----------------------
* float64 is required.  At d ≥ 40 the basin moments a_j, b_j on well-aligned
  directions are both close to 0 or 1; the discriminating signal b_j − a_j is
  small relative to those magnitudes and float32 corrupts ``cond_basin``.
  ``jax.config.update("jax_enable_x64", True)`` is already global in this
  project (see conftest.py); do not change it.

* No positive-part truncation. BMC weights may be negative; these are
  legitimate off-diagonal coupling corrections (same justification as the
  full-Gram simplex solver).

* Σ w_j is not constrained. Reporting only.

See :func:`sliced_committor.weights.full_gram_weights` for the sister
simplex solver.
"""

import logging

import jax
import jax.numpy as jnp

from .gram import (
    _assemble_gram_matrix,
    _compute_derivative_matrix,
    compute_shared_gram_diagnostics,
    resolve_cos_matrix,
    resolve_metric_diagonal,
)
from .weights import _interpolate_q_at_samples_masked, _mask_and_regularize_gram, _resolve_eta

logger = logging.getLogger(__name__)


class BMCRepresentationError(RuntimeError):
    """Raised when the basin-moment constraint matrix is rank-deficient.

    Signals that the projection basis ``{f_j}`` cannot independently match
    μ_A[q̂] = 0 and μ_B[q̂] = 1; typically because every slice yields
    a_j ≈ b_j (cannot distinguish A from B). The remedy is more or
    differently-aligned directions, not a smaller regulariser.
    """


# ---------------------------------------------------------------------------
# Basin moments
# ---------------------------------------------------------------------------


def compute_basin_moments(ctx, batch_size: int = 512) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Compute the basin-conditional moments (a, b).

    a_j = μ_A[ q_j(θ_j · x) ] = (1/|A|) Σ_{x ∈ A} q_j(θ_j · x),
    b_j = μ_B[ q_j(θ_j · x) ] = (1/|B|) Σ_{x ∈ B} q_j(θ_j · x).

    Args:
        ctx: WeightingContext with ``in_A``, ``in_B``, and
            ``projected_samples`` populated.
        batch_size: Number of directions per chunk. The interior interpolation
            materializes a (batch, N) committor buffer; with N ~ 2×10⁵ and
            M ~ 10⁴ a single-shot (M, N) buffer is tens of GB. Batching keeps
            peak memory at (batch, N) × dtype, independent of M.

    Returns:
        a: (M,) basin-A moments.
        b: (M,) basin-B moments.
    """
    if ctx.in_A is None or ctx.in_B is None:
        raise ValueError(
            "BMC: ctx.in_A / ctx.in_B are missing; basin moments require "
            "the basin labels of the input samples. Pass in_A / in_B to "
            "compute_sliced_committor so they propagate into the result."
        )
    if ctx.projected_samples is None:
        raise ValueError(
            "BMC: ctx.projected_samples is None; basin moments require "
            "stored projections. Use store_projected_samples=True in "
            "compute_sliced_committor."
        )

    in_A = ctx.in_A
    in_B = ctx.in_B

    n_A_int = int(jnp.sum(in_A))
    n_B_int = int(jnp.sum(in_B))
    if n_A_int < 50 or n_B_int < 50:
        logger.warning(
            f"BMC: basin sample count low (n_A={n_A_int}, n_B={n_B_int}); "
            f"basin moments will be noisy and cond_basin may be unreliable."
        )

    n_A = jnp.maximum(jnp.sum(in_A), 1.0)
    n_B = jnp.maximum(jnp.sum(in_B), 1.0)

    # Chunk over M to bound peak memory at (batch_size, N) instead of (M, N).
    # _interpolate_q_at_samples_masked is JIT'd; passing different leading
    # dims would trigger recompilation, so we use jax.lax.dynamic_slice and a
    # Python loop. Equivalent algorithm, just slabbed.
    M = ctx.projected_samples.shape[0]
    a_chunks: list = []
    b_chunks: list = []
    for start in range(0, M, batch_size):
        end = min(start + batch_size, M)
        s_grid = ctx.slice_coords[start:end]
        q_grid = ctx.committors_1d[start:end]
        ps = ctx.projected_samples[start:end]
        q_A_chunk = _interpolate_q_at_samples_masked(s_grid, q_grid, ps, in_A)
        q_B_chunk = _interpolate_q_at_samples_masked(s_grid, q_grid, ps, in_B)
        a_chunks.append(jnp.sum(q_A_chunk, axis=1) / n_A)
        b_chunks.append(jnp.sum(q_B_chunk, axis=1) / n_B)
        # Encourage XLA to free the chunk buffers before the next iteration.
        del q_A_chunk, q_B_chunk

    a = jnp.concatenate(a_chunks, axis=0)
    b = jnp.concatenate(b_chunks, axis=0)
    return a, b


# ---------------------------------------------------------------------------
# Closed-form KKT solve
# ---------------------------------------------------------------------------


@jax.jit
def _solve_basin_moment_kkt(G, a, b, valid_mask, eta_val):
    """JIT body for the 2×2 KKT closed-form solve.

    Returns a dict of JAX arrays. The Python wrapper handles eta='auto' and
    the BMCRepresentationError gate.
    """
    valid = valid_mask.astype(G.dtype)
    G_reg, _med_diag = _mask_and_regularize_gram(G, valid_mask, eta_val)

    a_m = a * valid
    b_m = b * valid

    # Single Cholesky factor, two RHS
    rhs = jnp.stack([a_m, b_m], axis=-1)  # (M, 2)
    U, lower = jax.scipy.linalg.cho_factor(G_reg)
    sol = jax.scipy.linalg.cho_solve((U, lower), rhs)
    Ginv_a = sol[:, 0]
    Ginv_b = sol[:, 1]

    A_scalar = jnp.dot(a_m, Ginv_a)
    B_scalar = jnp.dot(b_m, Ginv_b)
    C_scalar = jnp.dot(a_m, Ginv_b)

    det = A_scalar * B_scalar - C_scalar * C_scalar
    # Cauchy–Schwarz guarantees det >= 0; floor for the division.
    det_safe = jnp.where(det > 1e-14, det, 1e-14)

    lambda_a = -C_scalar / det_safe
    lambda_b = A_scalar / det_safe

    w = (lambda_a * Ginv_a + lambda_b * Ginv_b) * valid

    # Conditioning indicators
    cond_basin = det / jnp.maximum(jnp.sqrt(jnp.maximum(A_scalar * B_scalar, 1e-30)), 1e-30)
    U_diag = jnp.diag(U)
    U_abs = jnp.abs(U_diag)
    cond_chol = (jnp.max(U_abs) / jnp.maximum(jnp.min(U_abs), 1e-30)) ** 2

    return {
        "w": w,
        "A_scalar": A_scalar,
        "B_scalar": B_scalar,
        "C_scalar": C_scalar,
        "det": det,
        "lambda_a": lambda_a,
        "lambda_b": lambda_b,
        "cond_basin": cond_basin,
        "condition_number": cond_chol,
        "G_reg": G_reg,
        "Ginv_a": Ginv_a,
        "Ginv_b": Ginv_b,
    }


def solve_basin_moment(
    G: jnp.ndarray,
    a: jnp.ndarray,
    b: jnp.ndarray,
    valid_mask: jnp.ndarray,
    eta: float | str = "auto",
    sample_weights: jnp.ndarray | None = None,
    N: int | None = None,
    raise_on_degenerate: bool = True,
    cond_basin_threshold: float = 1e-6,
) -> dict:
    """Closed-form BMC solve with optional rank-deficiency gate.

    Args:
        G: (M, M) cross-Dirichlet Gram matrix.
        a, b: (M,) basin moment vectors.
        valid_mask: (M,) boolean mask for valid directions.
        eta: Tikhonov regularisation (relative to median diagonal). Float or
            ``'auto'`` (N_eff-adaptive: η = 1/√N_eff, clamped at 1e-12).
        sample_weights, N: used only when eta='auto' to derive N_eff (shared
            convention with full_gram_weights).
        raise_on_degenerate: If True (default), raise BMCRepresentationError
            when cond_basin falls below ``cond_basin_threshold``.
        cond_basin_threshold: Threshold below which the basin moments are
            considered too collinear to optimise. Default 1e-6.

    Returns dict with keys (all scalars converted to Python floats):
        w:          (M,) optimal weights (may contain negative entries).
        a, b:       echo of inputs.
        A, B, C:    Gram scalars aᵀG⁻¹a, bᵀG⁻¹b, aᵀG⁻¹b.
        det:        AB − C² (Cauchy–Schwarz discriminant, ≥ 0).
        lambda_a, lambda_b:  KKT multipliers.
        cond_basin: det / √(A·B); ≪ 1 indicates ill-posed basin separation.
        condition_number: Cholesky-diagonal estimate of cond(G_reg).
        G_reg:      Tikhonov-regularised Gram (for downstream diagnostics).
        eta_used, N_eff: provenance.
    """
    M = G.shape[0]
    eta_val, N_eff_val = _resolve_eta(eta, M, valid_mask, sample_weights, N, G=G)

    out = _solve_basin_moment_kkt(G, a, b, valid_mask, eta_val)

    cond_basin_f = float(out["cond_basin"])
    if raise_on_degenerate and cond_basin_f < cond_basin_threshold:
        raise BMCRepresentationError(
            f"cond_basin={cond_basin_f:.3g} < {cond_basin_threshold:.3g}: "
            "basin moments are too collinear; the projection basis cannot "
            "distinguish A from B. Add more directions or use basin-aligned "
            "sampling. Set raise_on_degenerate=False to override."
        )

    return {
        "w": out["w"],
        "a": a,
        "b": b,
        "A": float(out["A_scalar"]),
        "B": float(out["B_scalar"]),
        "C": float(out["C_scalar"]),
        "det": float(out["det"]),
        "lambda_a": float(out["lambda_a"]),
        "lambda_b": float(out["lambda_b"]),
        "cond_basin": cond_basin_f,
        "condition_number": float(out["condition_number"]),
        "G_reg": out["G_reg"],
        "eta_used": float(eta_val),
        "N_eff": float(N_eff_val) if N_eff_val is not None else None,
    }


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------


def _add_bmc_diagnostics(result, G, a, b, valid_mask, ctx=None):
    """Append BMC-specific and shared Gram diagnostics to ``result``.

    Delegates the shared block (negative-weight stats, off-diagonal
    magnitude, derivative-matched diagonal_sanity) to
    :func:`sliced_committor.gram.compute_shared_gram_diagnostics`, then
    appends BMC-specific scalars: the constraint residuals
    ``|aᵀw|`` and ``|bᵀw − 1|`` (must be machine epsilon for a well-posed
    KKT solve) and ``sum_w`` (reported but not constrained by BMC).
    """
    compute_shared_gram_diagnostics(
        result, G, valid_mask, ctx=ctx, metric_diag=resolve_metric_diagonal(ctx)
    )
    w = result["w"]
    result["constraint_residual_A"] = float(jnp.abs(jnp.dot(a, w)))
    result["constraint_residual_B"] = float(jnp.abs(jnp.dot(b, w) - 1.0))
    result["sum_w"] = float(jnp.sum(w))


# ---------------------------------------------------------------------------
# Top-level orchestrator
# ---------------------------------------------------------------------------


def basin_moment_weights(
    ctx,
    samples: jnp.ndarray | None = None,
    sample_weights: jnp.ndarray | None = None,
    tikhonov: float | str = "auto",
    raise_on_degenerate: bool = True,
    cond_basin_threshold: float = 1e-6,
    gram_dtype: str = "float64",
) -> dict:
    """Top-level BMC weight solver.

    Assembles the cross-Dirichlet Gram matrix G from a WeightingContext,
    computes the basin moments (a, b), and returns the closed-form BMC
    weights plus diagnostics.

    Args:
        ctx: WeightingContext (must carry ``in_A``, ``in_B``, and
            ``projected_samples``).
        samples: (N, dim) equilibrium samples. Ignored when
            ctx.projected_samples is available.
        sample_weights: (N,) optional MBAR weights; falls back to
            ctx.sample_weights, then to uniform 1/N (with a warning).
        tikhonov: Tikhonov factor passed through to ``solve_basin_moment``.
            Default 'auto' (N_eff-adaptive, matches full_gram_weights).
        raise_on_degenerate, cond_basin_threshold: see ``solve_basin_moment``.
        gram_dtype: dtype for the dominant (M, N)×(N, M) inner product.
            Default ``'float64'`` (legacy path). Pass ``'float32'`` to halve
            the (M, N) derivative-matrix buffer and run the matmul in single
            precision; the (M, M) result is upcast to ``cos_matrix.dtype``
            before the cosine factor and the KKT solve. Matches the
            full-Gram simplex solver's ``gram_dtype`` knob.

    Returns:
        dict: output of ``solve_basin_moment`` augmented with the assembled
        ``G``, negative-weight stats, off-diagonal magnitude,
        derivative-matched ``diagonal_sanity``, and BMC constraint residuals.
    """
    if not jax.config.read("jax_enable_x64"):
        raise ValueError(
            "BMC requires jax_enable_x64=True. "
            "Add `jax.config.update('jax_enable_x64', True)` before computing the result."
        )
    if sample_weights is None:
        sample_weights = ctx.sample_weights
    if sample_weights is None:
        logger.warning(
            "basin_moment_weights: sample_weights=None; assuming uniform 1/N. "
            "Pass MBAR weights explicitly if samples are non-equilibrium."
        )

    # Resolve projected samples
    if ctx.projected_samples is not None:
        projected_samples = ctx.projected_samples
        moments_ctx = ctx
    elif samples is not None:
        projected_samples = ctx.directions @ samples.T
        moments_ctx = ctx._replace(projected_samples=projected_samples)
    else:
        raise ValueError(
            "basin_moment_weights: pass samples explicitly or use "
            "store_projected_samples=True in compute_sliced_committor()."
        )

    N = projected_samples.shape[1]
    W = sample_weights if sample_weights is not None else jnp.ones(N) / N

    # Gram matrix. The (M, N) derivative matrix dominates memory at large M;
    # casting to float32 (gram_dtype='float32') halves the buffer and the
    # matmul cost, and the (M, M) result is upcast to cos_matrix.dtype inside
    # _assemble_gram_matrix before the cosine factor + KKT solve.
    F = _compute_derivative_matrix(ctx, projected_samples)
    cos_matrix = resolve_cos_matrix(ctx)
    matmul_dtype = jnp.dtype(gram_dtype)
    F_lo = F.astype(matmul_dtype)
    W_lo = W.astype(matmul_dtype)
    G = _assemble_gram_matrix(F_lo, W_lo, cos_matrix)
    # Free the (M, N) derivative buffers before basin-moment interpolation,
    # which itself allocates two more (M, N) float64 q_A / q_B buffers in
    # compute_basin_moments. Without this drop, peak memory at large M is
    # roughly 5 × (M, N) on top of projected_samples (~13 GB × 5 for villin
    # at M=8192) and OOMs even at gram_dtype='float32'.
    del F, F_lo, W_lo

    # Basin moments
    a, b = compute_basin_moments(moments_ctx)

    # Closed-form solve
    result = solve_basin_moment(
        G,
        a,
        b,
        ctx.valid_mask,
        eta=tikhonov,
        sample_weights=sample_weights,
        N=N,
        raise_on_degenerate=raise_on_degenerate,
        cond_basin_threshold=cond_basin_threshold,
    )
    result["G"] = G

    _add_bmc_diagnostics(result, G, a, b, ctx.valid_mask, ctx=ctx)
    return result
