"""
Weighting Functions for Sliced Committor
=========================================

Weighting schemes derived from the variational principle: minimise the
Dirichlet energy of the approximation error D[q-bar - q] under the
diagonal approximation G ~ diag(D_1, ..., D_M).

The general optimal weight has the form

    w_j  propto  (1 - eps_j) / D_j

where D_j is the 1D Dirichlet energy of the slice committor used in
reconstruction and eps_j is the flux-weighted boundary error.  This
follows from the Generalised Fundamental Identity (GFI), which holds
for *any* choice of 1D committor q_theta(s).

Self-consistency principle
--------------------------
D and eps must both come from the same 1D solver that provides q_theta
for the reconstruction q-bar = sum w_j q_j(theta_j . x).


Available weighting functions
------------------------------
  Diagonal (RD reconstruction):
    corrected_dirichlet_inv_rd:  w propto (1 - eps^eq,RD)+ / D^RD

  Full Gram matrix solver:
    full_gram_weights:  constrained w = u + D̂[q]·v with self-consistent D[q].
      Default ε estimator is the equilibrium estimator
      eps = mean(q at A) + mean(1 - q at B) for the b vector; pass
      ``epsilon_fn=compute_epsilon_rms`` for the RMS variant.

Pure JAX, JIT-compiled, log-space arithmetic.
"""

import logging
from collections.abc import Callable
from functools import partial
from typing import Dict, List, Optional

import jax
import jax.numpy as jnp
import numpy as np
from jax import jit, vmap

logger = logging.getLogger(__name__)

from ._internal import signed_logsumexp, to_log_abs_sign
from .gram import (
    _assemble_gram_matrix,
    _compute_derivative_matrix,
    compute_shared_gram_diagnostics,
)
from .solver import WeightingContext, _interp_1d_at_samples

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


@jit
def _normalize_from_log(log_w, valid):
    """Normalise weights from log-space via softmax, masking invalid."""
    log_w_masked = jnp.where(valid, log_w, -jnp.inf)
    return jax.nn.softmax(log_w_masked)


@jit
def _log_correction_from_eps(epsilon):
    """Convert boundary error eps to log((1 - eps)+)."""
    correction = jnp.maximum(1.0 - epsilon, 0.0)
    return jnp.where(correction > 0, jnp.log(correction), -jnp.inf)


def _sample_weights_match(a, b):
    """Cheap equivalence test for sample-weight arrays used by the Gram cache.

    Returns True iff a and b refer to the same effective weighting:
      - both None,
      - same Python object,
      - or arrays of equal shape and equal values (one O(N) sync).

    Identity is checked first to keep the common case free; array equality is
    only triggered when shapes match, so a wrong-N caller short-circuits on
    the shape test before paying for any device sync.
    """
    if a is None and b is None:
        return True
    if a is None or b is None:
        return False
    if a is b:
        return True
    if hasattr(a, "shape") and hasattr(b, "shape") and a.shape != b.shape:
        return False
    return bool(jnp.array_equal(a, b))


@jit
def _compute_corrected_weights(log_D, epsilon, valid_mask):
    """Fused: log((1-eps)+) - log(D) -> softmax -> normalised weights.

    Standard formula w ~ (1-eps)/D.
    """
    correction = jnp.maximum(1.0 - epsilon, 0.0)
    log_correction = jnp.where(correction > 0, jnp.log(correction), -jnp.inf)
    log_weights = log_correction - log_D
    log_weights = jnp.where(valid_mask, log_weights, -jnp.inf)
    return jax.nn.softmax(log_weights)


# ===========================================================================
# EPSILON ESTIMATORS
# ===========================================================================
#
# Per-direction boundary error ε_j ∈ [0, 2], used to form b_j = 1 − ε_j
# in the full Gram solve.
# ===========================================================================


@jit
def _interpolate_q_at_samples_masked(slice_coords, committors_1d, projected_samples, mask):
    """Interpolate 1D committors at masked sample positions.

    Args:
        slice_coords: (M, n_bins)
        committors_1d: (M, n_bins)
        projected_samples: (M, N)
        mask: (N,) boolean, which samples to include

    Returns:
        q_masked: (M, N) committor values, 0 where mask is False
    """

    def _interp_single(s_grid, q_grid, s_proj):
        q = _interp_1d_at_samples(s_grid, q_grid, s_proj)
        q = jnp.clip(q, 0.0, 1.0)
        return jnp.where(mask, q, 0.0)

    return vmap(_interp_single)(slice_coords, committors_1d, projected_samples)


def compute_epsilon_equilibrium(ctx, sample_weights=None):
    """Equilibrium-weighted boundary error: eps_A = mean(q at A), eps_B = mean(1-q at B).

    Flux model: j_A(s) ∝ ρ_A(s). Returns (M,) array.

    Args:
        ctx: WeightingContext with in_A/in_B and projected_samples.
        sample_weights: (N,) optional non-uniform weights (e.g. MBAR). When
            provided, ε is estimated under the reweighted measure so that it
            is consistent with a Gram matrix assembled with the same weights.
            If None, uniform 1/N (matches the cached RD estimator).
    """
    if ctx.in_A is None or ctx.in_B is None:
        raise ValueError("ctx.in_A / ctx.in_B required for epsilon computation")
    if ctx.projected_samples is None:
        raise ValueError("projected_samples required for epsilon computation")
    in_A, in_B = ctx.in_A, ctx.in_B
    N = in_A.shape[0]

    if sample_weights is None:
        W = jnp.ones(N, dtype=jnp.float64) / N
    else:
        W = sample_weights

    w_A = W * in_A.astype(W.dtype)
    w_B = W * in_B.astype(W.dtype)
    Z_A = jnp.maximum(jnp.sum(w_A), 1e-30)
    Z_B = jnp.maximum(jnp.sum(w_B), 1e-30)

    q_A = _interpolate_q_at_samples_masked(
        ctx.slice_coords, ctx.committors_1d, ctx.projected_samples, in_A
    )
    q_B = _interpolate_q_at_samples_masked(
        ctx.slice_coords, ctx.committors_1d, ctx.projected_samples, in_B
    )

    eps_A = jnp.sum(q_A * w_A[None, :], axis=1) / Z_A
    eps_B = 1.0 - jnp.sum(q_B * w_B[None, :], axis=1) / Z_B

    return eps_A + eps_B


def compute_epsilon_rms(ctx, sample_weights=None):
    """RMS boundary error: weighted L² norm on A/B samples.

    ε_A = sqrt( Σ_A w·q² / Σ_A w )
    ε_B = sqrt( Σ_B w·(1 − q)² / Σ_B w )

    By Cauchy–Schwarz, ε_rms ≥ ε_equilibrium pointwise; useful when the
    equilibrium average underweights tail samples (e.g. an extended state
    with long projection tails). JIT-friendly weighted second moment, so
    respects ``sample_weights``.
    """
    if ctx.in_A is None or ctx.in_B is None:
        raise ValueError("ctx.in_A / ctx.in_B required for epsilon computation")
    if ctx.projected_samples is None:
        raise ValueError("projected_samples required for epsilon computation")
    in_A, in_B = ctx.in_A, ctx.in_B
    N = in_A.shape[0]

    if sample_weights is None:
        W = jnp.ones(N, dtype=jnp.float64) / N
    else:
        W = sample_weights

    w_A = W * in_A.astype(W.dtype)
    w_B = W * in_B.astype(W.dtype)
    Z_A = jnp.maximum(jnp.sum(w_A), 1e-30)
    Z_B = jnp.maximum(jnp.sum(w_B), 1e-30)

    q_A = _interpolate_q_at_samples_masked(
        ctx.slice_coords, ctx.committors_1d, ctx.projected_samples, in_A
    )
    q_B = _interpolate_q_at_samples_masked(
        ctx.slice_coords, ctx.committors_1d, ctx.projected_samples, in_B
    )

    ms_A = jnp.sum(w_A[None, :] * q_A**2, axis=1) / Z_A
    ms_B = jnp.sum(w_B[None, :] * (1.0 - q_B) ** 2, axis=1) / Z_B
    eps_A = jnp.sqrt(jnp.maximum(ms_A, 0.0))
    eps_B = jnp.sqrt(jnp.maximum(ms_B, 0.0))
    return eps_A + eps_B


@jit
def _interpolate_qprime_at_samples_masked(slice_coords, committors_1d, projected_samples, mask):
    """Interpolate dq/ds at masked sample positions via central differences.

    Non-uniform-grid aware: uses (q[i+1]-q[i-1])/(s[i+1]-s[i-1]) on the
    interior and one-sided differences at endpoints. Matches the
    searchsorted-based linear interp in _interpolate_q_at_samples_masked,
    so quantile-binned slice_coords (the project default) are handled
    correctly.

    Args:
        slice_coords: (M, n_bins)
        committors_1d: (M, n_bins)
        projected_samples: (M, N)
        mask: (N,) boolean

    Returns:
        qprime_masked: (M, N), 0 where mask is False
    """

    def _qprime_single(s_grid, q_grid, s_proj):
        ds_interior = jnp.maximum(s_grid[2:] - s_grid[:-2], 1e-30)
        ds_left = jnp.maximum(s_grid[1] - s_grid[0], 1e-30)
        ds_right = jnp.maximum(s_grid[-1] - s_grid[-2], 1e-30)
        qp = jnp.zeros_like(q_grid)
        qp = qp.at[1:-1].set((q_grid[2:] - q_grid[:-2]) / ds_interior)
        qp = qp.at[0].set((q_grid[1] - q_grid[0]) / ds_left)
        qp = qp.at[-1].set((q_grid[-1] - q_grid[-2]) / ds_right)
        qp_i = _interp_1d_at_samples(s_grid, qp, s_proj)
        return jnp.where(mask, qp_i, 0.0)

    return vmap(_qprime_single)(slice_coords, committors_1d, projected_samples)


@jit
def _epsilon_flux1d_chunk(slice_coords_c, committors_1d_c, projected_samples_c, in_A, in_B, W):
    """Per-chunk (M_c,) reduction for compute_epsilon_flux1d.

    Lifted to module scope so JAX caches the trace across chunks of identical
    shape; the function would otherwise re-trace on every outer call.
    """
    q_A = _interpolate_q_at_samples_masked(
        slice_coords_c, committors_1d_c, projected_samples_c, in_A
    )
    q_B = _interpolate_q_at_samples_masked(
        slice_coords_c, committors_1d_c, projected_samples_c, in_B
    )
    qp_A = _interpolate_qprime_at_samples_masked(
        slice_coords_c, committors_1d_c, projected_samples_c, in_A
    )
    qp_B = _interpolate_qprime_at_samples_masked(
        slice_coords_c, committors_1d_c, projected_samples_c, in_B
    )

    w_A = jnp.abs(qp_A) * W[None, :]
    w_B = jnp.abs(qp_B) * W[None, :]
    sum_w_A = jnp.sum(w_A, axis=1)
    sum_w_B = jnp.sum(w_B, axis=1)

    eps_A = jnp.where(
        sum_w_A > 1e-30,
        jnp.sum(w_A * q_A, axis=1) / jnp.maximum(sum_w_A, 1e-30),
        0.0,
    )
    eps_B = jnp.where(
        sum_w_B > 1e-30,
        jnp.sum(w_B * (1.0 - q_B), axis=1) / jnp.maximum(sum_w_B, 1e-30),
        0.0,
    )
    return eps_A + eps_B


def compute_epsilon_flux1d(ctx, sample_weights=None, chunk_size: int = 128):
    """Flux-importance-reweighted boundary error (flux1D).

    Re-weights ε under the 1D reactive-flux density
    j_j^{1D}(s) = ρ_j(s) q_j'(s) rather than the equilibrium density
    ρ_j(s). Importance reweighting equilibrium → flux on A-samples gives
    per-sample weight ∝ |q_j'(s_i)|, combined multiplicatively with any
    MBAR ``sample_weights`` W_i:

        ε_A = Σ_{i∈A} |q'(s_i)| W_i q(s_i)       / Σ_{i∈A} |q'(s_i)| W_i
        ε_B = Σ_{i∈B} |q'(s_i)| W_i (1 - q(s_i)) / Σ_{i∈B} |q'(s_i)| W_i

    On the 2D Wolfe-Quapp benchmark (β ∈ {2, 4}) this estimator has
    bias ≈ +0.014 and RMSE ≈ 0.07-0.08, overestimating ~78 % of
    directions; it beats both equilibrium (biased low) and source
    (biased high) on RMSE. Returns (M,) array.

    The estimator materialises four (chunk_size, N) interpolation
    buffers (q_A, q_B, q'_A, q'_B), so peak memory scales with
    ``chunk_size``. The default 128 keeps peak ≲ 2 × the equilibrium
    estimator at the same M; pass ``chunk_size=0`` (or any value
    ≥ M) to disable chunking.

    Args:
        ctx: WeightingContext with in_A/in_B, projected_samples,
            slice_coords, committors_1d.
        sample_weights: (N,) optional non-uniform sample weights
            (e.g. MBAR). If None, uniform 1/N.
        chunk_size: number of directions per XLA call. Set to 0 (or any
            value ≥ M) to process all directions in a single call.
    """
    if ctx.in_A is None or ctx.in_B is None:
        raise ValueError("ctx.in_A / ctx.in_B required for epsilon computation")
    if ctx.projected_samples is None:
        raise ValueError("projected_samples required for epsilon computation")
    in_A, in_B = ctx.in_A, ctx.in_B
    N = in_A.shape[0]
    M = ctx.slice_coords.shape[0]

    if sample_weights is None:
        W = jnp.ones(N, dtype=jnp.float64) / N
    else:
        W = sample_weights

    if chunk_size <= 0 or chunk_size >= M:
        return _epsilon_flux1d_chunk(
            ctx.slice_coords,
            ctx.committors_1d,
            ctx.projected_samples,
            in_A,
            in_B,
            W,
        )

    chunks = [
        _epsilon_flux1d_chunk(
            ctx.slice_coords[i : i + chunk_size],
            ctx.committors_1d[i : i + chunk_size],
            ctx.projected_samples[i : i + chunk_size],
            in_A,
            in_B,
            W,
        )
        for i in range(0, M, chunk_size)
    ]
    return jnp.concatenate(chunks, axis=0)


def _interpolate_q_at_all_samples(
    slice_coords, committors_1d, projected_samples, chunk_size: int = 128
):
    """Interpolate 1D committors at every sample (unmasked, clipped to [0,1]).

    Chunked over the M direction axis to bound peak memory. Each vmap call
    over a chunk of size K materializes ~5 × (K, N) intermediates (bin
    indices, interp weights, gathered q_low/q_high, output); at K=128,
    N=1.08M, float64 this is ≈5 GB peak vs ≈45 GB for an unchunked
    vmap over M=1024: the difference between fitting in 48 GB GPU VRAM
    and OOM.

    Note: deliberately NOT @jit'd at this level so the Python ``for`` loop
    over chunks dispatches each ``interp_vmap`` call as a separate XLA
    execution. ``interp_vmap`` is JIT-traced internally per chunk shape
    (cached across chunks of identical size), and memory from one chunk is
    released before the next is dispatched. A wrapping ``@jit`` would unroll
    the loop into a single graph and silently re-materialize the unchunked
    peak.
    """

    def _interp_single(s_grid, q_grid, s_proj):
        q = _interp_1d_at_samples(s_grid, q_grid, s_proj)
        return jnp.clip(q, 0.0, 1.0)

    interp_vmap = vmap(_interp_single)
    M = projected_samples.shape[0]
    if chunk_size <= 0 or chunk_size >= M:
        return interp_vmap(slice_coords, committors_1d, projected_samples)
    chunks = [
        interp_vmap(
            slice_coords[i : i + chunk_size],
            committors_1d[i : i + chunk_size],
            projected_samples[i : i + chunk_size],
        )
        for i in range(0, M, chunk_size)
    ]
    return jnp.concatenate(chunks, axis=0)


def _epsilon_estimators():
    """Single source of truth for the string-name -> callable mapping."""
    return {
        "equilibrium": compute_epsilon_equilibrium,
        "rms": compute_epsilon_rms,
        "flux1d": compute_epsilon_flux1d,
    }


EPSILON_ESTIMATORS = _epsilon_estimators()


def _resolve_epsilon_fn(epsilon_fn):
    """Map ``None`` / string name / callable to a concrete estimator callable.

    String checks come first so that user-supplied class objects (e.g. ``int``)
    do not bypass validation via ``callable()``; unknown strings raise
    ``ValueError`` with the list of valid choices.
    """
    if epsilon_fn is None:
        return compute_epsilon_equilibrium
    if isinstance(epsilon_fn, str):
        if epsilon_fn not in EPSILON_ESTIMATORS:
            valid = ", ".join(sorted(EPSILON_ESTIMATORS))
            raise ValueError(
                f"compute_epsilon: unknown estimator name {epsilon_fn!r}. "
                f"Valid choices: {valid}."
            )
        return EPSILON_ESTIMATORS[epsilon_fn]
    if isinstance(epsilon_fn, type) or not callable(epsilon_fn):
        raise TypeError(
            "compute_epsilon: epsilon_fn must be None, a string name "
            f"({', '.join(sorted(EPSILON_ESTIMATORS))}), or a callable; "
            f"got {type(epsilon_fn).__name__}."
        )
    return epsilon_fn


def compute_epsilon(
    ctx, sample_weights=None, *, epsilon_fn=None, clamp=False, clamp_warn_threshold=0.1
):
    """Boundary-error estimator with optional clamp.

    ``epsilon_fn`` can be:
      * ``None`` (default): uses :func:`compute_epsilon_equilibrium`.
      * A string name: ``'equilibrium'``, ``'rms'``, or ``'flux1d'``. Unknown
        names raise ``ValueError`` with the list of valid choices.
      * A callable ``ctx -> (M,)`` accepting ``sample_weights``.

    The diagonal path already applies ``(1 - ε)_+`` inside
    ``_compute_corrected_weights``, so ``clamp`` is redundant there.
    The full-Gram path does NOT truncate, so ``clamp=True`` turns
    uniformly-negative ``b = 1 - ε`` into zero, a detectable degenerate
    solve instead of a silently wrong one.

    ``clamp_warn_threshold`` (default 0.1) is the fraction-above-1 above
    which a warning fires when ``clamp=True``.
    """
    epsilon_fn = _resolve_epsilon_fn(epsilon_fn)
    eps = epsilon_fn(ctx, sample_weights=sample_weights)
    if clamp:
        frac_clamped = float(jnp.mean((eps > 1.0).astype(eps.dtype)))
        if frac_clamped > clamp_warn_threshold:
            logger.warning(
                f"compute_epsilon: {frac_clamped:.1%} of directions had "
                f"ε > 1 and were clamped. Saturation regime; check the "
                f"projection ansatz."
            )
        eps = jnp.minimum(eps, 1.0)
    return eps


# ===========================================================================
# WEIGHTING FUNCTIONS
# ===========================================================================

# ---------------------------------------------------------------------------
# RD reconstruction: corrected with equilibrium eps
# ---------------------------------------------------------------------------


def corrected_dirichlet_inv_rd(ctx):
    """w_j propto (1 - eps^eq,RD)+ / D^RD

    Variationally consistent (both eps and D from RD committor), but the
    equilibrium eps estimator has reduced discriminative power for RD:
    absorption enforces q ~ 0 in bulk of A and q ~ 1 in bulk of B,
    making eps^eq small for all directions regardless of alignment.

    The correction is approximately redundant with 1/D^RD because D^RD
    already encodes boundary violations (absorption identity).  Reduces
    to 1/D^RD for strong absorption (large kappa).
    """
    log_D = ctx.log_dirichlet
    epsilon = ctx.boundary_errors

    if log_D is None or epsilon is None:
        raise ValueError(
            "RD diagnostics (log_dirichlet, boundary_errors) "
            "not pre-computed. Use compute_sliced_committor()."
        )

    return _compute_corrected_weights(log_D, epsilon, ctx.valid_mask)


# ===========================================================================
# FULL CROSS-DIRICHLET GRAM MATRIX SOLVE
# ===========================================================================
#
# Assembles the full M×M cross-Dirichlet Gram matrix G and solves the
# constrained variational optimum, eliminating all four structural
# approximations of the diagonal scheme (spec Section 2):
#
#   (i)   Diagonal G → full G
#   (ii)  Unknown D[q] → self-consistent estimate D̂[q]
#   (iii) Normalized-unconstrained → constrained optimum (Σw_j = 1)
#   (iv)  Independent positive-part truncation → no truncation
#         (negative weights are legitimate coupling corrections)
#
# Mathematical summary:
#
#   E(w) = wᵀ G w − 2 D[q] bᵀw + D[q]
#
#   Constrained optimum:  w_con = u + D̂[q] · v
#     u = G⁻¹1 / (1ᵀG⁻¹1)           min-energy base weight (Σu = 1)
#     v = G⁻¹b − (Q/β_G) G⁻¹1       zero-sum correction (Σv = 0)
#
#   D̂[q] from self-consistency:  Δ_G D² − D + 1/β_G = 0
#     P = bᵀG⁻¹b,  Q = 1ᵀG⁻¹b,  β_G = 1ᵀG⁻¹1,  Δ_G = P − Q²/β_G
#     D̂[q] = (1 − √(1 − 4Δ_G/β_G)) / (2Δ_G)      (smaller root)
#
# In the diagonal limit G → diag(D_1,...,D_M) and Δ_G → 0, this
# recovers the existing formula w_j ∝ (1 − ε_j)/D_j.
# ===========================================================================


@jit
def _assemble_gram_and_inner(F, W, cos_matrix):
    """Assemble both the Gram matrix and its cosine-stripped inner factor.

    The dominant ``F_scaled @ F_scaled.T`` matmul is performed in F's dtype
    (caller controls via ``gram_dtype``) and the result is upcast to
    ``cos_matrix.dtype`` before mixing with the cosine matrix. This lets the
    caller run the matmul in float32 while keeping the downstream solve in
    float64.

    Returns
    -------
    G : (M, M)
        Cross-Dirichlet Gram matrix, ``G_jk = (e_j·e_k) · ⟨F_j F_k⟩_W``,
        in ``cos_matrix.dtype``.
    H : (M, M)
        Slice-derivative inner-product matrix, ``H_jk = ⟨F_j F_k⟩_W``,
        upcast to ``cos_matrix.dtype``. This is the cosine-stripped factor;
        the gradient-correlation readout ``r_jk = H_jk / √(H_jj H_kk)`` is
        basis-independent and is the right object for interaction graphs /
        community detection.
    """
    F_scaled = F * jnp.sqrt(W)[None, :]  # (M, N) in F's dtype
    H_lo = F_scaled @ F_scaled.T  # (M, M) in F's dtype
    H = H_lo.astype(cos_matrix.dtype)
    G = cos_matrix * H
    return G, H


@jit
def _assemble_overlap_matrix(Q, W):
    """Assemble the L² overlap matrix of slice committors.

    M_jk = Σ_n W_n q_j(θ_j·x_n) q_k(θ_k·x_n)

    No cos_matrix prefactor: this is a sample-weighted L² inner product, not a
    directional gradient overlap. Used by the generalized eigenproblem
    G v = λ M v in CV-discovery downstream.

    Args:
        Q: (M, N) per-direction interpolated committor at samples
           (clipped to [0, 1] by ``_interpolate_q_at_all_samples``).
        W: (N,) sample weights (normalised).

    Returns:
        M_overlap: (M, M) symmetric positive semi-definite overlap matrix.
    """
    Q_scaled = Q * jnp.sqrt(W)[None, :]  # (M, N)
    return Q_scaled @ Q_scaled.T  # (M, M)


def _resolve_eta(eta, M, valid_mask, sample_weights=None, N=None):
    """Resolve the Tikhonov parameter.

    Accepts a float (used verbatim) or the string ``'auto'`` (default).  In
    ``'auto'`` mode, η is scaled to the estimated effective sample size:

        η = max(1e-12, 1 / √N_eff)

    with ``N_eff = (Σw)² / Σw²`` for non-uniform weights and ``N_eff = N`` for
    uniform weights.  This matches the Full_Solve recommendation δ ~ 1/√N_eff
    and suppresses regularisation bias in well-conditioned, high-N regimes.
    """
    if isinstance(eta, str):
        if eta != "auto":
            raise ValueError(f"Unknown eta string: {eta!r}")
        if sample_weights is not None:
            W = jnp.asarray(sample_weights)
            Z = jnp.sum(W)
            Z2 = jnp.sum(W**2)
            N_eff = jnp.where(Z2 > 0, Z**2 / Z2, 1.0)
            N_eff_val = float(N_eff)
        elif N is not None:
            N_eff_val = float(N)
        else:
            raise ValueError("eta='auto' requires either sample_weights or N.")
        return max(1e-12, 1.0 / (N_eff_val**0.5)), N_eff_val
    return float(eta), None


@partial(jit, static_argnames=("constraint",))
def _solve_constrained_gram_jit(G, b, valid_mask, eta_val, constraint="sum"):
    """JIT-compiled numerical body of the constrained Gram solve.

    Returns a dict of JAX arrays. The Python wrapper ``_solve_constrained_gram``
    handles the ``eta='auto'`` resolution, the optional SVD condition number,
    and conversion of scalar fields to Python floats. Keeping all numerics in
    a single JIT region eliminates ~10 host syncs per call and lets XLA fuse
    the masking, Tikhonov, Cholesky, and self-consistency arithmetic.
    """
    M = G.shape[0]
    valid = valid_mask.astype(G.dtype)

    # Mask invalid directions: zero rows/cols, identity on diagonal
    mask_2d = valid[:, None] * valid[None, :]
    G_masked = G * mask_2d + jnp.diag(1.0 - valid)  # invalid → identity row

    # Adaptive Tikhonov: η × median(valid diagonal entries)
    G_diag_valid = jnp.where(valid > 0, jnp.diag(G_masked), jnp.inf)
    med_diag = jnp.median(G_diag_valid)
    # Fallback if all invalid
    med_diag = jnp.where(jnp.isfinite(med_diag) & (med_diag > 0), med_diag, 1.0)
    G_reg = G_masked + eta_val * med_diag * jnp.eye(M)

    # Prepare RHS
    b_masked = b * valid
    ones = valid  # 1 for valid directions, 0 for invalid

    # Solve G⁻¹b and G⁻¹1 simultaneously via explicit Cholesky (G_reg is SPD).
    # cho_factor defaults to lower=False, so U is upper-triangular with
    # G_reg = UᵀU.
    rhs = jnp.stack([b_masked, ones], axis=-1)  # (M, 2)
    U, lower = jax.scipy.linalg.cho_factor(G_reg)
    sol = jax.scipy.linalg.cho_solve((U, lower), rhs)  # (M, 2)
    Ginv_b = sol[:, 0]
    Ginv_1 = sol[:, 1]

    # Key scalars
    P = jnp.dot(b_masked, Ginv_b)  # bᵀ G⁻¹ b
    Q = jnp.dot(ones, Ginv_b)  # 1ᵀ G⁻¹ b
    beta_G = jnp.dot(ones, Ginv_1)  # 1ᵀ G⁻¹ 1

    # Guard: β_G > 0 under Tikhonov regularisation, but fall back defensively
    # in case the entire valid block collapses.
    beta_G_safe = jnp.where(jnp.abs(beta_G) > 1e-30, beta_G, 1e-30)

    # Decomposition vectors
    u = Ginv_1 / beta_G_safe  # min-energy base (Σu = 1)
    v = Ginv_b - (Q / beta_G_safe) * Ginv_1  # zero-sum correction (Σv = 0)

    # Self-consistent D[q] from Δ_G · D² − D + 1/β_G = 0
    # Cauchy–Schwarz: Δ_G = P − Q²/β_G ≥ 0 exactly, but Tikhonov round-off can
    # produce small-negative values. Clamp before forming disc to keep the
    # rationalised root stable and to avoid accidental division by a tiny
    # negative number in the degenerate-root fallback below (Issue 1).
    Delta_G = jnp.maximum(P - Q**2 / beta_G_safe, 0.0)
    disc = 1.0 - 4.0 * Delta_G / beta_G_safe
    disc_safe = jnp.maximum(disc, 0.0)

    # Rationalised smaller root: algebraically equivalent to (1 − √disc)/(2Δ_G)
    # but free of the 1 − √(1 − x) cancellation for small Δ_G/β_G, and
    # smooth at Δ_G = 0 (gives exactly 1/β_G when disc = 1).
    Dq_hat_main = 2.0 / (beta_G_safe * (1.0 + jnp.sqrt(disc_safe)))
    # Degenerate-root fallback (spec eq. 32): when 4Δ_G > β_G the quadratic
    # has no real root; use the repeated-root value 1/(2Δ_G). In this region
    # Δ_G > β_G/4 > 0, so the division is well-defined. Require strict
    # positivity (not |Δ_G| > tol) so small-negative floating-point noise
    # does not flip sign and blow up the weights.
    Dq_hat_fallback = 1.0 / (2.0 * jnp.where(Delta_G > 1e-30, Delta_G, 1e-30))
    Dq_hat = jnp.where(disc >= 0.0, Dq_hat_main, Dq_hat_fallback)

    # --- Constrained optimal weights ---
    # 'sum' (default): Σw = 1 via w = u + D̂[q]·v with self-consistent D̂[q].
    # 'flux' (Issue 5): bᵀw = 1 closed-form w = G⁻¹b / P, Dq_hat = 1/P.
    if constraint == "sum":
        w_con = u + Dq_hat * v
        Dq_hat_out = Dq_hat
    elif constraint == "flux":
        P_safe = jnp.where(jnp.abs(P) > 1e-30, P, 1e-30)
        w_con = Ginv_b / P_safe
        Dq_hat_out = 1.0 / P_safe  # self-consistent D̂[q] under bᵀw=1
    else:
        raise ValueError(f"Unknown constraint: {constraint!r}")
    u_out = u
    v_out = v

    # Zero invalid directions; no positive-part truncation: negative weights
    # are legitimate off-diagonal coupling corrections (spec Section 6).
    # Under 'sum', Σw = 1 by construction (Σu = 1, Σv = 0).
    w = w_con * valid

    sign_w, log_abs_w = to_log_abs_sign(w)

    # --- Silent-degradation diagnostics (Issue A) ---
    # When b is near-proportional to 1, v → 0 and w → u (inverse-Dirichlet
    # base): the Gram/GFI machinery contributes nothing. These three scalars
    # make that failure mode observable without changing the solution.
    Dq_v = Dq_hat_out * v_out
    u_norm = jnp.linalg.norm(u_out)
    gram_correction_ratio = jnp.linalg.norm(Dq_v) / jnp.maximum(u_norm, 1e-30)
    n_valid_f = jnp.maximum(jnp.sum(valid), 1.0)
    mean_b_magnitude = jnp.sum(jnp.abs(b_masked)) / n_valid_f
    fraction_b_negative = jnp.sum(((b_masked < 0) & (valid > 0)).astype(G.dtype)) / n_valid_f

    # σ_M = 1 / ((1−ε)ᵀ G⁻¹ (1−ε)) = 1/P: the GFI upper bound on D[q].
    # Under constraint='flux', σ_M coincides with Dq_hat by construction;
    # under 'sum', Dq_hat is the self-consistent estimate and σ_M is the
    # bound (always ≥ true D[q] in the noise-free limit). Use |P| because
    # b can carry sign in the multi-constraint generalisation.
    P_safe_all = jnp.where(jnp.abs(P) > 1e-30, P, 1e-30)
    sigma_M = 1.0 / P_safe_all

    return {
        "w": w,
        "sign_w": sign_w,
        "log_abs_w": log_abs_w,
        "u": u_out,
        "v": v_out,
        "P": P,
        "Q": Q,
        "beta_G": beta_G,
        "Delta_G": Delta_G,
        "Dq_hat": Dq_hat_out,
        "sigma_M": sigma_M,
        "disc": disc,
        "w_unconstrained": jnp.where(jnp.abs(Q) > 1e-30, Ginv_b / Q, u),
        "G_reg": G_reg,
        "gram_correction_ratio": gram_correction_ratio,
        "mean_b_magnitude": mean_b_magnitude,
        "fraction_b_negative": fraction_b_negative,
    }


def _solve_constrained_gram(
    G,
    b,
    valid_mask,
    eta="auto",
    constraint="sum",
    sample_weights=None,
    N=None,
    compute_condition_number=False,
):
    """Constrained Gram solve: w = u + D̂[q]·v with self-consistent D[q].

    Solves the constrained optimization of the Dirichlet-energy error
    functional, using either the u/v decomposition under Σw = 1 (default) or
    the closed-form flux-constrained solution under bᵀw = 1.  No positive-part
    truncation is applied: negative weights are legitimate off-diagonal
    coupling corrections.

    Thin Python wrapper around the JIT-compiled body
    ``_solve_constrained_gram_jit``: resolves ``eta='auto'`` against the
    sample-derived N_eff, optionally computes the SVD condition number, and
    converts scalar JAX outputs to Python floats for diagnostic consumption.

    Args:
        G: (M, M) Gram matrix.
        b: (M,) boundary correction vector = 1 − ε (arbitrary real vector).
        valid_mask: (M,) boolean mask for valid directions.
        eta: Tikhonov regularisation (relative to median diagonal). Float or
            the string ``'auto'`` (default, N_eff-adaptive: η = 1/√N_eff,
            clamped at 1e-12).
        constraint: ``'sum'`` (Σw = 1, default) or ``'flux'`` (bᵀw = 1,
            closed-form w = G⁻¹b / P; see Issue 5). Under ``'flux'`` there is
            no self-consistency quadratic and ``Dq_hat = 1 / P = sigma_M``.
        sample_weights: (N,) used only when ``eta='auto'`` to derive N_eff.
        N: fallback sample count for ``eta='auto'`` when sample_weights is None.
        compute_condition_number: if True, compute an exact SVD-based condition
            number of the valid sub-block of G_reg (O(M³) extra work). Default
            False; `condition_number` is set to NaN. Pass True only when the
            diagnostic is actually consumed; on M~1000 systems the SVD rivals
            the Cholesky in wall-clock.

    Returns:
        dict. Units note (Issue 6): the Gram matrix is a Monte Carlo estimator
        of the inner product ⟨q'_j, q'_k⟩_p where p(s) has unit integral over
        the grid, so P, Q, β_G, Δ_G are in units of the empirical measure (no
        explicit partition function).  ``Dq_hat`` therefore estimates D[q] in
        those same units; the dimensionless ratio ``Dq_hat * P`` is the
        explained-variance R_M.  The returned weights are invariant under
        rescaling the empirical measure by a constant.

        Keys: w, u, v, P, Q, beta_G, Delta_G, Dq_hat, sigma_M, disc,
        w_unconstrained, G_reg, condition_number (exact SVD-based cond of
        G_reg, or NaN when ``compute_condition_number=False``), eta_used,
        N_eff (populated when eta='auto'), constraint.

        ``sigma_M = 1 / ((1−ε)ᵀ G⁻¹ (1−ε))`` is the GFI upper bound on D[q]:
        always ≥ true D[q] in the noise-free limit. Under ``constraint='flux'``
        it equals ``Dq_hat`` by construction; under ``'sum'`` ``Dq_hat`` is a
        self-consistent estimate while ``sigma_M`` is the bound. Use
        ``sigma_M`` as the diagnostic and as the basis-incompleteness signal
        (smaller σ_M ⇒ basis explains more of D[q]).
    """
    M = G.shape[0]

    # Resolve Tikhonov (Issue 3: 'auto' mode)
    eta_val, N_eff_val = _resolve_eta(eta, M, valid_mask, sample_weights, N)

    out = _solve_constrained_gram_jit(G, b, valid_mask, eta_val, constraint=constraint)

    # --- Condition number (exact SVD on valid sub-block) ---
    # Opt-in: at M~1000 the SVD is another O(M³) factorisation, comparable to
    # the Cholesky we just did. Most callers never read this and only print
    # the value, so default to NaN. Set ``compute_condition_number=True`` when
    # the diagnostic is actually consumed.
    if compute_condition_number:
        valid = valid_mask.astype(G.dtype)
        valid_np = np.asarray(valid > 0)
        if int(valid_np.sum()) >= 2:
            valid_idx = jnp.asarray(np.flatnonzero(valid_np))
            G_sub = out["G_reg"][valid_idx][:, valid_idx]
            cond_exact = float(jnp.linalg.cond(G_sub))
        else:
            cond_exact = float("nan")
    else:
        cond_exact = float("nan")

    return {
        "w": out["w"],
        "sign_w": out["sign_w"],
        "log_abs_w": out["log_abs_w"],
        "u": out["u"],
        "v": out["v"],
        "P": float(out["P"]),
        "Q": float(out["Q"]),
        "beta_G": float(out["beta_G"]),
        "Delta_G": float(out["Delta_G"]),
        "Dq_hat": float(out["Dq_hat"]),
        "sigma_M": float(out["sigma_M"]),
        "disc": float(out["disc"]),
        "w_unconstrained": out["w_unconstrained"],
        "G_reg": out["G_reg"],
        "condition_number": cond_exact,
        "eta_used": float(eta_val),
        "N_eff": float(N_eff_val) if N_eff_val is not None else None,
        "constraint": constraint,
        "gram_correction_ratio": float(out["gram_correction_ratio"]),
        "mean_b_magnitude": float(out["mean_b_magnitude"]),
        "fraction_b_negative": float(out["fraction_b_negative"]),
    }


# ---------------------------------------------------------------------------
# Multi-constraint KKT solve  (used by thin-shell BCM, Component F)
# ---------------------------------------------------------------------------


@jit
def _solve_multi_constraint_gram_jit(G, A, c, b, valid_mask, eta_val):
    """Solve min_w wᵀ G w subject to A w = c via the KKT block-Schur form.

    The KKT system
        [G   Aᵀ] [w]   [0]
        [A    0] [λ] = [c]
    reduces to ``(A G⁻¹ Aᵀ) λ = c`` and ``w = G⁻¹ Aᵀ λ``. Both forward
    solves share a single Cholesky factor of the Tikhonov-regularised G,
    so the overhead over the single-constraint solver is a (k, M) tri-
    solve, a (k, k) inversion, and a (M,) matvec.

    ``b = 1 − ε`` is passed only so σ_M can be computed from the same
    Cholesky factor; σ_M does not enter the solution itself.

    Args:
        G: (M, M) Gram matrix.
        A: (k, M) linear-constraint matrix.
        c: (k,) RHS of the constraints.
        b: (M,) boundary-error vector (1 − ε) for σ_M diagnostic only.
        valid_mask: (M,) boolean mask of valid directions.
        eta_val: float Tikhonov coefficient (resolved by ``_resolve_eta``).

    Returns:
        dict with ``w``, ``lambda``, ``G_reg``, ``A_masked``, ``M_lambda``,
        ``sigma_M``, ``constraint_residual`` (= ‖A w − c‖∞), and the same
        silent-degradation scalars as the single-constraint solver
        (``mean_b_magnitude``, ``fraction_b_negative``) so downstream
        diagnostics can be reused.
    """
    M = G.shape[0]
    k = A.shape[0]
    valid = valid_mask.astype(G.dtype)
    mask_2d = valid[:, None] * valid[None, :]
    G_masked = G * mask_2d + jnp.diag(1.0 - valid)

    G_diag_valid = jnp.where(valid > 0, jnp.diag(G_masked), jnp.inf)
    med_diag = jnp.median(G_diag_valid)
    med_diag = jnp.where(jnp.isfinite(med_diag) & (med_diag > 0), med_diag, 1.0)
    G_reg = G_masked + eta_val * med_diag * jnp.eye(M)

    # Mask invalid columns of A (forced w_j = 0 anyway).
    A_masked = A * valid[None, :]
    b_masked = b * valid

    U, lower = jax.scipy.linalg.cho_factor(G_reg)

    # Solve G y_j = A_j for j = 1, ..., k → Y is (M, k).
    Y = jax.scipy.linalg.cho_solve((U, lower), A_masked.T)
    # M_λ = A G⁻¹ Aᵀ → (k, k) Schur complement.
    M_lambda = A_masked @ Y
    # λ = M_λ⁻¹ c  (k is small, k ≤ ~10 in practice). No Tikhonov on M_λ:
    # the constraints should be satisfied *exactly* given G_reg, which is the
    # actual operator we minimise against. Regularising M_λ would loosen the
    # constraints; diagnose ill-conditioning via constraint_residual below.
    lam = jnp.linalg.solve(M_lambda, c)
    w = (Y @ lam) * valid

    # σ_M = 1 / (bᵀ G⁻¹ b): uses the same Cholesky factor, independent of
    # the constraint structure.
    Ginv_b = jax.scipy.linalg.cho_solve((U, lower), b_masked)
    P = jnp.dot(b_masked, Ginv_b)
    P_safe = jnp.where(jnp.abs(P) > 1e-30, P, 1e-30)
    sigma_M = 1.0 / P_safe

    constraint_residual = jnp.max(jnp.abs(A_masked @ w - c))

    n_valid_f = jnp.maximum(jnp.sum(valid), 1.0)
    mean_b_magnitude = jnp.sum(jnp.abs(b_masked)) / n_valid_f
    fraction_b_negative = jnp.sum(((b_masked < 0) & (valid > 0)).astype(G.dtype)) / n_valid_f

    return {
        "w": w,
        "lambda": lam,
        "G_reg": G_reg,
        "A_masked": A_masked,
        "M_lambda": M_lambda,
        "sigma_M": sigma_M,
        "P": P,
        "constraint_residual": constraint_residual,
        "mean_b_magnitude": mean_b_magnitude,
        "fraction_b_negative": fraction_b_negative,
    }


def _solve_multi_constraint_gram(
    G, A, c, b, valid_mask, eta="auto", sample_weights=None, N=None, compute_condition_number=False
):
    """Python wrapper around :func:`_solve_multi_constraint_gram_jit`.

    Resolves ``eta='auto'`` against the sample-derived N_eff, optionally
    computes the exact condition number of the constraint Schur complement
    M_λ (small, k × k, so this is cheap), and converts scalar JAX outputs
    to Python floats.

    Args:
        G: (M, M) Gram matrix.
        A: (k, M) linear-constraint matrix.
        c: (k,) RHS of the constraints.
        b: (M,) boundary-error vector (1 − ε); used only for σ_M and the
            silent-degradation scalars, not for the solution.
        valid_mask: (M,) boolean mask of valid directions.
        eta: Tikhonov regularisation (float or ``'auto'``).
        sample_weights, N: used only when ``eta='auto'``.
        compute_condition_number: if True, also return ``cond(M_lambda)``;
            small (k × k) so cost is negligible.

    Returns:
        dict: see ``_solve_multi_constraint_gram_jit`` plus ``eta_used``,
        ``N_eff``, ``condition_number_M_lambda``, ``sign_w``, ``log_abs_w``.
    """
    M = G.shape[0]
    eta_val, N_eff_val = _resolve_eta(eta, M, valid_mask, sample_weights, N)

    out = _solve_multi_constraint_gram_jit(G, A, c, b, valid_mask, eta_val)

    if compute_condition_number:
        cond_ml = float(jnp.linalg.cond(out["M_lambda"]))
    else:
        cond_ml = float("nan")

    sign_w, log_abs_w = to_log_abs_sign(out["w"])

    return {
        "w": out["w"],
        "sign_w": sign_w,
        "log_abs_w": log_abs_w,
        "lambda": out["lambda"],
        "G_reg": out["G_reg"],
        "A_masked": out["A_masked"],
        "M_lambda": out["M_lambda"],
        "sigma_M": float(out["sigma_M"]),
        "P": float(out["P"]),
        "constraint_residual": float(out["constraint_residual"]),
        "mean_b_magnitude": float(out["mean_b_magnitude"]),
        "fraction_b_negative": float(out["fraction_b_negative"]),
        "condition_number_M_lambda": cond_ml,
        "eta_used": float(eta_val),
        "N_eff": float(N_eff_val) if N_eff_val is not None else None,
    }


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------


def precompute_gram(ctx, samples=None, sample_weights=None):
    """Compute the Gram matrix G and boundary vector b (without solving).

    Use with ``solve_gram_weights`` to sweep Tikhonov values efficiently:
    compute G once, then solve for each eta.

    Args:
        ctx: WeightingContext (standard interface).
        samples: (N, dim) equilibrium samples.  Ignored when
            ctx.projected_samples is available.
        sample_weights: (N,) optional MBAR weights; if None, defaults to
            ctx.sample_weights (which is None for uniform measure).
            A ``None`` default silently assumes uniform 1/N; this is
            inappropriate for biased sampling (umbrella, metadynamics,
            replica exchange).  A warning is logged in that case
            (Issue 8); pass explicit weights to suppress it.

    Returns:
        G: (M, M) cross-Dirichlet Gram matrix.
        b: (M,) boundary correction vector (1 − ε).
    """
    if sample_weights is None:
        sample_weights = ctx.sample_weights
    if sample_weights is None:
        logger.warning(
            "precompute_gram: sample_weights=None; assuming uniform 1/N. "
            "Pass MBAR weights explicitly if samples are non-equilibrium."
        )

    epsilon = ctx.boundary_errors
    if epsilon is None:
        raise ValueError("Boundary errors not available.")

    if ctx.projected_samples is not None:
        projected_samples = ctx.projected_samples
    elif samples is not None:
        projected_samples = ctx.directions @ samples.T  # (M, N)
    else:
        raise ValueError(
            "Either pass samples explicitly or use "
            "store_projected_samples=True in compute_sliced_committor()."
        )

    N = projected_samples.shape[1]
    W = sample_weights if sample_weights is not None else jnp.ones(N) / N

    F = _compute_derivative_matrix(ctx, projected_samples)
    cos_matrix = ctx.cos_matrix if ctx.cos_matrix is not None else ctx.directions @ ctx.directions.T
    G = _assemble_gram_matrix(F, W, cos_matrix)
    b = 1.0 - epsilon

    return G, b


def precompute_gram_and_overlap(ctx, samples=None, sample_weights=None):
    """Compute G, H, M, F, Q, W, b in a single sweep.

    Companion to ``precompute_gram`` that additionally assembles the L²
    overlap matrix M_jk = ⟨q_j q_k⟩_W, the cosine-stripped slice-derivative
    inner factor H_jk = ⟨F_j F_k⟩_W, and returns the intermediate F and Q
    matrices for downstream CV-discovery use.

    Args:
        ctx: WeightingContext (standard interface).
        samples: (N, dim) equilibrium samples. Ignored when
            ctx.projected_samples is available.
        sample_weights: (N,) optional MBAR weights; if None, defaults to
            ctx.sample_weights (uniform 1/N with a warning if both are None).

    Returns:
        dict with keys ``G, H, M, F, Q_samples, W, b, projected_samples``.

        ``H`` is the basis-cosine-stripped inner factor: ``G = cos * H``.
        Gradient-correlation readouts (r, C[q̂]) consume ``H`` rather than
        ``G`` so that off-diagonal structure is not zeroed by orthogonal
        basis directions; see ``src/analysis/slice_correlation.py``.
    """
    if sample_weights is None:
        sample_weights = ctx.sample_weights
    if sample_weights is None:
        logger.warning(
            "precompute_gram_and_overlap: sample_weights=None; assuming "
            "uniform 1/N. Pass MBAR weights explicitly if samples are "
            "non-equilibrium."
        )

    epsilon = ctx.boundary_errors
    if epsilon is None:
        raise ValueError("Boundary errors not available.")

    if ctx.projected_samples is not None:
        projected_samples = ctx.projected_samples
    elif samples is not None:
        projected_samples = ctx.directions @ samples.T  # (M, N)
    else:
        raise ValueError(
            "Either pass samples explicitly or use "
            "store_projected_samples=True in compute_sliced_committor()."
        )

    N = projected_samples.shape[1]
    W = sample_weights if sample_weights is not None else jnp.ones(N) / N

    F = _compute_derivative_matrix(ctx, projected_samples)
    Q_samples = _interpolate_q_at_all_samples(
        ctx.slice_coords,
        ctx.committors_1d,
        projected_samples,
    )
    cos_matrix = ctx.cos_matrix if ctx.cos_matrix is not None else ctx.directions @ ctx.directions.T
    G, H = _assemble_gram_and_inner(F, W, cos_matrix)
    M_overlap = _assemble_overlap_matrix(Q_samples, W)
    b = 1.0 - epsilon

    return {
        "G": G,
        "H": H,
        "M": M_overlap,
        "F": F,
        "Q_samples": Q_samples,
        "W": W,
        "b": b,
        "projected_samples": projected_samples,
    }


def solve_gram_weights(
    G,
    b,
    valid_mask,
    eta="auto",
    constraint="sum",
    sample_weights=None,
    N=None,
    compute_condition_number=False,
):
    """Solve the constrained Gram problem for a given regularisation.

    Use with ``precompute_gram`` to sweep Tikhonov values without
    recomputing the Gram matrix each time.

    Args:
        G: (M, M) Gram matrix from ``precompute_gram``.
        b: (M,) boundary vector from ``precompute_gram``.
        valid_mask: (M,) boolean mask for valid directions.
        eta: Tikhonov regularisation. Float or ``'auto'`` (default,
            N_eff-adaptive).
        constraint: ``'sum'`` (Σw=1, default) or ``'flux'`` (bᵀw=1).
        sample_weights: only used when ``eta='auto'`` to estimate N_eff.
        N: fallback N for ``eta='auto'`` when sample_weights is None.
        compute_condition_number: if True, compute the exact SVD-based
            condition number (O(M³) extra work). Default False (NaN).

    Returns:
        dict: same keys as ``_solve_constrained_gram``.
    """
    return _solve_constrained_gram(
        G,
        b,
        valid_mask,
        eta=eta,
        constraint=constraint,
        sample_weights=sample_weights,
        N=N,
        compute_condition_number=compute_condition_number,
    )


def _add_gram_diagnostics(result, G, b, valid_mask, ctx=None):
    """Append the full-Gram solver's diagnostics to ``result``.

    Wraps :func:`compute_shared_gram_diagnostics` (negative weights,
    off-diagonal magnitude, derivative-matched diagonal_sanity) and adds
    the Gram-specific ``R_M_ratio = P / Σ_j b_j² / D_matched_j`` quality
    score: values near 1.0 indicate the constrained solve is in the
    Gram-correction regime; values near 0 indicate it has degenerated
    to the inverse-Dirichlet base.
    """
    D_matched_norm, valid_m = compute_shared_gram_diagnostics(
        result,
        G,
        valid_mask,
        ctx=ctx,
    )
    if D_matched_norm is None:
        result["R_M_ratio"] = float("nan")
        return
    R_diag = float(jnp.sum(jnp.where(valid_m, b**2 / D_matched_norm, 0.0)))
    R_full = result["P"]
    result["R_M_ratio"] = R_full / R_diag if R_diag > 1e-30 else float("nan")


def full_gram_weights(
    ctx,
    samples=None,
    sample_weights=None,
    tikhonov="auto",
    constraint="sum",
    clamp_epsilon=False,
    return_overlap=False,
    gram_dtype="float32",
    compute_condition_number=False,
    *,
    epsilon_fn=None,
):
    """Full Gram matrix weight solver.

    Assembles the M×M cross-Dirichlet matrix G and solves the constrained
    optimization problem exactly (up to a self-consistent estimate of D[q]).

    Eliminates all four structural approximations of the diagonal scheme:
      (i)   Diagonal G → full G
      (ii)  Unknown D[q] → self-consistent estimate D̂[q]
      (iii) Normalized-unconstrained → constrained optimum
      (iv)  Independent positive-part truncation → no truncation

    Units note: the returned ``Dq_hat`` estimates D[q] in the same units as
    the empirical inner product ⟨q'_j, q'_k⟩_p, where p is the normalised
    (∫p = 1) sample density.  If you need physical units that correspond to
    an unnormalised Boltzmann measure e^{−βV} dx with partition function Z,
    multiply ``Dq_hat`` by Z.  The returned weights are invariant under
    rescaling the empirical measure.

    Args:
        ctx: WeightingContext (standard interface).
        samples: (N, dim) equilibrium samples for Gram estimation.
            Ignored when ctx.projected_samples is available.
        sample_weights: (N,) optional MBAR weights; if None, defaults to
            ctx.sample_weights (which is None for uniform measure). A warning
            is logged when both are None.
        tikhonov: Regularisation η (relative to median diagonal). Float or
            the string ``'auto'`` (default, N_eff-adaptive: η = 1/√N_eff,
            clamped at 1e-12).
        constraint: ``'sum'`` (Σw=1, default) or ``'flux'`` (bᵀw=1).
        gram_dtype: dtype for the dominant (M, N) × (N, M) inner product
            ``F √W @ Fᵀ √W``. Default ``'float32'``: runs the matmul in
            single precision (1.7–2× faster on CPU, up to 5–10× on GPU with
            TF32) and upcasts the (M, M) result to float64 before solving.
            Pass ``'float64'`` to keep the entire path in double precision
            (bit-equivalent to the legacy implementation).
        compute_condition_number: if True, compute an exact SVD-based
            condition number (O(M³) extra work). Default False (returns NaN).
        epsilon_fn: optional ``ctx -> (M,)`` ε estimator (e.g.
            :func:`compute_epsilon_rms`). When ``None`` (default), uses the
            cached equilibrium ε from ``ctx.boundary_errors`` whenever
            ``sample_weights`` matches the cache and ``clamp_epsilon`` is
            off; otherwise recomputes equilibrium. A non-None ``epsilon_fn``
            always recomputes (no cache reuse).

    Returns:
        dict: see ``_solve_constrained_gram`` plus the assembled ``G``,
        negative-weight statistics, and ``diagonal_sanity`` (derivative-matched,
        Z_rel-rescaled).
    """
    # Default sample_weights from ctx (identity-compare determines cache hit).
    if sample_weights is None:
        sample_weights = ctx.sample_weights
    if sample_weights is None:
        logger.warning(
            "full_gram_weights: sample_weights=None; assuming uniform 1/N. "
            "Pass MBAR weights explicitly if samples are non-equilibrium."
        )

    # Projected samples
    if ctx.projected_samples is not None:
        projected_samples = ctx.projected_samples
        eps_ctx = ctx
    elif samples is not None:
        projected_samples = ctx.directions @ samples.T
        # Install the freshly-computed projections so downstream ε estimators
        # (which read ctx.projected_samples directly) see them without a
        # second matmul.
        eps_ctx = ctx._replace(projected_samples=projected_samples)
    else:
        raise ValueError(
            "Either pass samples explicitly or use "
            "store_projected_samples=True in compute_sliced_committor()."
        )

    N = projected_samples.shape[1]
    W = sample_weights if sample_weights is not None else jnp.ones(N) / N

    # Gram matrix (G = cos_matrix * H, with H = ⟨F_j F_k⟩_W).  H is the
    # cosine-stripped factor; expose it for gradient-correlation readouts.
    # The dominant cost is the (M, N)×(N, M) matmul inside
    # ``_assemble_gram_and_inner``; running it in float32 (default) and
    # upcasting the (M, M) result is a near-free 1.7–2× speedup on CPU and
    # much more on TF32-capable GPUs.
    F = _compute_derivative_matrix(ctx, projected_samples)
    cos_matrix = ctx.cos_matrix if ctx.cos_matrix is not None else ctx.directions @ ctx.directions.T
    matmul_dtype = jnp.dtype(gram_dtype)
    F_lo = F.astype(matmul_dtype)
    W_lo = W.astype(matmul_dtype)
    G, H = _assemble_gram_and_inner(F_lo, W_lo, cos_matrix)

    # Boundary correction vector b = 1 − ε. The cached ctx.boundary_errors
    # corresponds to ctx.sample_weights under the equilibrium estimator, so
    # we reuse it whenever (a) the caller didn't request a custom ε, (b)
    # sample_weights matches the cache, and (c) clamping is off (the cache
    # holds raw values). The match check is by Python identity (free) or
    # array equality (cheap O(N) but saves two (M, N) interpolations).
    cache_matches = epsilon_fn is None and _sample_weights_match(sample_weights, ctx.sample_weights)
    if cache_matches and ctx.boundary_errors is not None and not clamp_epsilon:
        epsilon = ctx.boundary_errors
    else:
        epsilon = compute_epsilon(
            eps_ctx,
            sample_weights=sample_weights,
            epsilon_fn=epsilon_fn,
            clamp=clamp_epsilon,
        )
    b = 1.0 - epsilon

    # Solve
    result = _solve_constrained_gram(
        G,
        b,
        ctx.valid_mask,
        eta=tikhonov,
        constraint=constraint,
        sample_weights=sample_weights,
        N=N,
        compute_condition_number=compute_condition_number,
    )
    result["G"] = G
    result["H"] = H
    result["b"] = b

    if return_overlap:
        Q_samples = _interpolate_q_at_all_samples(
            ctx.slice_coords,
            ctx.committors_1d,
            projected_samples,
        )
        result["M"] = _assemble_overlap_matrix(Q_samples, W)
        result["Q_samples"] = Q_samples
        result["F"] = F
        result["W"] = W
        result["projected_samples"] = projected_samples

    _add_gram_diagnostics(result, G, b, ctx.valid_mask, ctx=ctx)

    # Silent-degradation warning (Issue A): when the Gram correction ‖D̂·v‖
    # is small relative to ‖u‖, the constrained solve reduces to the
    # inverse-Dirichlet base u and the Gram/GFI machinery has contributed
    # nothing. This usually signals that b is near-proportional to 1 (e.g.
    # uniform or uniformly-negative after saturation).
    if result["gram_correction_ratio"] < 0.05:
        logger.warning(
            f"full_gram_weights: Gram correction ratio "
            f"{result['gram_correction_ratio']:.3g} < 0.05; the constrained "
            f"solve has silently degenerated to the inverse-Dirichlet base "
            f"(w ≈ u). mean|b|={result['mean_b_magnitude']:.3g}, "
            f"fraction(b<0)={result['fraction_b_negative']:.1%}. "
            f"Check whether (1 − ε) is uniformly small / negative."
        )

    return result


def compute_gram_diagnostics(ctx, samples=None, sample_weights=None, tikhonov="auto"):
    """Compute Gram matrix diagnostics (convenience wrapper).

    Calls full_gram_weights and returns the full result dict.
    See full_gram_weights for documentation.
    """
    return full_gram_weights(ctx, samples, sample_weights, tikhonov)


# ===========================================================================
# THIN-SHELL BCM WEIGHTS (Component F from the optimal-weights document)
# ===========================================================================
#
# Solves the constrained Dirichlet-energy problem
#
#   min_w wᵀ G w   s.t.   a_δᵀ w = 0,   b_δᵀ w = 1,
#
# where a_δ_j = μ_{A_δ}[q_θj(θj·x)], b_δ_j = μ_{B_δ}[q_θj(θj·x)] are basin
# moments over near-boundary shells
#
#   A_δ := {x ∉ A : dist(x, A) ≤ quantile_δ(dist(·, A) | x ∉ A)},
#
# and similarly B_δ. As δ → 0 the shells contract to the boundaries
# ∂A, ∂B and the constraints approach the GFI flux-weighted BC integral
# that controls ε̄. As δ → 1 they reduce to the full basin-conditional
# moment constraints (a_doc, b_doc in the document's (P) statement).
# ===========================================================================


def thin_shell_bcm_weights(
    ctx,
    samples,
    sample_weights=None,
    *,
    delta=0.1,
    in_A=None,
    in_B=None,
    dist_to_A=None,
    dist_to_B=None,
    tikhonov="auto",
    gram_dtype="float32",
    compute_condition_number=False,
    epsilon_fn=None,
    clamp_epsilon=False,
):
    """Thin-shell basin-conditional moment constraints (Component F).

    Builds the cross-Dirichlet Gram G the same way as
    :func:`full_gram_weights`, but replaces the single linear constraint
    (Σw = 1 or bᵀw = 1) with the two near-boundary basin-moment constraints

        a_δᵀ w = 0    (q̄ averages to ~0 over the near-A shell A_δ)
        b_δᵀ w = 1    (q̄ averages to ~1 over the near-B shell B_δ)

    See module docstring for the shell definition. No new hyperparameter γ
    (no Nitsche penalty form): the shell quantile δ is the only free knob.

    Args:
        ctx: ``WeightingContext`` (must have ``slice_coords``,
            ``committors_1d``, ``directions`` populated; carries the cached
            ε via ``ctx.boundary_errors`` for σ_M / silent-degradation
            diagnostics).
        samples: (N, dim) equilibrium samples in the same feature space as
            ``ctx.directions``.
        sample_weights: (N,) optional MBAR weights. None ⇒ uniform 1/N.
        delta: shell quantile in (0, 1]. δ = 0.1 (default) takes the
            nearest-to-A 10% of out-of-A samples (by feature-space Euclidean
            distance to the nearest in-A sample) as A_δ. δ = 1.0 reduces
            to the full basin-conditional moments.
        in_A, in_B: (N,) optional precomputed basin masks. If None, taken
            from ``ctx.in_A`` / ``ctx.in_B``.
        dist_to_A, dist_to_B: (N,) sample-to-nearest-basin-sample distances.
            Required: geometric distance utilities are outside the library
            scope, so callers must compute these in their own feature space
            (e.g. ``scipy.spatial.distance.cdist`` or a custom RMSD) and pass
            them in.
        tikhonov, gram_dtype, compute_condition_number, epsilon_fn,
        clamp_epsilon: same semantics as :func:`full_gram_weights`.

    Returns:
        dict with keys (in addition to the multi-constraint solver output):
            ``G``, ``H``: Gram matrix and its cosine-stripped factor.
            ``b``: 1 − ε used for σ_M.
            ``a_delta``, ``b_delta``: (M,) basin-moment vectors at δ-shell.
            ``delta``: δ used.
            ``n_A_delta``, ``n_B_delta``: shell sample counts (int).
            ``dist_threshold_A``, ``dist_threshold_B``: distance cutoffs
                defining each shell (Python floats).

    Raises:
        ValueError: if either shell is empty, or if basin masks/distances
            cannot be derived from ctx and were not passed explicitly.
    """
    if not (0.0 < delta <= 1.0):
        raise ValueError(f"delta must be in (0, 1], got {delta!r}")

    if sample_weights is None:
        sample_weights = ctx.sample_weights
    if sample_weights is None:
        logger.warning(
            "thin_shell_bcm_weights: sample_weights=None; assuming uniform "
            "1/N. Pass MBAR weights explicitly for biased sampling."
        )

    samples = jnp.asarray(samples)

    # --- Basin masks ---
    if in_A is None:
        in_A = ctx.in_A
    if in_B is None:
        in_B = ctx.in_B
    in_A_arr = jnp.asarray(in_A).astype(bool)
    in_B_arr = jnp.asarray(in_B).astype(bool)

    # --- Distances to A/B ---
    if dist_to_A is None or dist_to_B is None:
        raise ValueError(
            "thin_shell_bcm_weights requires dist_to_A and dist_to_B to be "
            "supplied as (N,) arrays. Geometric distance utilities are "
            "outside the library scope; compute them from your samples and "
            "basin definitions (e.g. via scipy.spatial.distance) and pass "
            "them explicitly."
        )
    dist_to_A = jnp.asarray(dist_to_A)
    dist_to_B = jnp.asarray(dist_to_B)

    # --- Define the δ-shells over out-of-state samples ---
    out_A_mask = ~in_A_arr
    out_B_mask = ~in_B_arr
    n_out_A = int(jnp.sum(out_A_mask))
    n_out_B = int(jnp.sum(out_B_mask))
    if n_out_A == 0:
        raise ValueError("thin_shell_bcm_weights: no out-of-A samples; cannot define A_δ.")
    if n_out_B == 0:
        raise ValueError("thin_shell_bcm_weights: no out-of-B samples; cannot define B_δ.")

    # Quantile is computed over out-of-state distances only; in-state
    # samples have dist = 0 and would otherwise pull the quantile down.
    out_A_dist = jnp.where(out_A_mask, dist_to_A, jnp.inf)
    out_B_dist = jnp.where(out_B_mask, dist_to_B, jnp.inf)
    sorted_dist_A = jnp.sort(out_A_dist)
    sorted_dist_B = jnp.sort(out_B_dist)
    # Index ⌈δ·N_out⌉ − 1 (clamped) into the ascending sorted distances.
    idx_A = max(0, int(np.ceil(delta * n_out_A)) - 1)
    idx_B = max(0, int(np.ceil(delta * n_out_B)) - 1)
    thresh_A = float(sorted_dist_A[idx_A])
    thresh_B = float(sorted_dist_B[idx_B])

    in_A_delta = out_A_mask & (dist_to_A <= thresh_A)
    in_B_delta = out_B_mask & (dist_to_B <= thresh_B)
    n_A_delta = int(jnp.sum(in_A_delta))
    n_B_delta = int(jnp.sum(in_B_delta))
    if n_A_delta == 0 or n_B_delta == 0:
        raise ValueError(
            f"thin_shell_bcm_weights: empty shell after threshold "
            f"(n_A_δ={n_A_delta}, n_B_δ={n_B_delta}). "
            f"Increase δ or check distance/basin definitions."
        )

    # --- Projected samples and Gram (mirror full_gram_weights setup) ---
    if ctx.projected_samples is not None:
        projected_samples = ctx.projected_samples
        eps_ctx = ctx
    else:
        projected_samples = ctx.directions @ samples.T
        eps_ctx = ctx._replace(projected_samples=projected_samples)

    N = projected_samples.shape[1]
    W = sample_weights if sample_weights is not None else jnp.ones(N) / N

    F = _compute_derivative_matrix(ctx, projected_samples)
    cos_matrix = ctx.cos_matrix if ctx.cos_matrix is not None else ctx.directions @ ctx.directions.T
    matmul_dtype = jnp.dtype(gram_dtype)
    F_lo = F.astype(matmul_dtype)
    W_lo = jnp.asarray(W).astype(matmul_dtype)
    G, H = _assemble_gram_and_inner(F_lo, W_lo, cos_matrix)

    # --- ε vector (only for σ_M; not used in the constraint structure) ---
    cache_matches = epsilon_fn is None and _sample_weights_match(sample_weights, ctx.sample_weights)
    if cache_matches and ctx.boundary_errors is not None and not clamp_epsilon:
        epsilon = ctx.boundary_errors
    else:
        epsilon = compute_epsilon(
            eps_ctx,
            sample_weights=sample_weights,
            epsilon_fn=epsilon_fn,
            clamp=clamp_epsilon,
        )
    b_vec = 1.0 - epsilon

    # --- Basin moments on the δ-shells ---
    Q_samples = _interpolate_q_at_all_samples(
        ctx.slice_coords,
        ctx.committors_1d,
        projected_samples,
    )  # (M, N), slice committors at each sample, clipped to [0, 1]

    W_arr = jnp.asarray(W)
    w_A_shell = jnp.where(in_A_delta, W_arr, 0.0)
    w_B_shell = jnp.where(in_B_delta, W_arr, 0.0)
    wA_sum = jnp.sum(w_A_shell)
    wB_sum = jnp.sum(w_B_shell)
    # Both denominators are > 0 by the empty-shell check above.
    a_delta = (Q_samples * w_A_shell[None, :]).sum(axis=1) / jnp.maximum(wA_sum, 1e-30)
    b_delta = (Q_samples * w_B_shell[None, :]).sum(axis=1) / jnp.maximum(wB_sum, 1e-30)

    # --- Solve the multi-constraint KKT (k = 2) ---
    A_constraint = jnp.stack([a_delta, b_delta], axis=0)  # (2, M)
    c_constraint = jnp.array([0.0, 1.0], dtype=G.dtype)

    solve_out = _solve_multi_constraint_gram(
        G,
        A_constraint,
        c_constraint,
        b_vec,
        ctx.valid_mask,
        eta=tikhonov,
        sample_weights=sample_weights,
        N=N,
        compute_condition_number=compute_condition_number,
    )

    result = {
        **solve_out,
        "G": G,
        "H": H,
        "b": b_vec,
        "a_delta": a_delta,
        "b_delta": b_delta,
        "delta": float(delta),
        "n_A_delta": n_A_delta,
        "n_B_delta": n_B_delta,
        "dist_threshold_A": thresh_A,
        "dist_threshold_B": thresh_B,
        "constraint": "thin_shell_bcm",
    }

    if result["constraint_residual"] > 1e-5:
        logger.warning(
            f"thin_shell_bcm_weights: ‖A w − c‖∞ = "
            f"{result['constraint_residual']:.3g} > 1e-5. The KKT solve "
            f"left a non-trivial residual; check Tikhonov scaling or "
            f"M_λ conditioning (cond={result['condition_number_M_lambda']})."
        )

    return result


# ===========================================================================
# GFI RESIDUAL DIAGNOSTICS (Item B from the optimal-weights document)
# ===========================================================================
#
# These functions implement the GFI-derived diagnostics that decompose the
# Dirichlet residual E(w*) into a BC-shrinkage piece (ε̄*)²D[q] and a basis-gap
# piece D[r] (orthogonal in the Dirichlet inner product). The residual
# decomposition uses three computable quantities:
#
#   D[q̄*]  = w*ᵀ G w*                       (Dirichlet energy of the sliced
#                                              approximation, exact from G, w*)
#   σ_M    = 1 / ((1−ε)ᵀ G⁻¹ (1−ε))         (GFI upper bound on D[q])
#   α_D    ≈ μ_B[q̄*] − μ_A[q̄*]              (Dirichlet-projection slope estimator
#                                              via basin-conditional separation)
#
# and yields the bound
#
#   D[r] ≥ D[q̄*] − α_D² · σ_M
#
# which is the certified basis-gap piece: the residual that direction
# enrichment can remove. The complement α_D²·σ_M upper-bounds the shrinkage
# piece, which 1D-RD recalibration (sliced_committor.calibration.recalibrate)
# can remove.
# Use ``eta_basis = D[r]_lower / D[q̄*]`` to route: small → recalibrate,
# large → enrich directions.
# ===========================================================================


def compute_alpha_d_sepdist(q_bar_samples, in_A, in_B, sample_weights=None) -> float:
    """Basin-conditional separation estimator of the Dirichlet-projection slope.

    Estimates α_D ≈ ⟨∇q̄, ∇q⟩_μ / D[q] by the basin-conditional separation

        α_D ≈ μ_B[q̄] − μ_A[q̄]

    For the true committor q this is exactly 1 (since q|_A = 0, q|_B = 1);
    for any q̄ ∈ span{slice committors} it tracks the alignment of q̄'s
    level sets with q's. This is the ``θ²_sepdist`` form referenced in the
    optimal-weights document. The calibrated variant (which absorbs a
    bias correction from incomplete basin coverage) is deferred.

    Args:
        q_bar_samples: (N,) the sliced committor evaluated at samples.
        in_A, in_B: (N,) boolean basin masks for the same samples.
        sample_weights: (N,) optional MBAR weights; if None, uniform 1/N.

    Returns:
        α_D as a Python float. Returns 0.0 if either basin has zero
        (weighted) sample count.
    """
    q = jnp.asarray(q_bar_samples)
    in_A = jnp.asarray(in_A).astype(bool)
    in_B = jnp.asarray(in_B).astype(bool)

    if sample_weights is None:
        w_A_sum = jnp.sum(in_A.astype(q.dtype))
        w_B_sum = jnp.sum(in_B.astype(q.dtype))
        mean_A = jnp.where(
            w_A_sum > 0, jnp.sum(jnp.where(in_A, q, 0.0)) / jnp.maximum(w_A_sum, 1.0), 0.0
        )
        mean_B = jnp.where(
            w_B_sum > 0, jnp.sum(jnp.where(in_B, q, 0.0)) / jnp.maximum(w_B_sum, 1.0), 0.0
        )
    else:
        W = jnp.asarray(sample_weights)
        w_A = jnp.where(in_A, W, 0.0)
        w_B = jnp.where(in_B, W, 0.0)
        w_A_sum = jnp.sum(w_A)
        w_B_sum = jnp.sum(w_B)
        mean_A = jnp.where(w_A_sum > 0, jnp.sum(w_A * q) / jnp.maximum(w_A_sum, 1e-30), 0.0)
        mean_B = jnp.where(w_B_sum > 0, jnp.sum(w_B * q) / jnp.maximum(w_B_sum, 1e-30), 0.0)

    if float(w_A_sum) <= 0.0 or float(w_B_sum) <= 0.0:
        return 0.0
    return float(mean_B - mean_A)


def compute_residual_decomposition(
    gram_result, q_bar_samples, in_A, in_B, sample_weights=None
) -> dict[str, float]:
    """GFI-anchored decomposition of the Dirichlet residual at the optimum.

    Decomposes E(w*) = D[q̄* − q] into

        (ε̄*)² · D[q]   (BC-shrinkage piece, removable by recalibration G)
        D[r]            (basis-gap piece, removable only by direction enrichment)

    via the (B) lower bound D[r] ≥ D[q̄*] − α_D² · σ_M, where σ_M is the GFI
    upper bound on D[q] (already in ``gram_result['sigma_M']``) and α_D is
    estimated by :func:`compute_alpha_d_sepdist`.

    Use ``eta_basis`` ∈ [0, 1] as a routing signal:
      * ``eta_basis`` near 1 ⇒ residual is dominated by basis incompleteness;
        adding directions (Components C, D) will help; recalibration is a no-op.
      * ``eta_basis`` near 0 ⇒ residual is dominated by BC-shrinkage; G's
        1D-RD recalibration will remove most of it.

    Args:
        gram_result: dict from :func:`full_gram_weights` (must contain ``'w'``,
            ``'G'``, and ``'sigma_M'``).
        q_bar_samples: (N,) sliced committor evaluated at the same samples
            used to fit the Gram (typically
            ``evaluate_committor(result, samples, weights)``).
        in_A, in_B: (N,) boolean basin masks for samples.
        sample_weights: (N,) optional MBAR weights for α_D.

    Returns:
        dict with keys:
            ``D_bar_q``: D[q̄*] = w*ᵀ G w* (exact).
            ``sigma_M``: GFI upper bound on D[q] (from gram_result).
            ``alpha_D``: Dirichlet-projection slope estimator (sepdist).
            ``D_r_lower``: (B) lower bound on D[r] = D[q̄*] − α_D² σ_M,
                clamped non-negative.
            ``D_bc_shrinkage_upper``: D[q̄*] − D_r_lower; equals α_D² σ_M
                when D_r_lower is the unclamped bound. Upper bound on
                the BC-shrinkage piece.
            ``eta_basis``: D_r_lower / D[q̄*] ∈ [0, 1]; fraction of the
                residual *certified* to be basis-gap.

    Notes:
        * D_r_lower can be 0 if σ_M is loose enough that α_D²·σ_M ≥ D[q̄*];
          this is inconclusive (the residual may still be partly basis-gap,
          but the bound is not tight enough to certify it).
        * Units track ``gram_result['sigma_M']``: invariant under uniform
          rescaling of the empirical measure, so ratios like ``eta_basis``
          are dimensionless.
    """
    if "w" not in gram_result or "G" not in gram_result:
        raise ValueError(
            "compute_residual_decomposition requires 'w' and 'G' in "
            "gram_result. Call full_gram_weights with return_overlap=False "
            "(the default), which still populates G."
        )
    if "sigma_M" not in gram_result:
        raise ValueError(
            "compute_residual_decomposition requires 'sigma_M' in "
            "gram_result. Re-run full_gram_weights after the GFI patch."
        )

    w = jnp.asarray(gram_result["w"])
    G = jnp.asarray(gram_result["G"])
    sigma_M = float(gram_result["sigma_M"])

    D_bar_q = float(w @ (G @ w))
    alpha_D = compute_alpha_d_sepdist(q_bar_samples, in_A, in_B, sample_weights)

    D_r_lower_raw = D_bar_q - alpha_D**2 * sigma_M
    D_r_lower = max(0.0, D_r_lower_raw)
    D_bc_shrinkage_upper = D_bar_q - D_r_lower
    eta_basis = D_r_lower / D_bar_q if D_bar_q > 1e-30 else 0.0

    return {
        "D_bar_q": D_bar_q,
        "sigma_M": sigma_M,
        "alpha_D": alpha_D,
        "D_r_lower": D_r_lower,
        "D_r_lower_raw": float(D_r_lower_raw),
        "D_bc_shrinkage_upper": D_bc_shrinkage_upper,
        "eta_basis": float(eta_basis),
    }


# ===========================================================================
# REGISTRIES
# ===========================================================================

_ALL_WEIGHT_FUNCTIONS = [
    corrected_dirichlet_inv_rd,
]

_DEFAULT_WEIGHT_FUNCTIONS = [
    corrected_dirichlet_inv_rd,
]


def get_all_weight_functions():
    """Return all diagonal weight functions usable through ``compute_weights_multi``.

    Currently a single-element list containing :func:`corrected_dirichlet_inv_rd`.
    The full-Gram simplex solver and the Basin-Moment-Constrained solver have
    richer signatures (Tikhonov, ``epsilon_fn``, ``gram_dtype``, ...) and are
    invoked directly via :func:`compute_full_gram_weights` and
    :func:`compute_basin_moment_weights` rather than through this dispatcher.
    """
    return list(_ALL_WEIGHT_FUNCTIONS)


def get_default_weight_functions():
    """Return the cheap-default weight-function set for production runs.

    Equivalent to :func:`get_all_weight_functions` today; kept distinct so
    future heavier diagonal estimators can be added to "all" without
    perturbing the default loop.
    """
    return list(_DEFAULT_WEIGHT_FUNCTIONS)
