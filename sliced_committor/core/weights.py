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
follows from the Flux–Fidelity Identity (FFI), which holds
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
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from jax import jit, vmap

logger = logging.getLogger(__name__)

from ._internal import to_log_abs_sign
from .gram import (
    _compute_derivative_matrix,
    compute_shared_gram_diagnostics,
)
from .solver import _interp_1d_at_samples

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


def _basin_weight_sums(in_A, in_B, sample_weights):
    """Per-basin sample weights and their floored normalisers.

    Shared by the equilibrium / RMS epsilon estimators: builds
    ``w_A = W·1_A`` and ``w_B = W·1_B`` from the (optional) sample weights
    ``W`` (uniform ``1/N`` when ``sample_weights`` is None) and returns their
    sums floored at ``1e-30`` to guard the downstream divisions.

    Returns:
        ``(w_A, w_B, Z_A, Z_B)``.
    """
    N = in_A.shape[0]
    W = jnp.ones(N, dtype=jnp.float64) / N if sample_weights is None else sample_weights
    w_A = W * in_A.astype(W.dtype)
    w_B = W * in_B.astype(W.dtype)
    Z_A = jnp.maximum(jnp.sum(w_A), 1e-30)
    Z_B = jnp.maximum(jnp.sum(w_B), 1e-30)
    return w_A, w_B, Z_A, Z_B


def _basin_moment_base(ctx, sample_weights):
    """Shared front-half of the equilibrium / RMS epsilon estimators.

    Validates basin labels + projections, builds per-basin sample weights
    (:func:`_basin_weight_sums`), and interpolates the per-slice committor at the
    A- and B-masked samples. Returns ``(q_A, q_B, w_A, w_B, Z_A, Z_B)``; each
    estimator applies only its own final reduction.
    """
    if ctx.in_A is None or ctx.in_B is None:
        raise ValueError("ctx.in_A / ctx.in_B required for epsilon computation")
    if ctx.projected_samples is None:
        raise ValueError("projected_samples required for epsilon computation")
    in_A, in_B = ctx.in_A, ctx.in_B
    w_A, w_B, Z_A, Z_B = _basin_weight_sums(in_A, in_B, sample_weights)
    q_A = _interpolate_q_at_samples_masked(
        ctx.slice_coords, ctx.committors_1d, ctx.projected_samples, in_A
    )
    q_B = _interpolate_q_at_samples_masked(
        ctx.slice_coords, ctx.committors_1d, ctx.projected_samples, in_B
    )
    return q_A, q_B, w_A, w_B, Z_A, Z_B


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
    q_A, q_B, w_A, w_B, Z_A, Z_B = _basin_moment_base(ctx, sample_weights)
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
    q_A, q_B, w_A, w_B, Z_A, Z_B = _basin_moment_base(ctx, sample_weights)
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
                f"compute_epsilon: unknown estimator name {epsilon_fn!r}. Valid choices: {valid}."
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


def _effective_n(sample_weights, N):
    """``N_eff = (sum w)^2 / sum w^2``, or ``N`` for uniform weights; None if neither."""
    if sample_weights is not None:
        W = jnp.asarray(sample_weights)
        Z, Z2 = jnp.sum(W), jnp.sum(W**2)
        if float(Z2) > 0:
            return float(Z**2 / Z2)
    if N is not None:
        return float(N)
    return None


def _resolve_eta(eta, M, valid_mask, sample_weights=None, N=None, G=None):
    """Resolve the Tikhonov parameter.

    Accepts a float (used verbatim) or one of two strings.

    ``'auto'`` (default) scales η to the estimated effective sample size only:

        η = max(1e-12, 1 / √N_eff)

    with ``N_eff = (Σw)² / Σw²`` for non-uniform weights and ``N_eff = N`` for
    uniform weights.  This matches the Full_Solve recommendation δ ~ 1/√N_eff
    and suppresses regularisation bias in well-conditioned, high-N regimes.

    ``'auto_lambda'`` is the recommended setting:

        ridge = 1.8e-4 * (1e5 / N_eff) * M_valid * geomean(diag G_valid)

    ``('lambda', lam)`` is the same rule with the constant exposed.  Both fall
    back to ``η = lam * M_valid`` when ``G`` is not supplied.

    The ``1/N_eff`` factor is as load-bearing as the ``M`` factor and for the same
    reason: without it the rule is only correct at the N it was calibrated on.
    Anchoring on ``M * geomean(diag G)`` alone, best single constant, worst ratio
    to each configuration's own oracle ridge on the 2D benchmark:

        exponent p in (1e5/N_eff)^p:   0     0.5    1.0    1.3
        worst over N = 25k..400k:    1.632  1.208  1.045  1.027
        at N = 25 000 alone:         1.632  1.139  1.045  1.027
        at N = 400 000 alone:        1.501  1.208  1.023  1.004

    And the M-degradation itself comes back without it: at N = 25 000 the N-blind
    rule loses 1.41x over M = 64..512 (4.16e-4 -> 5.87e-4), against 1.04x at
    N = 1e5.  ``mean(diag G)`` is itself N-independent to 1.1x across that range,
    so this is a genuinely separate degree of freedom, not double-counting.
    ``p = 1`` is used because it is the round, Wishart-flavoured choice and
    ``p = 1.3`` (the least-squares fit to three N values on one system) buys only
    1.7% more.

    Be clear about which parts of this are derived and which are fitted.

    * ``ridge ∝ M`` is DERIVED, exactly, with no constant: replace the M
      directions by k copies of each and the trial space is unchanged, so the fit
      must be unchanged, and the restoring ridge is exactly ``k*eta`` (measured
      1.000, 1.957, 4.019, 7.867 for k = 1, 2, 4, 8).
    * The anchor must be EXTENSIVE in M -- forced by the same argument -- and must
      be a DIAGONAL statistic rather than a spectral one, because in high
      dimension ``G`` is numerically diagonal and ``lambda_max(G)`` is constant in
      M (on AIB9 it is 251655 to six figures across M = 128..2048), so a
      ``lambda_max`` anchor would be M-independent and provably wrong.
    * WHICH diagonal statistic is NOT derived.  Duplication cannot separate
      ``geomean``, ``mean`` (= tr(G)/M) or any other, since all are extensive.
      ``geomean`` is an empirical choice: worst ratio 1.054 against 1.079 for
      ``M*median`` and 1.086 for ``tr(G)``.
    * The constant ``1.8e-4`` is a MAGIC NUMBER.  It was fitted by minimising the
      worst-case ratio to each configuration's own ORACLE ridge over 21
      configurations -- 2D Wolfe-Quapp (d=2, exact PDE oracle), AIB9 (d=52, three
      direction seeds) and villin (d=350), M = 64..2048, two error metrics.  The
      per-system optima span 8x (2D 2.7e-4, AIB9 4.6e-5, villin 3.5e-5); the
      compromise survives that only because the objective is flat.

    Measured, against each configuration's own oracle ridge:

        eta = 'auto' (the M-blind default)   median 1.072x   worst 4.793x
        'auto_lambda'                        median 1.008x   worst 1.054x
        held-out-energy selection            median 1.034x   worst 1.209x

    Refitting the constant with each configuration held out gives worst 1.074x,
    so it is not merely fitting its own test set.  Sensitivity to the constant,
    worst case over the same 21 configurations: 2x off costs <=1.18, 3x off <=1.27,
    10x off 1.7-2.7, 100x off 4-19.  So a factor of a few is cheap and an order of
    magnitude is not.  ``lam`` is dimensionless but NOT universal, and it does not
    transfer across N (roughly ``N^-1.3`` on 2D over N = 25k..400k); it is
    calibrated only over N = 1e5..2.7e5.  Outside that, re-derive it with the
    held-out cap selector (``experiments/ridge_stopping_rule.py``), which needs no
    oracle and no constant at all, and costs about 1.3x instead of 1.05x.

    Why a diagonal anchor at all.  The solve is ``w = G_reg⁻¹d / (dᵀG_reg⁻¹d)`` with
    ``G_reg = G + η·median(diag G)·I``, which is *exactly invariant* under
    ``G -> αG`` because the anchor co-scales.  So rescaling the operator -- the
    usual "K/n -> Mercer operator" move -- is a provable no-op here, and cannot be
    the thing that is mis-normalised.  What is mis-normalised is the norm placed
    on the coefficient vector: ``G`` is a *sum* over directions (only the sample
    index is averaged, via ``W``), so ``tr(G) ∝ M`` exactly, while the anchor
    ``median(diag G)`` does not move with M at all.  A fixed ``eta`` is therefore
    a vanishing relative penalty.

    The exponent is exactly 1, and it is measurable without any theory or any
    reference solution: replace the M directions by k copies of each.  The trial
    space is unchanged, so the fitted function must be unchanged, and the ridge
    that restores it is ``k*eta`` (measured 1.000, 1.957, 4.019, 7.867 for
    k = 1, 2, 4, 8 -- ``experiments/msweep_2d_normalization.py``).  ``tr(G)``
    scales by exactly k under that duplication while ``median(diag G)`` does not
    move, which is the mis-normalisation in two numbers.

    Why ``geomean(diag G)`` and not ``median(diag G)`` or ``tr(G)/M``.  Duplication
    fixes only that the anchor must be *extensive*; every extensive candidate
    (``tr(G)``, ``M``, ``||G||_F``, ``lambda_max``, top-k eigenvalue sums) scales by
    k under it, so it cannot choose between them.  The choice is empirical, and
    the discriminator is how well one constant transfers.  The median is a
    knife-edge order statistic: under the LDA sampler the draw has two strata
    whose per-direction Dirichlet energies differ by orders of magnitude, and the
    pooled median sits in the gap between the modes.  ``tr(G)`` is a sample mean
    of a heavy-tailed variable (the top 1% of directions carry 11-39% of it).  The
    geometric mean is neither.  Worst ratio to the per-configuration oracle with a
    single constant, over the same 21 configurations:

        M * geomean(diag)   1.054      (leave-one-out 1.074)
        M * median(diag)    1.079      (1.101)
        tr(G) = M * mean    1.086      (1.101)
        M alone             1.127      (1.209)
        median(diag)        1.481      (2.720)   <- the shipped M-blind shape

    Only the last is qualitatively wrong; the rest differ by little, and the
    honest summary is that the anchor must be extensive and ``M*geomean`` is the
    best of the family rather than uniquely correct.

    ``'auto_m'`` is ``η = M_valid / N_eff``: right M-dependence, median anchor,
    system-specific constant (the measured optimum is ~40x it on 2D and ~8x on
    AIB9).  Prefer ``'auto_lambda'``.

    Caveats worth knowing before trusting any of this on a new system.  The ridge
    matters enormously on 2D (it moves the error 79-475x) and hardly at all on
    villin (1.02x across seven decades), so a rule can look excellent there while
    being untested.  On AIB9 the optimal ridge varies ~400x across three direction
    seeds at fixed M, so its argmin is not a stable target -- what is stable is
    that any extensive rule stays inside the (very wide) optimal basin.

    ``'auto'`` (default) scales to the effective sample size only:

        η = max(1e-12, 1 / √N_eff)

    with ``N_eff = (Σw)² / Σw²`` for non-uniform weights and ``N_eff = N`` for
    uniform weights.  It is M-blind, which is what makes the error degrade as
    directions are added: on 2D against the exact PDE oracle the true Dirichlet
    error rises 3.7x over M = 32..2048 under ``'auto'`` and is flat to 2% under a
    single constant ``lam`` (``docs/degradation_with_M.md``).  ``'auto'`` remains
    the default only for backward compatibility.

    Returns ``(eta_value, N_eff)``; ``N_eff`` is None when ``eta`` is a bare float.
    """
    AUTO_LAMBDA = 1.8e-4
    AUTO_LAMBDA_N0 = 1.0e5  # the N the constant was calibrated at

    if isinstance(eta, (tuple, list)) or eta == "auto_lambda":
        if isinstance(eta, str):
            val = AUTO_LAMBDA
            n_eff = _effective_n(sample_weights, N)
            if n_eff is None:
                raise ValueError(
                    "eta='auto_lambda' requires either sample_weights or N: the "
                    "rule scales as 1/N_eff, and dropping that factor is only "
                    "correct at N = 1e5. Pass ('lambda', lam) to opt out."
                )
            val = val * AUTO_LAMBDA_N0 / n_eff
        else:
            kind, val = eta
            if kind == "ridge_abs":
                # An ABSOLUTE ridge, in the units of G itself.  The caller
                # multiplies eta by median(diag G_valid), so divide it out.  This
                # is how the calibration-free selectors hand their answer back:
                # they choose a ridge in physical units, not a dimensionless
                # multiplier, and must not be re-scaled by an anchor on the way in.
                valid = jnp.asarray(valid_mask).astype(jnp.float64)
                if G is None:
                    raise ValueError("eta=('ridge_abs', r) requires G.")
                diag = np.asarray(jnp.diag(jnp.asarray(G)), dtype=np.float64)
                keep = np.asarray(valid > 0) & np.isfinite(diag) & (diag > 0)
                med = float(np.median(diag[keep])) if keep.sum() else 0.0
                if not (np.isfinite(med) and med > 0):
                    raise ValueError("eta=('ridge_abs', r): median(diag G_valid) is not positive.")
                return max(1e-12, float(val) / med), None
            if kind != "lambda":
                raise ValueError(
                    f"Unknown eta form: {eta!r}; expected ('lambda', value) or "
                    f"('ridge_abs', value)."
                )
        valid = jnp.asarray(valid_mask).astype(jnp.float64)
        M_valid = max(float(jnp.sum(valid)), 1.0)
        if G is None:
            return max(1e-12, float(val) * M_valid), None
        diag = np.asarray(jnp.diag(jnp.asarray(G)), dtype=np.float64)
        keep = np.asarray(valid > 0) & np.isfinite(diag) & (diag > 0)
        if keep.sum() < 1:
            return max(1e-12, float(val) * M_valid), None
        geo = float(np.exp(np.mean(np.log(diag[keep]))))
        med = float(np.median(diag[keep]))
        if not (np.isfinite(geo) and geo > 0 and np.isfinite(med) and med > 0):
            return max(1e-12, float(val) * M_valid), None
        # The caller multiplies eta by median(diag G_valid), so dividing it out
        # here makes the delivered absolute ridge exactly lam * M * geomean.
        return max(1e-12, float(val) * M_valid * geo / med), None
    if isinstance(eta, str):
        if eta not in ("auto", "auto_m"):
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
            raise ValueError(f"eta={eta!r} requires either sample_weights or N.")
        if eta == "auto_m":
            M_valid = float(jnp.sum(jnp.asarray(valid_mask).astype(jnp.float64)))
            return max(1e-12, max(M_valid, 1.0) / N_eff_val), N_eff_val
        return max(1e-12, 1.0 / (N_eff_val**0.5)), N_eff_val
    return float(eta), None


def _mask_and_regularize_gram(G, valid_mask, eta_val):
    """Mask invalid directions and add the median-diagonal Tikhonov ridge.

    Invalid directions get zeroed rows/cols and a unit diagonal (so the
    regularised Gram stays SPD); the ridge is scaled by the median of the
    valid diagonal entries. Shared by every Gram-based KKT solver
    (``_solve_constrained_gram_jit`` here, plus the basin-moment and enriched
    solvers in ``_bmc`` / ``_bmc_enriched``). Traced into each caller's JIT
    region; not decorated to avoid a redundant jit boundary.

    Returns ``(G_reg, med_diag)``: the regularised Gram and the median valid
    diagonal used to scale the ridge.
    """
    M = G.shape[0]
    valid = valid_mask.astype(G.dtype)
    mask_2d = valid[:, None] * valid[None, :]
    G_masked = G * mask_2d + jnp.diag(1.0 - valid)  # invalid → identity row
    G_diag_valid = jnp.where(valid > 0, jnp.diag(G_masked), jnp.inf)
    med_diag = jnp.median(G_diag_valid)
    med_diag = jnp.where(jnp.isfinite(med_diag) & (med_diag > 0), med_diag, 1.0)
    G_reg = G_masked + eta_val * med_diag * jnp.eye(M)
    return G_reg, med_diag


@partial(jit, static_argnames=("constraint",))
def _solve_constrained_gram_jit(G, b, valid_mask, eta_val, constraint="sum"):
    """JIT-compiled numerical body of the constrained Gram solve.

    Returns a dict of JAX arrays. The Python wrapper ``_solve_constrained_gram``
    handles the ``eta='auto'`` resolution, the optional SVD condition number,
    and conversion of scalar fields to Python floats. Keeping all numerics in
    a single JIT region eliminates ~10 host syncs per call and lets XLA fuse
    the masking, Tikhonov, Cholesky, and self-consistency arithmetic.
    """
    valid = valid_mask.astype(G.dtype)
    G_reg, _med_diag = _mask_and_regularize_gram(G, valid_mask, eta_val)

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
    # base): the Gram/FFI machinery contributes nothing. These three scalars
    # make that failure mode observable without changing the solution.
    Dq_v = Dq_hat_out * v_out
    u_norm = jnp.linalg.norm(u_out)
    gram_correction_ratio = jnp.linalg.norm(Dq_v) / jnp.maximum(u_norm, 1e-30)
    n_valid_f = jnp.maximum(jnp.sum(valid), 1.0)
    mean_b_magnitude = jnp.sum(jnp.abs(b_masked)) / n_valid_f
    fraction_b_negative = jnp.sum(((b_masked < 0) & (valid > 0)).astype(G.dtype)) / n_valid_f

    # σ_M = 1 / ((1−ε)ᵀ G⁻¹ (1−ε)) = 1/P: the FFI upper bound on D[q].
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

        ``sigma_M = 1 / ((1−ε)ᵀ G⁻¹ (1−ε))`` is the FFI upper bound on D[q]:
        always ≥ true D[q] in the noise-free limit. Under ``constraint='flux'``
        it equals ``Dq_hat`` by construction; under ``'sum'`` ``Dq_hat`` is a
        self-consistent estimate while ``sigma_M`` is the bound. Use
        ``sigma_M`` as the diagnostic and as the basis-incompleteness signal
        (smaller σ_M ⇒ basis explains more of D[q]).
    """
    M = G.shape[0]

    # Resolve Tikhonov (Issue 3: 'auto' mode)
    eta_val, N_eff_val = _resolve_eta(eta, M, valid_mask, sample_weights, N, G=G)

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
    # inverse-Dirichlet base u and the Gram/FFI machinery has contributed
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
