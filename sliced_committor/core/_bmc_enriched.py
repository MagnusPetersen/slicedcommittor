"""Enriched Basin-Moment-Constrained (EBMC) variational solver.

Strict extension of :mod:`sliced_committor._bmc`. The vanilla BMC solver enforces

    μ_A[q̄] = aᵀw = 0,    μ_B[q̄] = bᵀw = 1

on the pure-sum ansatz ``q̄(x) = Σ_j w_j q_{θ_j}(θ_j·x)``. EBMC adds a free
global bias ``c`` to the ansatz:

    q̄(x) = c + Σ_j w_j q_{θ_j}(θ_j·x).

Eliminating ``c = -aᵀw`` collapses the two basin-moment constraints into the
single linear constraint ``(b − a)ᵀ w = 1``. Closed form:

    M_gap := (b − a)ᵀ G⁻¹ (b − a) = A + B − 2 C,
    w_EBMC = (G⁻¹b − G⁻¹a) / M_gap,
    c_EBMC = (A − C) / M_gap,
    𝓓[q̄*] = 1 / M_gap

where ``A = aᵀG⁻¹a``, ``B = bᵀG⁻¹b``, ``C = aᵀG⁻¹b`` are the same scalars BMC
computes. EBMC therefore reuses the BMC Cholesky and shares ``(a, b)``.

Galerkin–Rayleigh–Ritz on the strictly larger trial space ``V_w ⊕ span{1}``
gives ``𝓓_BMC / 𝓓_EBMC = 1 + (A − C)² / Δ ≥ 1`` with ``Δ = AB − C²``. Two
further properties:

* Shift-invariance of ``w``: replacing ``q_{θ_j}`` by ``q_{θ_j} + α_j``
  shifts only ``c`` by ``-Σ_j w_j α_j``; ``w`` is unchanged. BMC weights
  are not shift-invariant.
* Algebraic flat-slice cancellation: ``q_{θ_j} ≡ const ⇒ a_j ≈ b_j ⇒
  (b−a)_j ≈ 0``, so ``w_j`` contributes nothing to the constraint.

Failure mode: ``M_gap → 0`` (every slice has identical basin moments, a
genuine representation failure rather than the constraint-geometry
collinearity that trips BMC's ``cond_basin`` gate).

Power-Enriched Slice Basis (PESB-EBMC)
--------------------------------------

The basic EBMC trial space is ``span{q_{θ_j}}_{j=1..M} ⊕ span{1}``. PESB-EBMC
enriches each per-ridge basis to ``{Ψ_{n_k}(q_{θ_j})}_{k=1..P}`` with the
*smoothstep* family

    Ψ_n(v) = v^n / (v^n + (1 − v)^n),    n ≥ 1.

Properties: ``Ψ_n(0) = 0``, ``Ψ_n(0.5) = 0.5``, ``Ψ_n(1) = 1`` (basin
endpoints and the half-committor level invariant). ``n = 1`` reproduces
identity; larger ``n`` steepens the transition. Restriction ``n ≥ 1``
keeps ``Ψ'_n`` bounded at basin endpoints where equilibrium samples
concentrate.

The KKT solve runs unchanged on the ``(MP, MP)`` augmented Gram via
:func:`_solve_enriched_bmc_kkt`. Galerkin: the PESB trial space contains the
plain-EBMC one (set ``w_{j,k} = 0`` for ``k ≥ 2`` to recover EBMC), so
``M_gap_PESB ≥ M_gap_EBMC`` at the population level.

Only the smoothstep basis is shipped: the monomial basis ``v^p`` was
disconfirmed (asymmetric around v=0.5, Hilbert-matrix-like within-block
conditioning at large P). Defaults ``P=2`` and ``n_values = [1.0, 2.0]``
match the combined paper-figure runs across the aib9, villin, and
chignolin benchmarks.
"""

import logging

import jax
import jax.numpy as jnp
import numpy as np
from jax import vmap

from ._bmc import compute_basin_moments
from .gram import (
    _assemble_gram_matrix,
    _compute_derivative_matrix,
    compute_shared_gram_diagnostics,
    resolve_cos_matrix,
    resolve_metric_diagonal,
)
from .solver import _interp_1d_at_samples
from .weights import _mask_and_regularize_gram, _resolve_eta

logger = logging.getLogger(__name__)


class EnrichedBMCRepresentationError(RuntimeError):
    """Raised when ``M_gap = (b−a)ᵀG⁻¹(b−a)`` collapses.

    Every slice has effectively identical basin moments
    (``a_j ≈ b_j`` across the basis), so no projection direction
    distinguishes A from B at the moment level. Remedy: more or
    differently-aligned directions, not a smaller regulariser.
    """


# ---------------------------------------------------------------------------
# Closed-form KKT solve (shared by basic EBMC and PESB-EBMC)
# ---------------------------------------------------------------------------


@jax.jit
def _solve_enriched_bmc_kkt(G, a, b, valid_mask, eta_val):
    """JIT body for the single-constraint closed-form solve with bias."""
    valid = valid_mask.astype(G.dtype)
    G_reg, _med_diag = _mask_and_regularize_gram(G, valid_mask, eta_val)

    a_m = a * valid
    b_m = b * valid

    rhs = jnp.stack([a_m, b_m], axis=-1)
    U, lower = jax.scipy.linalg.cho_factor(G_reg)
    sol = jax.scipy.linalg.cho_solve((U, lower), rhs)
    Ginv_a = sol[:, 0]
    Ginv_b = sol[:, 1]

    A_s = jnp.dot(a_m, Ginv_a)
    B_s = jnp.dot(b_m, Ginv_b)
    C_s = jnp.dot(a_m, Ginv_b)
    M_gap = A_s + B_s - 2.0 * C_s
    Delta = A_s * B_s - C_s * C_s

    M_gap_safe = jnp.where(M_gap > 1e-14, M_gap, 1e-14)
    w = ((Ginv_b - Ginv_a) / M_gap_safe) * valid
    c = (A_s - C_s) / M_gap_safe

    Delta_safe = jnp.where(Delta > 1e-14, Delta, jnp.nan)
    improvement_over_bmc = 1.0 + (A_s - C_s) ** 2 / Delta_safe

    scale = jnp.maximum(jnp.maximum(A_s, B_s), 1e-30)
    cond_enriched = M_gap / scale

    U_diag = jnp.diag(U)
    U_abs = jnp.abs(U_diag)
    cond_chol = (jnp.max(U_abs) / jnp.maximum(jnp.min(U_abs), 1e-30)) ** 2

    return {
        "w": w,
        "c": c,
        "M_gap": M_gap,
        "A_scalar": A_s,
        "B_scalar": B_s,
        "C_scalar": C_s,
        "Delta": Delta,
        "improvement_over_bmc": improvement_over_bmc,
        "cond_enriched": cond_enriched,
        "condition_number": cond_chol,
        "G_reg": G_reg,
        "Ginv_a": Ginv_a,
        "Ginv_b": Ginv_b,
    }


def solve_enriched_basin_moment(
    G: jnp.ndarray,
    a: jnp.ndarray,
    b: jnp.ndarray,
    valid_mask: jnp.ndarray,
    eta: float | str = "auto",
    sample_weights: jnp.ndarray | None = None,
    N: int | None = None,
    raise_on_degenerate: bool = True,
    cond_enriched_threshold: float = 1e-6,
) -> dict:
    """Closed-form EBMC solve with optional M_gap-collapse gate.

    Args:
        G: (M, M) cross-Dirichlet Gram matrix.
        a, b: (M,) basin moment vectors.
        valid_mask: (M,) boolean mask for valid directions.
        eta: Tikhonov factor. Float or ``'auto'`` (N_eff-adaptive).
        sample_weights, N: used only when ``eta='auto'``.
        raise_on_degenerate: if True, raise
            :class:`EnrichedBMCRepresentationError` when
            ``cond_enriched < cond_enriched_threshold``.
        cond_enriched_threshold: threshold on ``M_gap / max(A, B)``.
            Default 1e-6.

    Returns:
        dict with ``w``, ``c``, ``M_gap``, the shared Gram scalars
        ``A, B, C, Delta``, ``improvement_over_bmc``, ``cond_enriched``,
        ``condition_number``, ``G_reg``, ``eta_used``, ``N_eff``.
    """
    M = G.shape[0]
    eta_val, N_eff_val = _resolve_eta(eta, M, valid_mask, sample_weights, N, G=G)

    out = _solve_enriched_bmc_kkt(G, a, b, valid_mask, eta_val)

    cond_enriched_f = float(out["cond_enriched"])
    if raise_on_degenerate and cond_enriched_f < cond_enriched_threshold:
        raise EnrichedBMCRepresentationError(
            f"cond_enriched={cond_enriched_f:.3g} < {cond_enriched_threshold:.3g}: "
            "moment gap (b−a) is collapsing in G⁻¹ norm; the projection basis "
            "cannot distinguish A from B at the moment level. Add more or "
            "differently-aligned directions. Set raise_on_degenerate=False "
            "to override."
        )

    return {
        "w": out["w"],
        "c": float(out["c"]),
        "a": a,
        "b": b,
        "M_gap": float(out["M_gap"]),
        "A": float(out["A_scalar"]),
        "B": float(out["B_scalar"]),
        "C": float(out["C_scalar"]),
        "Delta": float(out["Delta"]),
        "improvement_over_bmc": float(out["improvement_over_bmc"]),
        "cond_enriched": cond_enriched_f,
        "condition_number": float(out["condition_number"]),
        "G_reg": out["G_reg"],
        "eta_used": float(eta_val),
        "N_eff": float(N_eff_val) if N_eff_val is not None else None,
    }


# ---------------------------------------------------------------------------
# Halfset-eigen Gram regularization (tikhonov='halfset_eigen')
# ---------------------------------------------------------------------------
#
# Vendored from ``lib/recovar/regularize.py`` (``halfset_eigen_regularize``,
# with the fold recombination of ``halfset_grams_from_arrays`` and the solve
# convention of ``solve_with_G``) so that ``lib/sliced_committor`` stays
# self-contained; this package must not import ``lib/recovar``. Validation
# and provenance: ``docs/recovar_transfers.md`` (EXP-A and the Composition
# experiments).

_HALFSET_N_FOLDS = 10
_HALFSET_N_BANDS = 12


def _halfset_ssnr_from_corr(c):
    """FSC to SSNR of the combined (full) dataset: SSNR = 2c/(1-c)."""
    c = np.clip(c, 0.0, 0.999999)
    return 2.0 * c / (1.0 - c)


def _halfset_eigen_regularize(G1, G2, n_bands=_HALFSET_N_BANDS):
    """Wiener ridge in eigen-bands of Gbar = (G1 + G2)/2.

    Split the frames in two halves, assemble a Gram matrix from each, and
    read the sampling noise off their DISAGREEMENT band by band, exactly as
    cryoEM FSC reads the noise off two half-map reconstructions. The SSNR of
    band k comes from the correlation of the band rows of ``V'G1V`` against
    ``V'G2V`` (the halfset Grams expressed in the eigenbasis of their
    average) via the FSC identity ``SSNR = 2c/(1-c)``; each eigenvalue is
    then inflated as ``lam_reg = max(lam, 0) * (1 + 1/SSNR)``, so the
    INVERSE the solve uses is damped by the Wiener factor
    ``SSNR/(1+SSNR)``. PSD by construction (floor ``1e-12 * max lam_reg``);
    no tuned constant anywhere.

    Vendored verbatim (numerics unchanged, including the ``scipy.linalg.eigh``
    driver) from ``lib/recovar/regularize.py``; see that module and
    ``docs/recovar_transfers.md`` for the validated lesson that the sampling
    noise of G is diagonal in the EIGENBASIS of G, not in any a-priori
    slice-property basis.

    Returns ``(G_reg, info)`` with ``info = dict(band_ssnr, lam, lam_reg)``.
    """
    from scipy.linalg import eigh

    G1 = np.asarray(G1, dtype=np.float64)
    G2 = np.asarray(G2, dtype=np.float64)
    M = G1.shape[0]
    Gbar = 0.5 * (G1 + G2)
    lam, V = eigh(Gbar)
    order = np.argsort(-lam)
    lam, V = lam[order], V[:, order]
    A1 = V.T @ G1 @ V
    A2 = V.T @ G2 @ V

    edges = np.linspace(0, M, n_bands + 1).astype(int)
    ssnr = np.zeros(M)
    band_ssnr = np.zeros(n_bands)
    for k in range(n_bands):
        sl = slice(edges[k], edges[k + 1])
        v1 = A1[sl, :].ravel()
        v2 = A2[sl, :].ravel()
        if v1.size < 3 or v1.std() == 0 or v2.std() == 0:
            s = 1e6
        else:
            s = _halfset_ssnr_from_corr(np.corrcoef(v1, v2)[0, 1])
        band_ssnr[k] = s
        ssnr[sl] = s
    lam_reg = np.maximum(lam, 0.0) * (1.0 + 1.0 / np.maximum(ssnr, 1e-12))
    floor = 1e-12 * max(lam_reg.max(), 1e-300)
    lam_reg = np.maximum(lam_reg, floor)
    G_reg = (V * lam_reg) @ V.T
    return G_reg, dict(band_ssnr=band_ssnr, lam=lam, lam_reg=lam_reg)


def heldout_cap_halfset(G_folds, w_folds, a_folds, b_folds, wA, wB, valid_mask):
    """Held-out Dirichlet cap evaluated with the half-set solve itself.

    The cap of Eq. (cap), ``E[u] / F[u]^2``, bounds ``nu_AB`` for whatever trial
    space produced it, so it ranks direction sets on the same variational
    footing the weights already obey and no reference committor enters it. To
    rank trial spaces rather than fit them it has to be read out of sample, and
    to rank the trial spaces that are actually deployed it has to be read
    through the solve that is actually deployed.

    For each fold ``k`` the remaining folds are the training set. Their even and
    odd halves give the two half-Grams the regulariser reads the sampling noise
    off, the constrained solve of Eq. (wstar) runs on the regularised training
    Gram, and the cap

        w' G_k w / ((b_k - a_k) . w)^2

    is evaluated on the fold that was held out. Both factors are out of sample,
    which is what makes the number comparable across different trial spaces;
    the in-sample Dirichlet energy is monotone in M by construction and cannot
    rank them.

    There is no ridge grid here, because the half-set filter has no scalar to
    select. The return value is therefore one number per trial space, which is
    what a model-selection criterion needs.

    Two asymmetries against the deployed fit are worth naming, because both are
    inherent to reading a criterion out of sample and neither disturbs a
    ranking. Each training set is ``K-1`` folds rather than the whole sample, so
    its half-Grams are noisier and the filter damps a little harder than it will
    at deployment. And with ``K`` even the ``K-1`` training folds split
    ``K/2`` against ``K/2 - 1``, so the two halves are not exactly equal in
    weight. Both effects are common to every trial space compared, so they shift
    the level of the cap and not the order of it.

    Returns ``dict`` with ``cap`` (the fold mean), ``per_fold``, ``se``,
    ``n_folds`` and ``n_ok``.
    """
    G_folds = np.asarray(G_folds, np.float64)
    w_folds = np.asarray(w_folds, np.float64)
    a_folds = np.asarray(a_folds, np.float64)
    b_folds = np.asarray(b_folds, np.float64)
    wA = np.asarray(wA, np.float64)
    wB = np.asarray(wB, np.float64)
    keep = np.asarray(valid_mask, bool)
    K = G_folds.shape[0]
    caps = np.full(K, np.nan)

    for k in range(K):
        other = np.array([j for j in range(K) if j != k])
        # A fold that holds no samples of some basin cannot define the moment
        # gap, on either side of the split.
        if wA[k] <= 0 or wB[k] <= 0:
            continue
        if wA[other].sum() <= 0 or wB[other].sum() <= 0:
            continue
        even, odd = other[0::2], other[1::2]
        if even.size == 0 or odd.size == 0:
            continue

        wo = w_folds[other]
        Gtr = np.tensordot(wo, G_folds[other], axes=(0, 0)) / max(wo.sum(), 1e-300)
        atr = (wA[other] @ a_folds[other]) / max(wA[other].sum(), 1e-300)
        btr = (wB[other] @ b_folds[other]) / max(wB[other].sum(), 1e-300)
        # The half-set split is taken over the TRAINING folds only, so the
        # regulariser never sees the fold the cap is read on.
        G1 = (np.tensordot(w_folds[even], G_folds[even], axes=(0, 0))
              / max(w_folds[even].sum(), 1e-300))
        G2 = (np.tensordot(w_folds[odd], G_folds[odd], axes=(0, 0))
              / max(w_folds[odd].sum(), 1e-300))
        try:
            res = _solve_halfset_eigen(
                Gtr, atr, btr, keep, G1, G2, raise_on_degenerate=False)
        except (np.linalg.LinAlgError, ValueError):
            continue
        w = np.asarray(res["w"], np.float64)
        gap = float((b_folds[k] - a_folds[k]) @ w)
        if abs(gap) > 1e-30:
            caps[k] = float(w @ G_folds[k] @ w) / gap ** 2

    n_ok = int(np.isfinite(caps).sum())
    if n_ok == 0:
        raise ValueError("heldout_cap_halfset: the cap is undefined on every fold.")
    with np.errstate(invalid="ignore"):
        mean = float(np.nanmean(caps))
        se = float(np.nanstd(caps, ddof=1) / np.sqrt(n_ok)) if n_ok > 1 else float("nan")
    return dict(cap=mean, per_fold=caps, se=se, n_folds=int(K), n_ok=n_ok)


def _solve_halfset_eigen(
    G,
    a,
    b,
    valid_mask,
    G1,
    G2,
    raise_on_degenerate=True,
    cond_enriched_threshold=1e-6,
):
    """Closed-form EBMC solve on the halfset-eigen regularized Gram.

    Restricts the two half-Grams to the valid directions BEFORE the
    eigendecomposition (masked rows are zero, and their zero eigenvalues
    would poison the low eigen-bands' SSNR), regularizes with
    :func:`_halfset_eigen_regularize`, and solves with ZERO additional
    Tikhonov ridge; ``G_reg`` already carries its own PSD floor. The solve
    mirrors ``lib/recovar/regularize.py`` ``solve_with_G`` (Cholesky with an
    lstsq fallback) on the valid sub-block, then embeds back to the full
    direction set with identity rows for invalid directions, matching the
    mask contract of ``_mask_and_regularize_gram``.

    Returns the key set of :func:`solve_enriched_basin_moment` (with
    ``eta_used = 0.0``) plus ``'halfset_info'``, the raw regularizer info
    dict for the caller's diagnostics block.
    """
    from scipy.linalg import cho_factor, cho_solve

    valid = np.asarray(valid_mask, bool)
    iv = np.flatnonzero(valid)
    M = int(np.asarray(G).shape[0])
    G1v = np.asarray(G1, np.float64)[np.ix_(iv, iv)]
    G2v = np.asarray(G2, np.float64)[np.ix_(iv, iv)]
    G_reg_v, info = _halfset_eigen_regularize(G1v, G2v, n_bands=_HALFSET_N_BANDS)

    a_np = np.asarray(a, np.float64)
    b_np = np.asarray(b, np.float64)
    av, bv = a_np[iv], b_np[iv]
    delta = bv - av
    cond_chol = float("inf")
    try:
        cf = cho_factor(G_reg_v, lower=True)
        w_dual = cho_solve(cf, delta)
        Ginv_a = cho_solve(cf, av)
        Ginv_b = cho_solve(cf, bv)
        L_abs = np.abs(np.diag(cf[0]))
        cond_chol = float((L_abs.max() / max(L_abs.min(), 1e-30)) ** 2)
    except np.linalg.LinAlgError:
        w_dual = np.linalg.lstsq(G_reg_v, delta, rcond=None)[0]
        Ginv_a = np.linalg.lstsq(G_reg_v, av, rcond=None)[0]
        Ginv_b = np.linalg.lstsq(G_reg_v, bv, rcond=None)[0]

    A_s = float(av @ Ginv_a)
    B_s = float(bv @ Ginv_b)
    C_s = float(av @ Ginv_b)
    M_gap = float(delta @ w_dual)
    Delta = A_s * B_s - C_s * C_s

    w_v = w_dual / M_gap if abs(M_gap) > 1e-300 else w_dual
    c = float(-(av @ w_v))
    w = np.zeros(M)
    w[iv] = w_v

    scale = max(A_s, B_s, 1e-30)
    cond_enriched = M_gap / scale
    if raise_on_degenerate and cond_enriched < cond_enriched_threshold:
        raise EnrichedBMCRepresentationError(
            f"cond_enriched={cond_enriched:.3g} < {cond_enriched_threshold:.3g}: "
            "moment gap (b−a) is collapsing in G⁻¹ norm; the projection basis "
            "cannot distinguish A from B at the moment level. Add more or "
            "differently-aligned directions. Set raise_on_degenerate=False "
            "to override."
        )

    G_reg_full = np.eye(M)
    G_reg_full[np.ix_(iv, iv)] = G_reg_v

    improvement = 1.0 + (A_s - C_s) ** 2 / Delta if Delta > 1e-14 else float("nan")
    return {
        "w": jnp.asarray(w),
        "c": c,
        "a": a,
        "b": b,
        "M_gap": M_gap,
        "A": A_s,
        "B": B_s,
        "C": C_s,
        "Delta": Delta,
        "improvement_over_bmc": improvement,
        "cond_enriched": float(cond_enriched),
        "condition_number": cond_chol,
        "G_reg": jnp.asarray(G_reg_full),
        "eta_used": 0.0,
        "N_eff": None,
        "halfset_info": info,
    }


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------


def _add_ebmc_diagnostics(result, G, a, b, valid_mask, ctx=None):
    """Append shared Gram diagnostics and EBMC-specific scalars to ``result``.

    Reuses :func:`compute_shared_gram_diagnostics` for the negative-weight,
    off-diagonal-magnitude, and derivative-matched diagonal-sanity blocks,
    then adds:

      * ``constraint_residual_moment_gap``: ``|(b−a)ᵀw − 1|`` (must be ~ε).
      * ``mu_A_check``, ``mu_B_check``: ``aᵀw + c`` and ``bᵀw + c``
        (must be ~0 and ~1).
      * ``sum_w``: reported but not constrained.
      * ``optimal_dirichlet_energy``: ``1 / M_gap``.
    """
    compute_shared_gram_diagnostics(
        result, G, valid_mask, ctx=ctx, metric_diag=resolve_metric_diagonal(ctx)
    )
    w = result["w"]
    c = result["c"]

    result["constraint_residual_moment_gap"] = float(jnp.abs(jnp.dot(b - a, w) - 1.0))
    result["mu_A_check"] = float(jnp.dot(a, w) + c)
    result["mu_B_check"] = float(jnp.dot(b, w) + c)
    result["sum_w"] = float(jnp.sum(w))
    M_gap = result["M_gap"]
    result["optimal_dirichlet_energy"] = float(1.0 / M_gap) if M_gap > 0 else float("inf")


# ---------------------------------------------------------------------------
# Top-level basic EBMC solver
# ---------------------------------------------------------------------------


def enriched_basin_moment_weights(
    ctx,
    samples: jnp.ndarray | None = None,
    sample_weights: jnp.ndarray | None = None,
    tikhonov: float | str = "auto",
    raise_on_degenerate: bool = True,
    cond_enriched_threshold: float = 1e-6,
    gram_dtype: str = "float64",
    heldout_cap: bool = False,
) -> dict:
    """Top-level basic EBMC solver.

    Assembles the cross-Dirichlet Gram matrix, computes basin moments
    ``(a, b)``, and runs the closed-form single-constraint solve with bias.

    The returned dict carries ``'c'`` and ``'q_bar'`` (zeros), so passing it
    to :func:`evaluate_committor` activates the centered-basis path
    automatically: ``q̂(x) = c + Σ_j w_j q_{θ_j}(θ_j·x)``.

    Args:
        ctx: ``WeightingContext`` (must carry ``in_A``, ``in_B``, and
            ``projected_samples``; use ``store_projected_samples=True``
            in :func:`compute_sliced_committor`).
        samples: ``(N, dim)`` equilibrium samples. Ignored when
            ``ctx.projected_samples`` is populated.
        sample_weights: ``(N,)`` optional MBAR weights; falls back to
            ``ctx.sample_weights``, then to uniform ``1/N``.
        tikhonov: Gram regularisation. ``'auto'`` (default) is the closed-form
            ``η = 1/√N_eff`` and needs no extra pass over the data. ``'cv'``
            selects the ridge by minimising the held-out Dirichlet cap, so it
            carries no fitted constant, and is what the paper's figures use; it
            costs one extra assembly-equivalent (the folds partition the
            samples) plus K eigendecompositions, so it is opt-in rather than the
            default. ``'halfset_eigen'`` regularises the Gram matrix itself
            instead of selecting a scalar ridge: two interleaved contiguous
            basin-stratified half-Grams (10 folds) measure the per-eigenband
            SSNR of G and a Wiener factor damps each band; the solve then runs
            with zero additional ridge. Tuning-free; costs one extra
            assembly-equivalent plus one eigendecomposition. Frames must be
            TIME-ORDERED, the same contract as ``'cv'`` (the contiguous
            stratified folds assume serial order within each stratum; on
            shuffled frames the measured SSNR is optimistic). Validated in
            ``docs/recovar_transfers.md`` (EXP-A). Also accepts a float or
            ``'auto_lambda'``. ``'cv'`` and ``'halfset_eigen'`` are
            implemented here and *only* here -- the full-Gram, plain-BMC, PESB
            and Nitsche solvers do not accept them. See ``docs/ridge_rule.md``.
        raise_on_degenerate, cond_enriched_threshold: see
            :func:`solve_enriched_basin_moment`.
        gram_dtype: dtype for the dominant ``(M, N)×(N, M)`` inner product.
            Default ``'float64'``.

    Returns:
        dict: output of :func:`solve_enriched_basin_moment` augmented with
        the assembled ``G``, the basin moments ``a``, ``b``, zero
        ``q_bar``, and shared Gram + EBMC-specific diagnostics.
            heldout_cap: with ``tikhonov='halfset_eigen'``, additionally evaluate
            the held-out Dirichlet cap under that same solve and return it as
            ``result['heldout_cap']``. This is the quantity that ranks trial
            spaces (direction sets, M) without a reference committor; see
            :func:`heldout_cap_halfset`. Off by default because it costs the
            per-fold basin moments and K small solves, which a plain fit does
            not need.
"""
    if not jax.config.read("jax_enable_x64"):
        raise ValueError(
            "EBMC requires jax_enable_x64=True. "
            "Add `jax.config.update('jax_enable_x64', True)` before computing the result."
        )
    if ctx.in_A is None or ctx.in_B is None:
        raise ValueError(
            "enriched_basin_moment_weights: ctx.in_A / ctx.in_B are missing; "
            "basin moments require basin labels. Pass in_A / in_B to "
            "compute_sliced_committor so they propagate into the result."
        )

    if sample_weights is None:
        sample_weights = ctx.sample_weights
    if sample_weights is None:
        logger.warning(
            "enriched_basin_moment_weights: sample_weights=None; assuming "
            "uniform 1/N. Pass MBAR weights explicitly if samples are "
            "non-equilibrium."
        )

    if ctx.projected_samples is not None:
        projected_samples = ctx.projected_samples
        moments_ctx = ctx
    elif samples is not None:
        projected_samples = ctx.directions @ samples.T
        moments_ctx = ctx._replace(projected_samples=projected_samples)
    else:
        raise ValueError(
            "enriched_basin_moment_weights: pass samples explicitly or use "
            "store_projected_samples=True in compute_sliced_committor()."
        )

    N = projected_samples.shape[1]
    W = sample_weights if sample_weights is not None else jnp.ones(N) / N

    F = _compute_derivative_matrix(ctx, projected_samples)
    cos_matrix = resolve_cos_matrix(ctx)
    matmul_dtype = jnp.dtype(gram_dtype)
    F_lo = F.astype(matmul_dtype)
    W_lo = W.astype(matmul_dtype)
    G = _assemble_gram_matrix(F_lo, W_lo, cos_matrix)
    want_cv = isinstance(tikhonov, str) and tikhonov == "cv"
    want_halfset = isinstance(tikhonov, str) and tikhonov == "halfset_eigen"
    if not (want_cv or want_halfset):
        # tikhonov='cv' and 'halfset_eigen' need F again for the per-fold Gram
        # blocks; every other setting is done with it here, and at villin's
        # M = 2048 the (M, N) buffer is 3.3 GB, so keep the early free as the
        # default path.
        del F, F_lo, W_lo

    a, b = compute_basin_moments(moments_ctx)

    cv_info = None
    # With fewer than two valid directions there is nothing to select: the
    # constraint (b-a)'w = 1 then fixes w outright -- at M = 1,
    # w = 1/(b_1 - a_1) whatever the ridge -- so the ridge cannot change the
    # answer and cross-validating it is meaningless, not merely expensive.
    # Fall back rather than raise: a ladder that sweeps M upward from 1 is a
    # legitimate caller and should not have to special-case its first rung.
    if want_cv or want_halfset:
        n_valid = int(np.asarray(ctx.valid_mask, bool).sum())
        if n_valid < 2:
            logger.warning(
                "tikhonov=%r: only %d valid direction(s); the moment constraint "
                "already determines w, so the regularisation is immaterial. "
                "Falling back to 'auto'.",
                tikhonov,
                n_valid,
            )
            want_cv = False
            want_halfset = False
            tikhonov = "auto"
            del F, F_lo, W_lo
    if want_cv:
        # Calibration-free ridge: minimise the HELD-OUT cap, 1-SE rule.  Costs one
        # extra assembly-equivalent (the folds partition the samples) plus K
        # eigendecompositions.  See ``_ridge_cv`` and ``docs/ridge_rule.md``.
        from ._ridge_cv import (
            DEFAULT_N_FOLDS,
            fold_basin_moments,
            fold_gram_blocks,
            make_folds,
            select_ridge_cv,
        )

        # Stratify by basin so every fold holds A, B and transition samples;
        # blocks stay contiguous within each stratum, which is what keeps
        # serially-correlated frames off both sides of the split.
        strata = np.where(np.asarray(ctx.in_A, bool), 0, np.where(np.asarray(ctx.in_B, bool), 1, 2))
        fold_of = make_folds(N, DEFAULT_N_FOLDS, contiguous=True, strata=strata)
        G_folds, w_folds = fold_gram_blocks(F, W, cos_matrix, fold_of, DEFAULT_N_FOLDS)
        a_folds, b_folds, wA, wB = fold_basin_moments(moments_ctx, fold_of, DEFAULT_N_FOLDS)
        cv_info = select_ridge_cv(G_folds, w_folds, a_folds, b_folds, wA, wB, ctx.valid_mask)
        tikhonov = ("ridge_abs", cv_info["ridge"])
        del G_folds, a_folds, b_folds, F, F_lo, W_lo

    halfset_info = None
    heldout_cap_info = None
    if want_halfset:
        # Halfset-eigen Gram regularization (the RECOVAR sec. 1 transfer,
        # validated in docs/recovar_transfers.md EXP-A). The fold
        # recombination matches recovar.halfset_grams_from_arrays: each fold
        # block is normalised by its own weight sum, so the weighted mean of
        # the even (odd) fold blocks IS the even (odd) half-Gram, and
        # interleaving contiguous stratified blocks between the halves is
        # what keeps the halfset disagreement honest on serially correlated
        # frames.
        from ._ridge_cv import fold_basin_moments, fold_gram_blocks, make_folds

        strata = np.where(np.asarray(ctx.in_A, bool), 0, np.where(np.asarray(ctx.in_B, bool), 1, 2))
        fold_of = make_folds(N, _HALFSET_N_FOLDS, contiguous=True, strata=strata)
        G_folds, w_folds = fold_gram_blocks(F, W, cos_matrix, fold_of, _HALFSET_N_FOLDS)
        del F, F_lo, W_lo
        even = np.arange(0, _HALFSET_N_FOLDS, 2)
        odd = np.arange(1, _HALFSET_N_FOLDS, 2)
        G1 = np.tensordot(w_folds[even], G_folds[even], axes=(0, 0)) / w_folds[even].sum()
        G2 = np.tensordot(w_folds[odd], G_folds[odd], axes=(0, 0)) / w_folds[odd].sum()
        # The held-out cap under this same solve, for ranking trial spaces. The
        # fold Gram blocks are already assembled above, so the only extra cost
        # is the basin moments per fold and K small solves.
        if heldout_cap:
            a_folds, b_folds, wA_f, wB_f = fold_basin_moments(
                moments_ctx, fold_of, _HALFSET_N_FOLDS)
            heldout_cap_info = heldout_cap_halfset(
                G_folds, w_folds, a_folds, b_folds, wA_f, wB_f, ctx.valid_mask)
            del a_folds, b_folds
        del G_folds

        result = _solve_halfset_eigen(
            G,
            a,
            b,
            ctx.valid_mask,
            G1,
            G2,
            raise_on_degenerate=raise_on_degenerate,
            cond_enriched_threshold=cond_enriched_threshold,
        )
        halfset_info = result.pop("halfset_info")
    else:
        result = solve_enriched_basin_moment(
            G,
            a,
            b,
            ctx.valid_mask,
            eta=tikhonov,
            sample_weights=sample_weights,
            N=N,
            raise_on_degenerate=raise_on_degenerate,
            cond_enriched_threshold=cond_enriched_threshold,
        )
    result["G"] = G
    result["q_bar"] = jnp.zeros_like(result["w"])
    if heldout_cap_info is not None:
        result["heldout_cap"] = {
            "cap": heldout_cap_info["cap"],
            "se": heldout_cap_info["se"],
            "n_folds": heldout_cap_info["n_folds"],
            "n_ok": heldout_cap_info["n_ok"],
            "per_fold": np.asarray(heldout_cap_info["per_fold"]),
        }
    if halfset_info is not None:
        result["halfset_eigen"] = {
            "n_folds": _HALFSET_N_FOLDS,
            "n_bands": _HALFSET_N_BANDS,
            "band_ssnr": [float(s) for s in halfset_info["band_ssnr"]],
            "lam_max": float(np.max(halfset_info["lam"])),
            "lam_min_reg": float(np.min(halfset_info["lam_reg"])),
        }
    if cv_info is not None:
        result["ridge_cv"] = {
            k: cv_info[k] for k in ("ridge", "anchor", "idx", "idx_argmin", "at_edge", "n_folds")
        }
        result["ridge_cv"]["curve"] = np.asarray(cv_info["curve"])
        result["ridge_cv"]["grid"] = np.asarray(cv_info["grid"])
        result["ridge_cv"]["se"] = np.asarray(cv_info["se"])

    _add_ebmc_diagnostics(result, G, a, b, ctx.valid_mask, ctx=ctx)
    return result


# ---------------------------------------------------------------------------
# Power-Enriched Slice Basis (PESB-EBMC) with smoothstep basis
# ---------------------------------------------------------------------------


def _slice_values_at_samples(slice_coords, committors_1d, projected_samples):
    """``Q[j, n] = q_{θ_j}(θ_j · x_n)``, clipped to ``[0, 1]``.

    NaN-safe (replaces NaN per-slice committors with 0 before interpolation).
    """
    q_safe = jnp.where(jnp.isnan(committors_1d), 0.0, committors_1d)

    def _interp_single(s_grid, q_grid, s_proj):
        q = _interp_1d_at_samples(s_grid, q_grid, s_proj)
        return jnp.clip(q, 0.0, 1.0)

    return vmap(_interp_single)(slice_coords, q_safe, projected_samples)


@jax.jit
def _smoothstep_chunk_field(Q_chunk, Fprime_chunk, n_values):
    """``F_aug_chunk[(j, k), n] = Ψ'_{n_k}(Q[j, n]) · Fprime[j, n]``."""
    n_b = n_values[None, :, None].astype(Q_chunk.dtype)
    Q_b = Q_chunk[:, None, :]
    Qc = 1.0 - Q_b
    denom = jnp.maximum(Q_b**n_b + Qc**n_b, 1e-30)
    num_deriv = n_b * (Q_b ** (n_b - 1.0)) * (Qc ** (n_b - 1.0))
    Psi_deriv = num_deriv / (denom**2)
    F = Psi_deriv * Fprime_chunk[:, None, :]
    M_chunk, N = Q_chunk.shape
    P = n_values.shape[0]
    return F.reshape(M_chunk * P, N)


@jax.jit
def _smoothstep_chunk_moments(Q_chunk, in_A_f, in_B_f, n_values):
    """Per-chunk basin-moment partial sums for the smoothstep basis."""
    n_b = n_values[None, :, None].astype(Q_chunk.dtype)
    Q_b = Q_chunk[:, None, :]
    Qc = 1.0 - Q_b
    denom = jnp.maximum(Q_b**n_b + Qc**n_b, 1e-30)
    Psi = (Q_b**n_b) / denom
    M_chunk, N = Q_chunk.shape
    P = n_values.shape[0]
    a_partial = jnp.sum(Psi * in_A_f[None, None, :], axis=-1).reshape(M_chunk * P)
    b_partial = jnp.sum(Psi * in_B_f[None, None, :], axis=-1).reshape(M_chunk * P)
    return a_partial, b_partial


def _build_augmented_field_smoothstep(Q, Fprime, n_values, batch_size=512):
    """Chunked smoothstep augmented field ``(M*P, N)``.

    Chunks over the M axis to bound peak memory at one ``(batch_size*P, N)``
    chunk. Each chunk runs through the JIT-fused
    :func:`_smoothstep_chunk_field`.
    """
    M = Q.shape[0]
    chunks = []
    for start in range(0, M, batch_size):
        end = min(start + batch_size, M)
        chunks.append(
            _smoothstep_chunk_field(
                Q[start:end],
                Fprime[start:end],
                n_values,
            )
        )
    return jnp.concatenate(chunks, axis=0)


def _build_augmented_basin_moments_smoothstep(Q, in_A, in_B, n_values, batch_size=512):
    """Chunked basin moments for the smoothstep basis."""
    M = Q.shape[0]
    in_A_f = in_A.astype(Q.dtype)
    in_B_f = in_B.astype(Q.dtype)
    n_A = jnp.maximum(jnp.sum(in_A_f), 1.0)
    n_B = jnp.maximum(jnp.sum(in_B_f), 1.0)
    a_chunks, b_chunks = [], []
    for start in range(0, M, batch_size):
        end = min(start + batch_size, M)
        a_p, b_p = _smoothstep_chunk_moments(
            Q[start:end],
            in_A_f,
            in_B_f,
            n_values,
        )
        a_chunks.append(a_p)
        b_chunks.append(b_p)
    a = jnp.concatenate(a_chunks, axis=0) / n_A
    b = jnp.concatenate(b_chunks, axis=0) / n_B
    return a, b


def _resolve_smoothstep_n_values(P, n_values):
    """Validate / normalise the ``(P, n_values)`` pair for the smoothstep basis.

    Defaults to ``n_values = linspace(1.0, P, P)`` so the family always
    includes the identity ``Ψ_1`` at index 0 (P=1 reproduces basic EBMC).
    Rejects ``n_k < 1`` (the n < 1 regime has unbounded Ψ' at basin
    endpoints).
    """
    if n_values is None:
        if not isinstance(P, int) or P < 1:
            raise ValueError(f"P must be a positive integer (got {P!r}).")
        return jnp.linspace(1.0, float(P), int(P), dtype=jnp.float64)
    n_arr = jnp.asarray(n_values, dtype=jnp.float64).reshape(-1)
    if n_arr.shape[0] < 1:
        raise ValueError("n_values must be non-empty.")
    if not bool(jnp.all(n_arr >= 1.0)):
        raise ValueError(
            f"smoothstep basis requires n_values >= 1.0; got "
            f"min(n_values)={float(jnp.min(n_arr)):.3g}. The n<1 regime is "
            "not supported (Ψ' diverges at basin endpoints where "
            "equilibrium samples concentrate)."
        )
    return n_arr


def _augmented_cos_matrix(cos_matrix, P):
    """Kronecker lift: ``cos_aug[(j,p), (l,p')] = cos_matrix[j, l]``."""
    return jnp.kron(cos_matrix, jnp.ones((P, P), dtype=cos_matrix.dtype))


def _within_block_condition_numbers(G_reg, M, P):
    """Per-direction P×P within-block condition numbers of ``G_reg``.

    Diagnostic for the powered-up basis: each within-direction block is
    essentially a moment matrix of the on-slice density of ``q_{θ_j}`` in
    v-coordinates. Watch for values above 1e8.
    """
    G_blocks = G_reg.reshape(M, P, M, P).transpose(0, 2, 1, 3)
    diag_blocks = vmap(lambda j: G_blocks[j, j])(jnp.arange(M))

    def _block_cond(blk):
        s = jnp.linalg.svd(blk, compute_uv=False)
        return s[0] / jnp.maximum(s[-1], 1e-30)

    return vmap(_block_cond)(diag_blocks)


def enriched_basin_moment_weights_power(
    ctx,
    samples: jnp.ndarray | None = None,
    sample_weights: jnp.ndarray | None = None,
    P: int = 2,
    n_values: jnp.ndarray | None = None,
    tikhonov: float | str = "auto",
    raise_on_degenerate: bool = True,
    cond_enriched_threshold: float = 1e-6,
    gram_dtype: str = "float64",
    direction_batch_size: int = 512,
) -> dict:
    """Power-Enriched Slice Basis EBMC with the smoothstep basis.

    Enriches each per-ridge basis function ``q_{θ_j}`` to the family
    ``{Ψ_{n_k}(q_{θ_j})}_{k=1..P}`` while keeping the EBMC bias and
    basin-moment constraint structure. The KKT solve runs on the
    ``(MP, MP)`` augmented Gram via :func:`_solve_enriched_bmc_kkt`
    unchanged; only the dimensions and the auxiliary diagnostics differ.

    At ``P=1`` (or ``n_values=[1.0]``) this reproduces
    :func:`enriched_basin_moment_weights` bit-for-bit.

    Args:
        ctx, samples, sample_weights, tikhonov, raise_on_degenerate,
        cond_enriched_threshold, gram_dtype: see
            :func:`enriched_basin_moment_weights`.
        P: enrichment dimension per direction. Used when ``n_values`` is
            None (then ``n_values = linspace(1.0, P, P)``). Must be ≥ 1.
        n_values: smoothstep exponents ``(P,)`` with ``n_k ≥ 1``. When
            given, overrides ``P``.
        direction_batch_size: directions per JIT chunk during field and
            moment assembly. Controls peak memory: ~``batch * P * N``
            per chunk.

    Returns:
        dict with all basic-EBMC keys plus the PESB additions:

          * ``P``: int, effective enrichment dimension.
          * ``n_values``: ``(P,)`` smoothstep exponents.
          * ``w_by_power``: ``(M, P)`` reshape of ``w``; column k is the
            coefficient of ``Ψ_{n_k}``.
          * ``power_weight_mass``: ``(P,)`` normalised ``Σ_j |w_{j,k}|``;
            < 5% on k ≥ 2 indicates the enrichment is contributing
            nothing.
          * ``within_block_cond``: ``(M,)`` per-direction P×P block
            condition numbers of ``G_reg``.
          * ``improvement_over_ebmc``: ``M_gap_PESB / M_gap_EBMC`` at
            matched inputs; ≥ 1 in the population limit.

        ``w``, ``a``, ``b``, ``q_bar`` have shape ``(M*P,)`` (flattened
        row-major: index ``j*P + (k-1)`` for direction j, exponent k).
    """
    if not jax.config.read("jax_enable_x64"):
        raise ValueError(
            "PESB-EBMC requires jax_enable_x64=True. "
            "Add `jax.config.update('jax_enable_x64', True)` before computing the result."
        )
    if ctx.in_A is None or ctx.in_B is None:
        raise ValueError(
            "enriched_basin_moment_weights_power: ctx.in_A / ctx.in_B are "
            "missing; basin moments require basin labels. Pass in_A / in_B "
            "to compute_sliced_committor so they propagate into the result."
        )

    n_arr = _resolve_smoothstep_n_values(P, n_values)
    P_eff = int(n_arr.shape[0])

    if sample_weights is None:
        sample_weights = ctx.sample_weights
    if sample_weights is None:
        logger.warning(
            "enriched_basin_moment_weights_power: sample_weights=None; "
            "assuming uniform 1/N. Pass MBAR weights explicitly if samples "
            "are non-equilibrium."
        )

    if ctx.projected_samples is not None:
        projected_samples = ctx.projected_samples
    elif samples is not None:
        projected_samples = ctx.directions @ samples.T
    else:
        raise ValueError(
            "enriched_basin_moment_weights_power: pass samples explicitly or "
            "use store_projected_samples=True in compute_sliced_committor()."
        )

    M = ctx.directions.shape[0]
    N = projected_samples.shape[1]
    W = sample_weights if sample_weights is not None else jnp.ones(N) / N

    Q = _slice_values_at_samples(
        ctx.slice_coords,
        ctx.committors_1d,
        projected_samples,
    )
    Fprime = _compute_derivative_matrix(ctx, projected_samples)

    cos_matrix = resolve_cos_matrix(ctx)
    matmul_dtype = jnp.dtype(gram_dtype)
    F_aug = _build_augmented_field_smoothstep(
        Q,
        Fprime,
        n_arr,
        batch_size=direction_batch_size,
    )
    F_aug_lo = F_aug.astype(matmul_dtype)
    W_lo = W.astype(matmul_dtype)
    cos_aug = _augmented_cos_matrix(cos_matrix, P_eff)
    G = _assemble_gram_matrix(F_aug_lo, W_lo, cos_aug)
    del F_aug, F_aug_lo, W_lo, cos_aug, Fprime

    a, b = _build_augmented_basin_moments_smoothstep(
        Q,
        ctx.in_A,
        ctx.in_B,
        n_arr,
        batch_size=direction_batch_size,
    )
    valid_mp = jnp.repeat(ctx.valid_mask, P_eff)

    result = solve_enriched_basin_moment(
        G,
        a,
        b,
        valid_mp,
        eta=tikhonov,
        sample_weights=sample_weights,
        N=N,
        raise_on_degenerate=raise_on_degenerate,
        cond_enriched_threshold=cond_enriched_threshold,
    )
    result["G"] = G
    result["q_bar"] = jnp.zeros_like(result["w"])

    # ctx=None for diagonal_sanity: the density-weighted D_j matching argument
    # is specific to the p=1 (identity) basis function.
    _add_ebmc_diagnostics(result, G, a, b, valid_mp, ctx=None)

    result["P"] = int(P_eff)
    result["n_values"] = n_arr
    w_by_power = result["w"].reshape(M, P_eff)
    result["w_by_power"] = w_by_power
    power_mass = jnp.sum(jnp.abs(w_by_power), axis=0)
    result["power_weight_mass"] = power_mass / jnp.maximum(jnp.sum(power_mass), 1e-30)
    result["within_block_cond"] = _within_block_condition_numbers(
        result["G_reg"],
        M,
        P_eff,
    )

    if P_eff > 1:
        idx_first = jnp.arange(0, M * P_eff, P_eff)
        G_sub = G[idx_first][:, idx_first]
        a_sub = a[idx_first]
        b_sub = b[idx_first]
        eta_val_used = jnp.asarray(result["eta_used"], dtype=G.dtype)
        sub = _solve_enriched_bmc_kkt(
            G_sub,
            a_sub,
            b_sub,
            ctx.valid_mask,
            eta_val_used,
        )
        m_gap_first = float(sub["M_gap"])
        result["improvement_over_ebmc"] = (
            float(result["M_gap"] / m_gap_first) if m_gap_first > 1e-30 else float("nan")
        )
    else:
        result["improvement_over_ebmc"] = 1.0

    return result
