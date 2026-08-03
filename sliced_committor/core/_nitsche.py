"""Second-moment (Nitsche) boundary conditions for the sliced committor.

EBMC enforces the boundary conditions only in the *mean*,

    mu_A[q_bar] = c + a^T w = 0,     mu_B[q_bar] = c + b^T w = 1,

which after eliminating ``c`` collapses to the single scalar constraint
``(b - a)^T w = 1``.  One constraint on an M-dimensional weight vector leaves
M-1 unconstrained directions, and the objective ``w^T G w`` is a *seminorm*
that cannot see how ``q_bar`` behaves pointwise inside the basins.  As the
slice basis grows the minimiser spends the extra freedom lowering its own
objective below the true committor's -- ``E_bar < nu_AB`` is routinely observed
-- which is only possible because the *true* flux fidelity ``F[q_bar]`` is
drifting below 1.  The error

    e_M = E_M / nu_AB - 2 F_M + 1

then rises even though ``E_M`` falls monotonically, which it must, by nesting.

This module replaces the mean-only constraint by a Nitsche penalty on the basin
*second* moments,

    min_{w,c}  w^T G w  +  beta [ <q_bar^2>_A + <(q_bar - 1)^2>_B ],

a full-rank quadratic form on each basin rather than one scalar.  Setting the
gradient in ``(w, c)`` to zero gives the ``(M+1) x (M+1)`` system

    [ G + beta (P_A + P_B)   beta (a + b) ] [w]   [ beta b ]
    [ beta (a + b)^T            2 beta    ] [c] = [ beta   ]

with ``P_A[j,k] = <q_j q_k>_A`` and ``P_B[j,k] = <q_j q_k>_B``.

This is the equilibrium form of Solve C from ``src/noneq`` (the same system with
``G`` replaced by the non-symmetric stiffness ``A = G + N``); see
``docs/noneq_design.md`` Sec. 6a, where it is the only solver whose committor
RMSE decreases with M.

**Status: experimental, and its motivating premise did not survive testing.**  The
argument above rests on the claim that the deployed EBMC error *turns around* past
some M.  That claim came from sweeps drawing independent directions at each M,
which confounds the trend with draw variance -- on those sweeps the per-rung
*oracle* rises too, on 3 of 6 configurations.  A nested ladder, where the coarse
rung's directions are a subset of the fine rung's, does not reproduce it: on the
2D benchmark at N = 1e5 the degradation is absent (the largest rung is the best,
``last/min = 1.000``), and on AIB9 the per-rung oracle *plateaus*
(0.0593/0.0601/0.0593 at M = 256/512/1024, minimum at the top rung), which places
the residual rise entirely on the ridge selector rather than on the boundary
conditions.  So the M-degradation this module was written to cure is not, on
present evidence, intrinsic anywhere it has been measured.

None of that makes the solver wrong -- second-moment BCs are a genuinely stronger
constraint than one scalar, and the ``O(1/beta)`` analysis below stands -- but it
is not the default, it is not what the paper deploys, and the claim that it is
"the first solver whose RMSE decreases with M" is inherited from ``src/noneq``,
where it was measured for the non-symmetric NESS problem, not re-established here.
Prefer ``ebmc`` unless you are specifically investigating boundary treatment.

Caveats, stated up front.  The penalty pins the *equilibrium* basin moments, not
the flux-weighted boundary averages that define ``F``.  It removes the
dimensionality of the freedom to exploit ``f_hat - f``, not the bias itself, and
it carries an ``O(1/beta)`` consistency error, so ``beta`` is a real knob:
too small and the BCs leak, too large and the system over-constrains and the
conditioning degrades.  The returned ``bc_resid_A`` / ``bc_resid_B`` make the
first failure mode observable; unlike EBMC's ``mu_A_check`` / ``mu_B_check``,
which the solve pins by construction, these residuals are informative.
"""

from __future__ import annotations

import logging

import jax
import jax.numpy as jnp

from ._bmc import compute_basin_moments
from .gram import _assemble_gram_matrix, _compute_derivative_matrix
from .weights import _resolve_eta

logger = logging.getLogger(__name__)

__all__ = [
    "compute_basin_second_moments",
    "solve_nitsche",
    "nitsche_weights",
]


# --------------------------------------------------------------------------- #
# basin second moments
# --------------------------------------------------------------------------- #
@jax.jit
def _q_at(slice_coords, committors_1d, projected_samples):
    """``Q[j, n] = clip(q_j(theta_j . x_n), 0, 1)`` for one slab of samples."""
    from .solver import _interp_1d_at_samples

    def _one(s_grid, q_grid, s_proj):
        return jnp.clip(_interp_1d_at_samples(s_grid, q_grid, s_proj), 0.0, 1.0)

    return jax.vmap(_one)(slice_coords, committors_1d, projected_samples)


def compute_basin_second_moments(ctx, chunk: int = 8192):
    """Basin second-moment Grams ``(P_A, P_B)``, each ``(M, M)``.

        P_A[j, k] = (1/|A|) sum_{x in A} q_j(theta_j . x) q_k(theta_k . x)

    Uses the same unweighted basin average as :func:`compute_basin_moments`, so
    ``(a, b)`` and ``(P_A, P_B)`` are consistent members of the same linear
    system.

    Chunked over *samples* rather than directions: the accumulation
    ``P += Q_chunk @ Q_chunk.T`` keeps peak memory at ``(M, chunk)`` plus the
    ``(M, M)`` output, independent of the basin population.  A single
    ``(M, N_basin)`` buffer at villin's M = 2048 would be gigabytes.
    """
    if ctx.in_A is None or ctx.in_B is None:
        raise ValueError(
            "nitsche: ctx.in_A / ctx.in_B are missing; the second-moment "
            "penalty needs basin labels. Pass in_A / in_B to "
            "compute_sliced_committor so they propagate into the result."
        )
    if ctx.projected_samples is None:
        raise ValueError(
            "nitsche: ctx.projected_samples is None; use "
            "store_projected_samples=True in compute_sliced_committor."
        )

    slice_coords = ctx.slice_coords
    q_1d = jnp.where(jnp.isnan(ctx.committors_1d), 0.0, ctx.committors_1d)
    S = ctx.projected_samples
    M = S.shape[0]

    out = []
    for mask in (ctx.in_A, ctx.in_B):
        idx = jnp.asarray(jnp.flatnonzero(jnp.asarray(mask)))
        n = int(idx.shape[0])
        if n < 50:
            logger.warning(
                "nitsche: basin sample count low (n=%d); the second-moment Gram will be noisy.", n
            )
        P = jnp.zeros((M, M), dtype=jnp.float64)
        for start in range(0, n, chunk):
            sub = idx[start : start + chunk]
            Q = _q_at(slice_coords, q_1d, S[:, sub])
            P = P + Q @ Q.T
            del Q
        out.append(P / max(n, 1))
    return out[0], out[1]


# --------------------------------------------------------------------------- #
# the (M+1) x (M+1) solve
# --------------------------------------------------------------------------- #
def solve_nitsche(
    G, PA, PB, a, b, valid_mask, *, beta_tilde=1e3, eta="auto", sample_weights=None, N=None
):
    """Solve the second-moment-penalised system; returns a weight dict.

    ``beta = beta_tilde * mean(diag(G)_valid)`` non-dimensionalises the penalty:
    ``G`` carries the units of ``<(dq/ds)^2>`` while ``P_A``, ``P_B`` are
    ``O(1)``, so ``beta_tilde`` is dimensionless and transfers across systems.
    """
    G = jnp.asarray(G, dtype=jnp.float64)
    PA = jnp.asarray(PA, dtype=jnp.float64)
    PB = jnp.asarray(PB, dtype=jnp.float64)
    a = jnp.asarray(a, dtype=jnp.float64)
    b = jnp.asarray(b, dtype=jnp.float64)
    M = G.shape[0]
    valid = jnp.asarray(valid_mask).astype(jnp.float64)
    n_valid = jnp.maximum(jnp.sum(valid), 1.0)

    eta_val, N_eff_val = _resolve_eta(eta, M, valid_mask, sample_weights, N, G=G)

    mask_2d = valid[:, None] * valid[None, :]
    diag_G = jnp.where(valid > 0, jnp.diag(G), 0.0)
    scale = jnp.sum(diag_G) / n_valid  # tr(G)_valid / M_valid
    scale = jnp.where(jnp.isfinite(scale) & (scale > 0), scale, 1.0)
    beta = beta_tilde * scale

    # Match EBMC's ridge convention exactly (eta x median valid diagonal of G)
    # so the two solvers differ only in the boundary treatment.
    med_diag = jnp.median(jnp.where(valid > 0, jnp.diag(G), jnp.inf))
    med_diag = jnp.where(jnp.isfinite(med_diag) & (med_diag > 0), med_diag, 1.0)

    K = (G + beta * (PA + PB)) * mask_2d
    # Invalid directions: identity row/col at the block's own scale, so w_j = 0
    # without distorting the conditioning of the valid block.
    K = K + jnp.diag((1.0 - valid) * jnp.maximum(scale, 1e-30))
    K = K + eta_val * med_diag * jnp.eye(M)

    a_m = a * valid
    b_m = b * valid

    H = jnp.zeros((M + 1, M + 1), dtype=jnp.float64)
    H = H.at[:M, :M].set(K)
    H = H.at[:M, M].set(beta * (a_m + b_m))
    H = H.at[M, :M].set(beta * (a_m + b_m))
    H = H.at[M, M].set(2.0 * beta)
    H = H + 1e-11 * jnp.trace(H) / (M + 1) * jnp.eye(M + 1)

    g = jnp.zeros(M + 1, dtype=jnp.float64)
    g = g.at[:M].set(beta * b_m)
    g = g.at[M].set(beta)

    x = jnp.linalg.solve(H, g)
    w = x[:M] * valid
    c = float(x[M])

    # Diagnostics. The BC residuals are the quantity beta actually controls;
    # mu_A / mu_B are the EBMC constraints, no longer imposed, so their
    # departure from (0, 1) is informative here rather than vacuous.
    mu_A = float(jnp.dot(a_m, w) + c)
    mu_B = float(jnp.dot(b_m, w) + c)
    resid_A = float(c**2 + 2.0 * c * jnp.dot(a_m, w) + w @ PA @ w)
    resid_B = float((c - 1.0) ** 2 + 2.0 * (c - 1.0) * jnp.dot(b_m, w) + w @ PB @ w)

    return {
        "w": w,
        "c": c,
        "a": a,
        "b": b,
        "beta_used": float(beta),
        "beta_tilde": float(beta_tilde),
        "gram_scale": float(scale),
        "bc_resid_A": max(resid_A, 0.0),
        "bc_resid_B": max(resid_B, 0.0),
        "mu_A_check": mu_A,
        "mu_B_check": mu_B,
        "sum_w": float(jnp.sum(w)),
        "dirichlet_energy_gram": float(w @ G @ w),
        "eta_used": float(eta_val),
        "N_eff": float(N_eff_val) if N_eff_val is not None else None,
    }


# --------------------------------------------------------------------------- #
# top-level solver
# --------------------------------------------------------------------------- #
def nitsche_weights(
    ctx,
    samples=None,
    sample_weights=None,
    *,
    beta_tilde: float = 1e3,
    tikhonov: float | str = "auto",
    gram_dtype: str = "float64",
    chunk: int = 8192,
) -> dict:
    """Top-level second-moment (Nitsche) solver.

    Assembles ``G``, the basin moments ``(a, b)`` and the basin second-moment
    Grams ``(P_A, P_B)``, then runs :func:`solve_nitsche`.

    The returned dict carries ``'c'`` and ``'q_bar'`` (zeros), so passing it to
    ``evaluate_committor`` / ``build_committor`` activates the centered-basis
    path: ``q_hat(x) = c + sum_j w_j q_j(theta_j . x)``.  Without ``'q_bar'``
    the evaluator would take the normalised-array path and silently rescale the
    solution by ``1 / sum(w)``.

    Args:
        ctx: ``WeightingContext`` carrying ``in_A``, ``in_B`` and
            ``projected_samples``.
        samples: ``(N, dim)`` samples; only needed when the context has no
            stored projections.
        sample_weights: ``(N,)`` optional weights, used for the Gram and for the
            ``'auto'`` ridge.  The basin moments follow ``compute_basin_moments``
            and stay unweighted.
        beta_tilde: dimensionless penalty scale, ``beta = beta_tilde * tr(G)/M``.
        tikhonov: ridge on ``G``, matching EBMC's convention.
        gram_dtype: dtype of the dominant ``(M, N) x (N, M)`` product.
        chunk: sample-chunk size for the second-moment accumulation.
    """
    if not jax.config.read("jax_enable_x64"):
        raise ValueError(
            "nitsche requires jax_enable_x64=True. Add "
            "`jax.config.update('jax_enable_x64', True)` before computing the result."
        )
    if ctx.in_A is None or ctx.in_B is None:
        raise ValueError(
            "nitsche_weights: ctx.in_A / ctx.in_B are missing; pass in_A / in_B "
            "to compute_sliced_committor so they propagate into the result."
        )

    if sample_weights is None:
        sample_weights = ctx.sample_weights

    if ctx.projected_samples is not None:
        projected_samples = ctx.projected_samples
        moments_ctx = ctx
    elif samples is not None:
        projected_samples = ctx.directions @ samples.T
        moments_ctx = ctx._replace(projected_samples=projected_samples)
    else:
        raise ValueError(
            "nitsche_weights: pass samples explicitly or use "
            "store_projected_samples=True in compute_sliced_committor()."
        )

    N = projected_samples.shape[1]
    W = sample_weights if sample_weights is not None else jnp.ones(N) / N

    F = _compute_derivative_matrix(ctx, projected_samples)
    cos_matrix = ctx.cos_matrix if ctx.cos_matrix is not None else ctx.directions @ ctx.directions.T
    matmul_dtype = jnp.dtype(gram_dtype)
    G = _assemble_gram_matrix(F.astype(matmul_dtype), W.astype(matmul_dtype), cos_matrix)
    del F

    a, b = compute_basin_moments(moments_ctx)
    PA, PB = compute_basin_second_moments(moments_ctx, chunk=chunk)

    result = solve_nitsche(
        G,
        PA,
        PB,
        a,
        b,
        ctx.valid_mask,
        beta_tilde=beta_tilde,
        eta=tikhonov,
        sample_weights=sample_weights,
        N=N,
    )
    result["G"] = G
    result["q_bar"] = jnp.zeros_like(result["w"])
    return result
