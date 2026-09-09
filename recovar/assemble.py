"""
Assembly + dual solve for the sliced committor (JAX-aware port of
``_reference/slicedcv/core.py``).

    G_jk    = (theta_j' D theta_k) * E_pi[ q_j' q_k' ]
    a_j     = mu_A[qtilde_j],   b_j = mu_B[qtilde_j]
    w_dual  = G^{-1}(b - a),    M_gap = (b-a)' w_dual
    w*      = w_dual / M_gap,   c* = -a' w*
    nu_AB  <= 1 / M_gap                        (population-level bound)

Uses the dual (unconstrained) form of NOTE_dual_identity.md throughout:
    h(w) = 2 (b-a)'w - w'Gw,   max_w h = 1/nu_AB,  argmax = G^{-1}(b-a)
This makes held-out evaluation of h trivial (see the cv/dual diagnostics).

D = I in feature space throughout (as in the paper's molecular examples).

Delta vs the reference: assemble() streams over frame blocks with float64
accumulators (one pass, O(M^2 + M*frame_block) memory) instead of storing
every (qv, qpv) block (O(N*M)). Gram matrices and basin moments are float64.
"""
import numpy as np
from scipy.linalg import cho_factor, cho_solve

import jax
import jax.numpy as jnp

from .basis import _eval_at_kernel


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------

@jax.jit
def _accumulate_block(thetas, grids, q, qp, Xb, Wb, WAb, WBb):
    """Fused projection + interpolation + Gram/moment accumulation for one
    frame block. Mirrors the reference's rounding: qv/qpv are float32,
    qp is scaled by sqrt(W) in float32, the matmuls run in float64."""
    S = (Xb @ thetas.T).T                            # (M, F) float64
    qv, qpv = _eval_at_kernel(grids, q, qp, S)       # float32
    a_blk = qv.astype(jnp.float64) @ WAb
    b_blk = qv.astype(jnp.float64) @ WBb
    Pi = qpv * jnp.sqrt(Wb).astype(jnp.float32)[None, :]
    P64 = Pi.astype(jnp.float64)
    C_blk = P64 @ P64.T
    return C_blk, a_blk, b_blk


def assemble(basis, X, in_A, in_B, sample_weights=None, frame_block=8192):
    """Gram matrix G, basin moments a, b.

    G_jk = (theta_j . theta_k) * sum_n W_n q_j'(s_j^n) q_k'(s_k^n)

    Streaming: one pass over frame blocks, accumulating C, a, b in float64;
    the direction-cosine matrix Theta = thetas @ thetas.T is applied at the
    end, exactly as in the reference.
    """
    M = basis.M
    N = X.shape[0]
    W = (np.full(N, 1.0 / N) if sample_weights is None
         else np.asarray(sample_weights, dtype=np.float64))
    W = W / W.sum()

    WA = W * in_A; WB = W * in_B
    WA = WA / WA.sum(); WB = WB / WB.sum()

    C = np.zeros((M, M))
    a = np.zeros(M); b = np.zeros(M)
    for n0 in range(0, N, frame_block):
        n1 = min(n0 + frame_block, N)
        C_blk, a_blk, b_blk = _accumulate_block(
            basis.thetas, basis.grids, basis.q, basis.qp,
            X[n0:n1], W[n0:n1], WA[n0:n1], WB[n0:n1])
        C += np.asarray(C_blk)
        a += np.asarray(a_blk)
        b += np.asarray(b_blk)

    Theta = basis.thetas @ basis.thetas.T
    G = Theta * C
    return G, a, b


def assemble_slice_values(basis, X, block=256):
    """Return (qtilde, qprime) as (M, N) float32 arrays."""
    M = basis.M
    N = X.shape[0]
    QV = np.empty((M, N), dtype=np.float32)
    QP = np.empty((M, N), dtype=np.float32)
    for j0 in range(0, M, block):
        j1 = min(j0 + block, M)
        sub = basis.subset(np.arange(j0, j1))
        S = (X @ sub.thetas.T).T
        QV[j0:j1], QP[j0:j1] = sub.eval_at(S)
    return QV, QP


# ---------------------------------------------------------------------------
# Solve
# ---------------------------------------------------------------------------

def regularize(G, tikhonov=1e-3):
    """Hand-set ridge relative to the median diagonal (the current default)."""
    med = np.median(np.diag(G))
    med = med if np.isfinite(med) and med > 0 else 1.0
    return G + tikhonov * med * np.eye(G.shape[0])


def solve_dual(G, delta, tikhonov=1e-3, G_is_regularised=False):
    """w_dual = G^{-1} delta ; M_gap = delta' w_dual ; h* = M_gap.

    Returns dict with w (the normalised w* of the paper), w_dual, M_gap, nu_hat.
    """
    Gr = G if G_is_regularised else regularize(G, tikhonov)
    try:
        cf = cho_factor(Gr, lower=True)
        w_dual = cho_solve(cf, delta)
    except np.linalg.LinAlgError:
        w_dual = np.linalg.lstsq(Gr, delta, rcond=None)[0]
    M_gap = float(delta @ w_dual)
    w = w_dual / M_gap if abs(M_gap) > 1e-300 else w_dual
    return dict(w=w, w_dual=w_dual, M_gap=M_gap,
                nu_hat=1.0 / M_gap if abs(M_gap) > 1e-300 else np.inf,
                G_reg=Gr)


def dual_objective(w_dual, G, delta):
    """h(w) = 2 delta'w - w'Gw.   Max over w is M_gap = 1/nu_AB."""
    return float(2.0 * delta @ w_dual - w_dual @ (G @ w_dual))


def evaluate(basis, w, c, Xq, block=65536):
    """qbar(x) = c + sum_j w_j q_j(theta_j . x). Chunked over query points."""
    out = np.full(Xq.shape[0], c, dtype=np.float64)
    for n0 in range(0, Xq.shape[0], block):
        n1 = min(n0 + block, Xq.shape[0])
        S = (Xq[n0:n1] @ basis.thetas.T).T
        qv, _ = basis.eval_at(S)
        out[n0:n1] += w @ qv.astype(np.float64)
    return out


def fit(basis, X, in_A, in_B, tikhonov=1e-3, sample_weights=None):
    """Full pipeline: assemble + dual solve. Returns a result dict."""
    G, a, b = assemble(basis, X, in_A, in_B, sample_weights=sample_weights)
    delta = b - a
    sol = solve_dual(G, delta, tikhonov)
    sol.update(G=G, a=a, b=b, delta=delta,
               c=float(-a @ sol['w']), basis=basis)
    return sol


# ---------------------------------------------------------------------------
# Error metrics
# ---------------------------------------------------------------------------

def transition_rmse(q_hat, q_ref, in_A, in_B):
    m = ~(in_A | in_B)
    return float(np.sqrt(np.mean((q_hat[m] - q_ref[m]) ** 2)))


def clip01(x):
    return np.clip(x, 0.0, 1.0)
