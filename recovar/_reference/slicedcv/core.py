"""
Core sliced committor (numpy port of the paper's method).

    qbar(x) = c + sum_j w_j q_j(theta_j . x)

    G_jk    = (theta_j' D theta_k) * E_pi[ q_j' q_k' ]
    a_j     = mu_A[qtilde_j],   b_j = mu_B[qtilde_j]
    w_dual  = G^{-1}(b - a),    M_gap = (b-a)' w_dual
    w*      = w_dual / M_gap,   c* = -a' w*
    nu_AB  <= 1 / M_gap                        (population-level bound)

Uses the dual (unconstrained) form of NOTE_dual_identity.md throughout:
    h(w) = 2 (b-a)'w - w'Gw,   max_w h = 1/nu_AB,  argmax = G^{-1}(b-a)
This makes held-out evaluation of h trivial (see recovar.cv_dual).

D = I in feature space throughout (as in the paper's molecular examples).
"""
import numpy as np
from scipy.linalg import solve_banded, cho_factor, cho_solve


# ---------------------------------------------------------------------------
# Directions
# ---------------------------------------------------------------------------

def isotropic_directions(M, d, rng):
    rng = np.random.default_rng(rng)
    v = rng.standard_normal((M, d))
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def lda_directions(M, X, in_A, in_B, rng, kappa=20.0):
    """Power-spherical-like sampler concentrated on the Fisher LDA axis.

    mu = normalised (mean_B - mean_A) whitened by the pooled within-class
    covariance; directions are drawn as mu tilted by an isotropic
    perturbation whose size is set by `kappa` (larger = tighter).
    """
    rng = np.random.default_rng(rng)
    d = X.shape[1]
    mA = X[in_A].mean(0)
    mB = X[in_B].mean(0)
    Sw = np.cov(X[in_A].T) * in_A.sum() + np.cov(X[in_B].T) * in_B.sum()
    Sw = np.atleast_2d(Sw) / (in_A.sum() + in_B.sum())
    Sw += 1e-6 * np.trace(Sw) / d * np.eye(d)
    mu = np.linalg.solve(Sw, mB - mA)
    mu /= np.linalg.norm(mu)
    pert = rng.standard_normal((M, d)) / np.sqrt(kappa)
    v = mu[None, :] + pert
    return v / np.linalg.norm(v, axis=1, keepdims=True)


# ---------------------------------------------------------------------------
# Histogram smoothing with (optionally) position-dependent bandwidth
# ---------------------------------------------------------------------------

def _smooth_matrix(n_bins, bw_bins):
    """Mass-conserving Gaussian smoothing matrix.

    bw_bins: scalar or (n_bins,) per-source-bin bandwidth in units of bins.
    Returns S with smoothed = S @ counts.
    """
    bw = np.broadcast_to(np.asarray(bw_bins, dtype=float), (n_bins,))
    i = np.arange(n_bins)
    D = i[:, None] - i[None, :]                     # (target, source)
    sig = np.maximum(bw[None, :], 1e-6)
    K = np.exp(-0.5 * (D / sig) ** 2)
    K /= K.sum(axis=0, keepdims=True)               # conserve source mass
    return K


def _histograms(s, in_A, in_B, edges, bw_bins, w=None):
    n_bins = len(edges) - 1
    if w is None:
        h_all, _ = np.histogram(s, bins=edges)
        h_A, _ = np.histogram(s[in_A], bins=edges)
        h_B, _ = np.histogram(s[in_B], bins=edges)
    else:
        h_all, _ = np.histogram(s, bins=edges, weights=w)
        h_A, _ = np.histogram(s[in_A], bins=edges, weights=w[in_A])
        h_B, _ = np.histogram(s[in_B], bins=edges, weights=w[in_B])
    if bw_bins is not None:
        S = _smooth_matrix(n_bins, bw_bins)
        h_all = S @ h_all.astype(float)
        h_A = S @ h_A.astype(float)
        h_B = S @ h_B.astype(float)
    return h_all.astype(float), h_A.astype(float), h_B.astype(float)


# ---------------------------------------------------------------------------
# 1D reaction-diffusion slice profile
# ---------------------------------------------------------------------------

def rd_profile(rho, rho_A, rho_B, h, kappa_rel=1e6):
    """Solve d/ds[rho q'] = kappa [rho_A q - rho_B (1-q)] with Neumann ends.

    kappa is set relative to the diffusion scale so the stiff limit is
    reached in a scale-free way:  kappa = kappa_rel * max(rho)/h^2 / max(rho_A+rho_B).
    """
    n = len(rho)
    rho = np.maximum(rho, 1e-300)
    face = 0.5 * (rho[:-1] + rho[1:]) / h**2        # (n-1,)

    absorb = rho_A + rho_B
    scale_diff = face.max() if face.size else 1.0
    scale_abs = absorb.max() if absorb.max() > 0 else 1.0
    kappa = kappa_rel * scale_diff / scale_abs

    lower = np.zeros(n); upper = np.zeros(n)
    lower[1:] = face
    upper[:-1] = face
    diag = -(lower + upper) - kappa * absorb
    rhs = -kappa * rho_B

    # Guard: with no absorption anywhere the pure-Neumann operator is
    # singular (constant null space). This happens for random directions in
    # high d that see no state density. Add a tiny well-scaled ridge and
    # fall back to a constant profile if it is still singular.
    dead = np.abs(diag) < 1e-300
    diag[dead] = -1.0
    rhs[dead] = 0.0
    # Only the absorption-free case is singular (pure Neumann Laplacian has a
    # constant null space). Scale the ridge to the DIFFUSION scale -- scaling
    # it to max|diag| would be dominated by the stiff absorption term and
    # would swamp the diffusion-scale entries.
    if absorb.max() <= 0:
        diag = diag - 1e-8 * max(scale_diff, 1e-300)

    ab = np.zeros((3, n))
    ab[0, 1:] = upper[:-1]
    ab[1, :] = diag
    ab[2, :-1] = lower[1:]
    try:
        q = solve_banded((1, 1), ab, rhs)
    except Exception:
        return np.full(n, 0.5)
    if not np.all(np.isfinite(q)):
        return np.full(n, 0.5)
    return np.clip(q, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Slice basis
# ---------------------------------------------------------------------------

class SliceBasis:
    """Per-direction 1D profiles q_j(s) and derivatives q_j'(s) on a grid."""

    def __init__(self, thetas, grids, q, qp):
        self.thetas = thetas            # (M, d)
        self.grids = grids              # (M, n_bins)
        self.q = q                      # (M, n_bins)
        self.qp = qp                    # (M, n_bins)
        self.M, self.n_bins = q.shape

    def subset(self, idx):
        return SliceBasis(self.thetas[idx], self.grids[idx],
                          self.q[idx], self.qp[idx])

    def eval_at(self, S):
        """Interpolate q and q' at projections S (M, N).

        Grids are uniform per direction (linspace), so this is fully
        vectorised: no Python loop over directions.
        """
        nb = self.n_bins
        g0 = self.grids[:, 0][:, None]
        h = (self.grids[:, 1] - self.grids[:, 0])[:, None]
        t = (S - g0) / h
        i = np.clip(np.floor(t), 0, nb - 2).astype(np.int32)
        f = np.clip(t - i, 0.0, 1.0)
        lo = t < 0
        hi = t > nb - 1

        def gather(Y):
            y0 = np.take_along_axis(Y, i, axis=1)
            y1 = np.take_along_axis(Y, i + 1, axis=1)
            return y0 * (1 - f) + y1 * f

        qv = gather(self.q)
        qv = np.where(lo, self.q[:, :1], np.where(hi, self.q[:, -1:], qv))
        qpv = gather(self.qp)
        qpv = np.where(lo | hi, 0.0, qpv)
        return qv.astype(np.float32), qpv.astype(np.float32)


def build_basis(thetas, X, in_A, in_B, n_bins=200, bw_bins=1.5,
                kappa_rel=1e6, bw_fn=None, pad=0.02, sample_weights=None):
    """Build slice profiles for every direction.

    bw_fn: optional callable(j, edges, h_all, h_A, h_B) -> per-bin bandwidth,
           used by the adaptive-bandwidth experiment.
    """
    M = thetas.shape[0]
    grids = np.empty((M, n_bins))
    Q = np.empty((M, n_bins))
    QP = np.empty((M, n_bins))
    S_all = X @ thetas.T                            # (N, M)

    for j in range(M):
        s = S_all[:, j]
        lo, hi = s.min(), s.max()
        span = hi - lo
        lo -= pad * span; hi += pad * span
        edges = np.linspace(lo, hi, n_bins + 1)
        centers = 0.5 * (edges[:-1] + edges[1:])
        h = centers[1] - centers[0]

        bw = bw_bins
        if bw_fn is not None:
            bw = bw_fn(j, edges, s, in_A, in_B)
        h_all, h_A, h_B = _histograms(s, in_A, in_B, edges, bw,
                                      w=sample_weights)

        rho = h_all / (h_all.sum() * h + 1e-300)
        rA = h_A / (h_A.sum() * h + 1e-300) if h_A.sum() > 0 else h_A
        rB = h_B / (h_B.sum() * h + 1e-300) if h_B.sum() > 0 else h_B

        q = rd_profile(rho, rA, rB, h, kappa_rel=kappa_rel)
        qp = np.gradient(q, h)

        grids[j] = centers
        Q[j] = q
        QP[j] = qp

    return SliceBasis(thetas, grids, Q, QP)


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------

def assemble(basis, X, in_A, in_B, sample_weights=None, block=256,
             return_parts=False):
    """Gram matrix G, basin moments a, b.

    G_jk = (theta_j . theta_k) * sum_n W_n q_j'(s_j^n) q_k'(s_k^n)
    """
    M = basis.M
    N = X.shape[0]
    W = np.full(N, 1.0 / N) if sample_weights is None else np.asarray(sample_weights)
    W = W / W.sum()

    C = np.zeros((M, M))
    a = np.zeros(M); b = np.zeros(M)
    WA = W * in_A; WB = W * in_B
    WA = WA / WA.sum(); WB = WB / WB.sum()

    Fs = []
    for j0 in range(0, M, block):
        j1 = min(j0 + block, M)
        sub = basis.subset(np.arange(j0, j1))
        S = (X @ sub.thetas.T).T                    # (blk, N)
        qv, qpv = sub.eval_at(S)
        Fs.append((qv, qpv))
        a[j0:j1] = qv @ WA
        b[j0:j1] = qv @ WB

    for i, (qi, pi) in enumerate(Fs):
        i0 = i * block
        Pi = pi * np.sqrt(W)[None, :].astype(np.float32)
        for k, (qk, pk) in enumerate(Fs):
            if k < i:
                continue
            k0 = k * block
            Pk = pk * np.sqrt(W)[None, :].astype(np.float32)
            blk = (Pi.astype(np.float64) @ Pk.astype(np.float64).T)
            C[i0:i0 + blk.shape[0], k0:k0 + blk.shape[1]] = blk
            if k != i:
                C[k0:k0 + blk.shape[1], i0:i0 + blk.shape[0]] = blk.T

    Theta = basis.thetas @ basis.thetas.T
    G = Theta * C
    if return_parts:
        return G, a, b, C, Theta
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


def evaluate(basis, w, c, Xq, block=256):
    """qbar(x) = c + sum_j w_j q_j(theta_j . x)."""
    out = np.full(Xq.shape[0], c, dtype=np.float64)
    for j0 in range(0, basis.M, block):
        j1 = min(j0 + block, basis.M)
        sub = basis.subset(np.arange(j0, j1))
        S = (Xq @ sub.thetas.T).T
        qv, _ = sub.eval_at(S)
        out += w[j0:j1] @ qv.astype(np.float64)
    return out


def fit(basis, X, in_A, in_B, tikhonov=1e-3, sample_weights=None):
    """Full pipeline: assemble + dual solve. Returns a result dict."""
    G, a, b = assemble(basis, X, in_A, in_B, sample_weights)
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
