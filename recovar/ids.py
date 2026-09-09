"""IDS -- Iterated Direction Sampling (path-scatter subspace refinement).

The flagship RECOVAR transfer: a rotation-equivariant, system-knowledge-free
replacement for both the coordinate feature mask and the tuned Fisher-LDA
direction sampler. JAX-aware port of the IDS section of the numpy prototype
(``_reference/slicedcv/recovar.py``). Only the validated exact, linear-in-d
pipeline is kept (the t21 configuration), plus the dense path-scatter /
dense-Horn reference implementations used as ground truth by the tests.

Why path scatter (and not the Dirichlet structure tensor)
---------------------------------------------------------
The Dirichlet structure tensor of a first-pass ansatz is rank-1 dominated
(measured lam_2/lam_1 ~ 0.03): a monotone trial committor only varies along
one direction, so its gradient cannot reveal the curvature direction. The
chicken-and-egg is broken by using the first pass only to ORDER frames along
the reaction, not to supply a gradient. Bin frames by the first-pass
committor and take the between-bin scatter: this is multi-class Fisher
discriminant analysis along reaction progress. Two classes (A vs B) give
rank 1 -- the ordinary LDA axis. L bins along a CURVED path give rank up to
L-1, because the path's tangent differs at different points along it. That
is exactly the reactive subspace. Whitening by the pooled within-bin
covariance makes the estimator equivariant under any invertible linear map
of feature space, not merely rotations -- badly scaled features are handled
automatically.

Lessons inherited from the retired dense / low-rank variants
------------------------------------------------------------
(Recorded here because the code that carried them was retired; every one of
them was paid for with a measured regression.)

* Direction map-back uses W', not W. Whitening acts on POINTS, y = W x with
  W Sw W' = I. A direction is a linear FUNCTIONAL: theta'x = (W^{-T}theta)'y,
  so a whitened direction u maps back to theta = W' u. Since W'W = Sw^{-1}
  for ANY valid whitener, the W' map-back is invariant to which whitener you
  picked, and the rank-1 case reproduces the LDA axis Sw^{-1}(m_B - m_A).
  Applying W instead gives W W (m_B - m_A), which is whitener-dependent and
  wrong. The dense path was accidentally correct because Sw^{-1/2} is
  symmetric. (The exact pipeline below sidesteps whitening entirely: the
  generalised eigenvectors of Sb v = lambda Sw v ARE the feature-space
  discriminant directions -- no whitener, no map-back.)

* Marchenko-Pastur shrinkage of a within-bin spectrum must be SOFT, not a
  hard edge cut. Thresholding at the MP edge keeps zero spikes once
  d >~ 200, collapsing the whitener to pure diagonal and throwing away the
  modest-but-systematic anisotropy that actually tilts the discriminant.
  The spiked inversion lambda -> ell already maps near-bulk eigenvalues back
  to the bulk, so it IS a soft threshold: keep every direction and let the
  map decide how much of each to trust.

* shrink=0.1, and Sw must be EXACT. Sw^{-1} is applied to the bin means, so
  under-regularising it amplifies noise in the low-variance directions --
  invisible at d = 48 and fatal by d = 384 (shrink=0.02 lost the whole IDS
  gain there). And every low-rank approximation of Sw was refuted in t15:
  with an EXACT Sw the linear-in-d pipeline matches or beats the dense
  version (0.0080/0.0108/0.0161 vs 0.0105/0.0114/0.0176 at d = 96/192/384),
  while every approximate Sw model loses a factor ~2 -- randomized PCA
  cannot isolate the informative directions of a near-flat within-bin
  spectrum (power iteration needs a spectral gap, and there isn't one).
  So do not approximate Sw at all: solve with it via CG matvecs.

The exact, linear-in-d formulation
----------------------------------
The algorithm never wants Sw^{-1/2} as an operator. It wants the generalised
eigenproblem

      Sb v = lambda Sw v          (multi-class LDA)

and Sb = A'A has rank <= L-1 by construction. So solving Sw Y = A' for the
L bin-mean vectors reduces everything to an L x L eigenproblem, and the
generalised eigenvectors ARE the feature-space discriminant directions.
Sw x = a is solved by Jacobi-preconditioned block conjugate gradients using
only Sw-matvecs, which stream over frames in O(rows d). Total cost
O(k L N d) with k ~ 10-40 CG iterations: linear in d, exact.

JAX is used exactly where the FLOPs are: the Sw matvec Z'(Z V) + eps V is a
jitted kernel over the float32 within-group-centred matrix Z (float64
epsilon term), matching the numpy prototype within float32 tolerance. The
CG loop, histogramming, small eigenproblems and rng draws stay in numpy so
the prototype's random stream is reproduced draw for draw.

Cross-module imports (.basis / .assemble / .splits / .regularize) are done
lazily inside the driver functions, so the subspace machinery in this module
is importable and testable on its own.
"""

import numpy as np
from scipy.linalg import eigh

import jax
import jax.numpy as jnp


# ===========================================================================
# Path labels and the dense (reference) path scatter
# ===========================================================================

def path_labels(qhat, in_A, in_B, w, n_bins):
    """Weighted-quantile bin labels along reaction progress; A and B terminal."""
    N = len(qhat)
    lab = np.full(N, -1)
    tr = ~(in_A | in_B)
    if tr.sum() > n_bins * 5:
        q = qhat[tr]; wt = w[tr]
        o = np.argsort(q)
        cw = np.cumsum(wt[o]); cw /= cw[-1]
        edges = np.interp(np.linspace(0, 1, n_bins + 1), cw, q[o])
        edges[0] -= 1e-9; edges[-1] += 1e-9
        lab[tr] = np.clip(np.searchsorted(edges, q, side='right') - 1,
                          0, n_bins - 1) + 1
    lab[in_A] = 0
    lab[in_B] = n_bins + 1
    return lab


def path_scatter_w(X, qhat, in_A, in_B, w=None, n_bins=8, shrink=0.1,
                   whiten=True, labels=None):
    """Weight-aware between-bin scatter. w = per-frame equilibrium weights
    (e.g. MBAR), normalised internally; None means uniform."""
    N, d = X.shape
    w = np.full(N, 1.0 / N) if w is None else np.asarray(w, float)
    w = w / w.sum()
    if labels is None:
        labels = path_labels(qhat, in_A, in_B, w, n_bins)
    groups = [g for g in np.unique(labels) if g >= 0 and (labels == g).sum() >= d + 2]
    if len(groups) < 3:
        return np.eye(d), np.eye(d), labels
    ms, ps, Sw = [], [], np.zeros((d, d))
    for g in groups:
        m = labels == g
        wl = w[m]; sw = wl.sum()
        if sw <= 0:
            continue
        Z = X[m]
        mu = (wl[:, None] * Z).sum(0) / sw
        Zc = Z - mu
        Sw += (wl[:, None] * Zc).T @ Zc
        ms.append(mu); ps.append(sw)
    ms = np.array(ms); ps = np.array(ps); ps = ps / ps.sum()
    Sw /= max(w[labels >= 0].sum(), 1e-300)
    mbar = (ps[:, None] * ms).sum(0)
    Mc = ms - mbar
    Sb = (Mc * ps[:, None]).T @ Mc
    if not whiten:
        return Sb, np.eye(d), labels
    Sw = Sw + shrink * (np.trace(Sw) / d) * np.eye(d)
    lam, V = eigh(Sw)
    lam = np.maximum(lam, 1e-12 * max(lam.max(), 1e-300))
    # Wh = Sw^{-1/2}. A direction u in whitened space corresponds to the
    # feature-space direction Sw^{-1/2} u (since theta.x = theta' Sw^{1/2} y).
    # Mapping back with Sw^{+1/2} would be wrong -- and would amplify exactly
    # the high-variance nuisance directions whitening is meant to suppress.
    # With Sw^{-1/2} the rank-1 case reproduces the LDA axis Sw^{-1}(m_B - m_A).
    Wh = (V / np.sqrt(lam)) @ V.T
    return Wh @ Sb @ Wh, Wh, labels


def select_rank_permutation(X, labels, Wh, w=None, n_perm=25, q=0.95,
                            rmax=None, rng=None):
    """Horn parallel analysis: compare the observed whitened between-bin
    spectrum with the spectrum obtained after shuffling the group labels
    among frames, holding group sizes and the whitener fixed.

    Uses no split, no ordering and no extra data -- only the sample set and
    the labels the method already has. r = largest r with
    lambda_i > q-quantile of the null lambda_i for every i <= r.
    """
    rng = np.random.default_rng(rng)
    N, d = X.shape
    w = np.full(N, 1.0 / N) if w is None else np.asarray(w, float) / np.sum(w)
    rmax = rmax or max(1, min(d, 20))
    Y = X @ Wh.T                                   # whitened coordinates

    def spec(lb):
        gs = [g for g in np.unique(lb) if g >= 0]
        ms, ps = [], []
        for g in gs:
            m = lb == g
            sw = w[m].sum()
            if sw <= 0:
                continue
            ms.append((w[m][:, None] * Y[m]).sum(0) / sw); ps.append(sw)
        ms = np.array(ms); ps = np.array(ps); ps = ps / ps.sum()
        Mc = ms - (ps[:, None] * ms).sum(0)
        Sb = (Mc * ps[:, None]).T @ Mc
        return np.sort(np.linalg.eigvalsh(Sb))[::-1]

    obs = spec(labels)
    keep = labels >= 0
    idx = np.flatnonzero(keep)
    null = np.zeros((n_perm, d))
    for t in range(n_perm):
        lb = labels.copy()
        lb[idx] = labels[rng.permutation(idx)]
        null[t] = spec(lb)
    thr = np.quantile(null, q, axis=0)
    r = 0
    for i in range(min(rmax, d)):
        if obs[i] > thr[i]:
            r = i + 1
        else:
            break
    return max(r, 1), obs, thr


# ===========================================================================
# Exact Sw as an operator (JAX matvec), block CG, generalised eigenproblem
# ===========================================================================

@jax.jit
def _sw_matvec_kernel(Z, V32, V64, eps):
    """out = Z'(Z V) in float32 (matching the numpy prototype's sgemm),
    cast to float64, plus the exact float64 ridge eps * V."""
    out = (Z.T @ (Z @ V32)).astype(jnp.float64)
    return out + eps * V64


class _SwOperator:
    """Streaming within-group covariance operator. Sw v in O(rows*d).

    The float32 group-centred, sqrt(w)-scaled matrix Z lives on the JAX
    device; matvecs run through the jitted kernel above. ``.Z``, ``.diag``
    and ``.eps`` keep the prototype's numpy attributes (``sample_inv_sw``
    consumes ``.Z`` directly).
    """

    def __init__(self, X, labels, w, n_sub=None, rng=None, shrink=0.1):
        # shrink must match what the dense path used (0.1). Sw^{-1} is applied
        # to the bin means, so under-regularising it amplifies noise in the
        # low-variance directions -- invisible at d = 48 and fatal by d = 384.
        rng = np.random.default_rng(rng)
        keep = np.flatnonzero(labels >= 0)
        if n_sub is not None and len(keep) > n_sub:
            keep = rng.choice(keep, size=n_sub, replace=False)
        lab = labels[keep]
        ww = (w[keep] if w is not None else np.full(len(keep), 1.0))
        ww = ww / ww.sum()
        d = X.shape[1]
        Xk = X[keep]
        Z = np.empty((len(keep), d), dtype=np.float32)
        for g in np.unique(lab):
            m = lab == g
            sw = ww[m].sum()
            mu = (ww[m][:, None] * Xk[m]).sum(0) / max(sw, 1e-300)
            Z[m] = (Xk[m] - mu).astype(np.float32)
        self.Z = Z * np.sqrt(ww)[:, None].astype(np.float32)
        self.d = d
        self.diag = (self.Z.astype(np.float64) ** 2).sum(0)
        self.eps = shrink * self.diag.mean()
        self.diag = self.diag + self.eps
        self._Zj = jnp.asarray(self.Z)

    def matvec(self, V):
        """V: (d,) or (d,k)."""
        V = np.asarray(V)
        out = _sw_matvec_kernel(self._Zj, V.astype(np.float32),
                                V.astype(np.float64), self.eps)
        return np.asarray(out)


def _block_cg(op, B, Minv, tol=1e-8, maxiter=200):
    """CG on Sw X = B for each column of B (block, Jacobi-preconditioned)."""
    X = np.zeros_like(B)
    R = B - op.matvec(X)
    Zk = Minv[:, None] * R
    P = Zk.copy()
    rz = (R * Zk).sum(0)
    b0 = np.maximum(np.linalg.norm(B, axis=0), 1e-300)
    it = 0
    for it in range(1, maxiter + 1):
        AP = op.matvec(P)
        pap = (P * AP).sum(0)
        pap = np.where(np.abs(pap) < 1e-300, 1e-300, pap)
        al = rz / pap
        X += al[None, :] * P
        R -= al[None, :] * AP
        if np.max(np.linalg.norm(R, axis=0) / b0) < tol:
            break
        Zk = Minv[:, None] * R
        rz_new = (R * Zk).sum(0)
        P = Zk + (rz_new / np.where(np.abs(rz) < 1e-300, 1e-300, rz))[None, :] * P
        rz = rz_new
    return X, it


def path_directions_exact(X, qhat, in_A, in_B, w=None, n_bins=8, n_sub=None,
                          rng=None, tol=1e-8, maxiter=200, labels=None,
                          shrink=0.1):
    """Generalised eigenproblem Sb v = lambda Sw v, solved with L CG solves.

    Returns (lam, Theta, n_cg, labels, op): eigenvalues, the feature-space
    discriminant directions (d x k, k <= L-1, Sw-orthonormal), the CG
    iteration count, the bin labels and the Sw operator (reused by the rank
    rule and the floor sampler).
    """
    N, d = X.shape
    w = np.full(N, 1.0 / N) if w is None else np.asarray(w, float)
    w = w / w.sum()
    if labels is None:
        labels = path_labels(qhat, in_A, in_B, w, n_bins)
    op = _SwOperator(X, labels, w, n_sub=n_sub, rng=rng, shrink=shrink)

    gs = [g for g in np.unique(labels) if g >= 0]
    Mm, ps = [], []
    for g in gs:
        m = labels == g
        sw = w[m].sum()
        if sw <= 0:
            continue
        Mm.append((w[m][:, None] * X[m]).sum(0) / sw); ps.append(sw)
    Mm = np.array(Mm); ps = np.array(ps); ps = ps / ps.sum()
    Mc = Mm - (ps[:, None] * Mm).sum(0)
    A = Mc * np.sqrt(ps)[:, None]                        # L x d, Sb = A'A

    # Jacobi (diagonal) preconditioner: only has to precondition, not be right
    Minv = 1.0 / np.maximum(op.diag, 1e-300)
    Y, n_cg = _block_cg(op, A.T.copy(), Minv, tol=tol, maxiter=maxiter)  # d x L
    K = A @ Y                                            # L x L = A Sw^-1 A'
    K = 0.5 * (K + K.T)
    ev, W = np.linalg.eigh(K)
    o = np.argsort(-ev); ev, W = np.maximum(ev[o], 0.0), W[:, o]
    keep = ev > 1e-12 * max(ev[0], 1e-300)
    ev, W = ev[keep], W[:, keep]
    Theta = Y @ W                                        # d x k
    nrm = np.sqrt(np.maximum((Theta * op.matvec(Theta)).sum(0), 1e-300))
    return ev, Theta / nrm[None, :], n_cg, labels, op


def select_rank_perm_exact(X, labels, obs, w=None, n_perm=25, q=0.95,
                           rmax=None, rng=None, op=None, Minv=None,
                           tol=1e-6, maxiter=200, **kw):
    """Horn parallel analysis on the generalised spectrum, with Sw HELD FIXED.

    The null must shuffle only the between-bin structure. Rebuilding Sw from
    the shuffled labels makes the null's Sw ~ the TOTAL covariance (shuffled
    groups have no within-group structure left), which is much larger, which
    depresses the null generalised eigenvalues, which inflates the selected
    rank. Measured: ranks came out ~7 instead of ~4, and the extra noise
    directions cost a factor ~2 in RMSE at d = 192.
    """
    rng = np.random.default_rng(rng)
    N = X.shape[0]
    w = np.full(N, 1.0 / N) if w is None else np.asarray(w, float) / np.sum(w)
    idx = np.flatnonzero(labels >= 0)
    if Minv is None:
        Minv = 1.0 / np.maximum(op.diag, 1e-300)

    def spec(lb):
        gs = [g for g in np.unique(lb) if g >= 0]
        Mm, ps = [], []
        for g in gs:
            m = lb == g
            sw = w[m].sum()
            if sw <= 0:
                continue
            Mm.append((w[m][:, None] * X[m]).sum(0) / sw); ps.append(sw)
        Mm = np.array(Mm); ps = np.array(ps); ps = ps / ps.sum()
        Mc = Mm - (ps[:, None] * Mm).sum(0)
        A = Mc * np.sqrt(ps)[:, None]
        Y, _ = _block_cg(op, A.T.copy(), Minv, tol=tol, maxiter=maxiter)
        K = A @ Y
        return np.sort(np.linalg.eigvalsh(0.5 * (K + K.T)))[::-1]

    null = np.zeros((n_perm, len(obs)))
    for t in range(n_perm):
        lb = labels.copy()
        lb[idx] = labels[rng.permutation(idx)]
        sp = spec(lb)
        null[t, :min(len(sp), null.shape[1])] = sp[:null.shape[1]]
    thr = np.quantile(null, q, axis=0)
    r = 0
    for i in range(min(rmax or 20, len(obs))):
        if obs[i] > thr[i]:
            r = i + 1
        else:
            break
    return max(r, 1), obs, thr


# ===========================================================================
# The exact isotropic floor, without a square root
#
# The dense sampler draws theta ~ N(0, Sigma) with
#
#     Sigma = tau Sw^{-1} + sum_{i<=r} (lam_i - tau) th_i th_i'
#
# (th_i the Sw-orthonormal generalised eigenvectors, th' Sw th = 1). Sampling
# N(0, Sw^{-1}) looks like it needs Sw^{-1/2}. It does not:
#
#     u = Z'g + sqrt(eps) h   has Cov(u) = Z'Z + eps I = Sw     (free)
#     x = Sw^{-1} u           has Cov(x) = Sw^{-1} Sw Sw^{-1} = Sw^{-1}
#
# so one block-CG solve with M right-hand sides gives M exact draws from
# N(0, Sw^{-1}) using only Sw-matvecs. The floor is a background distribution,
# so a loose CG tolerance is fine.
# ===========================================================================

def sample_inv_sw(op, M, rng, tol=1e-3, maxiter=60, Minv=None):
    """M exact draws from N(0, Sw^{-1}); returns (d x M, n_cg).

    u = Z'g + sqrt(eps) h has Cov(u) = Z'Z + eps I = Sw, so x = Sw^{-1} u
    has Cov(x) = Sw^{-1} -- exact draws from only Sw-matvecs.
    """
    rng = np.random.default_rng(rng)
    rows, d = op.Z.shape
    g = rng.standard_normal((rows, M)).astype(np.float32)
    U = (op.Z.T @ g).astype(np.float64)
    U += np.sqrt(op.eps) * rng.standard_normal((d, M))
    if Minv is None:
        Minv = 1.0 / np.maximum(op.diag, 1e-300)
    return _block_cg(op, U, Minv, tol=tol, maxiter=maxiter)


def directions_exact_floor(lam, Theta, r, op, M, rng, tol=1e-3, maxiter=60):
    """theta ~ N(0, tau Sw^{-1} + sum_{i<=r}(lam_i - tau) th_i th_i'),
    i.e. exactly the dense sampler, built from matvecs only."""
    rng = np.random.default_rng(rng)
    d = op.d
    k = Theta.shape[1]
    r = int(np.clip(r, 1, k))
    # tau is the average of the DISCARDED spectrum over the whole space:
    # sum(lam[r:]) / (d - r), not the mean of the <= L-1 nonzero generalised
    # eigenvalues. The scatter has rank <= L-1, so the other d-L+1 eigenvalues
    # are zero and must be counted. Averaging over ~9 entries instead of ~186
    # makes the floor ~60x too large at d = 192, which flattens the
    # amplification of the informative directions back towards isotropic.
    tail = lam[r:]
    tau = (tail.sum() / max(d - r, 1)) if len(tail) else 0.0
    tau = max(tau, 1e-8 * max(lam[0], 1e-300))
    Xf, n_cg = sample_inv_sw(op, M, rng, tol=tol, maxiter=maxiter)   # d x M
    Y = np.sqrt(tau) * Xf.T                                          # M x d
    amp = np.sqrt(np.maximum(lam[:r] - tau, 0.0))
    Y += (rng.standard_normal((M, r)) * amp[None, :]) @ Theta[:, :r].T
    n = np.linalg.norm(Y, axis=1, keepdims=True)
    n[n[:, 0] <= 0] = 1.0
    return Y / n, n_cg


# ===========================================================================
# Feature mask (two-pass) -- kept for the coordinate-aligned baseline
# ===========================================================================

def feature_importance(basis, delta, w=None):
    """Per-feature contribution to the state separation.

    imp_k = sum_j |delta_j| * theta_jk^2   (normalised) -- how much of the
    A/B separating signal each feature carries, read off the first pass.
    """
    v = np.abs(delta) if w is None else np.abs(delta * w)
    imp = (v[:, None] * basis.thetas ** 2).sum(0)
    return imp / imp.sum()


def masked_directions(M, mask, d, rng):
    """Isotropic directions supported only on the masked features."""
    rng = np.random.default_rng(rng)
    k = int(mask.sum())
    v = np.zeros((M, d))
    v[:, mask] = rng.standard_normal((M, k))
    return v / np.linalg.norm(v, axis=1, keepdims=True)


# ===========================================================================
# Drivers
# ===========================================================================

def _fit_pass_w(thetas, X, in_A, in_B, basis_kw, split, w=None, n_bands=12):
    """One pass: build slices (weighted), halfset-eigen-regularise G, dual solve.

    The two halfset Grams are assembled on the index subsets of ``split``;
    ``assemble`` renormalises the sample weights internally, so the weights
    are renormalised within each half (matching the prototype).
    """
    from .basis import build_basis
    from .assemble import assemble, solve_dual
    from .regularize import halfset_eigen_regularize
    bs = build_basis(thetas, X, in_A, in_B, sample_weights=w, **basis_kw)
    i1, i2 = split
    w1 = None if w is None else w[i1]
    w2 = None if w is None else w[i2]
    G1, *_ = assemble(bs, X[i1], in_A[i1], in_B[i1], sample_weights=w1)
    G2, *_ = assemble(bs, X[i2], in_A[i2], in_B[i2], sample_weights=w2)
    Gr, info = halfset_eigen_regularize(G1, G2, n_bands=n_bands)
    G, a, b = assemble(bs, X, in_A, in_B, sample_weights=w)
    sol = solve_dual(Gr, b - a, G_is_regularised=True)
    sol.update(basis=bs, a=a, b=b, delta=b - a, c=float(-a @ sol['w']))
    return sol


def ids_fit(X, in_A, in_B, *, fit_fn, M=256, n_passes=3, sample_weights=None,
            rng=None, n_path_bins=8, n_perm=25, rmax=None, n_sub=None,
            n_floor_sub=None, tol=1e-6, floor_tol=1e-4, maxiter=200,
            shrink=0.1, verbose=False):
    """Generic IDS outer loop over an arbitrary committor fitter.

    pass 0 : isotropic directions on S^{d-1}
    pass k : bin frames by the pass-(k-1) committor (weighted quantiles),
             solve the generalised eigenproblem Sb v = lambda Sw v with CG
             matvecs, pick the rank by fixed-Sw Horn parallel analysis, and
             draw the next directions from the exact spiked sampler
             tau Sw^{-1} + sum_{i<=r} (lam_i - tau) th_i th_i'.

    ``fit_fn(thetas, pass_idx)`` must return ``(qhat, handle)`` with qhat the
    committor estimate at all frames, clipped to [0, 1], float64 (N,);
    ``handle`` is opaque to this loop and is returned for the final pass.
    After ``n_passes`` direction updates one final ``fit_fn`` call is made.

    The rng consumption ORDER is the prototype's (regression-tested):
    isotropic init, then per pass: Sw subsampling (only if n_sub), the
    n_perm rank permutations, the floor operator subsampling (only if
    n_floor_sub), then the floor draws. The caller may pass a Generator to
    share the stream (``np.random.default_rng`` passes Generators through).

    Returns dict(final=handle of the last fit, thetas=final directions,
    history=[dict(r, lam, n_cg, n_cg_floor, rank_thresholds) per pass]).
    """
    from .basis import isotropic_directions
    rng = np.random.default_rng(rng)
    N, d = X.shape
    w = sample_weights
    thetas = isotropic_directions(M, d, rng=rng)
    history = []
    kw = dict(n_bins=n_path_bins, n_sub=n_sub, tol=tol, maxiter=maxiter,
              shrink=shrink)
    for k in range(n_passes):
        qhat, _handle = fit_fn(thetas, k)
        lam, Th, n_cg, lab, op = path_directions_exact(X, qhat, in_A, in_B,
                                                       w=w, rng=rng, **kw)
        r, obs, thr = select_rank_perm_exact(X, lab, lam, w=w, n_perm=n_perm,
                                             rmax=rmax, rng=rng, op=op,
                                             tol=tol, maxiter=maxiter)
        op_f = op if n_floor_sub is None else _SwOperator(
            X, lab, w if w is not None else np.full(N, 1.0 / N),
            n_sub=n_floor_sub, rng=rng, shrink=shrink)
        thetas, n_cg2 = directions_exact_floor(lam, Th, r, op_f, M, rng,
                                               tol=floor_tol)
        history.append(dict(r=r, lam=np.asarray(lam).tolist(), n_cg=n_cg,
                            n_cg_floor=n_cg2,
                            rank_thresholds=np.asarray(thr)))
        if verbose:
            print(f"    pass {k}: r={r}  CG={n_cg}  floorCG={n_cg2}")
    _qhat, handle = fit_fn(thetas, n_passes)
    return dict(final=handle, thetas=thetas, history=history)


def ids_path_exact2(X, in_A, in_B, w=None, M=256, n_passes=3, basis_kw=None,
                    rng=None, split=None, n_path_bins=8, n_perm=25, rmax=None,
                    n_sub=None, n_floor_sub=None, tol=1e-6, floor_tol=1e-4,
                    maxiter=200, shrink=0.1, verbose=False):
    """IDS-path: exact Sw, exact Sw^{-1} floor, linear in d, no d x d object.

    Thin wrapper: ``ids_fit`` with the halfset-eigen-regularised dual solve
    (``_fit_pass_w``) as the fitter.

    DELTA vs the numpy prototype's DEFAULTS: ``n_floor_sub=None`` and
    ``floor_tol=1e-4`` here. The prototype shipped ``n_floor_sub=6000,
    floor_tol=1e-3``, but those were NOT the validated configuration -- the
    t21 validation (exact-linear IDS matching/beating dense IDS up to
    d = 1536) ran with the full-frame floor operator and the tighter floor
    tolerance. Pass the old values explicitly to reproduce the prototype's
    own defaults.

    Returns the prototype's structure -- dict(final=sol, history=[dict(pass_,
    r, sol, n_cg, n_cg_floor, lam) per pass + a final entry]) -- plus
    ``thetas`` (final directions) and ``ids_history`` (the ``ids_fit``
    history with the per-pass rank thresholds).
    """
    from .assemble import evaluate, clip01
    from .splits import iid_split
    rng = np.random.default_rng(rng)
    basis_kw = basis_kw or dict(n_bins=200, bw_bins=1.5)
    N, d = X.shape
    if split is None:
        split = iid_split(N, rng=rng)

    sols = []

    def fit_fn(thetas, k):
        sol = _fit_pass_w(thetas, X, in_A, in_B, basis_kw, split, w)
        qh = clip01(evaluate(sol['basis'], sol['w'], sol['c'], X))
        sols.append(sol)
        return np.asarray(qh, dtype=np.float64), sol

    res = ids_fit(X, in_A, in_B, fit_fn=fit_fn, M=M, n_passes=n_passes,
                  sample_weights=w, rng=rng, n_path_bins=n_path_bins,
                  n_perm=n_perm, rmax=rmax, n_sub=n_sub,
                  n_floor_sub=n_floor_sub, tol=tol, floor_tol=floor_tol,
                  maxiter=maxiter, shrink=shrink, verbose=verbose)

    hist = []
    for k, hk in enumerate(res['history']):
        hist.append(dict(pass_=k, r=hk['r'], sol=sols[k], n_cg=hk['n_cg'],
                         n_cg_floor=hk['n_cg_floor'],
                         lam=np.asarray(hk['lam'])))
    r_last = hist[-1]['r'] if hist else None
    hist.append(dict(pass_=n_passes, r=r_last, sol=sols[-1], n_cg=0,
                     n_cg_floor=0))
    return dict(final=res['final'], history=hist, thetas=res['thetas'],
                ids_history=res['history'])
