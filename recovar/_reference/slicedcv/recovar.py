"""
RECOVAR-transfer methods for the sliced committor.

Implements the ideas in NOTE_cryoem_transfers.md:

  §1 halfset_regularize      FSC-style automatic Wiener regularization of G
  §2 committor_resolution    half-split reproducibility -> a resolution
  §3 cv_dual                 held-out dual objective, overfitting control
  §4 GramOperator            matrix-free G matvec + CG (no M^2 object)
  §5 adaptive_bandwidth      per-region cross-validated profile bandwidth
  §6 flux_weighted_moments   deconvolution -> current-weighted basin moments
  §7 uq_delta / uq_bootstrap uncertainty on qbar and nu_AB
  §8 feature_mask            two-pass feature masking
  §9 nested_M_curve          scree curve over nested slice subsets
"""
import numpy as np
from scipy.linalg import cho_factor, cho_solve, eigh
from . import core as co


# ===========================================================================
# Splitting  (block splits for time-correlated data)
# ===========================================================================

def iid_split(n, rng):
    rng = np.random.default_rng(rng)
    p = rng.permutation(n)
    return p[:n // 2], p[n // 2:]


def block_split(n, block_len, rng=None, group=None):
    """Alternate contiguous blocks between halves.

    `group` (e.g. walker id) keeps blocks from crossing trajectory boundaries.
    """
    if group is None:
        group = np.zeros(n, dtype=int)
    h1, h2 = [], []
    for g in np.unique(group):
        idx = np.flatnonzero(group == g)
        nb = int(np.ceil(len(idx) / block_len))
        for k in range(nb):
            blk = idx[k * block_len:(k + 1) * block_len]
            (h1 if k % 2 == 0 else h2).append(blk)
    return np.concatenate(h1), np.concatenate(h2)


# ===========================================================================
# §3  Held-out dual objective
# ===========================================================================

def cv_dual(thetas, X, in_A, in_B, split, tikhonov=1e-3, basis_kw=None,
            rebuild_basis=True, return_basis=False):
    """Train the dual solve on half 1, evaluate h on half 2.

    h(w) = 2 (b-a)'w - w'Gw ,  max_w h = M_gap = 1/nu_AB (population).

    Overfitting inflates h_train, so 1/h_train is biased *below* nu_AB.
    h_test is the honest estimate.

    If rebuild_basis, the 1D profiles are also fitted on half 1 only, so the
    held-out evaluation is fully out-of-sample (the strict gold-standard).
    """
    basis_kw = basis_kw or {}
    i1, i2 = split
    X1, X2 = X[i1], X[i2]
    A1, B1 = in_A[i1], in_B[i1]
    A2, B2 = in_A[i2], in_B[i2]

    if rebuild_basis:
        basis = co.build_basis(thetas, X1, A1, B1, **basis_kw)
    else:
        basis = co.build_basis(thetas, X, in_A, in_B, **basis_kw)

    G1, a1, b1 = co.assemble(basis, X1, A1, B1)
    G2, a2, b2 = co.assemble(basis, X2, A2, B2)
    d1, d2 = b1 - a1, b2 - a2

    sol = co.solve_dual(G1, d1, tikhonov)
    w = sol['w_dual']
    h_tr = co.dual_objective(w, G1, d1)
    h_te = co.dual_objective(w, G2, d2)

    out = dict(h_train=h_tr, h_test=h_te,
               nu_train=1.0 / h_tr if h_tr > 0 else np.inf,
               nu_test=1.0 / h_te if h_te > 0 else np.inf,
               w_dual=w, w=sol['w'], c=float(-a1 @ sol['w']),
               G1=G1, G2=G2, d1=d1, d2=d2)
    if return_basis:
        out['basis'] = basis
    return out


# ===========================================================================
# §1  Halfset shell regularization
# ===========================================================================

def profile_shell_index(basis, n_shells=8):
    """Shell index from the bandwidth of each slice's own profile.

    Scale = participation ratio of |q_j'| over bins: large = broad, smooth
    profile (the low-frequency, well-determined slice); small = sharply
    peaked (high-frequency, estimated from few frames).
    Computed a priori, before any solve.
    """
    ap = np.abs(basis.qp)
    pr = (ap.sum(1) ** 2) / np.maximum((ap ** 2).sum(1), 1e-300)
    order = np.argsort(-pr)                       # broad (low-freq) first
    shell = np.empty(basis.M, dtype=int)
    edges = np.linspace(0, basis.M, n_shells + 1).astype(int)
    for p in range(n_shells):
        shell[order[edges[p]:edges[p + 1]]] = p
    return shell, pr


def _ssnr_from_corr(c):
    """FSC -> SSNR of the combined (full) dataset: SSNR = 2c/(1-c)."""
    c = np.clip(c, 0.0, 0.999999)
    return 2.0 * c / (1.0 - c)


def _ssnr_from_power(G1, G2, mask):
    """Direct noise-power estimate (no FSC model assumption).

    e = G1-G2 has variance 2 sigma^2; Gbar has noise variance sigma^2/2.
    SSNR(Gbar) = 2 * signal_power / sigma^2.
    """
    d = (G1 - G2)[mask]
    m = (0.5 * (G1 + G2))[mask]
    sig2 = 0.5 * np.mean(d ** 2)                  # sigma^2
    sig_pow = max(np.mean(m ** 2) - 0.5 * sig2, 0.0)
    return 2.0 * sig_pow / max(sig2, 1e-300)


def halfset_shell_regularize(G1, G2, shell, n_shells=None, mode='schur',
                             report=False):
    """Automatic, tuning-free regularization of Gbar from halfset agreement.

    mode='schur' : entrywise shrink Gbar o Lambda with Lambda = f f' +
                   diag(1-f^2) (PSD, unit diagonal, Schur product theorem),
                   f_p = SSNR_p/(1+SSNR_p) per shell. Shrinks correlations
                   between noisy shells toward a diagonal estimator.
    Returns (G_reg, info).
    """
    M = G1.shape[0]
    Gbar = 0.5 * (G1 + G2)
    n_shells = n_shells or (shell.max() + 1)

    ssnr_c = np.zeros(n_shells)
    ssnr_p = np.zeros(n_shells)
    for p in range(n_shells):
        rows = shell == p
        mask = np.zeros((M, M), dtype=bool)
        mask[np.ix_(rows, rows)] = True
        v1, v2 = G1[mask], G2[mask]
        if v1.size < 3 or v1.std() == 0 or v2.std() == 0:
            ssnr_c[p] = ssnr_p[p] = 1e6
            continue
        c = np.corrcoef(v1, v2)[0, 1]
        ssnr_c[p] = _ssnr_from_corr(c)
        ssnr_p[p] = _ssnr_from_power(G1, G2, mask)

    f = ssnr_c / (1.0 + ssnr_c)                   # Wiener factor per shell
    fv = f[shell]
    Lam = np.outer(fv, fv)
    np.fill_diagonal(Lam, 1.0)
    G_reg = Gbar * Lam

    info = dict(ssnr_corr=ssnr_c, ssnr_power=ssnr_p, f=f, Lambda=Lam)
    if report:
        info['agreement'] = np.corrcoef(ssnr_c, ssnr_p)[0, 1] if n_shells > 2 else np.nan
    return G_reg, info


def halfset_eigen_regularize(G1, G2, n_bands=12):
    """Wiener ridge in eigen-bands of Gbar (the pragmatic shell index).

    G_reg = V diag(lam_k (1 + 1/SSNR_k)) V',  so the *inverse* is damped by
    the Wiener factor SSNR/(1+SSNR) band by band. PSD by construction.
    """
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
        v1 = A1[sl, :].ravel(); v2 = A2[sl, :].ravel()
        if v1.size < 3 or v1.std() == 0 or v2.std() == 0:
            s = 1e6
        else:
            s = _ssnr_from_corr(np.corrcoef(v1, v2)[0, 1])
        band_ssnr[k] = s
        ssnr[sl] = s
    lam_reg = np.maximum(lam, 0.0) * (1.0 + 1.0 / np.maximum(ssnr, 1e-12))
    floor = 1e-12 * max(lam_reg.max(), 1e-300)
    lam_reg = np.maximum(lam_reg, floor)
    G_reg = (V * lam_reg) @ V.T
    return G_reg, dict(band_ssnr=band_ssnr, lam=lam, lam_reg=lam_reg)


# ===========================================================================
# §4  Matrix-free Gram operator
# ===========================================================================

class GramOperator:
    """G = Theta o C with Theta = theta' D theta (rank <= d) and C = P P'.

        G v = sum_{a=1..d} Y_a o ( P ( P' ( Y_a o v ) ) )

    Cost O(d N M) per matvec, no M^2 storage. With store_P=False the
    projections are recomputed per block, so memory is O(M + block*M).
    """

    def __init__(self, basis, X, sample_weights=None, D=None, store_P=True,
                 block=4096, ridge=0.0):
        self.basis = basis
        self.X = X
        self.M = basis.M
        self.N = X.shape[0]
        W = np.full(self.N, 1.0 / self.N) if sample_weights is None else sample_weights
        self.sw = np.sqrt(W / W.sum()).astype(np.float32)
        d = X.shape[1]
        if D is None:
            self.Y = basis.thetas.T.copy()                    # (d, M), D = I
        else:
            L = np.linalg.cholesky(D)
            self.Y = (L.T @ basis.thetas.T)
        self.block = block
        self.ridge = ridge
        self.store_P = store_P
        self._P = self._build_P() if store_P else None
        self.nmv = 0

    def _build_P(self):
        _, QP = co.assemble_slice_values(self.basis, self.X)
        return (QP * self.sw[None, :]).astype(np.float32)

    def _P_blocks(self):
        if self._P is not None:
            for n0 in range(0, self.N, self.block):
                yield self._P[:, n0:n0 + self.block]
        else:
            for n0 in range(0, self.N, self.block):
                n1 = min(n0 + self.block, self.N)
                S = (self.X[n0:n1] @ self.basis.thetas.T).T
                _, qp = self.basis.eval_at(S)
                yield (qp * self.sw[None, n0:n1]).astype(np.float32)

    def matvec(self, v):
        self.nmv += 1
        v = np.asarray(v, dtype=np.float64)
        out = np.zeros(self.M)
        for a in range(self.Y.shape[0]):
            Ya = self.Y[a]
            u = (Ya * v).astype(np.float32)
            acc = np.zeros(self.M, dtype=np.float64)
            for Pb in self._P_blocks():
                acc += (Pb @ (Pb.T @ u)).astype(np.float64)
            out += Ya * acc
        return out + self.ridge * v

    def to_dense(self):
        return np.column_stack([self.matvec(e) for e in np.eye(self.M)])


def cg(matvec, b, tol=1e-10, maxiter=None, M_inv=None):
    """Plain CG with optional diagonal preconditioner."""
    n = len(b)
    maxiter = maxiter or 10 * n
    x = np.zeros(n)
    r = b - matvec(x)
    z = M_inv * r if M_inv is not None else r
    p = z.copy()
    rz = r @ z
    b0 = np.linalg.norm(b)
    hist = []
    for it in range(maxiter):
        Ap = matvec(p)
        pAp = p @ Ap
        if pAp <= 0:
            break
        al = rz / pAp
        x += al * p
        r -= al * Ap
        res = np.linalg.norm(r) / max(b0, 1e-300)
        hist.append(res)
        if res < tol:
            break
        z = M_inv * r if M_inv is not None else r
        rz_new = r @ z
        p = z + (rz_new / rz) * p
        rz = rz_new
    return x, np.array(hist)


def gram_diagonal(basis, X, sample_weights=None, block=4096):
    """diag(G)_j = |theta_j|_D^2 * sum_n W_n q_j'(s_j^n)^2  (for preconditioning)."""
    N = X.shape[0]
    W = np.full(N, 1.0 / N) if sample_weights is None else sample_weights
    W = W / W.sum()
    d = np.zeros(basis.M)
    for n0 in range(0, N, block):
        n1 = min(n0 + block, N)
        S = (X[n0:n1] @ basis.thetas.T).T
        _, qp = basis.eval_at(S)
        d += (qp.astype(np.float64) ** 2) @ W[n0:n1]
    return d * np.einsum('ij,ij->i', basis.thetas, basis.thetas)


# ===========================================================================
# §2  Committor resolution
# ===========================================================================

def smooth_field_knn(Xq, vals, k, rng=None, n_ref=None):
    """Average `vals` over the k nearest neighbours of each point (in feature
    space). k sets the smoothing scale; the corresponding length is reported
    as the mean k-NN radius."""
    from scipy.spatial import cKDTree
    tree = cKDTree(Xq)
    dd, ii = tree.query(Xq, k=k, workers=-1)
    return vals[ii].mean(1), float(np.mean(dd[:, -1]))


def committor_shell_correlation(q1, q2, Xq, ks=(1, 2, 4, 8, 16, 32, 64, 128, 256)):
    """Correlation of two half-committors as a function of smoothing scale.

    Returns (k, length_scale, corr). The finest scale at which the two halves
    still agree is the 'committor resolution'.
    """
    out = []
    for k in ks:
        s1, L = smooth_field_knn(Xq, q1, k)
        s2, _ = smooth_field_knn(Xq, q2, k)
        if s1.std() == 0 or s2.std() == 0:
            c = np.nan
        else:
            c = np.corrcoef(s1, s2)[0, 1]
        out.append((k, L, c))
    return np.array(out)


def laplacian_shell_correlation(q1, q2, Xq, n_sub=3000, n_eig=200, n_bands=10,
                                rng=None, eps=None):
    """Faithful FSC analogue: correlate the two committors' coefficients in
    bands of graph-Laplacian eigenvectors (the Fourier basis on the data
    manifold)."""
    rng = np.random.default_rng(rng)
    idx = rng.choice(Xq.shape[0], size=min(n_sub, Xq.shape[0]), replace=False)
    Z = Xq[idx]
    from scipy.spatial.distance import pdist, squareform
    Dm = squareform(pdist(Z))
    if eps is None:
        eps = np.median(Dm[Dm > 0]) ** 2 / 4
    K = np.exp(-Dm ** 2 / eps)
    dg = K.sum(1)
    Ln = np.eye(len(Z)) - (K / np.sqrt(np.outer(dg, dg)))
    lam, V = eigh(Ln)
    V = V[:, :n_eig]
    c1 = V.T @ (q1[idx] - q1[idx].mean())
    c2 = V.T @ (q2[idx] - q2[idx].mean())
    edges = np.linspace(0, n_eig, n_bands + 1).astype(int)
    out = []
    for k in range(n_bands):
        sl = slice(edges[k], edges[k + 1])
        a, b = c1[sl], c2[sl]
        c = float(np.sum(a * b) / np.sqrt(np.sum(a**2) * np.sum(b**2) + 1e-300))
        out.append((0.5 * (edges[k] + edges[k + 1]), float(lam[sl].mean()), c))
    return np.array(out)


# ===========================================================================
# §5  Adaptive per-region bandwidth
# ===========================================================================

def make_adaptive_bw_fn(bw_grid=(0.5, 1.0, 2.0, 4.0, 8.0, 16.0), n_regions=6,
                        split_seed=0):
    """Return a bw_fn for core.build_basis that chooses, per s-region, the
    bandwidth minimising halfset disagreement of the smoothed density.

    Cross-validation score per (region, bandwidth): integrated squared
    difference between the two half-sample smoothed densities, normalised by
    the density scale (a proxy for the variance term), plus the squared
    difference of each half's estimate from the finest-bandwidth estimate on
    the *other* half (the bias term). Standard KDE CV in spirit.
    """
    def bw_fn(j, edges, s, in_A, in_B):
        n_bins = len(edges) - 1
        rng = np.random.default_rng(split_seed + j)
        m = rng.random(len(s)) < 0.5
        hs = [np.histogram(s[m], bins=edges)[0].astype(float),
              np.histogram(s[~m], bins=edges)[0].astype(float)]
        hs = [h / max(h.sum(), 1) for h in hs]
        redges = np.linspace(0, n_bins, n_regions + 1).astype(int)
        bw = np.empty(n_bins)
        for r in range(n_regions):
            sl = slice(redges[r], redges[r + 1])
            best, best_bw = np.inf, bw_grid[0]
            for b in bw_grid:
                S = co._smooth_matrix(n_bins, b)
                f1, f2 = S @ hs[0], S @ hs[1]
                # CV risk: ||f||^2 - 2 <f1, f2>  (Rudemo/Bowman ISE estimator)
                risk = (np.mean(((f1 + f2) / 2)[sl] ** 2)
                        - 2 * np.mean((f1 * f2)[sl]))
                if risk < best:
                    best, best_bw = risk, b
            bw[sl] = best_bw
        return bw
    return bw_fn


# ===========================================================================
# §6  Flux-weighted basin moments
# ===========================================================================

def true_flux_fidelity(basis, X, in_A, in_B, grad_q_ref, shell_frac=0.25):
    """Measure the TRUE flux-weighted phi_j using a reference committor's
    gradient, for comparison with phihat = b - a.

    phi_j = <qtilde_j>^flux_{dB} - <qtilde_j>^flux_{dA}, with the boundary
    averages taken over a thin shell just inside each basin edge and weighted
    by the outward normal component of J = pi D grad q.
    """
    QV, _ = co.assemble_slice_values(basis, X)
    out = {}
    for name, mask, center in (('A', in_A, None), ('B', in_B, None)):
        idx = np.flatnonzero(mask)
        c = X[idx].mean(0)
        r = np.linalg.norm(X[idx] - c, axis=1)
        thr = np.quantile(r, 1.0 - shell_frac)
        sh = idx[r >= thr]
        nrm = X[sh] - c
        nrm /= np.linalg.norm(nrm, axis=1, keepdims=True)[:, :1] * 0 + \
            np.linalg.norm(nrm, axis=1, keepdims=True)
        wflux = np.abs(np.einsum('ij,ij->i', grad_q_ref[sh], nrm))
        wflux = wflux / max(wflux.sum(), 1e-300)
        out[name] = QV[:, sh].astype(np.float64) @ wflux
    return out['B'] - out['A']


def flux_weighted_moments(basis, X, in_A, in_B, w, shell_frac=0.25):
    """Self-consistent step: use the current ansatz gradient
    grad qbar = sum_j w_j q_j' theta_j to define the flux weighting, then
    recompute the basin moments as current-weighted boundary averages."""
    QV, QP = co.assemble_slice_values(basis, X)
    grad = (w[:, None] * QP.astype(np.float64)).T @ basis.thetas   # (N, d)
    out = {}
    for name, mask in (('A', in_A), ('B', in_B)):
        idx = np.flatnonzero(mask)
        c = X[idx].mean(0)
        v = X[idx] - c
        r = np.linalg.norm(v, axis=1)
        thr = np.quantile(r, 1.0 - shell_frac)
        sh = idx[r >= thr]
        nrm = (X[sh] - c)
        nrm = nrm / np.linalg.norm(nrm, axis=1, keepdims=True)
        wf = np.abs(np.einsum('ij,ij->i', grad[sh], nrm))
        if wf.sum() <= 0:
            wf = np.ones(len(sh))
        wf = wf / wf.sum()
        out[name] = QV[:, sh].astype(np.float64) @ wf
    return out['A'], out['B']


# ===========================================================================
# §7  Uncertainty quantification
# ===========================================================================

def uq_bootstrap(basis, X, in_A, in_B, n_boot=40, block_len=1, rng=None,
                 tikhonov=1e-3, group=None, Xq=None):
    """Block bootstrap over frames: resample blocks, refit, collect the
    spread of w, nu_hat = 1/M_gap, and (optionally) qbar at query points."""
    rng = np.random.default_rng(rng)
    N = X.shape[0]
    QV, QP = co.assemble_slice_values(basis, X)
    Theta = basis.thetas @ basis.thetas.T
    if group is None:
        group = np.zeros(N, dtype=int)
    starts = np.arange(0, N, block_len)
    nus, ws = [], []
    qs = []
    for _ in range(n_boot):
        pick = rng.choice(len(starts), size=len(starts), replace=True)
        idx = np.concatenate([np.arange(starts[p],
                                        min(starts[p] + block_len, N))
                              for p in pick])
        qv = QV[:, idx].astype(np.float64)
        qp = QP[:, idx].astype(np.float64)
        n = len(idx)
        C = (qp @ qp.T) / n
        G = Theta * C
        A = in_A[idx]; B = in_B[idx]
        if A.sum() < 5 or B.sum() < 5:
            continue
        a = qv[:, A].mean(1); b = qv[:, B].mean(1)
        sol = co.solve_dual(G, b - a, tikhonov)
        nus.append(sol['nu_hat']); ws.append(sol['w'])
        if Xq is not None:
            qs.append(co.evaluate(basis, sol['w'], float(-a @ sol['w']), Xq))
    res = dict(nu=np.array(nus), w=np.array(ws))
    if Xq is not None:
        res['q'] = np.array(qs)
    return res


def uq_delta(basis, X, in_A, in_B, tikhonov=1e-3, Xq=None):
    """Analytic (delta-method) covariance of w* and nu_hat from the sampling
    covariance of the three sample averages (G, a, b).

    w_dual = G^{-1} d,  d = b - a.
    dw = G^{-1} (dd - dG w_dual);  M_gap = d'w_dual;
    dM_gap = 2 w_dual'dd - w_dual'dG w_dual  (using symmetry of G).
    Var of each term from the per-frame influence functions.
    """
    N = X.shape[0]
    QV, QP = co.assemble_slice_values(basis, X)
    QV = QV.astype(np.float64); QP = QP.astype(np.float64)
    Theta = basis.thetas @ basis.thetas.T
    C = (QP @ QP.T) / N
    G = Theta * C
    a = QV[:, in_A].mean(1); b = QV[:, in_B].mean(1)
    d = b - a
    sol = co.solve_dual(G, d, tikhonov)
    wd = sol['w_dual']; Mg = sol['M_gap']

    # per-frame influence on M_gap:
    #   from d : 2 wd' (b - a) contributions
    #   from G : - wd' G wd  ->  per frame -(theta-weighted) (wd'.qp_n)^2
    u = (basis.thetas.T @ (wd[:, None] * QP)).sum(0) if False else None
    # grad qbar-like projection: r_n = sum_j wd_j qp_jn theta_j   (d-vector)
    R = (wd[:, None] * QP).T @ basis.thetas                       # (N, d)
    gterm = np.einsum('ij,ij->i', R, R)                           # (wd' . )^2
    infl = -(gterm - gterm.mean())
    qA = QV[:, in_A]; qB = QV[:, in_B]
    nA = in_A.sum(); nB = in_B.sum()
    iA = -2.0 * (wd @ (qA - a[:, None])) / nA * N
    iB = 2.0 * (wd @ (qB - b[:, None])) / nB * N
    tot = infl.copy()
    tot[in_A] += iA
    tot[in_B] += iB
    var_Mgap = np.var(tot) / N
    se_nu = np.sqrt(var_Mgap) / Mg ** 2

    out = dict(nu_hat=1.0 / Mg, se_nu=se_nu, M_gap=Mg, se_Mgap=np.sqrt(var_Mgap))
    if Xq is not None:
        # Var(qbar) via the linearisation dw = G^{-1}(dd - dG wd)/Mgap ...
        # dominant term: sampling error of d, propagated.
        Gi = np.linalg.pinv(sol['G_reg'])
        Sd = np.cov(np.concatenate([qB - b[:, None], -(qA - a[:, None])], axis=1))
        Sd = Sd / min(nA, nB)
        Cw = Gi @ Sd @ Gi.T / Mg ** 2
        QVq, _ = co.assemble_slice_values(basis, Xq)
        out['q_se'] = np.sqrt(np.maximum(
            np.einsum('jn,jk,kn->n', QVq.astype(np.float64), Cw,
                      QVq.astype(np.float64)), 0.0))
    return out


# ===========================================================================
# §8  Feature mask (two-pass)
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
# §9  Nested-M curve
# ===========================================================================

def nested_M_curve(thetas, X, in_A, in_B, split, Ms, tikhonov=1e-3,
                   basis_kw=None, q_ref=None, Xq=None):
    """h_train / h_test over nested slice subsets (first M of a fixed pool)."""
    basis_kw = basis_kw or {}
    i1, i2 = split
    X1, X2 = X[i1], X[i2]
    A1, B1, A2, B2 = in_A[i1], in_B[i1], in_A[i2], in_B[i2]
    full = co.build_basis(thetas, X1, A1, B1, **basis_kw)
    rows = []
    for M in Ms:
        sub = full.subset(np.arange(M))
        G1, a1, b1 = co.assemble(sub, X1, A1, B1)
        G2, a2, b2 = co.assemble(sub, X2, A2, B2)
        d1, d2 = b1 - a1, b2 - a2
        sol = co.solve_dual(G1, d1, tikhonov)
        wd = sol['w_dual']
        h_tr = co.dual_objective(wd, G1, d1)
        h_te = co.dual_objective(wd, G2, d2)
        rmse = np.nan
        if q_ref is not None and Xq is not None:
            qh = co.clip01(co.evaluate(sub, sol['w'], float(-a1 @ sol['w']), Xq))
            rmse = float(np.sqrt(np.mean((qh - q_ref) ** 2)))
        rows.append((M, h_tr, h_te, 1.0 / h_tr, 1.0 / h_te if h_te > 0 else np.inf, rmse))
    return np.array(rows)


# ===========================================================================
# §4b  Nystrom / randomised low-rank factorisation of G
# ===========================================================================

def nystrom_gram(op, r, rng=None, oversample=10):
    """Randomised Nystrom approximation G ~ Y (Om' Y)^+ Y' using r+p matvecs.

    Never forms G. Returns (U, s) with G ~ U diag(s) U'.
    """
    rng = np.random.default_rng(rng)
    M = op.M
    p = min(oversample, max(0, M - r))
    Om = rng.standard_normal((M, r + p))
    Y = np.column_stack([op.matvec(Om[:, k]) for k in range(r + p)])
    nu = 1e-10 * np.sqrt(M) * max(np.linalg.norm(Y), 1e-300)
    Yn = Y + nu * Om
    B = Om.T @ Yn
    B = 0.5 * (B + B.T)
    try:
        L = np.linalg.cholesky(B)
        F = np.linalg.solve(L, Yn.T).T
    except np.linalg.LinAlgError:
        ev, V = eigh(B)
        ev = np.maximum(ev, 1e-14 * ev.max())
        F = Yn @ (V / np.sqrt(ev)) @ V.T
    U, sv, _ = np.linalg.svd(F, full_matrices=False)
    s = np.maximum(sv ** 2 - nu, 0.0)
    return U[:, :r], s[:r]


def nystrom_columns(basis, X, cols, sample_weights=None, block=8192):
    """Explicit column block G[:, cols] (the literal 'subset of columns'
    Nystrom of RECOVAR).  Cost O(N M |cols|), no M^2 object."""
    M = basis.M
    N = X.shape[0]
    W = np.full(N, 1.0 / N) if sample_weights is None else sample_weights
    W = W / W.sum()
    Cc = np.zeros((M, len(cols)))
    sub = basis.subset(np.asarray(cols))
    for n0 in range(0, N, block):
        n1 = min(n0 + block, N)
        Xb = X[n0:n1]
        _, qp = basis.eval_at((Xb @ basis.thetas.T).T)
        _, qpc = sub.eval_at((Xb @ sub.thetas.T).T)
        wb = W[n0:n1]
        Cc += (qp.astype(np.float64) * wb) @ qpc.astype(np.float64).T
    Theta_c = basis.thetas @ sub.thetas.T
    return Theta_c * Cc


def solve_lowrank(U, s, delta, rel_floor=1e-8):
    """Solve (U diag(s) U' + null) w = delta restricted to the range of U."""
    keep = s > rel_floor * max(s.max(), 1e-300)
    Uk, sk = U[:, keep], s[keep]
    c = Uk.T @ delta
    w = Uk @ (c / sk)
    Mg = float(delta @ w)
    return w, Mg, int(keep.sum())


def make_halfset_bw_fn(bw_grid=(0.5, 1.0, 2.0, 4.0, 8.0, 16.0), n_regions=6,
                       split_seed=0, target='profile'):
    """Per-region bandwidth chosen by halfset risk on the *committor profile*
    (not on the density).

    For each candidate bandwidth b, build the profile on each half of the
    frames and score it against the *other* half's finest-bandwidth profile:

        R(b) = mean_region[ (q_1^b - q_2^fine)^2 + (q_2^b - q_1^fine)^2 ] / 2

    which is the standard CV risk against a noisy but unbiased target, so it
    trades bias against variance with the right minimiser. target='deriv'
    scores q' instead of q (q' is what enters the Gram).
    """
    def bw_fn(j, edges, s, in_A, in_B):
        n_bins = len(edges) - 1
        h = edges[1] - edges[0]
        rng = np.random.default_rng(split_seed + j)
        m = rng.random(len(s)) < 0.5
        halves = []
        for sel in (m, ~m):
            ss, aa, bb = s[sel], in_A[sel], in_B[sel]
            ha, _ = np.histogram(ss, bins=edges)
            hA, _ = np.histogram(ss[aa], bins=edges)
            hB, _ = np.histogram(ss[bb], bins=edges)
            halves.append((ha.astype(float), hA.astype(float), hB.astype(float)))

        def prof(hist, b):
            S = co._smooth_matrix(n_bins, b)
            ha, hA, hB = (S @ hist[0], S @ hist[1], S @ hist[2])
            rho = ha / max(ha.sum() * h, 1e-300)
            rA = hA / max(hA.sum() * h, 1e-300)
            rB = hB / max(hB.sum() * h, 1e-300)
            q = co.rd_profile(rho, rA, rB, h)
            return np.gradient(q, h) if target == 'deriv' else q

        bfine = min(bw_grid)
        fine = [prof(halves[0], bfine), prof(halves[1], bfine)]
        cand = {b: [prof(halves[0], b), prof(halves[1], b)] for b in bw_grid}

        redges = np.linspace(0, n_bins, n_regions + 1).astype(int)
        bw = np.empty(n_bins)
        for r in range(n_regions):
            sl = slice(redges[r], redges[r + 1])
            best, best_b = np.inf, bw_grid[0]
            for b in bw_grid:
                q1, q2 = cand[b]
                risk = 0.5 * (np.mean((q1[sl] - fine[1][sl]) ** 2)
                              + np.mean((q2[sl] - fine[0][sl]) ** 2))
                if risk < best:
                    best, best_b = risk, b
            bw[sl] = best_b
        return bw
    return bw_fn


def bandpass_shell_correlation(q1, q2, Xq, ks=(2, 4, 8, 16, 32, 64, 128, 256, 512, 1024)):
    """Band-pass version of the committor shell correlation.

    Correlating the *smoothed* fields saturates at ~1 because the committor's
    low-frequency component (0 -> 1 across the domain) dominates every scale.
    The FSC analogue must be a shell: correlate the difference between
    successive smoothing scales, i.e. the content in the band between them.
    """
    from scipy.spatial import cKDTree
    tree = cKDTree(Xq)
    kmax = max(ks)
    dd, ii = tree.query(Xq, k=kmax, workers=-1)
    sm1, sm2, lens = {}, {}, {}
    for k in ks:
        sm1[k] = q1[ii[:, :k]].mean(1)
        sm2[k] = q2[ii[:, :k]].mean(1)
        lens[k] = float(np.mean(dd[:, k - 1]))
    out = []
    ks = list(ks)
    for a, b in zip(ks[:-1], ks[1:]):
        d1 = sm1[a] - sm1[b]
        d2 = sm2[a] - sm2[b]
        if d1.std() == 0 or d2.std() == 0:
            c = np.nan
        else:
            c = float(np.corrcoef(d1, d2)[0, 1])
        out.append((a, b, 0.5 * (lens[a] + lens[b]), c))
    return np.array(out)


# ===========================================================================
# IDS -- Iterated Dirichlet-Structure subspace refinement
#
# Rotation-equivariant, system-knowledge-free replacement for both the
# coordinate feature mask and the tuned Fisher-LDA sampler.
#
# The key object is the Dirichlet structure tensor
#
#     S = E_pi[ grad qbar  grad qbar^T ] ,   grad qbar = sum_j w_j q_j' theta_j
#
# a d x d tensor whose trace IS the Dirichlet energy, tr(S D) = D[qbar].
# Its eigen-decomposition is therefore the decomposition of the ansatz's own
# Dirichlet energy by direction in configuration space -- exactly the
# "where does the committor actually vary" question, answered in a basis-free
# way. Directions for the next pass are drawn from a spiked covariance built
# from S, so the scheme degenerates gracefully to isotropic when there is no
# structure to find.
#
# Nothing here refers to a coordinate system, so the whole scheme commutes
# with any rotation (or, with the whitened variant, any invertible linear map)
# of feature space.
# ===========================================================================

def structure_tensor(basis, X, w=None, delta=None, mode='grad', mask=None,
                     block=8192):
    """d x d Dirichlet structure tensor.

    mode='grad'  : S = E[g g'],  g_n = sum_j w_j q_j'(s_jn) theta_j.
                   tr(S) = D[qbar]. Needs a solved w.
    mode='delta' : S = sum_j delta_j^2 E[q_j'^2] theta_j theta_j'.
                   Solve-free and regularisation-free -- the robust choice
                   for the first pass, when w is not yet trustworthy.
    mask         : boolean over frames (e.g. transition region only).
    """
    d = X.shape[1]
    N = X.shape[0]
    sel = np.ones(N, bool) if mask is None else mask
    if mode == 'delta':
        assert delta is not None
        e = np.zeros(basis.M)
        for n0 in range(0, N, block):
            n1 = min(n0 + block, N)
            if not sel[n0:n1].any():
                continue
            _, qp = basis.eval_at((X[n0:n1] @ basis.thetas.T).T)
            e += (qp.astype(np.float64) ** 2) @ sel[n0:n1].astype(float)
        e /= max(sel.sum(), 1)
        c = (delta ** 2) * e
        return (basis.thetas * c[:, None]).T @ basis.thetas
    assert w is not None
    S = np.zeros((d, d))
    cnt = 0
    for n0 in range(0, N, block):
        n1 = min(n0 + block, N)
        m = sel[n0:n1]
        if not m.any():
            continue
        _, qp = basis.eval_at((X[n0:n1] @ basis.thetas.T).T)
        g = (w[:, None] * qp.astype(np.float64)).T @ basis.thetas   # (n, d)
        g = g[m]
        S += g.T @ g
        cnt += g.shape[0]
    return S / max(cnt, 1)


def subspace_overlap(S1, S2, rmax=None):
    """Mean squared cosine of principal angles between the leading-r
    eigenspaces of two halfset structure tensors, for r = 1..rmax.

    overlap(r) = ||P_r^(1) P_r^(2)||_F^2 / r.  Equals 1 for identical
    subspaces and ~ r/d for unrelated ones.
    """
    d = S1.shape[0]
    rmax = rmax or d
    l1, V1 = eigh(S1); l2, V2 = eigh(S2)
    V1 = V1[:, np.argsort(-l1)]
    V2 = V2[:, np.argsort(-l2)]
    Cm = V1.T @ V2
    out = np.zeros(rmax)
    for r in range(1, rmax + 1):
        out[r - 1] = (Cm[:r, :r] ** 2).sum() / r
    return out


def select_rank(S1, S2, thresh=0.9, rmax=None, margin=2.0):
    """Largest r whose leading-r subspace is reproducible across halves.

    Requires overlap(r) >= thresh AND overlap(r) >= margin * (r/d), the
    latter guarding against the r -> d limit where overlap is trivially 1.
    """
    d = S1.shape[0]
    rmax = rmax or max(1, d - 1)
    ov = subspace_overlap(S1, S2, rmax)
    r_best = 1
    for r in range(1, rmax + 1):
        if ov[r - 1] >= thresh and ov[r - 1] >= margin * (r / d):
            r_best = r
    return r_best, ov


def spiked_covariance(S, r, floor=1e-8):
    """Keep the top-r eigenvalues of S; replace the tail by its mean.

    This is the probabilistic-PCA / spiked-covariance structure: the
    discarded energy is redistributed isotropically over the orthogonal
    complement, so there is no free 'exploration' parameter. A flat spectrum
    (no structure) returns a multiple of the identity, i.e. plain isotropic
    sampling -- the scheme degrades gracefully rather than collapsing.
    """
    d = S.shape[0]
    lam, V = eigh(S)
    o = np.argsort(-lam)
    lam, V = np.maximum(lam[o], 0.0), V[:, o]
    r = int(np.clip(r, 1, d - 1))
    tail = lam[r:].mean() if r < d else 0.0
    lam_new = lam.copy()
    lam_new[r:] = tail
    lam_new = np.maximum(lam_new, floor * max(lam[0], 1e-300))
    return (V * lam_new) @ V.T


def directions_from_cov(Sigma, M, rng, power=1.0):
    """theta ~ N(0, Sigma^power), normalised. Rotation-equivariant."""
    rng = np.random.default_rng(rng)
    lam, V = eigh(Sigma)
    lam = np.maximum(lam, 0.0)
    scale = lam ** (0.5 * power)
    z = rng.standard_normal((M, Sigma.shape[0]))
    v = (z * scale[None, :]) @ V.T
    nrm = np.linalg.norm(v, axis=1, keepdims=True)
    bad = (nrm[:, 0] <= 0)
    if bad.any():
        v[bad] = rng.standard_normal((bad.sum(), Sigma.shape[0]))
        nrm = np.linalg.norm(v, axis=1, keepdims=True)
    return v / nrm


def _fit_pass(thetas, X, in_A, in_B, basis_kw, split, n_bands=12):
    """One pass: build slices, halfset-regularise G, dual solve."""
    bs = co.build_basis(thetas, X, in_A, in_B, **basis_kw)
    i1, i2 = split
    G1, *_ = co.assemble(bs, X[i1], in_A[i1], in_B[i1])
    G2, *_ = co.assemble(bs, X[i2], in_A[i2], in_B[i2])
    Gr, info = halfset_eigen_regularize(G1, G2, n_bands=n_bands)
    G, a, b = co.assemble(bs, X, in_A, in_B)
    sol = co.solve_dual(Gr, b - a, G_is_regularised=True)
    sol.update(basis=bs, a=a, b=b, delta=b - a, c=float(-a @ sol['w']),
               band_ssnr=info['band_ssnr'])
    return sol


def ids_refine(X, in_A, in_B, M=256, n_passes=3, basis_kw=None, rng=None,
               split=None, mode='grad', power=1.0, thresh=0.9,
               transition_only=True, rmax=None, verbose=False):
    """Iterated Dirichlet-Structure subspace refinement.

    pass 0 : isotropic directions on S^{d-1}
    pass k : directions ~ N(0, Sigma^power), Sigma = spiked_covariance(S, r),
             S the Dirichlet structure tensor of pass k-1, r chosen by
             halfset subspace reproducibility.

    No system knowledge, no coordinate basis, no tuned concentration. Cost per
    pass is that of the base method plus O(N d^2) for S.
    """
    rng = np.random.default_rng(rng)
    basis_kw = basis_kw or dict(n_bins=200, bw_bins=1.5)
    N, d = X.shape
    if split is None:
        split = iid_split(N, rng=rng)
    i1, i2 = split
    trans = ~(in_A | in_B) if transition_only else np.ones(N, bool)

    thetas = co.isotropic_directions(M, d, rng=rng)
    hist = []
    Sig = None
    for k in range(n_passes):
        sol = _fit_pass(thetas, X, in_A, in_B, basis_kw, split)
        bs = sol['basis']
        kw = dict(mode=mode, mask=trans)
        if mode == 'grad':
            kw['w'] = sol['w']
        else:
            kw['delta'] = sol['delta']
        S = structure_tensor(bs, X, **kw)
        # halfset structure tensors for the rank choice
        kw1 = dict(kw); kw1['mask'] = trans & np.isin(np.arange(N), i1)
        kw2 = dict(kw); kw2['mask'] = trans & np.isin(np.arange(N), i2)
        S1 = structure_tensor(bs, X, **kw1)
        S2 = structure_tensor(bs, X, **kw2)
        r, ov = select_rank(S1, S2, thresh=thresh, rmax=rmax)
        Sig_new = spiked_covariance(S, r)
        ang = np.nan
        if Sig is not None:
            ang = float(subspace_overlap(Sig, Sig_new, rmax=r)[-1])
        hist.append(dict(pass_=k, r=r, sol=sol, S=S, overlap=ov,
                         subspace_stability=ang))
        if verbose:
            print(f"    pass {k}: r={r}  1/Mgap={sol['nu_hat']:.4e}  "
                  f"stability={ang:.4f}")
        Sig = Sig_new
        thetas = directions_from_cov(Sig, M, rng, power=power)

    sol = _fit_pass(thetas, X, in_A, in_B, basis_kw, split)
    hist.append(dict(pass_=n_passes, r=hist[-1]['r'], sol=sol, S=None,
                     overlap=None, subspace_stability=np.nan))
    return dict(final=sol, history=hist, Sigma=Sig)


# ===========================================================================
# Path scatter -- the subspace estimator that actually works
#
# The Dirichlet structure tensor of a first-pass ansatz is rank-1 dominated
# (measured lam_2/lam_1 ~ 0.03): a monotone trial committor only varies along
# one direction, so its gradient cannot reveal the curvature direction. The
# chicken-and-egg is broken by using the first pass only to ORDER frames
# along the reaction, not to supply a gradient.
#
# Bin frames by the first-pass committor and take the between-bin scatter:
# this is multi-class Fisher discriminant analysis along reaction progress.
# Two classes (A vs B) give rank 1 -- the ordinary LDA axis. L bins along a
# CURVED path give rank up to L-1, because the path's tangent differs at
# different points along it. That is exactly the reactive subspace.
#
# Whitening by the pooled within-bin covariance makes the estimator
# equivariant under any invertible linear map of feature space, not merely
# rotations -- so badly scaled features are handled automatically.
# ===========================================================================

def path_scatter(X, qhat, in_A, in_B, n_bins=8, shrink=0.1, whiten=True):
    """Between-bin scatter along reaction progress, optionally whitened.

    Returns (S, W) with S the (whitened) scatter and W the map from whitened
    coordinates back to feature space (W = Sw^{1/2}, or I if whiten=False).
    """
    N, d = X.shape
    lab = np.full(N, -1)
    tr = ~(in_A | in_B)
    if tr.sum() > n_bins * 5:
        qt = qhat[tr]
        edges = np.quantile(qt, np.linspace(0, 1, n_bins + 1))
        edges[0] -= 1e-9; edges[-1] += 1e-9
        lab[tr] = np.clip(np.searchsorted(edges, qt, side='right') - 1,
                          0, n_bins - 1) + 1
    lab[in_A] = 0
    lab[in_B] = n_bins + 1

    groups = [g for g in np.unique(lab) if g >= 0 and (lab == g).sum() >= d + 2]
    if len(groups) < 3:
        return np.eye(d), np.eye(d)
    ms, ps, Sw = [], [], np.zeros((d, d))
    ntot = 0
    for g in groups:
        Z = X[lab == g]
        m = Z.mean(0)
        ms.append(m); ps.append(len(Z))
        Zc = Z - m
        Sw += Zc.T @ Zc
        ntot += len(Z)
    ms = np.array(ms); ps = np.array(ps, float) / sum(ps)
    Sw /= max(ntot, 1)
    mbar = (ps[:, None] * ms).sum(0)
    Mc = ms - mbar
    Sb = (Mc * ps[:, None]).T @ Mc

    if not whiten:
        return Sb, np.eye(d)
    Sw = Sw + shrink * (np.trace(Sw) / d) * np.eye(d)
    lam, V = eigh(Sw)
    lam = np.maximum(lam, 1e-12 * max(lam.max(), 1e-300))
    Wh = (V / np.sqrt(lam)) @ V.T          # Sw^{-1/2}
    # A direction u in whitened space corresponds to the feature-space
    # direction Sw^{-1/2} u (since theta.x = theta' Sw^{1/2} y). Mapping back
    # with Sw^{+1/2} would be wrong -- and would amplify exactly the
    # high-variance nuisance directions whitening is meant to suppress. With
    # Sw^{-1/2} the rank-1 case reproduces the LDA axis Sw^{-1}(m_B - m_A).
    return Wh @ Sb @ Wh, Wh


def select_rank_incremental(S1, S2, thresh=0.5, margin=3.0, rmax=None):
    """FSC-style stopping rule on the halfset subspace overlap.

    o_r = ||P_r^(1)'P_r^(2)||_F^2 - ||P_{r-1}^(1)'P_{r-1}^(2)||_F^2 in [0,1]
    is how much NEW reproducible content eigen-direction r adds. Scan upward
    and stop at the first r that drops below threshold (and below `margin`
    times the random-subspace baseline (2r-1)/d).

    The cumulative overlap used previously is the wrong statistic: it tends
    to 1 as r -> d for ANY pair of subspaces, so "largest r above threshold"
    selects r ~ d/2 regardless of the data.
    """
    d = S1.shape[0]
    rmax = rmax or max(1, min(d - 1, 20))
    l1, V1 = eigh(S1); l2, V2 = eigh(S2)
    V1 = V1[:, np.argsort(-l1)]; V2 = V2[:, np.argsort(-l2)]
    Cm = V1.T @ V2
    O = np.array([(Cm[:r, :r] ** 2).sum() for r in range(0, rmax + 1)])
    inc = np.diff(O)
    r = 0
    for i in range(rmax):
        base = margin * (2 * (i + 1) - 1) / d
        if inc[i] >= max(thresh, base):
            r = i + 1
        else:
            break
    return max(r, 1), inc


def ids_path(X, in_A, in_B, M=256, n_passes=3, basis_kw=None, rng=None,
             split=None, n_path_bins=8, whiten=True, power=1.0,
             thresh=0.5, rmax=None, verbose=False):
    """Iterated path-scatter subspace refinement.

    pass 0 : isotropic directions
    pass k : bin frames by the pass-(k-1) committor, take the whitened
             between-bin scatter, choose its rank by halfset reproducibility,
             and draw directions from the corresponding spiked covariance
             mapped back to feature space.

    Rotation- (and with whiten=True, general-linear-) equivariant. No system
    knowledge, no coordinate basis, no tuned concentration parameter.
    """
    rng = np.random.default_rng(rng)
    basis_kw = basis_kw or dict(n_bins=200, bw_bins=1.5)
    N, d = X.shape
    if split is None:
        split = iid_split(N, rng=rng)
    i1, i2 = split

    thetas = co.isotropic_directions(M, d, rng=rng)
    hist = []
    for k in range(n_passes):
        sol = _fit_pass(thetas, X, in_A, in_B, basis_kw, split)
        qh = co.clip01(co.evaluate(sol['basis'], sol['w'], sol['c'], X))
        S, W = path_scatter(X, qh, in_A, in_B, n_bins=n_path_bins, whiten=whiten)
        m1 = np.zeros(N, bool); m1[i1] = True
        S1, _ = path_scatter(X[m1], qh[m1], in_A[m1], in_B[m1],
                             n_bins=n_path_bins, whiten=whiten)
        S2, _ = path_scatter(X[~m1], qh[~m1], in_A[~m1], in_B[~m1],
                             n_bins=n_path_bins, whiten=whiten)
        r, inc = select_rank_incremental(S1, S2, thresh=thresh, rmax=rmax)
        Sig_w = spiked_covariance(S, r)
        Sigma = W @ Sig_w @ W.T                 # back to feature space
        Sigma = 0.5 * (Sigma + Sigma.T)
        hist.append(dict(pass_=k, r=r, sol=sol, inc=inc,
                         eigs=np.sort(np.linalg.eigvalsh(S))[::-1]))
        if verbose:
            print(f"    pass {k}: r={r}  inc={np.round(inc[:5], 3)}  "
                  f"1/Mgap={sol['nu_hat']:.3e}")
        thetas = directions_from_cov(Sigma, M, rng, power=power)

    sol = _fit_pass(thetas, X, in_A, in_B, basis_kw, split)
    hist.append(dict(pass_=n_passes, r=hist[-1]['r'], sol=sol, inc=None,
                     eigs=None))
    return dict(final=sol, history=hist)


# ===========================================================================
# Split-free, weight-aware variants
#
# Audit of what ids_path consumes: (X, in_A, in_B) -- configurations and the
# two state definitions. No lag time, no displacement, no frame ordering.
# "path scatter" bins STATIC frames by their committor value; the name refers
# to reaction progress, not to trajectories.
#
# Two places where it nonetheless assumed more, or less, than the method's
# real input contract, both fixed here:
#
#   (a) it ignored sample weights. The method accepts any ensemble that is
#       *reweightable* to equilibrium (MBAR/umbrella), so the scatter must be
#       weighted or it silently assumes unbiased sampling.
#   (b) rank selection used a halfset split. Splitting a bag of samples needs
#       no ordering, but on correlated frames a random split is optimistic and
#       doing it properly needs block splitting, i.e. frame ordering. The
#       permutation rule below removes the split entirely.
# ===========================================================================

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
    Sw /= max(sum(w[labels >= 0]), 1e-300)
    mbar = (ps[:, None] * ms).sum(0)
    Mc = ms - mbar
    Sb = (Mc * ps[:, None]).T @ Mc
    if not whiten:
        return Sb, np.eye(d), labels
    Sw = Sw + shrink * (np.trace(Sw) / d) * np.eye(d)
    lam, V = eigh(Sw)
    lam = np.maximum(lam, 1e-12 * max(lam.max(), 1e-300))
    Wh = (V / np.sqrt(lam)) @ V.T                  # Sw^{-1/2}; see note above
    return Wh @ Sb @ Wh, Wh, labels


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


def ids_path2(X, in_A, in_B, w=None, M=256, n_passes=3, basis_kw=None,
              rng=None, split=None, n_path_bins=8, whiten=True, power=1.0,
              rank_rule='permutation', thresh=0.5, rmax=None, n_perm=25,
              verbose=False):
    """IDS-path, weight-aware, with a split-free rank rule.

    Inputs are exactly the method's: configurations X, the two state
    definitions, and (optionally) per-frame equilibrium weights. With
    rank_rule='permutation' no data split is used anywhere in the subspace
    estimation, so no frame ordering is required or implied.
    """
    rng = np.random.default_rng(rng)
    basis_kw = basis_kw or dict(n_bins=200, bw_bins=1.5)
    N, d = X.shape
    if split is None:
        split = iid_split(N, rng=rng)
    thetas = co.isotropic_directions(M, d, rng=rng)
    hist = []
    for k in range(n_passes):
        sol = _fit_pass_w(thetas, X, in_A, in_B, basis_kw, split, w)
        qh = co.clip01(co.evaluate(sol['basis'], sol['w'], sol['c'], X))
        S, Wh, lab = path_scatter_w(X, qh, in_A, in_B, w=w,
                                    n_bins=n_path_bins, whiten=whiten)
        if rank_rule == 'permutation':
            r, obs, thr = select_rank_permutation(X, lab, Wh, w=w,
                                                  n_perm=n_perm, rmax=rmax,
                                                  rng=rng)
            diag = (obs[:4], thr[:4])
        else:
            i1, i2 = split
            m1 = np.zeros(N, bool); m1[i1] = True
            S1, _, _ = path_scatter_w(X[m1], qh[m1], in_A[m1], in_B[m1],
                                      w=None if w is None else w[m1],
                                      n_bins=n_path_bins, whiten=whiten)
            S2, _, _ = path_scatter_w(X[~m1], qh[~m1], in_A[~m1], in_B[~m1],
                                      w=None if w is None else w[~m1],
                                      n_bins=n_path_bins, whiten=whiten)
            r, inc = select_rank_incremental(S1, S2, thresh=thresh, rmax=rmax)
            diag = inc[:4]
        Sigma = Wh @ spiked_covariance(S, r) @ Wh.T
        Sigma = 0.5 * (Sigma + Sigma.T)
        hist.append(dict(pass_=k, r=r, sol=sol, diag=diag))
        if verbose:
            print(f"    pass {k}: r={r}")
        thetas = directions_from_cov(Sigma, M, rng, power=power)
    sol = _fit_pass_w(thetas, X, in_A, in_B, basis_kw, split, w)
    hist.append(dict(pass_=n_passes, r=hist[-1]['r'], sol=sol, diag=None))
    return dict(final=sol, history=hist)


def _fit_pass_w(thetas, X, in_A, in_B, basis_kw, split, w=None, n_bands=12):
    bs = co.build_basis(thetas, X, in_A, in_B, sample_weights=w, **basis_kw)
    i1, i2 = split
    w1 = None if w is None else w[i1]
    w2 = None if w is None else w[i2]
    G1, *_ = co.assemble(bs, X[i1], in_A[i1], in_B[i1], sample_weights=w1)
    G2, *_ = co.assemble(bs, X[i2], in_A[i2], in_B[i2], sample_weights=w2)
    Gr, info = halfset_eigen_regularize(G1, G2, n_bands=n_bands)
    G, a, b = co.assemble(bs, X, in_A, in_B, sample_weights=w)
    sol = co.solve_dual(Gr, b - a, G_is_regularised=True)
    sol.update(basis=bs, a=a, b=b, delta=b - a, c=float(-a @ sol['w']))
    return sol


# ===========================================================================
# Linear-in-d IDS-path
#
# The dense version forms d x d objects (Sw, its inverse square root, the
# scatter) and eigendecomposes them: O(N d^2 + d^3) per pass, against the
# core method's O(N d M). That is fine for the paper's feature spaces
# (d ~ 50-150 for torsion features) but breaks for raw Cartesian or contact
# representations. It is also unnecessary, because both objects are
# structurally low rank:
#
#   * the between-bin scatter Sb has rank <= L-1 BY CONSTRUCTION (L bins),
#     so it lives in the span of the L bin means -- never a d x d matrix;
#   * the useful part of the within-bin covariance Sw is diagonal + low rank,
#     which is the standard high-dimensional shrinkage model and is exactly
#     the spiked structure used elsewhere here.
#
# Everything below is O(N d (p + L + n_perm) + d (p+L)^2), linear in d.
# ===========================================================================

def _randomized_pca(Z, p, rng, oversample=10, n_iter=2):
    """Rank-p randomized SVD of Z (N x d): returns (U (d x p), s2 (p,)) with
    Z'Z/N ~ U diag(s2) U'.  Cost O(N d p)."""
    rng = np.random.default_rng(rng)
    N, d = Z.shape
    k = min(d, p + oversample)
    Om = rng.standard_normal((d, k)).astype(Z.dtype)
    Y = Z @ Om
    for _ in range(n_iter):
        Y = Z @ (Z.T @ Y)
    Q, _ = np.linalg.qr(Y)
    B = Q.T @ Z                       # k x d
    Ub, sb, Vt = np.linalg.svd(B, full_matrices=False)
    U = Vt[:p].T.astype(np.float64)
    s2 = (sb[:p].astype(np.float64) ** 2) / N
    return U, s2


class LowRankWhitener:
    """Sw ~ D^{1/2} [ U diag(s) U' + tau (I - UU') ] D^{1/2},  D = diag(v).

    Two corrections over the first version:

    (1) *Direction map-back uses W', not W.*  Whitening acts on points,
        y = W x with W Sw W' = I.  A direction is a linear FUNCTIONAL:
        theta'x = (W^{-T}theta)'y, so a whitened direction u maps back to
        theta = W' u.  Since W'W = Sw^{-1} for ANY valid whitener, the W'
        map-back is invariant to which whitener you picked, and the rank-1
        case reproduces the LDA axis Sw^{-1}(m_B - m_A).  Applying W instead
        gives W W (m_B - m_A), which is whitener-dependent and wrong.  The
        dense version was accidentally correct because Sw^{-1/2} is symmetric.

    (2) *The spike count is chosen, not fixed.*  Forcing a rank-p spike onto
        a nearly flat within-group spectrum injects spurious anisotropy: the
        top-p sample eigenvalues are upward-biased by finite sampling, so the
        whitener systematically shrinks p arbitrary directions.  Keep only
        eigenvalues above the Marchenko-Pastur edge (1+sqrt(gamma))^2 * bulk,
        gamma = d/n_eff, and debias the survivors by the standard spiked
        relation.  An isotropic Sw then keeps ZERO spikes and the whitener
        collapses to diagonal scaling, which is the right answer.
    """

    def __init__(self, X, labels, w, p=64, shrink=0.02, rng=None,
                 mp_factor=1.0, debias=True, exact=False):
        N, d = X.shape
        keep = labels >= 0
        w = w / w[keep].sum()
        gs = np.unique(labels[keep])
        rows = int(keep.sum())
        Z = np.empty((rows, d), dtype=np.float32)
        ww = np.empty(rows)
        pos = 0
        for g in gs:
            m = labels == g
            n = int(m.sum())
            mu = (w[m][:, None] * X[m]).sum(0) / w[m].sum()
            Z[pos:pos + n] = (X[m] - mu).astype(np.float32)
            ww[pos:pos + n] = w[m]
            pos += n
        sw = np.sqrt(ww / ww.sum())
        Zw = Z * sw[:, None].astype(np.float32)
        v = (Zw.astype(np.float64) ** 2).sum(0)
        v = v + shrink * v.mean()
        self.v = np.maximum(v, 1e-300)
        Zs = Zw / np.sqrt(self.v).astype(np.float32)

        n_eff = max((ww.sum() ** 2) / max((ww ** 2).sum(), 1e-300)
                    - len(gs), 2.0)
        tr = float((Zs.astype(np.float64) ** 2).sum())
        bulk = tr / d
        gamma = d / n_eff
        edge = bulk * (1.0 + np.sqrt(gamma)) ** 2 * mp_factor

        pmax = int(min(p, d - 1, rows - 1))
        if exact:
            Sw = (Zs.astype(np.float64).T @ Zs.astype(np.float64))
            ev, EV = eigh(Sw)
            ev = np.maximum(ev, 1e-12 * max(ev.max(), 1e-300))
            self.U, self.s = EV, ev
            self.tau = ev.min()
            self.d, self.n_spikes, self.edge, self.bulk = d, d, edge, bulk
            return
        U, s = _randomized_pca(Zs * np.sqrt(rows, dtype=np.float32), pmax, rng)
        s = np.maximum(s, 1e-12)
        if debias:
            # SOFT shrinkage, not a hard MP cut. Thresholding at the MP edge
            # keeps zero spikes once d >~ 200 here, collapsing the whitener to
            # pure diagonal and throwing away the modest-but-systematic
            # anisotropy that actually tilts the discriminant. The spiked
            # inversion lambda -> ell already maps near-bulk eigenvalues back
            # to the bulk, so it IS a soft threshold: keep every direction and
            # let the map decide how much of each to trust.
            lt = s / bulk
            disc = np.maximum((lt + 1.0 - gamma) ** 2 - 4.0 * lt, 0.0)
            lt = 0.5 * ((lt + 1.0 - gamma) + np.sqrt(disc))
            s = np.maximum(lt, 1.0) * bulk
        nk = int((s > edge).sum())
        self.U, self.s = U, s
        self.tau = max((tr - s.sum()) / max(d - s.shape[0], 1), 1e-12 * bulk)
        self.d, self.n_spikes, self.edge, self.bulk = d, nk, edge, bulk

    def _isqrt_tilde(self, Y):
        """Apply Sw_tilde^{-1/2} (symmetric) to rows of Y."""
        if self.U.shape[1] == 0:
            return Y / np.sqrt(self.tau)
        C = Y @ self.U
        return (Y - C @ self.U.T) / np.sqrt(self.tau) + \
            (C / np.sqrt(self.s)[None, :]) @ self.U.T

    def apply_W(self, Y):
        """W = Sw_tilde^{-1/2} D^{-1/2}: whitens POINTS. O(k d p)."""
        return self._isqrt_tilde(Y / np.sqrt(self.v)[None, :])

    def apply_WT(self, U_):
        """W' = D^{-1/2} Sw_tilde^{-1/2}: maps whitened DIRECTIONS back."""
        return self._isqrt_tilde(U_) / np.sqrt(self.v)[None, :]

    # backwards compatibility
    def inv_sqrt(self, Y):
        return self.apply_W(Y)


def path_scatter_lowrank(X, qhat, in_A, in_B, w=None, n_bins=8, p=64,
                         shrink=0.02, rng=None, labels=None, exact=False):
    """Whitened between-bin scatter in factored form.

    Returns (lam, V, tail, wh, labels): eigenvalues lam (<= L-1 of them) and
    eigenvectors V (d x len(lam)) of the whitened scatter, the tail level for
    the spiked model, and the whitener. No d x d matrix is ever formed.
    """
    N, d = X.shape
    w = np.full(N, 1.0 / N) if w is None else np.asarray(w, float)
    w = w / w.sum()
    if labels is None:
        labels = path_labels(qhat, in_A, in_B, w, n_bins)
    wh = LowRankWhitener(X, labels, w, p=p, shrink=shrink, rng=rng,
                         exact=exact)
    gs = [g for g in np.unique(labels) if g >= 0]
    Mm, ps = [], []
    for g in gs:
        m = labels == g
        sw = w[m].sum()
        if sw <= 0:
            continue
        Mm.append((w[m][:, None] * X[m]).sum(0) / sw); ps.append(sw)
    Mm = np.array(Mm); ps = np.array(ps); ps = ps / ps.sum()
    Yl = wh.apply_W(Mm)                        # L x d, whitened bin means
    Yc = Yl - (ps[:, None] * Yl).sum(0)
    A = (Yc * np.sqrt(ps)[:, None])            # L x d ; Sb = A' A
    Gm = A @ A.T                               # L x L
    ev, W = np.linalg.eigh(Gm)
    o = np.argsort(-ev); ev, W = np.maximum(ev[o], 0.0), W[:, o]
    keep = ev > 1e-14 * max(ev[0], 1e-300)
    ev, W = ev[keep], W[:, keep]
    V = (A.T @ W) / np.sqrt(ev)[None, :]       # d x k, orthonormal
    return ev, V, wh, labels


def select_rank_permutation_lowrank(X, labels, wh, w=None, n_perm=25, q=0.95,
                                    rmax=None, rng=None):
    """Horn parallel analysis in the factored representation. O(n_perm N d)."""
    rng = np.random.default_rng(rng)
    N, d = X.shape
    w = np.full(N, 1.0 / N) if w is None else np.asarray(w, float) / np.sum(w)
    keep = labels >= 0
    idx = np.flatnonzero(keep)
    rmax = rmax or 20

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
        Yl = wh.apply_W(Mm)
        Yc = Yl - (ps[:, None] * Yl).sum(0)
        A = Yc * np.sqrt(ps)[:, None]
        return np.sort(np.linalg.eigvalsh(A @ A.T))[::-1]

    obs = spec(labels)
    null = np.zeros((n_perm, len(obs)))
    for t in range(n_perm):
        lb = labels.copy()
        lb[idx] = labels[rng.permutation(idx)]
        s = spec(lb)
        null[t, :len(s)] = s[:null.shape[1]]
    thr = np.quantile(null, q, axis=0)
    r = 0
    for i in range(min(rmax, len(obs))):
        if obs[i] > thr[i]:
            r = i + 1
        else:
            break
    return max(r, 1), obs, thr


def directions_lowrank(lam, V, wh, r, d, M, rng, power=1.0):
    """Draw theta ~ N(0, Sigma) with Sigma = Sw^{-1/2}[tau I + sum_{i<=r}
    (lam_i - tau) v_i v_i'] Sw^{-1/2}, without forming Sigma.  O(M d (p+r))."""
    rng = np.random.default_rng(rng)
    r = int(np.clip(r, 1, len(lam)))
    tail = lam[r:]
    tau = (tail.sum() / max(d - r, 1)) if len(tail) else 0.0
    tau = max(tau, 1e-8 * max(lam[0], 1e-300))
    a = np.sqrt(np.maximum(lam[:r], tau) ** power)
    b = np.sqrt(tau ** power)
    Z = rng.standard_normal((M, d))
    C = Z @ V[:, :r]
    Y = b * Z + C * (a - b)[None, :] @ V[:, :r].T
    T = wh.apply_WT(Y)          # directions are functionals: W', not W
    n = np.linalg.norm(T, axis=1, keepdims=True)
    n[n[:, 0] <= 0] = 1.0
    return T / n


def ids_path_lowrank(X, in_A, in_B, w=None, M=256, n_passes=3, basis_kw=None,
                     rng=None, split=None, n_path_bins=8, p=64, power=1.0,
                     n_perm=25, rmax=None, verbose=False, exact=False):
    """IDS-path with no d x d object anywhere. Linear in d."""
    rng = np.random.default_rng(rng)
    basis_kw = basis_kw or dict(n_bins=200, bw_bins=1.5)
    N, d = X.shape
    if split is None:
        split = iid_split(N, rng=rng)
    thetas = co.isotropic_directions(M, d, rng=rng)
    hist = []
    for k in range(n_passes):
        sol = _fit_pass_w(thetas, X, in_A, in_B, basis_kw, split, w)
        qh = co.clip01(co.evaluate(sol['basis'], sol['w'], sol['c'], X))
        lam, V, wh, lab = path_scatter_lowrank(X, qh, in_A, in_B, w=w,
                                               n_bins=n_path_bins, p=p, rng=rng,
                                               exact=exact)
        r, obs, thr = select_rank_permutation_lowrank(X, lab, wh, w=w,
                                                      n_perm=n_perm,
                                                      rmax=rmax, rng=rng)
        hist.append(dict(pass_=k, r=r, sol=sol))
        if verbose:
            print(f"    pass {k}: r={r}")
        thetas = directions_lowrank(lam, V, wh, r, d, M, rng, power=power)
    sol = _fit_pass_w(thetas, X, in_A, in_B, basis_kw, split, w)
    hist.append(dict(pass_=n_passes, r=hist[-1]['r'], sol=sol))
    return dict(final=sol, history=hist)


# ===========================================================================
# IDS-path, exact and linear in d
#
# The diagnostic in t15 is decisive: with an EXACT Sw the low-rank pipeline
# matches or beats the dense version (0.0080/0.0108/0.0161 vs
# 0.0105/0.0114/0.0176 at d = 96/192/384), while every approximate Sw model
# loses a factor ~2. Randomized PCA cannot isolate the informative directions
# of a near-flat within-bin spectrum -- power iteration needs a spectral gap,
# and there isn't one. So do not approximate Sw at all.
#
# The observation that removes the need to: the algorithm never wants
# Sw^{-1/2} as an operator. It wants the generalised eigenproblem
#
#       Sb v = lambda Sw v          (multi-class LDA)
#
# and Sb = A'A has rank <= L-1 by construction. So solving Sw Y = A' for the
# L bin-mean vectors reduces everything to an L x L eigenproblem, and the
# generalised eigenvectors ARE the feature-space discriminant directions --
# no whitener, no map-back, no d x d object.
#
# Sw x = a is solved by conjugate gradients using only Sw-matvecs, which
# stream over frames in O(N d), preconditioned by the cheap diagonal +
# low-rank model. The model no longer has to be accurate -- only good enough
# to precondition -- so its failure mode is a few extra CG iterations rather
# than a wrong answer.
#
# Total: O(k L N d) with k ~ 10-40 CG iterations. Linear in d, exact.
# ===========================================================================

class _SwOperator:
    """Streaming within-group covariance operator. Sw v in O(rows*d)."""

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
        Z = np.empty((len(keep), d), dtype=np.float32)
        for g in np.unique(lab):
            m = lab == g
            sw = ww[m].sum()
            mu = (ww[m][:, None] * X[keep][m]).sum(0) / max(sw, 1e-300)
            Z[m] = (X[keep][m] - mu).astype(np.float32)
        self.Z = Z * np.sqrt(ww)[:, None].astype(np.float32)
        self.d = d
        self.diag = (self.Z.astype(np.float64) ** 2).sum(0)
        self.eps = shrink * self.diag.mean()
        self.diag = self.diag + self.eps

    def matvec(self, V):
        """V: (d,) or (d,k)."""
        V32 = np.asarray(V, dtype=np.float32)
        out = (self.Z.T @ (self.Z @ V32)).astype(np.float64)
        return out + self.eps * np.asarray(V, dtype=np.float64)


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
                          rng=None, tol=1e-8, maxiter=200, precond_rank=32,
                          labels=None, shrink=0.1):
    """Generalised eigenproblem Sb v = lambda Sw v, solved with L CG solves.

    Returns (lam, Theta, n_cg, labels): eigenvalues and the feature-space
    discriminant directions (d x k, k <= L-1), Sw-orthonormal.
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

    # diagonal + low-rank preconditioner: only has to precondition, not be right
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


def directions_exact(lam, Theta, r, d, M, rng, power=1.0, floor_frac=None):
    """theta ~ isotropic floor + the top-r discriminant directions.

    The floor is isotropic in FEATURE space (not in whitened space), which is
    both cheaper and closer to the advertised behaviour: with no structure the
    sampler returns plain isotropic directions.
    """
    rng = np.random.default_rng(rng)
    k = Theta.shape[1]
    r = int(np.clip(r, 1, k))
    tail = lam[r:]
    tau = (tail.sum() / max(d - r, 1)) if len(tail) else 0.0
    tau = max(tau, 1e-8 * max(lam[0], 1e-300))
    if floor_frac is not None:
        tau = floor_frac * lam[0]
    # QR-orthonormalise the retained discriminant directions before assigning
    # amplitudes. They are Sw-orthogonal, not Euclidean-orthogonal, so QR does
    # rotate them within their span -- but building Sigma from the raw
    # (non-orthogonal) directions double-counts variance wherever two of them
    # are near-parallel, which measured 2x worse. QR spans the same subspace
    # and keeps the covariance well behaved.
    Q, _ = np.linalg.qr(Theta[:, :r])
    a = np.sqrt(np.maximum(lam[:r] / tau, 1.0) ** power)
    Z = rng.standard_normal((M, d))
    C = Z @ Q
    Y = Z + C * (a - 1.0)[None, :] @ Q.T
    n = np.linalg.norm(Y, axis=1, keepdims=True)
    n[n[:, 0] <= 0] = 1.0
    return Y / n


def ids_path_exact(X, in_A, in_B, w=None, M=256, n_passes=3, basis_kw=None,
                   rng=None, split=None, n_path_bins=8, power=1.0, n_perm=25,
                   rmax=None, n_sub=None, tol=1e-8, maxiter=200,
                   shrink=0.1, verbose=False):
    """IDS-path: exact Sw, linear in d, no d x d object anywhere."""
    rng = np.random.default_rng(rng)
    basis_kw = basis_kw or dict(n_bins=200, bw_bins=1.5)
    N, d = X.shape
    if split is None:
        split = iid_split(N, rng=rng)
    thetas = co.isotropic_directions(M, d, rng=rng)
    hist = []
    kw = dict(n_bins=n_path_bins, n_sub=n_sub, tol=tol, maxiter=maxiter,
              shrink=shrink)
    for k in range(n_passes):
        sol = _fit_pass_w(thetas, X, in_A, in_B, basis_kw, split, w)
        qh = co.clip01(co.evaluate(sol['basis'], sol['w'], sol['c'], X))
        lam, Th, n_cg, lab, op = path_directions_exact(X, qh, in_A, in_B, w=w,
                                                       rng=rng, **kw)
        r, obs, thr = select_rank_perm_exact(X, lab, lam, w=w, n_perm=n_perm,
                                             rmax=rmax, rng=rng, op=op,
                                             tol=tol, maxiter=maxiter)
        hist.append(dict(pass_=k, r=r, sol=sol, n_cg=n_cg, lam=lam))
        if verbose:
            print(f"    pass {k}: r={r}  CG iters={n_cg}")
        thetas = directions_exact(lam, Th, r, d, M, rng, power=power)
    sol = _fit_pass_w(thetas, X, in_A, in_B, basis_kw, split, w)
    hist.append(dict(pass_=n_passes, r=hist[-1]['r'], sol=sol, n_cg=0))
    return dict(final=sol, history=hist)


# ===========================================================================
# The exact isotropic floor, without a square root
#
# The dense sampler draws theta ~ N(0, Sigma) with
#
#     Sigma = tau Sw^{-1} + sum_{i<=r} (lam_i - tau) th_i th_i'
#
# (th_i the Sw-orthonormal generalised eigenvectors, th' Sw th = 1). The
# linear-in-d version used tau*I for the floor because sampling N(0, Sw^{-1})
# looks like it needs Sw^{-1/2}. It does not:
#
#     u = Z'g + sqrt(eps) h   has Cov(u) = Z'Z + eps I = Sw     (free)
#     x = Sw^{-1} u           has Cov(x) = Sw^{-1} Sw Sw^{-1} = Sw^{-1}
#
# so one block-CG solve with M right-hand sides gives M exact draws from
# N(0, Sw^{-1}) using only Sw-matvecs. The floor is a background distribution,
# so a loose CG tolerance is fine.
# ===========================================================================

def sample_inv_sw(op, M, rng, tol=1e-3, maxiter=60, Minv=None):
    """M exact draws from N(0, Sw^{-1}); returns (d x M, n_cg)."""
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


def ids_path_exact2(X, in_A, in_B, w=None, M=256, n_passes=3, basis_kw=None,
                    rng=None, split=None, n_path_bins=8, n_perm=25, rmax=None,
                    n_sub=None, n_floor_sub=6000, tol=1e-6, floor_tol=1e-3,
                    maxiter=200, shrink=0.1, verbose=False):
    """IDS-path: exact Sw, exact Sw^{-1} floor, linear in d, no d x d object."""
    rng = np.random.default_rng(rng)
    basis_kw = basis_kw or dict(n_bins=200, bw_bins=1.5)
    N, d = X.shape
    if split is None:
        split = iid_split(N, rng=rng)
    thetas = co.isotropic_directions(M, d, rng=rng)
    hist = []
    kw = dict(n_bins=n_path_bins, n_sub=n_sub, tol=tol, maxiter=maxiter,
              shrink=shrink)
    for k in range(n_passes):
        sol = _fit_pass_w(thetas, X, in_A, in_B, basis_kw, split, w)
        qh = co.clip01(co.evaluate(sol['basis'], sol['w'], sol['c'], X))
        lam, Th, n_cg, lab, op = path_directions_exact(X, qh, in_A, in_B, w=w,
                                                       rng=rng, **kw)
        r, obs, thr = select_rank_perm_exact(X, lab, lam, w=w, n_perm=n_perm,
                                             rmax=rmax, rng=rng, op=op,
                                             tol=tol, maxiter=maxiter)
        op_f = op if n_floor_sub is None else _SwOperator(
            X, lab, w if w is not None else np.full(N, 1.0 / N),
            n_sub=n_floor_sub, rng=rng, shrink=shrink)
        thetas, n_cg2 = directions_exact_floor(lam, Th, r, op_f, M, rng,
                                               tol=floor_tol)
        hist.append(dict(pass_=k, r=r, sol=sol, n_cg=n_cg, n_cg_floor=n_cg2,
                         lam=lam))
        if verbose:
            print(f"    pass {k}: r={r}  CG={n_cg}  floorCG={n_cg2}")
    sol = _fit_pass_w(thetas, X, in_A, in_B, basis_kw, split, w)
    hist.append(dict(pass_=n_passes, r=hist[-1]['r'], sol=sol, n_cg=0,
                     n_cg_floor=0))
    return dict(final=sol, history=hist)
