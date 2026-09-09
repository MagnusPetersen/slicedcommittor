"""
Matrix-free Gram diagnostics (RECOVAR transfer, §4 / §4b).

The GramOperator matvec identity, the explicit diagonal, and the randomised
Nystrom low-rank factorisation of G -- none of which form the M x M Gram
matrix.

Deliberate omission: the prototype's plain cg() solver is NOT ported. CG on
this operator is refuted (REPORT.md §4): the redundancy that makes G low
rank is exactly what makes it ill-conditioned (lambda_1/lambda_M ~ 8e12 at
M = 512), and CG needed 470 iterations at M = 64 -- each a full O(d N M)
pass over the data. There is also no timing win over explicit BLAS-3
assembly below M ~ 2000, and the large-M strategy it was meant to enable is
itself refuted (isotropic M = 2048 does not close the gap to a concentrated
sampler once d >~ 12).

These tools are kept ONLY as an effective-rank diagnostic: rank-64 Nystrom
of an M = 512 Gram reproduces the full solve, and 99% of the trace sits in
~30 directions -- 512 random slices in d = 2 carry ~64 dimensions of
information.
"""
import numpy as np
from scipy.linalg import eigh

from .assemble import assemble_slice_values


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
        _, QP = assemble_slice_values(self.basis, self.X)
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


def gram_diagonal(basis, X, sample_weights=None, D=None, block=4096):
    """diag(G)_j = |theta_j|_D^2 * sum_n W_n q_j'(s_j^n)^2  (for preconditioning).

    ``D`` is the feature-space diffusion tensor; None means the identity. Before
    this argument existed the docstring promised the D-metric norm while the code
    computed the Euclidean one -- harmless while D = I was hardwired everywhere,
    wrong the moment a metric is supplied. Uses the same Cholesky factorisation
    as :class:`GramOperator`, so the two agree by construction.
    """
    N = X.shape[0]
    W = np.full(N, 1.0 / N) if sample_weights is None else sample_weights
    W = W / W.sum()
    d = np.zeros(basis.M)
    for n0 in range(0, N, block):
        n1 = min(n0 + block, N)
        S = (X[n0:n1] @ basis.thetas.T).T
        _, qp = basis.eval_at(S)
        d += (qp.astype(np.float64) ** 2) @ W[n0:n1]
    if D is None:
        return d * np.einsum('ij,ij->i', basis.thetas, basis.thetas)
    Y = np.linalg.cholesky(D).T @ basis.thetas.T          # (d, M), Y'Y = theta' D theta
    return d * np.einsum('ai,ai->i', Y, Y)


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
