"""
Uncertainty quantification for the sliced committor (RECOVAR transfer, §7).

Delta-method and block-bootstrap standard errors on nu_hat = 1/M_gap and,
optionally, on qbar at query points.

CAVEAT (REPORT.md §7): both estimators are VARIANCE-only. They agree with
each other to ~10% and match the empirical spread over independent datasets
when variance dominates (at N = 30000: 3.07e-4 vs 3.07e-4, 65% coverage
against a 68% target). But they are blind to bias: at N = 6000, where the
error is dominated by barrier-undersampling bias (mean nu_hat/nu = 0.74),
the reported standard errors are ~6x too small. Nothing here catches barrier
undersampling.

Ported from the numpy prototype at _reference/slicedcv/recovar.py; the
per-frame slice values, the dual solve and the evaluation are delegated to
recovar.assemble.
"""
import numpy as np

from .assemble import assemble_slice_values, solve_dual, evaluate


def uq_bootstrap(basis, X, in_A, in_B, n_boot=40, block_len=1, rng=None,
                 tikhonov=1e-3, group=None, Xq=None):
    """Block bootstrap over frames: resample blocks, refit, collect the
    spread of w, nu_hat = 1/M_gap, and (optionally) qbar at query points.

    Variance-only (see module docstring): the bootstrap spread is ~6x too
    small when bias dominates the error (REPORT.md §7). On time-correlated
    data use block_len >~ the integrated autocorrelation time.
    """
    rng = np.random.default_rng(rng)
    N = X.shape[0]
    QV, QP = assemble_slice_values(basis, X)
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
        sol = solve_dual(G, b - a, tikhonov)
        nus.append(sol['nu_hat']); ws.append(sol['w'])
        if Xq is not None:
            qs.append(evaluate(basis, sol['w'], float(-a @ sol['w']), Xq))
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

    Variance-only (see module docstring): ~6x too small when bias dominates
    (REPORT.md §7).
    """
    N = X.shape[0]
    QV, QP = assemble_slice_values(basis, X)
    QV = QV.astype(np.float64); QP = QP.astype(np.float64)
    Theta = basis.thetas @ basis.thetas.T
    C = (QP @ QP.T) / N
    G = Theta * C
    a = QV[:, in_A].mean(1); b = QV[:, in_B].mean(1)
    d = b - a
    sol = solve_dual(G, d, tikhonov)
    wd = sol['w_dual']; Mg = sol['M_gap']

    # per-frame influence on M_gap:
    #   from d : 2 wd' (b - a) contributions
    #   from G : - wd' G wd  ->  per frame -(theta-weighted) (wd'.qp_n)^2
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
        QVq, _ = assemble_slice_values(basis, Xq)
        out['q_se'] = np.sqrt(np.maximum(
            np.einsum('jn,jk,kn->n', QVq.astype(np.float64), Cw,
                      QVq.astype(np.float64)), 0.0))
    return out
