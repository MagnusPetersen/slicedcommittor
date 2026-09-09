"""Held-out dual objective and the nested-M scree curve (RECOVAR secs. 3, 9).

The dual (unconstrained) form of the EBMC solve,

    h(w) = 2 (b-a)'w - w'Gw,   max_w h = M_gap = 1/nu_AB (population),

makes held-out evaluation trivial: fit w on one half of the frames, evaluate
h on the other. Overfitting inflates h_train, so 1/h_train is biased *below*
nu_AB; h_test is the honest estimate.

Validated lessons (``_reference/REPORT.md`` secs. 3, 10):

* The train-test gap is a reliable overfitting diagnostic, and
  ``argmax h_test`` over nested M picks the RMSE-optimal M well. But
  ``1/h_train <= nu_AB <= 1/h_test`` is NOT a confidence bracket -- it held
  only 8/30 times, because other biases (barrier undersampling,
  regularization) can push both ends the same way.
* On time-correlated frames the split must be a BLOCK split
  (``splits.block_split``): a random frame split does not merely understate
  the overfitting diagnostic, it inverts its sign (measured -2.7% "no
  overfitting" against +31..64% for blocks at tau_int ~ 55 frames). Check
  convergence in the block length.
"""
import numpy as np

from .assemble import assemble, clip01, dual_objective, evaluate, solve_dual
from .basis import build_basis


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
        basis = build_basis(thetas, X1, A1, B1, **basis_kw)
    else:
        basis = build_basis(thetas, X, in_A, in_B, **basis_kw)

    G1, a1, b1 = assemble(basis, X1, A1, B1)
    G2, a2, b2 = assemble(basis, X2, A2, B2)
    d1, d2 = b1 - a1, b2 - a2

    sol = solve_dual(G1, d1, tikhonov)
    w = sol['w_dual']
    h_tr = dual_objective(w, G1, d1)
    h_te = dual_objective(w, G2, d2)

    out = dict(h_train=h_tr, h_test=h_te,
               nu_train=1.0 / h_tr if h_tr > 0 else np.inf,
               nu_test=1.0 / h_te if h_te > 0 else np.inf,
               w_dual=w, w=sol['w'], c=float(-a1 @ sol['w']),
               G1=G1, G2=G2, d1=d1, d2=d2)
    if return_basis:
        out['basis'] = basis
    return out


def nested_M_curve(thetas, X, in_A, in_B, split, Ms, tikhonov=1e-3,
                   basis_kw=None, q_ref=None, Xq=None):
    """h_train / h_test over nested slice subsets (first M of a fixed pool)."""
    basis_kw = basis_kw or {}
    i1, i2 = split
    X1, X2 = X[i1], X[i2]
    A1, B1, A2, B2 = in_A[i1], in_B[i1], in_A[i2], in_B[i2]
    full = build_basis(thetas, X1, A1, B1, **basis_kw)
    rows = []
    for M in Ms:
        sub = full.subset(np.arange(M))
        G1, a1, b1 = assemble(sub, X1, A1, B1)
        G2, a2, b2 = assemble(sub, X2, A2, B2)
        d1, d2 = b1 - a1, b2 - a2
        sol = solve_dual(G1, d1, tikhonov)
        wd = sol['w_dual']
        h_tr = dual_objective(wd, G1, d1)
        h_te = dual_objective(wd, G2, d2)
        rmse = np.nan
        if q_ref is not None and Xq is not None:
            qh = clip01(evaluate(sub, sol['w'], float(-a1 @ sol['w']), Xq))
            rmse = float(np.sqrt(np.mean((qh - q_ref) ** 2)))
        rows.append((M, h_tr, h_te, 1.0 / h_tr, 1.0 / h_te if h_te > 0 else np.inf, rmse))
    return np.array(rows)
