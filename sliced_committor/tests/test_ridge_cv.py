"""Gates for the calibration-free CV ridge (``tikhonov='cv'``).

The rule is benchmarked in ``experiments/ridge_theory_rules.py`` and written up in
``docs/ridge_rule.md``.  These tests pin the properties that make it correct
rather than the numbers it happens to produce on any one system.
"""

import numpy as np
import pytest

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402

from sliced_committor.core._ridge_cv import (  # noqa: E402
    RIDGE_HI, RIDGE_LO, fold_gram_blocks, make_folds, select_ridge_cv,
)
from sliced_committor.core.weights import _resolve_eta  # noqa: E402


# --------------------------------------------------------------------------- #
# folds
# --------------------------------------------------------------------------- #
def test_contiguous_folds_are_blocks_not_a_permutation():
    """MD frames are serially correlated; blocks are the only honest split.

    A permutation split puts consecutive (near-duplicate) frames on both sides,
    so the held-out cap comes out optimistic and the selector under-shrinks.
    """
    f = make_folds(100, 5, contiguous=True)
    assert (np.diff(f) >= 0).all(), "contiguous folds must be non-decreasing"
    assert np.bincount(f).tolist() == [20] * 5
    p = make_folds(100, 5, contiguous=False)
    assert not (np.diff(p) >= 0).all()


def test_folds_partition_every_sample_exactly_once():
    for N, K in ((97, 5), (1000, 7), (13, 13)):
        f = make_folds(N, K, contiguous=True)
        assert f.shape == (N,)
        assert f.min() == 0 and f.max() == K - 1
        assert np.bincount(f, minlength=K).sum() == N


# --------------------------------------------------------------------------- #
# the fold Gram blocks
# --------------------------------------------------------------------------- #
def _toy(M=12, N=600, dim=3, seed=0):
    rng = np.random.default_rng(seed)
    th = rng.normal(size=(M, dim))
    th /= np.linalg.norm(th, axis=1, keepdims=True)
    F = jnp.asarray(rng.normal(size=(M, N)) + 1.0)
    W = jnp.asarray(np.full(N, 1.0 / N))
    return th, jnp.asarray(th @ th.T), F, W


def test_fold_blocks_recombine_to_the_full_gram():
    """G is a weighted MEAN over samples, so the weighted average of the fold
    blocks must reproduce it exactly -- that identity is what lets the selector
    build every train/test combination without re-assembling."""
    from sliced_committor.core.gram import _assemble_gram_matrix

    th, cos, F, W = _toy()
    K = 5
    fold_of = make_folds(F.shape[1], K, contiguous=True)
    G_folds, w_folds = fold_gram_blocks(F, W, cos, fold_of, K)
    G_full = np.asarray(_assemble_gram_matrix(F, W, cos), np.float64)
    G_rec = np.tensordot(w_folds, G_folds, axes=(0, 0)) / w_folds.sum()
    assert np.allclose(G_rec, G_full, rtol=1e-10, atol=1e-14)


def test_fold_blocks_are_psd_and_weights_sum_to_one():
    th, cos, F, W = _toy()
    K = 4
    G_folds, w_folds = fold_gram_blocks(F, W, cos, make_folds(F.shape[1], K), K)
    assert np.isclose(w_folds.sum(), 1.0, rtol=1e-12)
    for Gk in G_folds:
        assert np.linalg.eigvalsh(0.5 * (Gk + Gk.T)).min() > -1e-9


def test_empty_fold_is_an_error_not_a_silent_zero():
    th, cos, F, W = _toy(N=3)
    with pytest.raises(ValueError, match="empty"):
        fold_gram_blocks(F, W, cos, np.array([0, 0, 0]), 5)


# --------------------------------------------------------------------------- #
# the selector
# --------------------------------------------------------------------------- #
def _cv_inputs(M=10, N=800, dim=3, K=5, seed=1):
    """Fold blocks for a toy problem with a genuine A/B split."""
    rng = np.random.default_rng(seed)
    th = rng.normal(size=(M, dim))
    th /= np.linalg.norm(th, axis=1, keepdims=True)
    cos = th @ th.T
    fold_of = make_folds(N, K, contiguous=True)
    G_folds, w_folds = [], []
    a_f, b_f, wA, wB = [], [], [], []
    for k in range(K):
        m = fold_of == k
        n = int(m.sum())
        Fk = rng.normal(size=(M, n)) * 0.4 + 1.0
        G_folds.append(cos * (Fk @ Fk.T / n))
        w_folds.append(n / N)
        a_f.append(rng.normal(size=M) * 0.05 + 0.1)
        b_f.append(rng.normal(size=M) * 0.05 + 0.9)
        wA.append(n / 3)
        wB.append(n / 3)
    return (np.stack(G_folds), np.array(w_folds), np.stack(a_f), np.stack(b_f),
            np.array(wA), np.array(wB), np.ones(M, bool))


def test_selector_returns_a_positive_ridge_inside_its_grid():
    args = _cv_inputs()
    out = select_ridge_cv(*args)
    assert out["ridge"] > 0
    assert RIDGE_LO * out["anchor"] <= out["ridge"] <= RIDGE_HI * out["anchor"]
    assert out["curve"].shape == out["grid"].shape == out["se"].shape


def test_one_se_rule_never_shrinks_less_than_the_bare_argmin():
    """A tie-break must move toward MORE regularisation, never less.

    Under-shrinking is the failure mode that brings the M-degradation back, so a
    tie-break that could go the other way would be actively harmful.
    """
    args = _cv_inputs()
    lo = select_ridge_cv(*args, one_se=False)
    for mode in (True, "paired"):
        hi = select_ridge_cv(*args, one_se=mode)
        assert hi["ridge"] >= lo["ridge"], mode
        assert hi["idx"] >= hi["idx_argmin"], mode


def test_paired_tolerance_is_never_looser_than_the_level_tolerance():
    """The paired SE removes the common per-fold offset in the cap LEVEL, so it
    can only be tighter -- which is what stops the tie-break walking up a flat
    curve (2.0-2.3x the oracle on the 2D benchmark with the level SE)."""
    args = _cv_inputs()
    lvl = select_ridge_cv(*args, one_se=True)
    par = select_ridge_cv(*args, one_se="paired")
    assert par["idx"] <= lvl["idx"]
    ok = np.isfinite(par["se_paired"]) & np.isfinite(par["se"])
    assert (par["se_paired"][ok] <= par["se"][ok] + 1e-12).all()


def test_default_is_the_bare_argmin():
    """The shipped default is the bare argmin.

    A 1-SE tie-break was the original default; on the paper's 2D panel it walks
    three grid points up a curve that is flat relative to its fold noise and
    costs +15.1% against +1.3% for the argmin, so it is opt-in now.
    """
    args = _cv_inputs()
    assert select_ridge_cv(*args)["idx"] == select_ridge_cv(*args, one_se=False)["idx"]
    assert select_ridge_cv(*args)["idx"] == select_ridge_cv(*args)["idx_argmin"]


def test_selector_is_invariant_to_rescaling_G():
    """The solve is invariant under G -> alpha G, so the SELECTED ridge must
    scale by exactly alpha and the chosen grid index must not move."""
    G, w, a, b, wA, wB, mask = _cv_inputs()
    base = select_ridge_cv(G, w, a, b, wA, wB, mask)
    for alpha in (1e-3, 1e3):
        s = select_ridge_cv(G * alpha, w, a, b, wA, wB, mask)
        assert s["idx"] == base["idx"]
        assert np.isclose(s["ridge"], base["ridge"] * alpha, rtol=1e-10)


def test_selector_ignores_invalid_directions():
    G, w, a, b, wA, wB, mask = _cv_inputs(M=10)
    mask2 = mask.copy()
    mask2[[2, 7]] = False
    out = select_ridge_cv(G, w, a, b, wA, wB, mask2)
    assert np.isfinite(out["ridge"]) and out["ridge"] > 0


def test_selector_needs_two_valid_directions():
    G, w, a, b, wA, wB, mask = _cv_inputs()
    bad = np.zeros_like(mask)
    bad[0] = True
    with pytest.raises(ValueError, match="fewer than two valid"):
        select_ridge_cv(G, w, a, b, wA, wB, bad)


# --------------------------------------------------------------------------- #
# the eta plumbing
# --------------------------------------------------------------------------- #
def test_ridge_abs_delivers_exactly_that_absolute_ridge():
    """``('ridge_abs', r)`` must survive the caller's median(diag G) rescale.

    The selectors choose a ridge in the units of G; if the anchor were applied on
    top, the delivered ridge would be off by median(diag G) -- silently, and by
    orders of magnitude.
    """
    rng = np.random.default_rng(0)
    M = 16
    d = rng.uniform(2.0, 5.0, M)
    G = jnp.asarray(np.diag(d))
    mask = jnp.ones(M, bool)
    med = float(np.median(d))
    for r in (1e-6, 3.3, 1e4):
        eta, _ = _resolve_eta(("ridge_abs", r), M, mask, N=1000, G=G)
        assert np.isclose(eta * med, r, rtol=1e-12), (eta * med, r)


def test_ridge_abs_requires_G():
    with pytest.raises(ValueError, match="requires G"):
        _resolve_eta(("ridge_abs", 1.0), 4, jnp.ones(4, bool), N=10, G=None)


def test_unknown_eta_tuple_is_rejected():
    with pytest.raises(ValueError, match="expected"):
        _resolve_eta(("nonsense", 1.0), 4, jnp.ones(4, bool), N=10, G=None)


# --------------------------------------------------------------------------- #
# stratification
# --------------------------------------------------------------------------- #
def test_stratified_folds_put_every_basin_in_every_fold():
    """Unstratified contiguous blocks fail on basin-sorted data.

    A trajectory that visits A before B -- or any array sorted by basin -- puts
    all of A in the early folds, and the held-out cap is undefined on a fold that
    holds no A or no B.  Stratifying keeps the blocks contiguous WITHIN each
    basin, so the anti-leakage property survives.
    """
    N, K = 3000, 5
    strata = np.concatenate([np.zeros(1000), np.ones(1000), np.full(1000, 2)])
    plain = make_folds(N, K, contiguous=True)
    assert len(set(strata[plain == 0])) == 1, "the unstratified split is basin-pure"

    strat = make_folds(N, K, contiguous=True, strata=strata)
    for k in range(K):
        assert set(np.unique(strata[strat == k])) == {0, 1, 2}
    # still blocks within a stratum: each stratum's fold labels are monotone
    for s in (0, 1, 2):
        lab = strat[strata == s]
        assert (np.diff(lab) >= 0).all()


def test_fold_blocks_recombine_under_a_non_contiguous_assignment():
    """Stratified labels are not sorted, so the block builder must gather."""
    from sliced_committor.core.gram import _assemble_gram_matrix

    th, cos, F, W = _toy(N=600)
    K = 4
    # UNBALANCED strata: a small basin advances through its folds much faster
    # than a large one, so the pooled labels are genuinely out of order.  Evenly
    # interleaved strata would still come out monotone and test nothing.
    strata = np.concatenate([np.zeros(400), np.ones(60), np.full(140, 2)])
    rng = np.random.default_rng(3)
    strata = strata[rng.permutation(600)]
    fold_of = make_folds(600, K, contiguous=True, strata=strata)
    assert not (np.diff(fold_of) >= 0).all()
    G_folds, w_folds = fold_gram_blocks(F, W, cos, fold_of, K)
    G_full = np.asarray(_assemble_gram_matrix(F, W, cos), np.float64)
    G_rec = np.tensordot(w_folds, G_folds, axes=(0, 0)) / w_folds.sum()
    assert np.allclose(G_rec, G_full, rtol=1e-10, atol=1e-14)


# --------------------------------------------------------------------------- #
# end to end
# --------------------------------------------------------------------------- #
def test_fit_committor_accepts_tikhonov_cv_and_reports_the_ridge():
    from sliced_committor.core.committor import fit_committor

    rng = np.random.default_rng(0)
    N = 8000
    X = rng.normal(size=(N, 2)) * 0.6
    X[: N // 2, 0] -= 1.5
    X[N // 2:, 0] += 1.5
    in_A, in_B = X[:, 0] < -1.2, X[:, 0] > 1.2

    q, fit = fit_committor(X, in_A=in_A, in_B=in_B, n_directions=24, seed=0,
                           return_details=True, n_bins=40, n_min=10,
                           binning_method="quantile", weights="ebmc",
                           weight_kwargs={"tikhonov": "cv"})
    info = fit.weights["ridge_cv"]
    assert info["ridge"] > 0 and not info["at_edge"]
    assert info["n_folds"] == 5
    vals = np.asarray(q(np.array([[-2.0, 0.0], [0.0, 0.0], [2.0, 0.0]])))
    assert vals[0] < 0.15 < vals[1] < 0.85 < vals[2]


def test_cv_refit_runs_end_to_end_and_is_deterministic():
    """``cv_refit`` rebuilds the 1D basis on each training fold.

    It is the only rule benchmarked that sees binning noise, and it is the one
    that works in both regimes (2D N=25000: 1.026/1.000/1.031 against 1.78/1.90/
    1.63 for the shared-basis `cv`).  It costs K basis builds, which is why it is
    opt-in rather than the default.
    """
    from sliced_committor.core.committor import fit_committor

    rng = np.random.default_rng(2)
    N = 6000
    X = rng.normal(size=(N, 2)) * 0.6
    X[: N // 2, 0] -= 1.5
    X[N // 2:, 0] += 1.5
    kw = dict(in_A=X[:, 0] < -1.2, in_B=X[:, 0] > 1.2, n_directions=16, seed=0,
              return_details=True, n_bins=30, n_min=10,
              binning_method="quantile", weights="ebmc")

    q1, f1 = fit_committor(X, weight_kwargs={"tikhonov": "cv_refit"}, **kw)
    q2, f2 = fit_committor(X, weight_kwargs={"tikhonov": "cv_refit"}, **kw)
    assert np.isclose(float(f1["M_gap"] if isinstance(f1, dict) else f1.weights["M_gap"]),
                      float(f2.weights["M_gap"]), rtol=1e-12)
    vals = np.asarray(q1(np.array([[-2.0, 0.0], [0.0, 0.0], [2.0, 0.0]])))
    assert vals[0] < 0.15 < vals[1] < 0.85 < vals[2]
    assert np.isfinite(float(f1.weights["M_gap"])) and float(f1.weights["M_gap"]) > 0


def test_cv_refit_does_not_leak_into_the_weight_solver():
    """``'cv_refit'`` must be translated to an absolute ridge BEFORE dispatch --
    the weight solver only sees an already-fitted basis and would reject it."""
    from sliced_committor.core.solver import compute_enriched_basin_moment_weights
    import sliced_committor as sc

    rng = np.random.default_rng(3)
    N = 4000
    X = rng.normal(size=(N, 2)) * 0.6
    X[: N // 2, 0] -= 1.5
    X[N // 2:, 0] += 1.5
    res = sc.compute_sliced_committor(X, in_A=X[:, 0] < -1.2, in_B=X[:, 0] > 1.2,
                                      n_directions=12, seed=0, n_bins=25,
                                      n_min=10, binning_method="quantile")
    with pytest.raises(ValueError):
        compute_enriched_basin_moment_weights(res, X, tikhonov="cv_refit")


def test_cv_ridge_is_larger_than_no_ridge_and_changes_the_energy():
    """Sanity: the selected ridge must actually be applied.

    ``E_bar = 1/M_gap`` is monotone increasing in the ridge, so a CV fit must
    report at least the un-ridged energy.  If ``('ridge_abs', r)`` were dropped
    on the floor the two would agree exactly.
    """
    from sliced_committor.core.solver import compute_enriched_basin_moment_weights
    import sliced_committor as sc

    rng = np.random.default_rng(1)
    N = 8000
    X = rng.normal(size=(N, 2)) * 0.6
    X[: N // 2, 0] -= 1.5
    X[N // 2:, 0] += 1.5
    res = sc.compute_sliced_committor(X, in_A=X[:, 0] < -1.2, in_B=X[:, 0] > 1.2,
                                      n_directions=24, seed=0, n_bins=40,
                                      n_min=10, binning_method="quantile")
    w_cv = compute_enriched_basin_moment_weights(res, X, tikhonov="cv")
    w_0 = compute_enriched_basin_moment_weights(res, X, tikhonov=1e-14)
    assert 1.0 / float(w_cv["M_gap"]) > 1.0 / float(w_0["M_gap"])
