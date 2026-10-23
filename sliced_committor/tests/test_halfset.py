"""The half-set toolkit: folds, half-Grams, the eigenband filter."""

import os

import jax
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from sliced_committor.core import _halfset as hs
from sliced_committor.core._ebmc import _solve

from ._helpers import spd_pair as _spd_pair


def test_folds_are_contiguous_blocks_within_each_stratum():
    N = 100
    strata = np.array([0] * 30 + [2] * 50 + [1] * 20)
    fold = hs.make_folds(N, 5, strata=strata)
    assert fold.shape == (N,) and set(fold) == set(range(5))
    for s in (0, 1, 2):
        idx = np.flatnonzero(strata == s)
        f = fold[idx]
        assert np.all(np.diff(f) >= 0), "blocks must be contiguous in time within a stratum"
        assert set(f) == set(range(5)), "every fold holds samples of every stratum"


def test_folds_without_strata_are_plain_blocks():
    fold = hs.make_folds(10, 5)
    np.testing.assert_array_equal(fold, [0, 0, 1, 1, 2, 2, 3, 3, 4, 4])


def test_basin_strata():
    in_A = np.array([True, False, False])
    in_B = np.array([False, True, False])
    np.testing.assert_array_equal(hs.basin_strata(in_A, in_B), [0, 1, 2])


def test_halfset_grams_deal_folds_alternately():
    rng = np.random.default_rng(0)
    G_folds = np.stack([np.eye(3) * (k + 1) for k in range(6)])
    w = rng.uniform(1, 2, 6)
    G1, G2 = hs.halfset_grams(G_folds, w)
    even, odd = [0, 2, 4], [1, 3, 5]
    np.testing.assert_allclose(G1, np.eye(3) * (w[even] @ (np.array(even) + 1)) / w[even].sum())
    np.testing.assert_allclose(G2, np.eye(3) * (w[odd] @ (np.array(odd) + 1)) / w[odd].sum())
    with pytest.raises(ValueError, match="two folds"):
        hs.halfset_grams(G_folds[:1], w[:1])


def test_identical_halves_recover_the_mean():
    """Perfect agreement saturates the SSNR and the filter does nothing."""
    G1, _ = _spd_pair(16, 15)
    R, info = hs.halfset_eigen_regularize(G1, G1.copy(), n_bands=4)
    np.testing.assert_allclose(R, G1, rtol=1e-5, atol=1e-8)
    assert np.all(info["band_ssnr"] > 1e5)


def test_noisier_halves_inflate_more():
    G1, G2 = _spd_pair(16, 3)
    R_small, _ = hs.halfset_eigen_regularize(G1, G2, n_bands=4)
    noisy = G2 + 0.5 * G1 * np.random.default_rng(1).uniform(0.5, 1.5, (16, 16))
    noisy = 0.5 * (noisy + noisy.T)
    R_big, _ = hs.halfset_eigen_regularize(G1, noisy, n_bands=4)
    assert np.trace(R_big) > np.trace(R_small)
    for R in (R_small, R_big):
        assert np.linalg.eigvalsh(R).min() > 0


def test_regularized_gram_embeds_identity_rows_for_invalid_directions():
    G1, G2 = _spd_pair(6, 4)
    valid = np.array([True, False, True, True, False, True])
    R, _ = hs.regularized_gram(G1, G2, valid, n_bands=2)
    iv = np.flatnonzero(valid)
    R_v, _ = hs.halfset_eigen_regularize(G1[np.ix_(iv, iv)], G2[np.ix_(iv, iv)], n_bands=2)
    np.testing.assert_array_equal(R[np.ix_(iv, iv)], R_v)
    for j in np.flatnonzero(~valid):
        assert R[j, j] == 1.0 and np.all(R[j, np.arange(6) != j] == 0.0)


def test_run_strata_give_every_fold_a_contiguous_share_of_every_run():
    """Window-ordered frames: without runs a fold covers a few windows; with
    ``run_ids`` every fold takes a contiguous block of every window."""
    n_win, n_per = 6, 200
    run = np.repeat(np.arange(n_win), n_per)
    in_A = np.zeros(run.size, bool)
    in_B = np.zeros(run.size, bool)
    in_A[:50], in_B[-50:] = True, True
    plain = hs.make_folds(run.size, 10, strata=hs.basin_strata(in_A, in_B))
    by_run = hs.make_folds(run.size, 10, strata=hs.basin_strata(in_A, in_B, run))
    trans = ~in_A & ~in_B
    assert min(len(set(run[(plain == k) & trans])) for k in range(10)) == 1  # the confound
    for k in range(10):
        assert set(run[(by_run == k) & ~in_A & ~in_B]) == set(range(n_win))
    for r in range(n_win):
        f = by_run[(run == r) & ~in_A & ~in_B]
        assert np.all(np.diff(f) >= 0), "blocks stay contiguous in time within a run"


def test_basin_strata_reject_misaligned_runs():
    with pytest.raises(ValueError, match="one entry per frame"):
        hs.basin_strata(np.zeros(4, bool), np.zeros(4, bool), np.zeros(3))


def test_the_filter_commutes_with_a_relabelling_of_the_slices():
    """The order of the slices is arbitrary; relabelling them permutes the
    eigenvectors and lets eigh choose their signs afresh, so the filter commutes
    with the permutation only if its band statistic ignores those signs."""
    G1, G2 = _spd_pair(12, 3)
    P = np.eye(12)[np.random.default_rng(4).permutation(12)]
    R, info = hs.regularized_gram(G1, G2, np.ones(12, bool), n_bands=3)
    R_p, info_p = hs.regularized_gram(P @ G1 @ P.T, P @ G2 @ P.T, np.ones(12, bool), n_bands=3)
    np.testing.assert_allclose(R_p, P @ R @ P.T, rtol=1e-9, atol=1e-15)
    np.testing.assert_allclose(info_p["band_ssnr"], info["band_ssnr"], rtol=1e-9)


def test_the_band_correlation_ignores_eigenvector_signs():
    """Flipping an eigenvector flips its row and column in both readings."""
    rng = np.random.default_rng(0)
    A1, A2 = rng.standard_normal((2, 6, 6))
    flip = np.diag([1.0, -1.0, 1.0, -1.0, -1.0, 1.0])
    band = slice(0, 3)
    c = hs._band_correlation(A1[band].ravel(), A2[band].ravel())
    B1, B2 = flip @ A1 @ flip, flip @ A2 @ flip
    assert hs._band_correlation(B1[band].ravel(), B2[band].ravel()) == pytest.approx(c, rel=1e-12)


def test_the_filtered_solve_keeps_rounding_at_rounding():
    """Seven of the twelve bands of this 64-direction Wolfe-Quapp fit are
    inflated by 1e12. A factorisation of the rebuilt filtered Gram reads the
    rounding of those eigenvalues on the signal bands (1e-5 of the weights);
    the solve in the filter's eigenbasis does not."""
    path = os.path.join(os.path.dirname(__file__), "golden", "halfset_reference.npz")
    with np.load(path) as z:
        G1, G2, a, b = (z[f"wq_fit::{k}"] for k in ("G1", "G2", "a", "b"))
        valid = z["wq_fit::valid_mask"]

    def weights(G1, G2, eig=True):
        G_reg, info = hs.regularized_gram(G1, G2, valid)
        return np.asarray(_solve(G_reg, a, b, valid, eig=info if eig else None)["w"])

    w = weights(G1, G2)
    scale = np.max(np.abs(w))
    rng = np.random.default_rng(0)
    for _ in range(5):
        E = rng.standard_normal(G1.shape)
        E = 1e-15 * (E + E.T)
        assert np.max(np.abs(weights(G1 * (1 + E), G2 * (1 - E)) - w)) < 1e-10 * scale
    np.testing.assert_allclose(weights(G1, G2, eig=False), w, rtol=0.0, atol=1e-3 * scale)
