"""The half-set toolkit: folds, half-Grams, the eigenband filter."""

import jax
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from sliced_committor.core import _halfset as hs

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
