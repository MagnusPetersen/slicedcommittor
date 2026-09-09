"""Pure invariant tests for lib/recovar; no reference (slicedcv) import.

These tests pin down mathematical properties that must hold regardless of
implementation details: split partitioning, label structure, Horn rank
recovery on a planted subspace, CG correctness, the Sw^{-1} sampler's
covariance, the spiked floor sampler's anisotropy, interpolation boundary
behavior, metric masking, and the separable test-system contract.
"""
import numpy as np
import pytest

from recovar.assemble import clip01, transition_rmse
from recovar.basis import SliceBasis
from recovar.ids import (
    _block_cg,
    _SwOperator,
    directions_exact_floor,
    path_labels,
    sample_inv_sw,
    select_rank_permutation,
)
from recovar.splits import block_split
from recovar.systems import nd_separable_dataset, random_rotation


# ---------------------------------------------------------------------------
# block_split
# ---------------------------------------------------------------------------

def test_block_split_partition_and_alternation():
    n, block_len = 100, 7
    group = np.repeat(np.arange(3), [41, 33, 26])   # three uneven groups
    h1, h2 = block_split(n, block_len, group=group)

    # Halves are disjoint and cover every frame.
    both = np.concatenate([h1, h2])
    assert len(np.intersect1d(h1, h2)) == 0
    np.testing.assert_array_equal(np.sort(both), np.arange(n))

    # Within each group, contiguous blocks alternate between the halves and
    # never cross a group boundary.
    set1, set2 = set(h1.tolist()), set(h2.tolist())
    for g in range(3):
        idx = np.flatnonzero(group == g)
        n_blocks = int(np.ceil(len(idx) / block_len))
        for k in range(n_blocks):
            blk = idx[k * block_len:(k + 1) * block_len]
            target = set1 if k % 2 == 0 else set2
            assert set(blk.tolist()) <= target

    # Deterministic: the rng argument does not perturb the assignment.
    h1b, h2b = block_split(n, block_len, rng=123, group=group)
    np.testing.assert_array_equal(h1, h1b)
    np.testing.assert_array_equal(h2, h2b)


# ---------------------------------------------------------------------------
# path_labels
# ---------------------------------------------------------------------------

def test_path_labels_structure_and_weight_balance():
    rng = np.random.default_rng(0)
    n, n_bins = 4000, 8
    qhat = rng.uniform(0.0, 1.0, n)
    in_A = qhat < 0.05
    in_B = qhat > 0.95
    w = 0.5 + rng.random(n)
    w = w / w.sum()
    lab = path_labels(qhat, in_A, in_B, w, n_bins)

    assert np.all(lab[in_A] == 0)
    assert np.all(lab[in_B] == n_bins + 1)
    tr = ~(in_A | in_B)
    assert np.all((lab[tr] >= 1) & (lab[tr] <= n_bins))

    # Weighted-quantile edges balance the total weight per interior bin.
    tot = w[tr].sum()
    for b in range(1, n_bins + 1):
        frac = w[lab == b].sum() / tot
        assert abs(frac - 1.0 / n_bins) < 0.2 / n_bins


# ---------------------------------------------------------------------------
# Horn parallel analysis on a planted rank-2 structure
# ---------------------------------------------------------------------------

def test_horn_rank_recovers_planted_rank2():
    # Six group means on a circle arc inside span(e0, e1), amplitude far
    # above the mean-estimation noise, isotropic within scatter: the
    # between-group structure is exactly rank 2 in whitened space.
    rng = np.random.default_rng(0)
    n_per, L, d = 120, 6, 10
    angles = np.linspace(0.0, np.pi, L)
    means = np.zeros((L, d))
    means[:, 0] = 1.5 * np.cos(angles)
    means[:, 1] = 1.5 * np.sin(angles)
    X = np.vstack([means[g] + rng.standard_normal((n_per, d))
                   for g in range(L)])
    labels = np.repeat(np.arange(L), n_per)
    r, obs, thr = select_rank_permutation(X, labels, np.eye(d), w=None,
                                          n_perm=30,
                                          rng=np.random.default_rng(1))
    assert r == 2
    # The two planted eigenvalues clear the null decisively.
    assert obs[0] > 2.0 * thr[0] and obs[1] > 2.0 * thr[1]


# ---------------------------------------------------------------------------
# _block_cg
# ---------------------------------------------------------------------------

class _DenseOp:
    def __init__(self, A):
        self.A = A

    def matvec(self, V):
        return self.A @ V


def test_block_cg_matches_dense_solve():
    rng = np.random.default_rng(1)
    d = 12
    R = rng.standard_normal((d, d))
    A = R @ R.T + d * np.eye(d)
    B = rng.standard_normal((d, 3))
    X, n_it = _block_cg(_DenseOp(A), B, 1.0 / np.diag(A),
                        tol=1e-12, maxiter=500)
    np.testing.assert_allclose(X, np.linalg.solve(A, B), atol=1e-8)
    assert n_it < 500


# ---------------------------------------------------------------------------
# sample_inv_sw
# ---------------------------------------------------------------------------

def test_sample_inv_sw_covariance():
    rng = np.random.default_rng(2)
    n, d, L = 1200, 6, 4
    labels = np.repeat(np.arange(L), n // L)
    scales = np.array([1.0, 2.0, 0.5, 1.0, 1.5, 0.8])
    X = rng.standard_normal((n, d)) * scales[None, :]
    X = X + 3.0 * rng.standard_normal((L, d))[labels]   # between-group offsets
    op = _SwOperator(X, labels, np.full(n, 1.0 / n))

    # The operator applies Sw = Z'Z + eps*I; reconstruct it from the
    # operator's own attributes and compare the draw covariance to Sw^{-1}.
    Z = op.Z.astype(np.float64)
    Sw = Z.T @ Z + op.eps * np.eye(d)
    Sinv = np.linalg.inv(Sw)
    M = 4000
    Xf, _ = sample_inv_sw(op, M, np.random.default_rng(3))
    emp = Xf @ Xf.T / M
    rel = np.linalg.norm(emp - Sinv) / np.linalg.norm(Sinv)
    assert rel < 0.15


# ---------------------------------------------------------------------------
# directions_exact_floor
# ---------------------------------------------------------------------------

def test_directions_exact_floor_norms_and_spike_ratio():
    # The spike must stay MODEST (lam0 = 2): the returned directions are
    # row-normalized, and a spike whose variance rivals the summed floor
    # mass gets shrunk by its own norm, deflating the measured ratio far
    # below lam0 / tau (measured 47 vs 184 at lam0 = 40). At lam0 = 2 the
    # normalization bias is mild and the factor-2 band is meaningful.
    rng = np.random.default_rng(4)
    n, d, L = 3000, 24, 5
    labels = np.repeat(np.arange(L), n // L)
    X = rng.standard_normal((n, d))          # within scatter ~ identity
    op = _SwOperator(X, labels, np.full(n, 1.0 / n))
    Theta, _ = np.linalg.qr(rng.standard_normal((d, 6)))
    lam = np.array([2.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    r = 1
    T, _ = directions_exact_floor(lam, Theta, r, op, 8000,
                                  np.random.default_rng(5))

    np.testing.assert_allclose(np.linalg.norm(T, axis=1), 1.0, atol=1e-12)

    # Spike-to-floor variance ratio approximates lam0 / tau with
    # tau = sum(lam[r:]) / (d - r), here 9.2 (measured 6.9). The wrong
    # denominator (len(lam) - r, the 60x floor bug's form) gives tau = 1
    # and a measured ratio near 2, below the factor-2 band.
    tau = lam[r:].sum() / (d - r)
    expected = lam[0] / tau
    v_perp = rng.standard_normal(d)
    v_perp -= Theta @ (Theta.T @ v_perp)     # orthogonal to every spike
    v_perp /= np.linalg.norm(v_perp)
    ratio = np.mean((T @ Theta[:, 0]) ** 2) / np.mean((T @ v_perp) ** 2)
    assert expected / 2.0 < ratio < expected * 2.0


# ---------------------------------------------------------------------------
# eval_at boundary behavior
# ---------------------------------------------------------------------------

def test_eval_at_out_of_grid_clamps_q_and_zeroes_qp():
    nb = 50
    grid = np.linspace(0.0, 1.0, nb, dtype=np.float32)[None, :]
    qv = np.linspace(0.2, 0.8, nb, dtype=np.float32)[None, :]
    qp = np.full((1, nb), 0.6, dtype=np.float32)
    basis = SliceBasis(np.array([[1.0, 0.0]]), grid, qv, qp)

    S = np.array([[-0.7, 1.9, 0.5, 0.0, 1.0]])
    q_out, qp_out = basis.eval_at(S)

    # Outside the grid: q clamped to the endpoint values, qp exactly zero.
    assert q_out[0, 0] == qv[0, 0]
    assert q_out[0, 1] == qv[0, -1]
    assert qp_out[0, 0] == 0.0
    assert qp_out[0, 1] == 0.0

    # Inside the grid: plain linear interpolation, nonzero derivative.
    assert abs(float(q_out[0, 2]) - 0.5) < 1e-6
    assert abs(float(qp_out[0, 2]) - 0.6) < 1e-6
    assert abs(float(q_out[0, 3]) - 0.2) < 1e-6
    assert abs(float(q_out[0, 4]) - 0.8) < 1e-6


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------

def test_transition_rmse_masks_basins():
    rng = np.random.default_rng(6)
    n = 20
    q_ref = rng.uniform(0.0, 1.0, n)
    in_A = np.zeros(n, bool)
    in_A[:5] = True
    in_B = np.zeros(n, bool)
    in_B[-5:] = True

    # Corrupting only the basins must not register at all.
    q_hat = q_ref.copy()
    q_hat[in_A] += 5.0
    q_hat[in_B] -= 3.0
    assert transition_rmse(q_hat, q_ref, in_A, in_B) == 0.0

    # One transition frame off by 0.3 over 10 transition frames.
    q_hat[7] = q_ref[7] + 0.3
    err = transition_rmse(q_hat, q_ref, in_A, in_B)
    assert abs(err - 0.3 / np.sqrt(10.0)) < 1e-12


def test_clip01_idempotent_and_bounded():
    x = np.array([-0.5, 0.0, 0.3, 1.0, 1.7])
    y = clip01(x)
    assert y.min() >= 0.0 and y.max() <= 1.0
    np.testing.assert_array_equal(clip01(y), y)


# ---------------------------------------------------------------------------
# nd_separable_dataset
# ---------------------------------------------------------------------------

def test_nd_separable_dataset_contract():
    kw = dict(beta=1.0, n_grid=150)
    X2, a2, b2, base2 = nd_separable_dataset(400, 2, rng=5, **kw)
    X8, a8, b8, base8 = nd_separable_dataset(400, 8, rng=5, **kw)

    # Labels depend only on the first 2 latent coordinates: adding nuisance
    # dimensions (same seed, so the same 2D base draw) changes nothing.
    np.testing.assert_array_equal(base2, base8)
    np.testing.assert_array_equal(a2, a8)
    np.testing.assert_array_equal(b2, b8)
    np.testing.assert_array_equal(X8[:, :2], base8)
    assert a8.sum() > 0 and b8.sum() > 0
    assert not np.any(a8 & b8)

    # The observed-basis transform is orthogonal and invertible; labels are
    # still computed from the latent base.
    T = random_rotation(8, rng=6)
    np.testing.assert_allclose(T @ T.T, np.eye(8), atol=1e-12)
    Xr, ar, br, baser = nd_separable_dataset(400, 8, rng=5, transform=T, **kw)
    np.testing.assert_array_equal(ar, a8)
    np.testing.assert_array_equal(br, b8)
    np.testing.assert_array_equal(baser, base8)
    np.testing.assert_allclose(Xr @ T, X8, atol=1e-10)
