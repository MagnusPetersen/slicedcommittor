"""The weight solve: constraints, ridge rules, the representation gate, the
held-out cap, invalid directions, and the block bootstrap."""

import jax
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp

import sliced_committor as sc

from ._helpers import TOL_SOLVER, two_basin_samples, wolfe_quapp_samples

pytestmark = pytest.mark.filterwarnings("ignore::UserWarning")


@pytest.fixture(scope="module")
def basis(golden_samples, golden_labels):
    in_A, in_B = golden_labels
    return sc.compute_sliced_committor(
        golden_samples, in_A=in_A, in_B=in_B, n_directions=16, seed=0
    )


def _wq(n=6000, m=64, seed=0, n_bins=50, binning="equal_width"):
    X, in_A, in_B = wolfe_quapp_samples(n, seed)
    return sc.compute_sliced_committor(
        jnp.asarray(X),
        in_A=jnp.asarray(in_A),
        in_B=jnp.asarray(in_B),
        n_directions=m,
        seed=seed,
        n_bins=n_bins,
        n_min=10,
        binning_method=binning,
    )


# --------------------------------------------------------------------------- #
# the constrained solve
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("tikhonov", ["halfset_eigen", "auto", 1e-3])
def test_constraints_hold(basis, tikhonov):
    w = sc.solve_weights(basis, tikhonov=tikhonov)
    assert isinstance(w, sc.Weights)
    assert w.w.shape == (16,)
    d = w.diagnostics
    assert abs(d["mu_A_check"]) < TOL_SOLVER
    assert abs(d["mu_B_check"] - 1.0) < TOL_SOLVER
    assert d["constraint_residual"] < TOL_SOLVER
    assert w.moment_gap > 0 and w.dirichlet_energy > 0 and np.isfinite(w.cond)
    assert w.tikhonov == tikhonov


def test_halfset_has_no_ridge_and_a_ridge_rule_has_one(basis):
    assert sc.solve_weights(basis, tikhonov="halfset_eigen").ridge == 0.0
    assert sc.solve_weights(basis, tikhonov="auto").ridge > 0.0
    assert sc.solve_weights(basis, tikhonov=2.5e-3).ridge == 2.5e-3


def test_absolute_ridge_is_absolute(basis):
    """A bigger ridge shrinks the solution toward the constraint-only answer."""
    small = sc.solve_weights(basis, tikhonov=1e-6)
    big = sc.solve_weights(basis, tikhonov=1e6)
    assert np.linalg.norm(np.asarray(big.w)) < np.linalg.norm(np.asarray(small.w))


def test_rejects_unknown_rule_and_negative_ridge(basis):
    with pytest.raises(ValueError, match="tikhonov must be"):
        sc.solve_weights(basis, tikhonov="cv")
    with pytest.raises(ValueError, match=">= 0"):
        sc.solve_weights(basis, tikhonov=-1.0)


def test_heldout_cap_requires_the_halfset_solve(basis):
    with pytest.raises(ValueError, match="heldout_cap"):
        sc.solve_weights(basis, tikhonov="auto", heldout_cap=True)
    assert sc.solve_weights(basis, tikhonov="auto").heldout_cap is None


def test_boundary_enforcement_at_samples(basis, golden_samples, golden_labels):
    in_A, in_B = golden_labels
    q = sc.build_committor(basis, sc.solve_weights(basis))(golden_samples, in_A=in_A, in_B=in_B)
    assert bool(jnp.all(q[in_A] == 0.0)) and bool(jnp.all(q[in_B] == 1.0))


def test_representation_error_when_no_slice_separates_the_basins(basis):
    """Flat slices give a_j = b_j for every j: the moment gap collapses."""
    flat = basis._replace(committors_1d=jnp.full_like(basis.committors_1d, 0.5))
    with pytest.raises(sc.RepresentationError, match="moment gap"):
        sc.solve_weights(flat, tikhonov="auto")
    w = sc.solve_weights(flat, tikhonov="auto", raise_on_degenerate=False)
    assert w.cond < 1e-6


def test_invalid_directions_get_zero_weight_and_decoupled_rows():
    """A hand-masked direction is excluded from the solve, including from the
    eigendecomposition of the half-set filter."""
    s, in_A, in_B = two_basin_samples(3000, dim=2)
    res = sc.compute_sliced_committor(s, in_A=in_A, in_B=in_B, n_directions=12, seed=0, n_bins=30)
    vm = np.asarray(res.valid_mask, bool).copy()
    vm[[0, 5]] = False
    res2 = res._replace(valid_mask=jnp.asarray(vm))
    w = sc.solve_weights(res2, tikhonov="halfset_eigen")
    assert np.all(np.asarray(w.w)[~vm] == 0.0) and np.all(np.isfinite(np.asarray(w.w)))
    G_reg = np.asarray(w.diagnostics["G_reg"])
    for j in np.flatnonzero(~vm):
        assert G_reg[j, j] == 1.0 and np.all(G_reg[j, np.arange(12) != j] == 0.0)
    assert np.linalg.eigvalsh(0.5 * (G_reg + G_reg.T)).min() > 0.0


def test_single_valid_direction_falls_back_to_auto():
    s, in_A, in_B = two_basin_samples(2000, dim=2)
    res = sc.compute_sliced_committor(s, in_A=in_A, in_B=in_B, n_directions=4, seed=0, n_bins=30)
    vm = np.zeros(4, bool)
    vm[1] = True
    w = sc.solve_weights(res._replace(valid_mask=jnp.asarray(vm)), tikhonov="halfset_eigen")
    assert w.ridge > 0.0 and np.isfinite(w.moment_gap)


def test_halfset_regularised_gram_is_psd_and_deterministic():
    res = _wq(n=3000, m=24, seed=2, binning="quantile")
    a = sc.solve_weights(res)
    b = sc.solve_weights(res)
    np.testing.assert_array_equal(np.asarray(a.w), np.asarray(b.w))
    assert a.diagnostics["band_ssnr"] == b.diagnostics["band_ssnr"]
    G_reg = np.asarray(a.diagnostics["G_reg"])
    assert np.linalg.eigvalsh(0.5 * (G_reg + G_reg.T)).min() > 0.0
    assert a.diagnostics["lam_min_reg"] > 0.0 and np.all(np.isfinite(a.diagnostics["band_ssnr"]))


# --------------------------------------------------------------------------- #
# the held-out cap
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def wq6000():
    return _wq()


def test_cap_is_finite_uses_every_fold_and_leaves_the_weights_alone(wq6000):
    plain = sc.solve_weights(wq6000)
    capped = sc.solve_weights(wq6000, heldout_cap=True)
    hc = capped.heldout_cap
    assert np.isfinite(hc["cap"]) and hc["cap"] > 0
    assert hc["n_ok"] == hc["n_folds"] == 10
    assert np.isfinite(hc["per_fold"]).all() and np.isfinite(hc["per_fold_train"]).all()
    assert np.isfinite(hc["gap"])
    np.testing.assert_array_equal(np.asarray(plain.w), np.asarray(capped.w))
    assert plain.moment_gap == capped.moment_gap
    # read out of sample: it is not the in-sample energy
    assert not np.isclose(hc["cap"], capped.dirichlet_energy, rtol=1e-9)


def _paired(alt, base):
    d = np.asarray(alt["per_fold"]) - np.asarray(base["per_fold"])
    se = float(np.std(d, ddof=1) / np.sqrt(d.size))
    return float(np.mean(d)), float(np.mean(d) / se), int((d > 0).sum()), d.size


@pytest.mark.parametrize(
    "label,kw", [("fewer_directions", {"m": 4}), ("coarser_histograms", {"n_bins": 4})]
)
def test_cap_ranks_a_degraded_trial_space_worse(label, kw):
    """The property the paper's (mu, alpha) selection rests on: shrinking or
    blunting the trial space raises the held-out cap. Paired across folds."""
    base = sc.solve_weights(_wq(n=20000), heldout_cap=True).heldout_cap
    alt = sc.solve_weights(_wq(n=20000, **kw), heldout_cap=True).heldout_cap
    mean_d, t, n_pos, n = _paired(alt, base)
    assert mean_d > 0, f"{label}: the cap fell for a worse trial space"
    assert t > 4.0, f"{label}: paired t = {t:.1f}"
    assert n_pos == n, f"{label}: only {n_pos}/{n} folds agree"


# --------------------------------------------------------------------------- #
# the block bootstrap
# --------------------------------------------------------------------------- #
def test_bootstrap_shapes_and_determinism():
    res = _wq(n=3000, m=16, seed=1, binning="quantile")
    w = sc.solve_weights(res)
    pts = jnp.asarray(np.asarray(res.projected_samples)[:0]).reshape(0, 2)
    b1 = sc.bootstrap_weights(
        res, w, n_boot=6, block_len=25, seed=3, points=jnp.asarray(wolfe_quapp_samples(5, 9)[0])
    )
    b2 = sc.bootstrap_weights(
        res, w, n_boot=6, block_len=25, seed=3, points=jnp.asarray(wolfe_quapp_samples(5, 9)[0])
    )
    assert isinstance(b1, sc.Bootstrap) and b1.n_ok == 6
    assert b1.w.shape == (6, 16) and b1.c.shape == (6,) and b1.q.shape == (6, 5)
    np.testing.assert_array_equal(b1.w, b2.w)
    assert b1.moment_gap.std() > 0
    del pts


def test_bootstrap_blocks_never_cross_runs():
    from sliced_committor.core._ebmc import _run_segments

    runs = np.array([0, 0, 0, 1, 1, 2, 2, 2, 2])
    assert _run_segments(runs) == [(0, 3), (3, 5), (5, 9)]
    assert _run_segments(np.zeros(4, int)) == [(0, 4)]


def test_bootstrap_spread_tracks_the_seed_to_seed_spread():
    """On i.i.d. samples (block_len = 1) the bootstrap SE of the moment gap
    should be the right order of magnitude against independent draws."""
    seeds = range(6)
    gaps = []
    for seed in seeds:
        res = _wq(n=2500, m=12, seed=seed, binning="quantile")
        gaps.append(sc.solve_weights(res, tikhonov="auto").moment_gap)
    res0 = _wq(n=2500, m=12, seed=0, binning="quantile")
    w0 = sc.solve_weights(res0, tikhonov="auto")
    boot = sc.bootstrap_weights(res0, w0, n_boot=30, block_len=1, seed=0)
    ratio = boot.moment_gap.std(ddof=1) / np.std(gaps, ddof=1)
    assert 0.3 < ratio < 3.0, ratio
