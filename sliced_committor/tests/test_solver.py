"""The slice basis: input contract, reproducibility, batching, and physics.

The physics gates compare the sliced committor against a 2D finite-difference
reference (double well) and the closed-form 1D Ornstein-Uhlenbeck committor.
"""

import math

import jax
import numpy as np
import pytest
from scipy.special import erf

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp

import sliced_committor as sc

from ._helpers import (
    double_well_eval_points,
    double_well_pde,
    double_well_samples,
    interpolate_grid,
    two_basin_samples,
)


def _toy(n=400, seed=0):
    return two_basin_samples(n, seed=seed, radius=0.6)


# --------------------------------------------------------------------------- #
# input contract
# --------------------------------------------------------------------------- #
def test_rejects_non_2d_samples():
    s, in_A, in_B = _toy()
    with pytest.raises(ValueError, match="2-D"):
        sc.compute_sliced_committor(s[:, 0], in_A=in_A, in_B=in_B, n_directions=8)


@pytest.mark.parametrize("which", ["A", "B"])
def test_rejects_empty_basin(which):
    s, in_A, in_B = _toy()
    empty = jnp.zeros(s.shape[0], bool)
    kw = dict(in_A=empty, in_B=in_B) if which == "A" else dict(in_A=in_A, in_B=empty)
    with pytest.raises(ValueError, match=f"basin {which} is empty"):
        sc.compute_sliced_committor(s, n_directions=8, **kw)


def test_rejects_overlapping_basins():
    s, _, _ = _toy()
    mask = jnp.ones(s.shape[0], bool)
    with pytest.raises(ValueError, match="disjoint"):
        sc.compute_sliced_committor(s, in_A=mask, in_B=mask, n_directions=8)


@pytest.mark.parametrize("bad", [jnp.nan, jnp.inf])
def test_rejects_non_finite_samples(bad):
    s, in_A, in_B = _toy()
    with pytest.raises(ValueError, match="non-finite"):
        sc.compute_sliced_committor(s.at[3, 1].set(bad), in_A=in_A, in_B=in_B, n_directions=8)


def test_rejects_mismatched_label_length():
    s, _, in_B = _toy()
    with pytest.raises(ValueError, match="bool arrays matching samples"):
        sc.compute_sliced_committor(
            s, in_A=jnp.zeros(s.shape[0] - 1, bool), in_B=in_B, n_directions=8
        )


def test_rejects_unknown_binning_method():
    s, in_A, in_B = _toy()
    with pytest.raises(ValueError, match="binning_method"):
        sc.compute_sliced_committor(
            s, in_A=in_A, in_B=in_B, n_directions=8, binning_method="equal-width"
        )


def test_rejects_boundary_quantile_out_of_range():
    s, in_A, in_B = _toy()
    with pytest.raises(ValueError, match="boundary_quantile"):
        sc.compute_sliced_committor(s, in_A=in_A, in_B=in_B, n_directions=8, boundary_quantile=1.5)


def test_rejects_directions_of_wrong_dim():
    s, in_A, in_B = _toy()
    with pytest.raises(ValueError, match="incompatible"):
        sc.compute_sliced_committor(s, in_A=in_A, in_B=in_B, directions=jnp.ones((4, 3)))


def test_high_d_warning():
    rng = np.random.default_rng(0)
    s = jnp.asarray(rng.standard_normal((20, 100)))
    in_A = jnp.zeros(20, bool).at[:5].set(True)
    in_B = jnp.zeros(20, bool).at[15:].set(True)
    with pytest.warns(UserWarning, match="dim .* exceeds N"):
        sc.compute_sliced_committor(s, in_A=in_A, in_B=in_B, n_directions=8)


# --------------------------------------------------------------------------- #
# reproducibility, batching, summary
# --------------------------------------------------------------------------- #
def test_seed_reproducibility():
    s, in_A, in_B = two_basin_samples(500, dim=3)
    a = sc.compute_sliced_committor(s, in_A=in_A, in_B=in_B, n_directions=32, seed=11)
    b = sc.compute_sliced_committor(s, in_A=in_A, in_B=in_B, n_directions=32, seed=11)
    np.testing.assert_array_equal(np.asarray(a.directions), np.asarray(b.directions))
    np.testing.assert_array_equal(np.asarray(a.committors_1d), np.asarray(b.committors_1d))


def test_direction_batching_matches_to_rounding():
    """Batching the directions bounds memory and changes nothing beyond the last
    bits of XLA's reductions (identical on one XLA build, ``1e-14`` across)."""
    s, in_A, in_B = two_basin_samples(600, dim=3, seed=2)
    one = sc.compute_sliced_committor(s, in_A=in_A, in_B=in_B, n_directions=40, seed=5)
    batched = sc.compute_sliced_committor(
        s, in_A=in_A, in_B=in_B, n_directions=40, seed=5, direction_batch_size=16
    )
    for name in (
        "committors_1d",
        "free_energies",
        "slice_coords",
        "log_dirichlet",
        "boundary_errors",
    ):
        np.testing.assert_allclose(
            np.asarray(getattr(one, name)),
            np.asarray(getattr(batched, name)),
            rtol=1e-11,
            atol=1e-11,
        )


def test_supplied_directions_override_the_draw_and_warn_on_config():
    s, in_A, in_B = two_basin_samples(500, dim=3)
    dirs = sc.directions_uniform(jax.random.PRNGKey(9), 7, 3)
    with pytest.warns(UserWarning, match="take precedence"):
        r = sc.compute_sliced_committor(
            s,
            in_A=in_A,
            in_B=in_B,
            directions=dirs,
            direction_sampling=sc.DirectionSamplingConfig(mode="lda"),
        )
    np.testing.assert_array_equal(np.asarray(r.directions), np.asarray(dirs))
    assert r.axis is None


def test_result_shapes_and_summary():
    s, in_A, in_B = two_basin_samples(500, dim=3)
    r = sc.compute_sliced_committor(s, in_A=in_A, in_B=in_B, n_directions=24, n_bins=50, seed=7)
    M, N = 24, 500
    assert r.directions.shape == (M, 3)
    assert r.slice_coords.shape == r.free_energies.shape == r.committors_1d.shape == (M, 50)
    assert r.boundary_indices.shape == (M, 2)
    assert r.projected_samples.shape == (M, N)
    assert r.valid_mask.shape == r.log_dirichlet.shape == r.boundary_errors.shape == (M,)
    assert bool(jnp.all(r.valid_mask))
    text = r.summary()
    for token in ("M = 24", "valid slices", "mean eps", "mean log D"):
        assert token in text


def test_sample_weights_reweight_the_boundary_error():
    """Uniform weights reproduce the unweighted path exactly; non-uniform ones move it;
    weights that do not sum to one are refused (the solve uses them verbatim)."""
    s, in_A, in_B = two_basin_samples(500, dim=2, seed=3)
    N = s.shape[0]
    base = sc.compute_sliced_committor(s, in_A=in_A, in_B=in_B, n_directions=16, seed=1)
    unif = sc.compute_sliced_committor(
        s, in_A=in_A, in_B=in_B, n_directions=16, seed=1, sample_weights=jnp.ones(N) / N
    )
    np.testing.assert_allclose(
        np.asarray(unif.boundary_errors), np.asarray(base.boundary_errors), rtol=1e-6
    )
    w = np.random.default_rng(0).uniform(0.1, 1.0, N)
    skew = sc.compute_sliced_committor(
        s, in_A=in_A, in_B=in_B, n_directions=16, seed=1, sample_weights=w / w.sum()
    )
    assert not np.allclose(np.asarray(skew.boundary_errors), np.asarray(base.boundary_errors))
    with pytest.raises(ValueError, match="sum to one"):
        sc.compute_sliced_committor(s, in_A=in_A, in_B=in_B, n_directions=16, sample_weights=w)


# --------------------------------------------------------------------------- #
# physics: 2D double well against a finite-difference reference
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def pde():
    return double_well_pde(n_grid=64)


def _rmse_vs_pde(pde, samples, in_A, in_B, M, seed, n_bins=120):
    q_grid, X, Y = pde
    pts = double_well_eval_points()
    q = sc.fit_committor(
        jnp.asarray(samples),
        in_A=jnp.asarray(in_A),
        in_B=jnp.asarray(in_B),
        n_directions=M,
        n_bins=n_bins,
        seed=seed,
    )
    q_pred = np.asarray(q(jnp.asarray(pts)))
    return float(np.sqrt(np.mean((q_pred - interpolate_grid(q_grid, X, Y, pts)) ** 2)))


def test_double_well_matches_pde(pde):
    samples, in_A, in_B = double_well_samples(n=2500)
    assert _rmse_vs_pde(pde, samples, in_A, in_B, M=256, seed=42) < 0.10


def test_rmse_decreases_with_n_directions(pde):
    """The paper's scaling claim: error falls with the number of slices."""
    samples, in_A, in_B = double_well_samples(n=2500, seed=1)
    rmses = [_rmse_vs_pde(pde, samples, in_A, in_B, M=M, seed=2) for M in (16, 64, 256)]
    assert rmses[1] < 0.97 * rmses[0], rmses
    assert rmses[2] < 0.97 * rmses[1], rmses
    assert rmses[-1] < 0.1, rmses


# --------------------------------------------------------------------------- #
# physics: 1D Ornstein-Uhlenbeck closed form
# --------------------------------------------------------------------------- #
def _ou_committor(x, a, b):
    return (erf(x / math.sqrt(2.0)) - erf(a / math.sqrt(2.0))) / (
        erf(b / math.sqrt(2.0)) - erf(a / math.sqrt(2.0))
    )


def test_recovers_ou_closed_form():
    a, b = -2.0, 2.0
    rng = np.random.default_rng(0)
    x = rng.standard_normal(20000)
    x = x[(x > a) & (x < b)][:4000]
    samples = jnp.asarray(x[:, None])
    in_A, in_B = jnp.asarray(x < a + 0.2), jnp.asarray(x > b - 0.2)
    q = sc.fit_committor(samples, in_A=in_A, in_B=in_B, n_directions=8, n_bins=80, seed=42)
    x_eval = np.linspace(a + 0.3, b - 0.3, 120)
    q_pred = np.asarray(q(jnp.asarray(x_eval[:, None])))
    rmse = float(np.sqrt(np.mean((q_pred - _ou_committor(x_eval, a, b)) ** 2)))
    assert rmse < 0.05, rmse
