"""The slice basis: input contract, reproducibility, batching, and physics.

The physics gates compare the sliced committor against a 2D finite-difference
reference (double well) and the closed-form 1D Ornstein-Uhlenbeck committor.
"""

import math

import jax
import numpy as np
import pytest
from scipy.special import erfi

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
    """The committor between ``a`` and ``b`` of the Ornstein-Uhlenbeck process with
    stationary density ``exp(-x^2/2)``: ``q' ~ 1/rho = exp(x^2/2)``."""
    return (erfi(x / math.sqrt(2.0)) - erfi(a / math.sqrt(2.0))) / (
        erfi(b / math.sqrt(2.0)) - erfi(a / math.sqrt(2.0))
    )


@pytest.mark.parametrize("binning", ["quantile", "equal_width"])
def test_recovers_ou_closed_form(binning):
    rng = np.random.default_rng(0)
    x = rng.standard_normal(20000)
    x = x[(x > -2.0) & (x < 2.0)][:4000]
    samples = jnp.asarray(x[:, None])
    a, b = -1.8, 1.8  # the inner edges of the two states
    in_A, in_B = jnp.asarray(x < a), jnp.asarray(x > b)
    q = sc.fit_committor(
        samples,
        in_A=in_A,
        in_B=in_B,
        n_directions=8,
        n_bins=80,
        seed=42,
        binning_method=binning,
    )
    x_eval = np.linspace(a + 0.1, b - 0.1, 120)
    q_pred = np.asarray(q(jnp.asarray(x_eval[:, None])))
    rmse = float(np.sqrt(np.mean((q_pred - _ou_committor(x_eval, a, b)) ** 2)))
    assert rmse < 0.03, rmse


# --------------------------------------------------------------------------- #
# the slice problem on non-uniform (equal-count) grids
# --------------------------------------------------------------------------- #
def _bimodal_1d(sep, seed=0):
    """Two Gaussian states and a uniform background, in one dimension."""
    rng = np.random.default_rng(seed)
    s_A = rng.normal(-sep / 2, 0.5, 20000)
    s_B = rng.normal(+sep / 2, 0.7, 10000)
    s_U = rng.uniform(-4.0, 4.0, 4000)
    s = np.concatenate([s_A, s_B, s_U])
    in_A = np.zeros(s.size, bool)
    in_B = np.zeros(s.size, bool)
    in_A[:20000] = True
    in_B[20000:30000] = True
    return s, in_A, in_B


def _slice_moments(res):
    """``a_j = <q_j>_A`` and ``b_j = <q_j>_B`` of every slice, at the samples."""
    from sliced_committor.core.solver import _interp_committor

    q = jax.vmap(_interp_committor)(
        res.slice_coords, jnp.nan_to_num(res.committors_1d), res.projected_samples
    )
    in_A, in_B = np.asarray(res.in_A), np.asarray(res.in_B)
    return np.asarray(q)[:, in_A].mean(1), np.asarray(q)[:, in_B].mean(1)


@pytest.mark.parametrize("binning,n_bins", [("equal_width", 400), ("quantile", 2000)])
def test_separated_states_give_unit_slice_fidelity(binning, n_bins):
    """The state densities sit on the grid nodes their samples fall in, so on
    equal-count bins too the slice reaches 0 across A and 1 across B."""
    s, in_A, in_B = _bimodal_1d(sep=4.0)
    res = sc.compute_sliced_committor(
        s[:, None],
        in_A=in_A,
        in_B=in_B,
        directions=np.ones((1, 1)),
        n_bins=n_bins,
        binning_method=binning,
    )
    a, b = _slice_moments(res)
    assert a[0] < 2e-3 and 1.0 - b[0] < 2e-3, (a, b)


@pytest.mark.parametrize(
    "binning,n_bins", [("equal_width", 200), ("quantile", 200), ("quantile", 1000)]
)
def test_slice_fidelity_is_conservative(binning, n_bins):
    """Summing the rows of the slice problem gives ``INT rho_A q = INT rho_B (1-q)``,
    so every slice has ``a_j ~ 1 - b_j`` and a fidelity ``b_j - a_j >= 0``. At the
    samples the two sides differ by the interpolation between nodes (about 24
    samples per bin here; the deployed fits have more)."""
    rng = np.random.default_rng(1)
    x_A = rng.normal([-1.0, 0.0], [0.6, 0.9], (12000, 2))
    x_B = rng.normal([1.0, 0.3], [0.8, 0.6], (8000, 2))
    x_U = rng.uniform(-3.0, 3.0, (4000, 2))
    x = np.concatenate([x_A, x_B, x_U])
    in_A = np.r_[np.ones(12000, bool), np.zeros(12000, bool)]
    in_B = np.r_[np.zeros(12000, bool), np.ones(8000, bool), np.zeros(4000, bool)]
    res = sc.compute_sliced_committor(
        x, in_A=in_A, in_B=in_B, n_directions=24, seed=3, n_bins=n_bins, binning_method=binning
    )
    a, b = _slice_moments(res)
    assert np.max(np.abs(a - (1.0 - b))) < 1e-2, np.abs(a - (1.0 - b))
    assert np.min(b - a) >= -1e-3, b - a


@pytest.mark.parametrize("binning,n_bins", [("quantile", 2000), ("equal_width", 400)])
def test_slice_follows_the_1d_committor_between_the_states(binning, n_bins):
    """Between the states' inner edges the slice solves ``(rho q')' = 0``, so it
    follows ``INT 1/rho``. A scheme that is not conservative on the equal-count
    grid solves ``rho q'' = 0`` there instead and comes out nearly linear in ``s``."""
    rng = np.random.default_rng(0)
    s_A = rng.normal(-3.0, 0.4, 20000)
    s_B = rng.normal(3.0, 0.4, 20000)
    s_U = rng.normal(0.0, 1.2, 40000)
    s_U = s_U[np.abs(s_U) < 2.0][:6000]
    s = np.concatenate([s_A, s_B, s_U])
    in_A = np.zeros(s.size, bool)
    in_B = np.zeros(s.size, bool)
    in_A[:20000] = True
    in_B[20000:40000] = True
    res = sc.compute_sliced_committor(
        s[:, None],
        in_A=in_A,
        in_B=in_B,
        directions=np.ones((1, 1)),
        n_bins=n_bins,
        binning_method=binning,
    )
    lo, hi = s_A.max(), s_B.min()
    z = np.linspace(lo, hi, 4001)
    inv_rho = np.exp(z**2 / (2 * 1.2**2))  # the background's; the states' tails vanish here
    ref = np.concatenate([[0.0], np.cumsum(0.5 * (inv_rho[1:] + inv_rho[:-1]) * np.diff(z))])
    ref /= ref[-1]
    s_grid = np.asarray(res.slice_coords[0])
    q = np.asarray(res.committors_1d[0])
    gap = (s_grid > lo) & (s_grid < hi)
    assert np.max(np.abs(q[gap] - np.interp(s_grid[gap], z, ref))) < 0.03


@pytest.mark.parametrize("binning", ["equal_width", "quantile"])
def test_discrete_slice_balance_is_exact(binning):
    """At the nodes the balance holds to rounding: ``sum m_A q = sum m_B (1 - q)``
    with the state masses binned on the density's own edges."""
    from sliced_committor.core.solver import compute_1d_free_energy, compute_1d_rd_committor

    s, in_A, in_B = _bimodal_1d(sep=1.5, seed=4)
    s = jnp.asarray(s)
    s_vals, F, edges = compute_1d_free_energy(s, 300, binning_method=binning, return_edges=True)
    s_A, s_B = s[in_A], s[in_B]
    q = compute_1d_rd_committor(
        s_vals,
        F,
        s_A,
        s_B,
        bin_edges=edges,
        rd_kappa=1e6,
        boundary_quantile=1.0,
        s_A_sorted=jnp.sort(s_A),
        s_B_sorted=jnp.sort(s_B),
    )
    n = s_vals.shape[0]
    idx = lambda v: np.clip(np.searchsorted(np.asarray(edges)[1:-1], np.asarray(v)), 0, n - 1)  # noqa: E731
    m_A = np.bincount(idx(s_A), minlength=n) / s_A.shape[0]
    m_B = np.bincount(idx(s_B), minlength=n) / s_B.shape[0]
    q = np.asarray(q)
    assert abs(m_A @ q - m_B @ (1.0 - q)) < 1e-8
    assert 0.0 <= m_A @ q < 0.5
