"""Gates for the two pieces added while diagnosing the degradation with M.

See ``docs/degradation_with_M.md``. The diagnosis is that the weight solve
overfits M coefficients to an N-sample Gram estimate and the deployed ridge
``eta = 1/sqrt(N_eff)`` never looks at M; the second-moment (Nitsche) solver was
the competing hypothesis and is kept because it is correct, not because it cures
this failure mode.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

import sliced_committor as sc
from sliced_committor.core.weights import _resolve_eta

from .test_validation_2d_double_well import _equilibrium_samples


@pytest.fixture(scope="module")
def fitted():
    samples, in_A, in_B = _equilibrium_samples(n_samples=4000, seed=3)
    result = sc.compute_sliced_committor(
        samples, in_A=in_A, in_B=in_B, n_directions=48, n_bins=80, seed=1
    )
    return result, np.asarray(samples), np.asarray(in_A), np.asarray(in_B)


# --------------------------------------------------------------------------- #
# the M-aware ridge
# --------------------------------------------------------------------------- #
def test_auto_m_is_linear_in_M_while_auto_is_M_blind():
    N = 100_000
    for M in (64, 1024):
        vm = jnp.ones(M, bool)
        eta_auto, _ = _resolve_eta("auto", M, vm, N=N)
        eta_m, _ = _resolve_eta("auto_m", M, vm, N=N)
        assert eta_auto == pytest.approx(1.0 / np.sqrt(N))  # M-blind
        assert eta_m == pytest.approx(M / N)  # linear in M
    # Doubling M must double the ridge: G sums over directions with no 1/M
    # quadrature weight, so its spectrum scales with M while median(diag G) does
    # not. The exponent is 1, not 1/2 -- see the duplication test in
    # experiments/msweep_2d_normalization.py.
    a, _ = _resolve_eta("auto_m", 256, jnp.ones(256, bool), N=N)
    b, _ = _resolve_eta("auto_m", 512, jnp.ones(512, bool), N=N)
    assert b / a == pytest.approx(2.0)


def test_lambda_form_is_eta_over_M():
    lam = 4e-4
    for M in (32, 2048):
        eta, n_eff = _resolve_eta(("lambda", lam), M, jnp.ones(M, bool), N=100)
        assert eta == pytest.approx(lam * M)
        assert n_eff is None
    with pytest.raises(ValueError):
        _resolve_eta(("relative", 1e-3), 8, jnp.ones(8, bool), N=100)


def test_lambda_anchors_the_ridge_on_M_times_geomean():
    """``('lambda', lam)`` must deliver an absolute ridge of ``lam*M*geomean(diag)``.

    The downstream regulariser multiplies eta by ``median(diag G)``, so the
    resolver has to divide it back out. The median is the wrong anchor: it is a
    knife-edge order statistic under the two-stratum LDA sampler, and it is the
    only candidate that is not extensive in M. The geometric mean is used in
    preference to the trace because it transfers best across systems, not because
    duplication singles it out -- duplication only fixes extensivity.
    """
    rng = np.random.default_rng(0)
    M = 256
    d = np.concatenate(
        [rng.lognormal(0.0, 0.3, M // 2), rng.lognormal(8.0, 0.3, M // 2)]
    )  # two strata, mean >> median
    G = jnp.asarray(np.diag(d))
    vm = jnp.ones(M, bool)
    lam = 1e-4
    med = float(np.median(d))
    geo = float(np.exp(np.mean(np.log(d))))

    eta, _ = _resolve_eta(("lambda", lam), M, vm, N=1000, G=G)
    assert eta * med == pytest.approx(lam * M * geo, rel=1e-9)

    eta_noG, _ = _resolve_eta(("lambda", lam), M, vm, N=1000)
    assert eta_noG == pytest.approx(lam * M)  # documented fallback
    assert eta != pytest.approx(eta_noG)  # and they differ here


def test_auto_lambda_scales_as_one_over_N_eff():
    """``ridge = 1.8e-4 * (1e5/N_eff) * M * geomean(diag G)``.

    The ``1/N_eff`` factor is not cosmetic: without it the rule is only right at
    the N the constant was calibrated on. On the 2D benchmark the N-blind form
    loses 1.63x against the per-configuration oracle over N = 25k..400k, and the
    M-degradation itself returns at N = 25 000 (1.41x over M = 64..512).
    """
    rng = np.random.default_rng(1)
    M = 128
    d = rng.lognormal(0.0, 2.0, M)
    G = jnp.asarray(np.diag(d))
    vm = jnp.ones(M, bool)
    geo, med = float(np.exp(np.mean(np.log(d)))), float(np.median(d))

    for N in (2.5e4, 1e5, 4e5):
        eta, _ = _resolve_eta("auto_lambda", M, vm, N=N, G=G)
        assert eta * med == pytest.approx(1.8e-4 * (1e5 / N) * M * geo, rel=1e-9)

    # halving N_eff must double the ridge
    lo, _ = _resolve_eta("auto_lambda", M, vm, N=5e4, G=G)
    hi, _ = _resolve_eta("auto_lambda", M, vm, N=1e5, G=G)
    assert lo / hi == pytest.approx(2.0, rel=1e-9)

    # non-uniform weights go through N_eff = (sum w)^2 / sum w^2, not len(w)
    w = np.concatenate([np.full(400, 1.0), np.full(3600, 1e-6)])
    n_eff = w.sum() ** 2 / (w**2).sum()
    e_w, _ = _resolve_eta("auto_lambda", M, vm, sample_weights=jnp.asarray(w), G=G)
    e_n, _ = _resolve_eta("auto_lambda", M, vm, N=n_eff, G=G)
    assert e_w == pytest.approx(e_n, rel=1e-9)
    assert n_eff < 0.2 * len(w)  # the weights really are concentrated


def test_auto_lambda_refuses_to_guess_N():
    """Silently dropping the 1/N_eff factor would be wrong except at N = 1e5."""
    G = jnp.asarray(np.diag(np.full(16, 2.0)))
    with pytest.raises(ValueError, match=r"N_eff|sample_weights or N"):
        _resolve_eta("auto_lambda", 16, jnp.ones(16, bool), G=G)


def test_lambda_ignores_nonpositive_diagonal_entries():
    """Masked-out or numerically zero directions must not sink the geometric mean."""
    M = 64
    d = np.full(M, 4.0)
    d[:8] = 0.0  # e.g. zeroed by the valid mask
    G = jnp.asarray(np.diag(d))
    vm = jnp.asarray([False] * 8 + [True] * (M - 8))
    eta, _ = _resolve_eta(("lambda", 1e-3), M, vm, N=1000, G=G)
    assert np.isfinite(eta)
    assert eta * 4.0 == pytest.approx(1e-3 * (M - 8) * 4.0, rel=1e-9)


def test_auto_m_and_lambda_count_only_valid_directions():
    vm = jnp.asarray([True] * 100 + [False] * 156)
    eta, _ = _resolve_eta("auto_m", 256, vm, N=10_000)
    assert eta == pytest.approx(100 / 10_000)
    eta2, _ = _resolve_eta(("lambda", 1e-3), 256, vm, N=10_000)
    assert eta2 == pytest.approx(1e-3 * 100)


def test_unknown_eta_string_rejected():
    with pytest.raises(ValueError):
        _resolve_eta("adaptive", 8, jnp.ones(8, bool), N=100)


def test_float_eta_passes_through():
    assert _resolve_eta(0.05, 8, jnp.ones(8, bool), N=100) == (0.05, None)


def test_duplicated_directions_need_a_proportionally_larger_ridge(fitted):
    """The exponent, measured rather than assumed.

    Duplicating every direction k-fold cannot change the trial space, so the
    fitted function must not change. It does change at fixed ``eta``, because the
    ridge penalises ``||w||^2`` while the duplicated solution splits its weight
    across copies. Solving the k-fold problem at ``k*eta`` restores it exactly.
    That fixes the ridge's M-dependence at exponent 1 with no continuum argument,
    no statistics and no reference solution.
    """
    from sliced_committor.core._bmc import compute_basin_moments
    from sliced_committor.core._bmc_enriched import solve_enriched_basin_moment
    from sliced_committor.core.gram import _assemble_gram_matrix, _compute_derivative_matrix
    from sliced_committor.core.solver import make_weighting_context

    result, samples, _, _ = fitted
    ctx = make_weighting_context(result)
    S = ctx.projected_samples
    W = jnp.ones(S.shape[1]) / S.shape[1]
    F = _compute_derivative_matrix(ctx, S)
    G = np.asarray(
        _assemble_gram_matrix(
            F.astype(jnp.float64), W.astype(jnp.float64), ctx.directions @ ctx.directions.T
        )
    )
    a, b = (np.asarray(x) for x in compute_basin_moments(ctx))
    M = G.shape[0]

    def solve(Gm, av, bv, eta):
        n = Gm.shape[0]
        out = solve_enriched_basin_moment(
            jnp.asarray(Gm),
            jnp.asarray(av),
            jnp.asarray(bv),
            jnp.ones(n, bool),
            eta=eta,
            N=S.shape[1],
            raise_on_degenerate=False,
        )
        return np.asarray(out["w"]), float(out["c"])

    eta0 = 1e-2
    w0, c0 = solve(G, a, b, eta0)
    for k in (2, 4):
        idx = np.repeat(np.arange(M), k)
        Gk, ak, bk = G[np.ix_(idx, idx)], a[idx], b[idx]

        wk, ck = solve(Gk, ak, bk, k * eta0)  # scaled ridge: invariant
        assert np.abs(wk.reshape(M, k).sum(1) - w0).max() < 1e-8 * max(np.abs(w0).max(), 1.0)
        assert abs(ck - c0) < 1e-8 * max(abs(c0), 1.0)

        wb, _ = solve(Gk, ak, bk, eta0)  # fixed ridge: NOT invariant
        assert np.abs(wb.reshape(M, k).sum(1) - w0).max() > 1e-6


# --------------------------------------------------------------------------- #
# the second-moment (Nitsche) solver
# --------------------------------------------------------------------------- #
def test_nitsche_returns_centered_basis_keys(fitted):
    result, samples, _, _ = fitted
    w = sc.compute_nitsche_weights(result, samples)
    # 'q_bar' is what routes evaluate_committor to the centered-basis path; its
    # absence would silently rescale the affine solution by 1/sum(w).
    assert "c" in w and "q_bar" in w and "w" in w
    assert np.asarray(w["q_bar"]).shape == np.asarray(w["w"]).shape
    assert np.all(np.asarray(w["q_bar"]) == 0.0)


def test_nitsche_penalty_tightens_boundary_conditions(fitted):
    """Raising beta must monotonically shrink the basin second moments.

    This is the only thing the penalty controls, and on the equilibrium 2D
    benchmark it is already ~1e-8 without it -- which is why tightening it four
    further orders buys almost nothing (docs/degradation_with_M.md Sec. 2).
    """
    result, samples, _, _ = fitted
    resid = []
    for bt in (1e1, 1e3, 1e5):
        w = sc.compute_nitsche_weights(result, samples, beta_tilde=bt)
        resid.append(w["bc_resid_A"] + w["bc_resid_B"])
        assert w["beta_used"] > 0
    assert resid[0] > resid[1] > resid[2]


def test_nitsche_matches_ebmc_when_penalty_is_negligible(fitted):
    """As beta -> 0 the boundary data drops out and both solves collapse.

    With no penalty the (M+1) system degenerates to the unconstrained
    minimum-energy solution w = 0, c = 1/2 (the constant halfway between the two
    basin values), which is the correct limit and a useful guard: it shows the
    penalty, not the Gram, is what carries the boundary information here.
    """
    result, samples, _, _ = fitted
    w = sc.compute_nitsche_weights(result, samples, beta_tilde=1e-10)
    assert abs(w["c"] - 0.5) < 0.05
    assert np.abs(np.asarray(w["w"])).sum() < 0.1


def test_nitsche_committor_respects_basins(fitted):
    result, samples, in_A, in_B = fitted
    w = sc.compute_nitsche_weights(result, samples, beta_tilde=1e3)
    q = np.asarray(sc.build_committor(result, w)(samples), dtype=float)
    assert np.isfinite(q).all()
    assert q[in_A].mean() < 0.1
    assert q[in_B].mean() > 0.9
