"""Tests for the Dirichlet-energy return and the settings-sweep API.

Covers :func:`committor_dirichlet_energy` (the slice-space variational
objective), the ``CommittorFit.dirichlet_energy`` field, and
:func:`sweep_committor` (Cartesian-grid fit + best-by-energy selection).

The strong checks are deterministic identities: for EBMC the reported energy is
``1/M_gap`` and must equal ``wᵀG w``; for full_gram / diagonal the energy must
equal an independent recomputation; and on a *nested* direction basis the energy
is variationally monotone (more directions cannot raise it).
"""

import warnings

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

import sliced_committor as sc

from .test_validation_2d_double_well import _equilibrium_samples


@pytest.fixture(scope="module")
def problem():
    return _equilibrium_samples(n_samples=1500)


# --------------------------------------------------------------------------- #
# committor_dirichlet_energy
# --------------------------------------------------------------------------- #


def test_ebmc_energy_equals_optimal_dirichlet_energy(problem):
    s, in_A, in_B = problem
    _, fit = sc.fit_committor(
        s,
        in_A=in_A,
        in_B=in_B,
        weights="ebmc",
        n_directions=128,
        n_bins=120,
        seed=1,
        return_details=True,
    )
    assert fit.dirichlet_energy is not None
    assert np.isfinite(fit.dirichlet_energy) and fit.dirichlet_energy > 0
    # The energy is the physical gradient energy wᵀG w (no ridge term).
    w = np.asarray(fit.weights["w"])
    G = np.asarray(fit.weights["G"])
    assert fit.dirichlet_energy == pytest.approx(float(w @ G @ w), rel=1e-9)
    # It is close to (but below) the solver's regularised 1/M_gap = wᵀ(G+ηI)w.
    reg = float(fit.weights["optimal_dirichlet_energy"])
    assert fit.dirichlet_energy <= reg * (1 + 1e-9)
    assert fit.dirichlet_energy == pytest.approx(reg, rel=0.1)


def test_full_gram_energy_matches_normalised_quadratic_form(problem):
    s, in_A, in_B = problem
    _, fit = sc.fit_committor(
        s,
        in_A=in_A,
        in_B=in_B,
        weights="full_gram",
        n_directions=128,
        n_bins=120,
        seed=1,
        return_details=True,
    )
    w = np.asarray(fit.weights["w"])
    G = np.asarray(fit.weights["G"])
    valid = np.asarray(fit.result.valid_mask, dtype=bool)
    w_eff = w * valid
    w_tilde = w_eff / w_eff.sum()
    e_manual = float(w_tilde @ G @ w_tilde)
    # Same G/w arrays; only jnp-vs-np matmul differs (full_gram G is float32).
    assert fit.dirichlet_energy == pytest.approx(e_manual, rel=1e-4)
    assert fit.dirichlet_energy > 0


def test_diagonal_energy_matches_logdirichlet_sum(problem):
    s, in_A, in_B = problem
    _, fit = sc.fit_committor(
        s,
        in_A=in_A,
        in_B=in_B,
        weights="diagonal",
        n_directions=128,
        n_bins=120,
        seed=1,
        return_details=True,
    )
    w = np.asarray(fit.weights)  # diagonal solver returns a bare (M,) array
    valid = np.asarray(fit.result.valid_mask, dtype=bool)
    w_eff = w * valid
    w_tilde = w_eff / w_eff.sum()
    log_D = np.asarray(fit.result.log_dirichlet)
    D_j = np.where(np.isfinite(log_D), np.exp(log_D), 0.0)
    e_manual = float(np.sum(w_tilde**2 * D_j))
    assert fit.dirichlet_energy == pytest.approx(e_manual, rel=1e-6)
    assert fit.dirichlet_energy > 0


def test_mode_gram_requires_a_gram(problem):
    s, in_A, in_B = problem
    _, fit = sc.fit_committor(
        s,
        in_A=in_A,
        in_B=in_B,
        weights="diagonal",
        n_directions=64,
        n_bins=120,
        seed=1,
        return_details=True,
    )
    # Bare array has no Gram and no reported energy -> mode='gram' must raise.
    with pytest.raises(ValueError):
        sc.committor_dirichlet_energy(fit.result, fit.weights, mode="gram")
    # but mode='diagonal' (and the default 'auto') succeed.
    e_diag = sc.committor_dirichlet_energy(fit.result, fit.weights, mode="diagonal")
    assert np.isfinite(e_diag) and e_diag > 0


def test_bad_mode_raises(problem):
    s, in_A, in_B = problem
    _, fit = sc.fit_committor(
        s,
        in_A=in_A,
        in_B=in_B,
        n_directions=64,
        n_bins=120,
        seed=1,
        return_details=True,
    )
    with pytest.raises(ValueError):
        sc.committor_dirichlet_energy(fit.result, fit.weights, mode="nonsense")


def test_default_fit_has_no_energy_and_returns_bare_callable(problem):
    s, in_A, in_B = problem
    q = sc.fit_committor(s, in_A=in_A, in_B=in_B, n_directions=32, n_bins=120, seed=1)
    assert callable(q)
    assert np.asarray(q(s[:5])).shape == (5,)


def test_energy_is_variationally_monotone_on_nested_basis(problem):
    s, in_A, in_B = problem
    rng = np.random.default_rng(0)
    D = rng.standard_normal((256, 2))
    D = jnp.asarray(D / np.linalg.norm(D, axis=1, keepdims=True))
    _, f128 = sc.fit_committor(
        s,
        in_A=in_A,
        in_B=in_B,
        directions=D[:128],
        n_directions=128,
        n_bins=120,
        return_details=True,
    )
    _, f256 = sc.fit_committor(
        s,
        in_A=in_A,
        in_B=in_B,
        directions=D,
        n_directions=256,
        n_bins=120,
        return_details=True,
    )
    # Nested basis + same N_eff-adaptive Tikhonov => 1/M_gap non-increasing.
    assert f256.dirichlet_energy <= f128.dirichlet_energy * (1 + 1e-4)


# --------------------------------------------------------------------------- #
# sweep_committor
# --------------------------------------------------------------------------- #


def test_sweep_cartesian_grid_and_best_selection(problem):
    s, in_A, in_B = problem
    res = sc.sweep_committor(
        s,
        in_A=in_A,
        in_B=in_B,
        grid={"n_directions": [64, 128], "weights": ["ebmc", "full_gram"]},
        n_bins=120,
        seed=1,
    )
    assert isinstance(res, sc.SweepResult)
    assert len(res.fits) == 4
    assert len(res.configs) == len(res.dirichlet_energies) == 4
    assert len(res.boundary_errors) == len(res.n_valid) == 4
    # Full product covered.
    got = {(c["n_directions"], c["weights"]) for c in res.configs}
    assert got == {(64, "ebmc"), (64, "full_gram"), (128, "ebmc"), (128, "full_gram")}
    # best_idx is the argmin of the Dirichlet energy.
    assert res.best_idx == min(range(4), key=lambda i: res.dirichlet_energies[i])
    # best_committor / best_fit are usable.
    assert res.best_fit is res.fits[res.best_idx]
    vals = np.asarray(res.best_committor(s[:10]))
    assert vals.shape == (10,)
    assert (vals >= -1e-9).all() and (vals <= 1 + 1e-9).all()
    assert "SweepResult" in res.summary()


def test_sweep_select_by_callable_is_honoured(problem):
    s, in_A, in_B = problem
    # Flip the objective: maximise the energy by minimising its negative.
    res = sc.sweep_committor(
        s,
        in_A=in_A,
        in_B=in_B,
        grid={"n_directions": [64, 128, 256]},
        weights="ebmc",
        n_bins=120,
        seed=1,
        select_by=lambda fit: -fit.dirichlet_energy,
    )
    assert res.best_idx == max(range(3), key=lambda i: res.dirichlet_energies[i])


def test_sweep_on_error_skip_and_raise(problem):
    s, in_A, in_B = problem

    def _boom(result, samples, **kw):
        raise RuntimeError("deliberate failure")

    res = sc.sweep_committor(
        s,
        in_A=in_A,
        in_B=in_B,
        grid={"weights": ["ebmc", _boom]},
        n_directions=64,
        n_bins=120,
        seed=1,
        on_error="skip",
    )
    assert len(res.fits) == 2
    failed = [i for i, f in enumerate(res.fits) if f is None]
    assert len(failed) == 1
    assert np.isinf(res.dirichlet_energies[failed[0]])
    # The winner is the surviving real fit, never the failed combo.
    assert res.fits[res.best_idx] is not None

    with pytest.raises(RuntimeError):
        sc.sweep_committor(
            s,
            in_A=in_A,
            in_B=in_B,
            grid={"weights": [_boom]},
            n_directions=64,
            n_bins=120,
            seed=1,
            on_error="raise",
        )

    # If EVERY combo fails under skip, raise a clear error (never return a
    # SweepResult whose best points at a None fit).
    with pytest.raises(RuntimeError):
        sc.sweep_committor(
            s,
            in_A=in_A,
            in_B=in_B,
            grid={"weights": [_boom, _boom]},
            n_directions=64,
            n_bins=120,
            seed=1,
            on_error="skip",
        )


def test_sweep_warns_on_mixed_energy_bases(problem):
    s, in_A, in_B = problem
    # Diagonal energy is on a different scale from the Gram-family energy, so
    # ranking a mixed sweep by "dirichlet" is invalid and must warn.
    with pytest.warns(UserWarning, match="different scales"):
        sc.sweep_committor(
            s,
            in_A=in_A,
            in_B=in_B,
            grid={"weights": ["ebmc", "diagonal"]},
            n_directions=64,
            n_bins=120,
            seed=1,
        )


def test_sweep_no_cross_family_warning_within_gram(problem):
    s, in_A, in_B = problem
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        sc.sweep_committor(
            s,
            in_A=in_A,
            in_B=in_B,
            grid={"weights": ["ebmc", "full_gram"]},
            n_directions=64,
            n_bins=120,
            seed=1,
        )
    assert not any("different scales" in str(w.message) for w in rec)


def test_sweep_input_validation(problem):
    s, in_A, in_B = problem
    with pytest.raises(ValueError):  # empty grid
        sc.sweep_committor(s, in_A=in_A, in_B=in_B, grid={})
    with pytest.raises(ValueError):  # reserved data key
        sc.sweep_committor(s, in_A=in_A, in_B=in_B, grid={"samples": [s]})
    with pytest.raises(ValueError):  # grid / fixed overlap
        sc.sweep_committor(
            s,
            in_A=in_A,
            in_B=in_B,
            grid={"n_directions": [64]},
            n_directions=128,
        )
    with pytest.raises(ValueError):  # non-list grid value
        sc.sweep_committor(s, in_A=in_A, in_B=in_B, grid={"n_directions": 64})


def test_pesb_energy_matches_quadratic_form(problem):
    # Closes the PESB gap: committor_dirichlet_energy on the (M*P,)-flattened
    # smoothstep weights must equal wᵀG w against the augmented Kronecker Gram.
    s, in_A, in_B = problem
    _, fit = sc.fit_committor(
        s,
        in_A=in_A,
        in_B=in_B,
        weights="pesb",
        n_directions=128,
        n_bins=120,
        seed=1,
        return_details=True,
    )
    w_flat = np.asarray(fit.weights["w_by_power"]).reshape(-1)
    G = np.asarray(fit.weights["G"])
    assert fit.dirichlet_energy == pytest.approx(float(w_flat @ G @ w_flat), rel=1e-6)
    assert fit.dirichlet_energy > 0


@pytest.mark.parametrize("solver", ["ebmc", "pesb", "full_gram"])
def test_energy_masks_invalid_directions(problem, solver):
    # The energy must use the SAME valid-masked weights build_committor applies,
    # so flipping directions to invalid zeroes their contribution consistently.
    s, in_A, in_B = problem
    _, fit = sc.fit_committor(
        s,
        in_A=in_A,
        in_B=in_B,
        weights=solver,
        n_directions=96,
        n_bins=120,
        seed=2,
        return_details=True,
    )
    res, w = fit.result, fit.weights
    vm = np.asarray(res.valid_mask).astype(bool).copy()
    vm[:8] = False  # force some invalid directions
    res2 = res._replace(valid_mask=jnp.asarray(vm))
    e = sc.committor_dirichlet_energy(res2, w)
    G = np.asarray(w["G"])
    if solver == "pesb":
        w_eff = (np.asarray(w["w_by_power"]) * vm[:, None]).reshape(-1)
    elif solver == "full_gram":  # normalised array combiner
        w_arr = np.asarray(w["w"]) * vm
        Z = w_arr.sum()
        w_eff = w_arr / (Z if abs(Z) > 0 else 1.0)
    else:  # ebmc centered: constant bias drops from the gradient
        w_eff = np.asarray(w["w"]) * vm
    np.testing.assert_allclose(e, float(w_eff @ G @ w_eff), rtol=1e-6, atol=1e-12)


def test_mixed_family_warning_with_callable_selector(problem):
    # The mixed-family warning must also fire for a callable select_by that
    # ranks on the Dirichlet energy, not only for the string default.
    s, in_A, in_B = problem
    with pytest.warns(UserWarning, match="different scales"):
        sc.sweep_committor(
            s,
            in_A=in_A,
            in_B=in_B,
            grid={"weights": ["ebmc", "diagonal"]},
            n_directions=64,
            n_bins=120,
            seed=1,
            select_by=lambda fit: fit.dirichlet_energy,
        )
