"""Port-vs-reference regression tests for lib/recovar.

Every test compares the JAX port against the numpy prototype vendored
verbatim at lib/recovar/_reference (importable as ``slicedcv``), on fixed
seeds. The prototype is frozen, so any disagreement is a change in the port;
these tests make the one-off equivalence checks from the porting session
durable.

Tolerances follow the dtype contract: pure-numpy code paths (rd_profile,
splits, systems, the dense scatter machinery) must agree essentially
bitwise; paths where the port stores float32 or runs a jitted float32 matvec
(basis arrays, assembly, the exact-Sw pipeline) get float32-scale
tolerances.
"""
import importlib
import inspect

import numpy as np
import pytest

# recovar/__init__ re-exports the assemble FUNCTION under the same name as
# the submodule, so "from recovar import assemble" yields the function;
# importlib resolves the module itself.
p_asm = importlib.import_module("recovar.assemble")
from recovar import basis as p_basis
from recovar import ids as p_ids
from recovar import regularize as p_reg
from recovar import splits as p_splits
from recovar import systems as p_sys

from slicedcv import core as r_core
from slicedcv import recovar as r_rec
from slicedcv import systems as r_sys


N_SAMPLES = 3000
N_DIRECTIONS = 32


def _norm_close(actual, desired, rtol, err_msg=""):
    """Elementwise closeness with an atol floor scaled to the array's own
    magnitude, so near-zero entries of sign-mixed matrices (G, w) do not
    demand impossible relative accuracy."""
    desired = np.asarray(desired)
    scale = float(np.max(np.abs(desired))) if desired.size else 1.0
    np.testing.assert_allclose(actual, desired, rtol=rtol,
                               atol=rtol * max(scale, 1e-300),
                               err_msg=err_msg)


def _max_principal_angle(U, V):
    """Largest principal angle (radians) between the column spans of U, V."""
    Qu, _ = np.linalg.qr(np.asarray(U, dtype=np.float64))
    Qv, _ = np.linalg.qr(np.asarray(V, dtype=np.float64))
    s = np.linalg.svd(Qu.T @ Qv, compute_uv=False)
    return float(np.arccos(np.clip(s.min(), -1.0, 1.0)))


# ---------------------------------------------------------------------------
# Shared fixtures: one small 2D Wolfe-Quapp dataset, built ONCE from the
# REFERENCE systems module so both sides see identical inputs, plus the two
# bases and the sparse-direct PDE ground truth.
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def wq_data():
    cfg = r_sys.wolfe_quapp_config(beta=1.0, dim=2)
    X = r_sys.grid_boltzmann_samples(cfg, N_SAMPLES, rng=0)
    in_A = cfg.in_A(X)
    in_B = cfg.in_B(X)
    thetas = r_core.isotropic_directions(N_DIRECTIONS, 2, rng=1)
    return cfg, X, in_A, in_B, thetas


@pytest.fixture(scope="module")
def bases(wq_data):
    cfg, X, in_A, in_B, thetas = wq_data
    ref = r_core.build_basis(thetas, X, in_A, in_B)
    port = p_basis.build_basis(thetas, X, in_A, in_B)
    return ref, port


@pytest.fixture(scope="module")
def ref_gab(wq_data, bases):
    cfg, X, in_A, in_B, _ = wq_data
    ref, _ = bases
    G, a, b = r_core.assemble(ref, X, in_A, in_B)
    return G, a, b


@pytest.fixture(scope="module")
def pde_grid():
    cfg = r_sys.wolfe_quapp_config(beta=1.0, dim=2)
    return r_sys.pde_committor_2d(cfg, n_grid=201)


# ---------------------------------------------------------------------------
# 1. rd_profile on canned histograms, including the degenerate paths.
# The absorption-free ridge must stay scaled to the DIFFUSION scale; a
# max-abs-diag-scaled ridge once degraded the committor 20x without raising
# (see _reference/REPORT.md, "A correction found by verifying").
# ---------------------------------------------------------------------------

def _rd_cases():
    n = 120
    s = np.linspace(0.0, 1.0, n)
    h = float(s[1] - s[0])
    rho = np.exp(-8.0 * (s - 0.5) ** 2) + 0.05
    rho_A = np.exp(-0.5 * ((s - 0.08) / 0.03) ** 2)
    rho_B = np.exp(-0.5 * ((s - 0.92) / 0.03) ** 2)
    zeros = np.zeros(n)
    return {
        # (a) the normal stiff-absorption path
        "stiff_absorption": (rho.copy(), rho_A.copy(), rho_B.copy(), h),
        # (b) no absorption anywhere: pure-Neumann singular operator,
        # handled by the diffusion-scaled ridge
        "absorption_free": (rho.copy(), zeros.copy(), zeros.copy(), h),
        # (c) dead diagonal: zero density with a huge bin width drives the
        # face conductances below 1e-300, so interior diag entries hit the
        # dead-row guard while the absorption nodes stay alive
        "dead_diagonal": (zeros.copy(), rho_A.copy(), rho_B.copy(), 100.0),
    }


@pytest.mark.parametrize("case", sorted(_rd_cases()))
def test_rd_profile_matches_reference(case):
    rho, rho_A, rho_B, h = _rd_cases()[case]
    q_port = p_basis.rd_profile(rho.copy(), rho_A.copy(), rho_B.copy(), h)
    q_ref = r_core.rd_profile(rho.copy(), rho_A.copy(), rho_B.copy(), h)
    assert np.all(np.isfinite(q_port))
    np.testing.assert_allclose(q_port, q_ref, rtol=0.0, atol=1e-14)


def test_rd_profile_stiff_case_is_a_committor():
    # Guard the guard test: the normal case must actually produce a
    # nontrivial 0-to-1 profile, not a constant fallback.
    rho, rho_A, rho_B, h = _rd_cases()["stiff_absorption"]
    q = p_basis.rd_profile(rho, rho_A, rho_B, h)
    assert q[5] < 0.05 and q[-5] > 0.95
    assert 0.0 <= q.min() and q.max() <= 1.0


# ---------------------------------------------------------------------------
# 2. build_basis + assemble + solve_dual + evaluate + fit, end to end.
# ---------------------------------------------------------------------------

def test_build_basis_matches_reference(bases):
    ref, port = bases
    np.testing.assert_array_equal(port.thetas, ref.thetas)
    assert float(np.max(np.abs(port.grids - ref.grids))) < 1e-5
    assert float(np.max(np.abs(port.q - ref.q))) < 1e-5
    assert float(np.max(np.abs(port.qp - ref.qp))) < 1e-5


@pytest.mark.parametrize("weighted", [False, True])
def test_assemble_matches_reference(wq_data, bases, weighted):
    cfg, X, in_A, in_B, _ = wq_data
    ref, port = bases
    wts = None
    if weighted:
        wts = 0.1 + np.random.default_rng(2).random(X.shape[0])
    G_r, a_r, b_r = r_core.assemble(ref, X, in_A, in_B, sample_weights=wts)
    G_p, a_p, b_p = p_asm.assemble(port, X, in_A, in_B, sample_weights=wts)
    _norm_close(G_p, G_r, 1e-6, "G")
    _norm_close(a_p, a_r, 1e-6, "a")
    _norm_close(b_p, b_r, 1e-6, "b")


def test_solve_dual_matches_reference_on_same_G(ref_gab):
    G, a, b = ref_gab
    delta = b - a
    sol_r = r_core.solve_dual(G, delta)
    sol_p = p_asm.solve_dual(G, delta)
    _norm_close(sol_p["w"], sol_r["w"], 1e-8, "w")
    _norm_close(sol_p["w_dual"], sol_r["w_dual"], 1e-8, "w_dual")
    assert abs(sol_p["M_gap"] - sol_r["M_gap"]) < 1e-8 * abs(sol_r["M_gap"])


def test_evaluate_matches_reference(wq_data, bases, ref_gab):
    # Same (w, c) on both sides isolates the interpolation and summation
    # machinery from solver conditioning.
    cfg, X, in_A, in_B, _ = wq_data
    ref, port = bases
    G, a, b = ref_gab
    sol = r_core.solve_dual(G, b - a)
    c = float(-a @ sol["w"])
    Xq = X[:800]
    q_r = r_core.evaluate(ref, sol["w"], c, Xq)
    q_p = p_asm.evaluate(port, sol["w"], c, Xq)
    assert float(np.max(np.abs(q_p - q_r))) < 1e-5


def test_fit_transition_rmse_matches_reference(wq_data, bases, pde_grid):
    cfg, X, in_A, in_B, _ = wq_data
    ref, port = bases
    q_grid, xs, ys = pde_grid
    q_true = r_sys.bilinear_interp(q_grid, xs, ys, X[:, :2])
    sol_r = r_core.fit(ref, X, in_A, in_B)
    sol_p = p_asm.fit(port, X, in_A, in_B)
    q_r = r_core.clip01(r_core.evaluate(ref, sol_r["w"], sol_r["c"], X))
    q_p = p_asm.clip01(p_asm.evaluate(port, sol_p["w"], sol_p["c"], X))
    rmse_r = r_core.transition_rmse(q_r, q_true, in_A, in_B)
    rmse_p = p_asm.transition_rmse(q_p, q_true, in_A, in_B)
    assert rmse_r < 0.08          # sanity: the comparison is meaningful
    assert abs(rmse_p - rmse_r) < 2e-4


# ---------------------------------------------------------------------------
# 3. Streaming invariance of the port's frame-block accumulation.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("weighted", [False, True])
def test_assemble_frame_block_invariance(wq_data, bases, weighted):
    cfg, X, in_A, in_B, _ = wq_data
    _, port = bases
    wts = None
    if weighted:
        wts = 0.1 + np.random.default_rng(3).random(X.shape[0])
    G1, a1, b1 = p_asm.assemble(port, X, in_A, in_B, sample_weights=wts,
                                frame_block=333)
    G2, a2, b2 = p_asm.assemble(port, X, in_A, in_B, sample_weights=wts,
                                frame_block=8192)
    _norm_close(G1, G2, 1e-12, "G")
    _norm_close(a1, a2, 1e-12, "a")
    _norm_close(b1, b2, 1e-12, "b")


# ---------------------------------------------------------------------------
# 4. halfset_eigen_regularize on random SPD pairs.
# ---------------------------------------------------------------------------

def _spd_pair(M, seed):
    rng = np.random.default_rng(seed)
    B = rng.standard_normal((M, M))
    base = B @ B.T / M
    E1 = rng.standard_normal((M, M))
    E2 = rng.standard_normal((M, M))
    G1 = base + 0.05 * (E1 @ E1.T) / M
    G2 = base + 0.05 * (E2 @ E2.T) / M
    return G1, G2


@pytest.mark.parametrize("M,n_bands,seed",
                         [(8, 2, 10), (12, 3, 11), (16, 4, 12),
                          (24, 8, 13), (33, 12, 14)])
def test_halfset_eigen_regularize_matches_reference(M, n_bands, seed):
    G1, G2 = _spd_pair(M, seed)
    R_p, info_p = p_reg.halfset_eigen_regularize(G1, G2, n_bands=n_bands)
    R_r, info_r = r_rec.halfset_eigen_regularize(G1, G2, n_bands=n_bands)
    _norm_close(R_p, R_r, 1e-10, "G_reg")
    np.testing.assert_allclose(info_p["band_ssnr"], info_r["band_ssnr"],
                               rtol=1e-10)
    # Invariants: PSD output; regularized spectrum never below the raw one.
    ev = np.linalg.eigvalsh(R_p)
    scale = float(ev.max())
    assert ev.min() >= -1e-10 * scale
    lam = info_p["lam"]
    lam_reg = info_p["lam_reg"]
    assert np.all(lam_reg >= np.maximum(lam, 0.0) - 1e-12 * scale)


def test_halfset_identical_halves_recover_mean():
    # G1 == G2 means perfect agreement: SSNR saturates at the clip value and
    # the Wiener inflation collapses to the tiny 1 + 1/SSNR factor, so
    # G_reg is Gbar = (G1 + G2) / 2 up to ~5e-7 relative.
    G1, _ = _spd_pair(16, 15)
    R_p, _ = p_reg.halfset_eigen_regularize(G1, G1.copy(), n_bands=4)
    _norm_close(R_p, G1, 1e-5, "G_reg vs Gbar")


# ---------------------------------------------------------------------------
# 5. IDS machinery on a synthetic curved-path dataset.
# ---------------------------------------------------------------------------

def _ids_synthetic(weighted):
    """Frames along a curved reaction path through 3 of d=16 features, with
    isotropic within-bin noise; qhat is the exact progress variable. The
    between-bin scatter then has 3 strong directions, keeping the compared
    eigen-subspace gap-protected."""
    rng = np.random.default_rng(21)
    n, d = 2000, 16
    t = rng.uniform(0.0, 1.0, n)
    X = 0.5 * rng.standard_normal((n, d))
    X[:, 0] += 3.0 * (t - 0.5)
    X[:, 1] += 2.0 * np.sin(np.pi * t)
    X[:, 2] += 1.5 * np.cos(2.0 * np.pi * t)
    qhat = t.copy()
    in_A = t < 0.08
    in_B = t > 0.92
    w = (0.5 + rng.random(n)) if weighted else None
    return X, qhat, in_A, in_B, w


def _uniform_or(w, n):
    return np.full(n, 1.0 / n) if w is None else np.asarray(w) / np.sum(w)


@pytest.mark.parametrize("weighted", [False, True])
def test_path_labels_and_scatter_match_reference(weighted):
    X, qhat, in_A, in_B, w = _ids_synthetic(weighted)
    w_arr = _uniform_or(w, len(qhat))
    lab_p = p_ids.path_labels(qhat, in_A, in_B, w_arr, 8)
    lab_r = r_rec.path_labels(qhat, in_A, in_B, w_arr, 8)
    np.testing.assert_array_equal(lab_p, lab_r)
    S_p, Wh_p, _ = p_ids.path_scatter_w(X, qhat, in_A, in_B, w=w)
    S_r, Wh_r, _ = r_rec.path_scatter_w(X, qhat, in_A, in_B, w=w)
    _norm_close(S_p, S_r, 1e-8, "S")
    _norm_close(Wh_p, Wh_r, 1e-8, "Wh")


@pytest.mark.parametrize("weighted", [False, True])
def test_path_directions_exact_matches_reference(weighted):
    X, qhat, in_A, in_B, w = _ids_synthetic(weighted)
    lam_p, Th_p, _, lab_p, _ = p_ids.path_directions_exact(
        X, qhat, in_A, in_B, w=w, rng=0)
    lam_r, Th_r, _, lab_r, _ = r_rec.path_directions_exact(
        X, qhat, in_A, in_B, w=w, rng=0)
    np.testing.assert_array_equal(lab_p, lab_r)
    assert len(lam_p) == len(lam_r)
    np.testing.assert_allclose(lam_p, lam_r, rtol=1e-4,
                               atol=1e-6 * float(lam_r[0]))
    # The three planted path directions are gap-protected; their span must
    # agree to well under a milliradian despite the float32 matvec.
    assert _max_principal_angle(Th_p[:, :3], Th_r[:, :3]) < 1e-3


@pytest.mark.parametrize("weighted", [False, True])
def test_rank_rule_and_floor_sampler_match_reference(weighted):
    X, qhat, in_A, in_B, w = _ids_synthetic(weighted)
    lam_p, Th_p, _, lab_p, op_p = p_ids.path_directions_exact(
        X, qhat, in_A, in_B, w=w, rng=0)
    lam_r, Th_r, _, lab_r, op_r = r_rec.path_directions_exact(
        X, qhat, in_A, in_B, w=w, rng=0)

    r_p, _, _ = p_ids.select_rank_perm_exact(
        X, lab_p, lam_p, w=w, n_perm=25, rng=np.random.default_rng(7), op=op_p)
    r_r, _, _ = r_rec.select_rank_perm_exact(
        X, lab_r, lam_r, w=w, n_perm=25, rng=np.random.default_rng(7), op=op_r)
    assert r_p == r_r
    # Structural bound only: with 10 bins the scatter rank is at most 9.
    assert 1 <= r_p <= 9

    # Floor sampler: share the reference spectrum and directions so the
    # comparison isolates the sampler + operator + CG; a tight CG tolerance
    # keeps both sides converged well past the stopping race.
    T_p, _ = p_ids.directions_exact_floor(
        lam_r, Th_r, 3, op_p, 64, np.random.default_rng(11), tol=1e-6)
    T_r, _ = r_rec.directions_exact_floor(
        lam_r, Th_r, 3, op_r, 64, np.random.default_rng(11), tol=1e-6)
    assert float(np.max(np.abs(T_p - T_r))) < 1e-3


def test_directions_exact_floor_tau_formula():
    # Locks the 60x floor bug: tau averages the DISCARDED spectrum over the
    # whole space, sum(lam[r:]) / (d - r), never over the len(lam) - r
    # nonzero generalised eigenvalues. The hand reconstruction below consumes
    # the same rng stream and only injects tau from the documented formula,
    # so any change to the denominator breaks the bitwise match.
    X, qhat, in_A, in_B, _ = _ids_synthetic(False)
    n, d = X.shape
    w_arr = np.full(n, 1.0 / n)
    labels = p_ids.path_labels(qhat, in_A, in_B, w_arr, 8)
    op = p_ids._SwOperator(X, labels, w_arr)
    Theta, _ = np.linalg.qr(np.random.default_rng(5).standard_normal((d, 6)))
    lam = np.array([50.0, 10.0, 4.0, 1.0, 0.5, 0.25])
    r, M = 2, 64
    out, _ = p_ids.directions_exact_floor(lam, Theta, r, op, M,
                                          np.random.default_rng(123))

    tau = lam[r:].sum() / (d - r)
    rng = np.random.default_rng(123)
    Xf, _ = p_ids.sample_inv_sw(op, M, rng, tol=1e-3, maxiter=60)
    Y = np.sqrt(tau) * Xf.T
    amp = np.sqrt(np.maximum(lam[:r] - tau, 0.0))
    Y = Y + (rng.standard_normal((M, r)) * amp[None, :]) @ Theta[:, :r].T
    Y = Y / np.linalg.norm(Y, axis=1, keepdims=True)
    np.testing.assert_allclose(out, Y, rtol=0.0, atol=1e-12)


# ---------------------------------------------------------------------------
# 6. splits and systems determinism.
# ---------------------------------------------------------------------------

def test_splits_match_reference():
    i1_p, i2_p = p_splits.iid_split(101, rng=4)
    i1_r, i2_r = r_rec.iid_split(101, rng=4)
    np.testing.assert_array_equal(i1_p, i1_r)
    np.testing.assert_array_equal(i2_p, i2_r)
    group = np.repeat(np.arange(3), [40, 35, 26])
    for h_p, h_r in zip(p_splits.block_split(101, 7, group=group),
                        r_rec.block_split(101, 7, group=group)):
        np.testing.assert_array_equal(h_p, h_r)


def test_nd_separable_dataset_matches_reference():
    T_p = p_sys.random_rotation(5, rng=9)
    T_r = r_sys.random_rotation(5, rng=9)
    np.testing.assert_array_equal(T_p, T_r)
    kw = dict(beta=1.0, rng=8, n_grid=150, nuisance_sigma=[1.0, 2.0, 0.5])
    out_p = p_sys.nd_separable_dataset(400, 5, transform=T_p, **kw)
    out_r = r_sys.nd_separable_dataset(400, 5, transform=T_r, **kw)
    for arr_p, arr_r in zip(out_p, out_r):
        np.testing.assert_array_equal(arr_p, arr_r)


def test_langevin_samples_match_reference():
    cfg_p = p_sys.double_well_config(beta=1.0, dim=2)
    cfg_r = r_sys.double_well_config(beta=1.0, dim=2)
    kw = dict(n_steps=40, n_walkers=2, rng=3, burn=15, thin=2)
    frames_p, wid_p = p_sys.langevin_samples(cfg_p, **kw)
    frames_r, wid_r = r_sys.langevin_samples(cfg_r, **kw)
    np.testing.assert_array_equal(frames_p, frames_r)
    np.testing.assert_array_equal(wid_p, wid_r)


# ---------------------------------------------------------------------------
# 7. ids_path_exact2 end to end.
# ---------------------------------------------------------------------------

def test_ids_path_exact2_matches_reference(pde_grid):
    # The reference's OLD defaults (n_floor_sub=6000, floor_tol=1e-3) were
    # not its validated configuration; both sides run the port's validated
    # defaults explicitly so the settings are identical. The float32 matvec
    # paths differ between numpy and JAX, so the criterion is banded
    # agreement of the final quality, not bitwise equality.
    X, in_A, in_B, base = r_sys.nd_separable_dataset(3000, 8, beta=1.0,
                                                     rng=17)
    q_grid, xs, ys = pde_grid
    q_true = r_sys.bilinear_interp(q_grid, xs, ys, base)
    kw = dict(M=48, n_passes=1, n_perm=25, n_floor_sub=None,
              floor_tol=1e-4, rng=33)
    res_p = p_ids.ids_path_exact2(X, in_A, in_B, **kw)
    res_r = r_rec.ids_path_exact2(X, in_A, in_B, **kw)

    sol_p = res_p["final"]
    sol_r = res_r["final"]
    q_p = p_asm.clip01(p_asm.evaluate(sol_p["basis"], sol_p["w"],
                                      sol_p["c"], X))
    q_r = r_core.clip01(r_core.evaluate(sol_r["basis"], sol_r["w"],
                                        sol_r["c"], X))
    e_p = p_asm.transition_rmse(q_p, q_true, in_A, in_B)
    e_r = r_core.transition_rmse(q_r, q_true, in_A, in_B)
    assert e_r < 0.2 and e_p < 0.2      # sanity: both fits are meaningful
    assert e_p < 1.3 * e_r
    assert e_r < 1.3 * e_p
    assert abs(res_p["history"][0]["r"] - res_r["history"][0]["r"]) <= 2


# ---------------------------------------------------------------------------
# 8. Defaults lock: the validated configuration must not drift.
# ---------------------------------------------------------------------------

def test_validated_defaults_locked():
    for fn in (p_ids.ids_path_exact2, p_ids.ids_fit):
        sig = inspect.signature(fn)
        assert sig.parameters["n_floor_sub"].default is None, fn.__name__
        assert sig.parameters["floor_tol"].default == 1e-4, fn.__name__
    for fn in (p_ids._SwOperator, p_ids.path_scatter_w,
               p_ids.path_directions_exact):
        sig = inspect.signature(fn)
        assert sig.parameters["shrink"].default == 0.1, getattr(
            fn, "__name__", str(fn))
