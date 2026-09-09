"""2D reproduction gates for the JAX port of the RECOVAR prototype.

Each gate re-runs one of the numpy prototype's validated 2D experiments
(protocol transcribed from the original t-scripts, vendored logs under
``_reference/logs/``) with THIS package, and asserts banded tolerances.
The bands are deliberately loose: BLAS order, float32 rounding and jitted
accumulation move the 3rd digit, so bitwise agreement is neither expected
nor asserted; what is asserted is that every headline conclusion of
``_reference/REPORT.md`` survives the port.

Gates (one function each, ``gate_<name>``):

    core            t01_validate.py  : M ladder, kappa insensitivity,
                    oracle-LS floor, PDE-oracle cross-check vs the repo
                    Jacobi baseline (lib/examples/_wolfe_quapp.py)
    halfset         t03_reg.py + t09b.log : halfset-eigen regularization vs
                    hand-set and oracle ridge, single seed + 5 seeds
    blocks          t02b_blocks.py   : random vs contiguous-block splits on
                    time-correlated Langevin data (overfit sign inversion)
    bandwidth       t06b_bandwidth.py: adaptive profile/derivative-CV
                    bandwidth vs tuned global; density-ISE negative control
    ids             t21_final.py     : exact linear-in-d IDS at d=96/192/384
    ids_multiseed   t19_multiseed.py : port exact IDS vs the VENDORED dense
                    reference (ids_path2), 3 seeds, d=96/192
    ids_consistency small shared-seed port-vs-reference ids_path_exact2 run

Every gate returns ``dict(name, passed, table, details, runtime_s)`` and is
driven by ``experiments/recovar_repro.py``.
"""
import sys
import time
from pathlib import Path

import numpy as np

from . import systems as sy
from .assemble import (assemble, assemble_slice_values, clip01, evaluate,
                       fit, solve_dual, transition_rmse)
from .basis import build_basis, isotropic_directions
from .bandwidth import make_adaptive_bw_fn, make_halfset_bw_fn
from .dual import nested_M_curve
from .ids import _fit_pass_w, ids_path_exact2
from .regularize import (halfset_eigen_regularize, halfset_grams_recovar,
                         solve_with_G)
from .splits import block_split, iid_split


# ===========================================================================
# Hard-coded pass bands. Sources: the vendored prototype logs under
# ``_reference/logs/`` and ``_reference/REPORT.md``. Bands are the contract;
# the cited numbers are what the numpy prototype measured.
# ===========================================================================

# t01.npz: nu_ref = 6.62539e-3 on the 401-grid sparse-direct solve.
PROTO_NU_REF = 6.6254e-3

# core gate; t01.npz rows: RMSE(M=512) = 0.004218, nu ratio 0.984 (M=16)
# and 0.970 (M=32); REPORT 'Validation': RMSE 0.0037..0.0042, kappa-flat
# from 1e2 to 1e8, oracle-LS 0.0014.
CORE_RMSE_M512_BAND = (0.0030, 0.0055)
CORE_NU_SMALL_M_BAND = (0.96, 1.04)
CORE_KAPPA_SPREAD_MAX = 1.5
CORE_ORACLE_LS_BAND = (0.0010, 0.0025)
# Both oracles solve the same PDE (401-grid sparse direct vs 300-grid
# Jacobi, tol 1e-9); disagreement is discretization only.
CORE_XCHECK_RMSE_MAX = 1.0e-3
# t01.npz reference rows per M: (RMSE, nu_hat/nu_ref).
PROTO_CORE_ROWS = {16: (0.00372, 0.984), 32: (0.00433, 0.970),
                   64: (0.00403, 0.969), 128: (0.00411, 0.965),
                   256: (0.00416, 0.960), 512: (0.00422, 0.956)}

# halfset gate; REPORT sec. 1 single-seed table: (N=60k,M=512)
# 0.0042/0.0037/0.0036, (15k,512) 0.0073/0.0038/0.0033, (6k,256)
# 0.0450/0.0241/0.0120 for default-ridge/oracle-ridge/halfset; halfset flux
# ratio 0.98..1.02 while the default ridge gives 0.076 at N=6000.
# t09b.log sec. 2 (WQ beta=1.0): 5/5 wins at N=6000 (mean ratio 0.494) and
# 5/5 at N=15000 (mean ratio 0.920, measured at M=256).
HALFSET_VS_ORACLE_MAX = 0.75      # at (6000, 256); prototype 0.0120/0.0241 = 0.50
# Band adjustment, measured 2026-08-14: the REPORT's "flux ratio 0.98..1.02"
# does NOT reproduce at (6000, 256) from the VENDORED prototype itself. On
# the exact t03 protocol (seed 0) the vendored numpy code gives flux ratio
# 0.7303 and the port gives 0.7303 (4-digit agreement, so the port is
# faithful); across seeds 0..5 the prototype spans 0.61..0.94 there. t03's
# own log is not vendored, so 0.98..1.02 cannot be pinned to these seeds.
# The tight near-1 band IS reproduced at the two larger configs (measured
# 0.978 at (60000,512), 1.009 at (15000,512)) and is asserted THERE; at
# (6000, 256) the reproducible claim is the contrast against the collapsed
# default ridge (0.730 vs 0.076, a ~10x gap; REPORT quotes 13x).
HALFSET_FLUX_BAND = (0.90, 1.10)        # at (15000,512) and (60000,512)
HALFSET_FLUX_BAND_HARD = (0.55, 1.15)   # at (6000,256); measured 0.61..0.94
DEFAULT_FLUX_MAX = 0.5            # at (6000, 256); prototype 0.076
HALFSET_WINS_MIN = 3              # of 5 seeds at (6000, 256); prototype 5/5
PROTO_HALFSET_ROWS = {(60000, 512): (0.0042, 0.0037, 0.0036),
                      (15000, 512): (0.0073, 0.0038, 0.0033),
                      (6000, 256): (0.0450, 0.0241, 0.0120)}

# blocks gate; REPORT sec. 10 at M=256: random-frame -2.7%, block-20 +31%,
# block-100 +59%, whole-walker +64%. Overfit% = 100*(h_train-h_test)/h_train,
# exactly as t02b_blocks.py prints it.
BLOCKS_RANDOM_MAX = 10.0          # percent; prototype -2.7
BLOCKS_B100_MIN = 25.0            # percent; prototype +59
BLOCKS_WALKER_MIN = 35.0          # percent; prototype +64
BLOCKS_MONOTONE_SLACK = 5.0       # percentage points on block20<=block100<=walker

# bandwidth gate; t06b.log: deriv-CV / best-global = 0.81 / 0.93 / 0.84 at
# N = 6k / 15k / 60k; density-ISE at N=6000 = 0.0154 vs best 0.0113 (1.36x).
BW_DERIV_VS_BEST_MAX = 1.05
BW_ISE_VS_BEST_MIN = 1.15
PROTO_BW_ROWS = {6000: dict(best=0.0113, profile=0.0104, deriv=0.0092, ise=0.0154),
                 15000: dict(best=0.0054, profile=0.0052, deriv=0.0051, ise=0.0146),
                 60000: dict(best=0.0039, profile=0.0038, deriv=0.0033, ise=0.0139)}

# ids gate; t21.log rotated block: exact-linear 0.0111(nu 1.00) at d=96,
# 0.0111(1.04) at 192, 0.0148(1.04) at 384; isotropic 0.2191/0.2578/0.3252;
# per-pass ranks 4..7.
IDS_RMSE_MAX = {96: 0.015, 192: 0.015, 384: 0.020}
IDS_NU_BAND = (0.8, 1.2)
IDS_ISO_RMSE_MIN = 0.20
IDS_RANK_BAND = (2, 9)
PROTO_IDS_ROWS = {96: dict(iso=0.2191, ids=0.0111, nu=1.00, ranks=[5, 5, 7]),
                  192: dict(iso=0.2578, ids=0.0111, nu=1.04, ranks=[7, 6, 7]),
                  384: dict(iso=0.3252, ids=0.0148, nu=1.04, ranks=[7, 6, 4])}

# ids-multiseed gate; t19c.log: mean exact / mean dense RMSE ratio 0.95 at
# d=96 and 1.10 at d=192 (seeds 7/17/27).
IDS_MS_RATIO_BAND = (0.80, 1.25)
PROTO_IDS_MS = {96: dict(dense=0.0110, exact=0.0104, ratio=0.95),
                192: dict(dense=0.0113, exact=0.0125, ratio=1.10)}

# ids-consistency gate: port vs vendored reference on shared seeds; both
# implementations run the validated t21 settings, so they should land on
# the same answer up to float32/BLAS noise, and both should crush the
# single-pass isotropic fit.
IDS_CONS_RATIO_BAND = (0.7, 1.4)
IDS_CONS_VS_ISO_MAX = 0.5


# ===========================================================================
# Shared helpers
# ===========================================================================

_REPO_ROOT = Path(__file__).resolve().parents[2]
_BKW = dict(n_bins=200, bw_bins=1.5)


def _default_cache(cache_dir):
    cache_dir = Path(cache_dir) if cache_dir is not None else (
        _REPO_ROOT / '.cache' / 'recovar')
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def _wq_oracle(cache_dir, verbose=True):
    """Cached WQ beta=1.0 PDE oracle (sparse direct, n_grid=401) + nu_ref.

    Returns (cfg, q_grid, xs, ys, nu_ref). Prototype: nu_ref = 6.6254e-3.
    """
    cfg = sy.wolfe_quapp_config(beta=1.0, dim=2)
    path = _default_cache(cache_dir) / 'wq_oracle_b1_n401.npz'
    if path.exists():
        d = np.load(path)
        return cfg, d['qg'], d['xs'], d['ys'], float(d['nu_ref'])
    t0 = time.time()
    qg, xs, ys = sy.pde_committor_2d(cfg, n_grid=401)
    nu_ref = sy.pde_reactive_flux(cfg, qg, xs, ys)
    np.savez(path, qg=qg, xs=xs, ys=ys, nu_ref=nu_ref)
    if verbose:
        print(f"  [oracle] PDE solve {time.time() - t0:.1f}s, "
              f"nu_ref = {nu_ref:.5e} (prototype {PROTO_NU_REF:.5e})")
    return cfg, qg, xs, ys, float(nu_ref)


def _jacobi_oracle(cache_dir, verbose=True):
    """Cached repo Jacobi committor (lib/examples/_wolfe_quapp.py defaults:
    BETA=1.0, PDE_N_GRID=300, tol 1e-9). Returns (q_xy, xs, ys); q_xy uses
    meshgrid indexing 'xy', so q_xy.T is 'ij'-indexed for bilinear_interp.
    """
    path = _default_cache(cache_dir) / 'wq_jacobi_b1_n300.npz'
    if path.exists():
        d = np.load(path)
        return d['q'], d['xs'], d['ys']
    ex_dir = str(_REPO_ROOT / 'lib' / 'examples')
    if ex_dir not in sys.path:
        sys.path.insert(0, ex_dir)
    import _wolfe_quapp as wq
    t0 = time.time()
    q, _, _, iters, delta = wq.jacobi_committor_2d()
    xs = np.linspace(*wq.DOMAIN, wq.PDE_N_GRID)
    np.savez(path, q=q, xs=xs, ys=xs, iters=iters, delta=delta)
    if verbose:
        print(f"  [jacobi] {iters} iters, final delta {delta:.1e}, "
              f"{time.time() - t0:.1f}s")
    return q, xs, xs


def _import_reference():
    """Import the vendored numpy prototype package (slicedcv)."""
    ref_dir = str(Path(__file__).resolve().parent / '_reference')
    if ref_dir not in sys.path:
        sys.path.insert(0, ref_dir)
    from slicedcv import core as ref_co
    from slicedcv import recovar as ref_rc
    return ref_co, ref_rc


def _nd_dataset(dim, seed, qg, xs, ys, n=20000):
    """t21/t19 'rotated' dataset: rotation seed = data seed + 1000."""
    T = sy.random_rotation(dim, rng=seed + 1000)
    X, in_A, in_B, base = sy.nd_separable_dataset(
        n, dim, beta=1.0, rng=np.random.default_rng(seed),
        nuisance_sigma=None, transform=T)
    q_ref = sy.bilinear_interp(qg, xs, ys, base)
    return X, in_A, in_B, q_ref


def _score_sol(sol, X, in_A, in_B, q_ref, nu_ref):
    """t21's sc(): transition RMSE of the clipped evaluation + nu ratio."""
    qh = clip01(evaluate(sol['basis'], sol['w'], sol['c'], X))
    return transition_rmse(qh, q_ref, in_A, in_B), sol['nu_hat'] / nu_ref


def _print_table(rows, verbose):
    if not verbose or not rows:
        return
    keys = list(dict.fromkeys(k for r in rows for k in r))
    print('  ' + '  '.join(f'{k:>14}' for k in keys))
    for r in rows:
        cells = []
        for k in keys:
            v = r.get(k, '')
            if isinstance(v, float):
                cells.append(f'{v:>14.4g}')
            else:
                cells.append(f'{str(v):>14}')
        print('  ' + '  '.join(cells))


def _result(name, passed, table, details, t0, verbose):
    if verbose:
        print(f"  gate {name}: {'PASS' if passed else 'FAIL'} "
              f"({time.time() - t0:.1f}s)")
    return dict(name=name, passed=bool(passed), table=table, details=details,
                runtime_s=float(time.time() - t0))


# ===========================================================================
# Gate 1: core (t01_validate.py)
# ===========================================================================

def gate_core(fast=False, seed=0, cache_dir=None, verbose=True):
    """M ladder + kappa insensitivity + oracle-LS floor + oracle cross-check."""
    t0 = time.time()
    cache_dir = _default_cache(cache_dir)
    cfg, qg, xs, ys, nu_ref = _wq_oracle(cache_dir, verbose)

    N = 60000
    X = sy.grid_boltzmann_samples(cfg, N, n_grid=500,
                                  rng=np.random.default_rng(seed))
    in_A, in_B = cfg.in_A(X), cfg.in_B(X)
    tr = ~in_A & ~in_B
    q_ref = sy.bilinear_interp(qg, xs, ys, X)

    table = []
    rmse_by_M, nu_by_M = {}, {}
    for M in [16, 32, 64, 128, 256, 512]:
        th = isotropic_directions(M, 2, rng=np.random.default_rng(100 + M))
        basis = build_basis(th, X, in_A, in_B, **_BKW)
        res = fit(basis, X, in_A, in_B, tikhonov=1e-3)
        qh = clip01(evaluate(basis, res['w'], res['c'], X))
        r = transition_rmse(qh, q_ref, in_A, in_B)
        ratio = res['nu_hat'] / nu_ref
        rmse_by_M[M], nu_by_M[M] = r, ratio
        pr, pn = PROTO_CORE_ROWS[M]
        table.append(dict(M=M, rmse=r, nu_ratio=ratio,
                          rmse_proto=pr, nu_ratio_proto=pn))
    _print_table(table, verbose)

    # kappa_rel insensitivity at M=256 (t01 structure, direction seed 7)
    th = isotropic_directions(256, 2, rng=np.random.default_rng(7))
    kappa_rmse = {}
    for kr in [1e2, 1e4, 1e6, 1e8]:
        basis = build_basis(th, X, in_A, in_B, kappa_rel=kr, **_BKW)
        res = fit(basis, X, in_A, in_B)
        qh = clip01(evaluate(basis, res['w'], res['c'], X))
        kappa_rmse[f'{kr:.0e}'] = transition_rmse(qh, q_ref, in_A, in_B)
    kvals = np.array(list(kappa_rmse.values()))
    kappa_spread = float(kvals.max() / kvals.min())

    # Oracle least-squares fit of q_ref in the same slice basis (t01):
    # regress q_ref on the M=256 slice features + a constant over ALL
    # samples, clip, and score on the transition region. Disclosed oracle:
    # it sees the truth, and it lower-bounds what the basis can express.
    th = isotropic_directions(256, 2, rng=np.random.default_rng(356))
    basis = build_basis(th, X, in_A, in_B, **_BKW)
    QV, _ = assemble_slice_values(basis, X)
    A_ls = np.concatenate([QV.T.astype(np.float64), np.ones((N, 1))], axis=1)
    coef = np.linalg.lstsq(A_ls, q_ref, rcond=None)[0]
    q_or = clip01(A_ls @ coef)
    oracle_ls_rmse = transition_rmse(q_or, q_ref, in_A, in_B)

    # Cross-check: this package's sparse-direct oracle vs the repo Jacobi
    # oracle (lib/examples/_wolfe_quapp.py), interpolated on the same
    # interior (transition-region) sample points. q_jac is 'xy'-indexed.
    q_jac, xj, yj = _jacobi_oracle(cache_dir, verbose)
    q_jac_at = sy.bilinear_interp(np.asarray(q_jac).T, xj, yj, X[tr])
    xcheck_rmse = float(np.sqrt(np.mean((q_jac_at - q_ref[tr]) ** 2)))

    checks = dict(
        rmse_M512=CORE_RMSE_M512_BAND[0] <= rmse_by_M[512] <= CORE_RMSE_M512_BAND[1],
        nu_ratio_M16=CORE_NU_SMALL_M_BAND[0] <= nu_by_M[16] <= CORE_NU_SMALL_M_BAND[1],
        nu_ratio_M32=CORE_NU_SMALL_M_BAND[0] <= nu_by_M[32] <= CORE_NU_SMALL_M_BAND[1],
        kappa_spread=kappa_spread < CORE_KAPPA_SPREAD_MAX,
        oracle_ls=CORE_ORACLE_LS_BAND[0] <= oracle_ls_rmse <= CORE_ORACLE_LS_BAND[1],
        oracle_xcheck=xcheck_rmse < CORE_XCHECK_RMSE_MAX,
    )
    details = dict(nu_ref=nu_ref, nu_ref_prototype=PROTO_NU_REF,
                   kappa_rmse=kappa_rmse, kappa_spread=kappa_spread,
                   oracle_ls_rmse=oracle_ls_rmse,
                   oracle_ls_rmse_prototype=0.0014,
                   oracle_xcheck_rmse=xcheck_rmse,
                   n_transition=int(tr.sum()),
                   barrier_frac=float(np.mean((q_ref[tr] > .1) & (q_ref[tr] < .9))),
                   checks=checks,
                   bands=dict(rmse_M512=CORE_RMSE_M512_BAND,
                              nu_small_M=CORE_NU_SMALL_M_BAND,
                              kappa_spread_max=CORE_KAPPA_SPREAD_MAX,
                              oracle_ls=CORE_ORACLE_LS_BAND,
                              xcheck_rmse_max=CORE_XCHECK_RMSE_MAX))
    if verbose:
        print(f"  kappa spread {kappa_spread:.3f}, oracle-LS "
              f"{oracle_ls_rmse:.4f}, oracle cross-check {xcheck_rmse:.2e}")
    return _result('core', all(checks.values()), table, details, t0, verbose)


# ===========================================================================
# Gate 2: halfset (t03_reg.py + t09_verify.py section 2)
# ===========================================================================

def _halfset_single(cfg, qg, xs, ys, nu_ref, N, M, seed):
    """t03's run(): one seed, three arms on a shared basis and split."""
    X = sy.grid_boltzmann_samples(cfg, N, n_grid=500,
                                  rng=np.random.default_rng(seed))
    in_A, in_B = cfg.in_A(X), cfg.in_B(X)
    q_ref = sy.bilinear_interp(qg, xs, ys, X)
    th = isotropic_directions(M, 2, rng=np.random.default_rng(seed + 77))
    basis = build_basis(th, X, in_A, in_B, **_BKW)
    split = iid_split(N, rng=np.random.default_rng(seed + 5))
    G1, G2, G, a, b = halfset_grams_recovar(basis, X, in_A, in_B, split)
    delta = b - a

    def score(sol, c):
        qh = clip01(evaluate(basis, sol['w'], c, X))
        return transition_rmse(qh, q_ref, in_A, in_B), sol['nu_hat'] / nu_ref

    # arm 1: the hand-set default ridge
    s_def = solve_dual(G, delta, tikhonov=1e-3)
    rmse_def, flux_def = score(s_def, float(-a @ s_def['w']))

    # arm 2: oracle ridge (picked by TRUE transition RMSE; disclosed oracle)
    best = dict(rmse=np.inf, flux=np.nan, eta=np.nan)
    for eta in np.logspace(-8, -1, 8):
        s = solve_dual(G, delta, tikhonov=float(eta))
        r, f = score(s, float(-a @ s['w']))
        if r < best['rmse']:
            best = dict(rmse=r, flux=f, eta=float(eta))

    # arm 3: halfset-eigen (tuning-free)
    Gr, _info = halfset_eigen_regularize(G1, G2, n_bands=12)
    s_h = solve_with_G(Gr, a, b)
    rmse_h, flux_h = score(s_h, s_h['c'])

    return dict(N=N, M=M, rmse_default=rmse_def, flux_default=flux_def,
                rmse_oracle=best['rmse'], oracle_eta=best['eta'],
                rmse_halfset=rmse_h, flux_halfset=flux_h)


def _halfset_five_seed(cfg, qg, xs, ys, N, M, seed_offset=0):
    """t09_verify.py section 2, transcribed: 5 seeds, even/odd halfset
    Grams, oracle ridge over [1e-1, 1e-2, 1e-3, 1e-4]."""
    wins, ratios = 0, []
    for s in range(5):
        X = sy.grid_boltzmann_samples(
            cfg, N, n_grid=500, rng=np.random.default_rng(400 + seed_offset + s))
        A, B = cfg.in_A(X), cfg.in_B(X)
        q_ref = sy.bilinear_interp(qg, xs, ys, X)
        th = isotropic_directions(M, 2, rng=np.random.default_rng(seed_offset + s + 9))
        bs = build_basis(th, X, A, B, **_BKW)
        G, a, b = assemble(bs, X, A, B)
        G1, *_ = assemble(bs, X[::2], A[::2], B[::2])
        G2, *_ = assemble(bs, X[1::2], A[1::2], B[1::2])
        Gr, _ = halfset_eigen_regularize(G1, G2, n_bands=12)
        s_h = solve_dual(Gr, b - a, G_is_regularised=True)
        r_h = transition_rmse(
            clip01(evaluate(bs, s_h['w'], float(-a @ s_h['w']), X)), q_ref, A, B)
        best = np.inf
        for eta in [1e-1, 1e-2, 1e-3, 1e-4]:
            sol = solve_dual(G, b - a, eta)
            r = transition_rmse(
                clip01(evaluate(bs, sol['w'], float(-a @ sol['w']), X)),
                q_ref, A, B)
            best = min(best, r)
        ratios.append(r_h / best)
        if r_h <= best:
            wins += 1
    return wins, [float(r) for r in ratios]


def gate_halfset(fast=False, seed=0, cache_dir=None, verbose=True):
    """Halfset-eigen Gram regularization vs default and oracle ridge."""
    t0 = time.time()
    cache_dir = _default_cache(cache_dir)
    cfg, qg, xs, ys, nu_ref = _wq_oracle(cache_dir, verbose)

    table = []
    by_cfg = {}
    for (N, M) in [(60000, 512), (15000, 512), (6000, 256)]:
        row = _halfset_single(cfg, qg, xs, ys, nu_ref, N, M, seed)
        pd, po, ph = PROTO_HALFSET_ROWS[(N, M)]
        row.update(rmse_default_proto=pd, rmse_oracle_proto=po,
                   rmse_halfset_proto=ph)
        by_cfg[(N, M)] = row
        table.append(row)
    _print_table(table, verbose)

    wins6, ratios6 = _halfset_five_seed(cfg, qg, xs, ys, 6000, 256, seed)
    wins15, ratios15 = _halfset_five_seed(cfg, qg, xs, ys, 15000, 512, seed)
    if verbose:
        print(f"  5-seed (6000,256): wins {wins6}/5, mean ratio "
              f"{np.mean(ratios6):.3f} (prototype 5/5, 0.494)")
        print(f"  5-seed (15000,512): wins {wins15}/5, mean ratio "
              f"{np.mean(ratios15):.3f} (prototype 5/5 at M=256, 0.920)")

    hard = by_cfg[(6000, 256)]
    checks = dict(
        halfset_vs_oracle=(hard['rmse_halfset']
                           <= HALFSET_VS_ORACLE_MAX * hard['rmse_oracle']),
        # near-1 flux where the prototype reproduces it (see the band note)
        halfset_flux_60000_512=(
            HALFSET_FLUX_BAND[0] <= by_cfg[(60000, 512)]['flux_halfset']
            <= HALFSET_FLUX_BAND[1]),
        halfset_flux_15000_512=(
            HALFSET_FLUX_BAND[0] <= by_cfg[(15000, 512)]['flux_halfset']
            <= HALFSET_FLUX_BAND[1]),
        halfset_flux_6000_256=(
            HALFSET_FLUX_BAND_HARD[0] <= hard['flux_halfset']
            <= HALFSET_FLUX_BAND_HARD[1]),
        default_flux_collapses=hard['flux_default'] < DEFAULT_FLUX_MAX,
        five_seed_wins=wins6 >= HALFSET_WINS_MIN,
    )
    details = dict(five_seed_6000_256=dict(wins=wins6, ratios=ratios6,
                                           mean_ratio=float(np.mean(ratios6)),
                                           prototype=dict(wins=5, mean_ratio=0.494)),
                   five_seed_15000_512=dict(wins=wins15, ratios=ratios15,
                                            mean_ratio=float(np.mean(ratios15)),
                                            prototype=dict(wins=5, mean_ratio=0.920,
                                                           note='prototype ran M=256')),
                   checks=checks,
                   band_note=('flux band at (6000,256) widened to '
                              f'{HALFSET_FLUX_BAND_HARD}: the vendored '
                              'prototype itself gives 0.7303 there on the '
                              't03 seeds (port 0.7303; seeds 0..5 span '
                              '0.61..0.94); REPORT 0.98..1.02 has no '
                              'vendored log. Near-1 asserted at the two '
                              'larger configs instead.'),
                   bands=dict(halfset_vs_oracle_max=HALFSET_VS_ORACLE_MAX,
                              halfset_flux=HALFSET_FLUX_BAND,
                              halfset_flux_6000_256=HALFSET_FLUX_BAND_HARD,
                              default_flux_max=DEFAULT_FLUX_MAX,
                              five_seed_wins_min=HALFSET_WINS_MIN))
    return _result('halfset', all(checks.values()), table, details, t0, verbose)


# ===========================================================================
# Gate 3: blocks (t02b_blocks.py)
# ===========================================================================

def _langevin_dataset(cache_dir, seed=0, verbose=True):
    """t02b Langevin data: 100 walkers started from exact Boltzmann samples
    (stationary marginal), 6000 steps, dt 5e-4, thin 10, no burn-in."""
    tag = '' if seed == 0 else f'_seed{seed}'
    path = _default_cache(cache_dir) / f'langevin_b1{tag}.npz'
    if path.exists():
        d = np.load(path)
        return d['X'], d['wid']
    cfg = sy.wolfe_quapp_config(beta=1.0, dim=2)
    NW, NSTEP, THIN = 100, 6000, 10
    t0 = time.time()
    x0 = sy.grid_boltzmann_samples(cfg, NW, n_grid=500,
                                   rng=np.random.default_rng(21 + seed))
    X, wid = sy.langevin_samples(cfg, NSTEP, dt=5e-4, D=1.0, n_walkers=NW,
                                 rng=np.random.default_rng(22 + seed),
                                 burn=0, thin=THIN, x0=x0,
                                 grad=sy.grad_rotated_wolfe_quapp)
    np.savez(path, X=X, wid=wid)
    if verbose:
        print(f"  [langevin] {X.shape[0]} frames in {time.time() - t0:.1f}s")
    return X, wid


def gate_blocks(fast=False, seed=0, cache_dir=None, verbose=True):
    """Random-frame vs contiguous-block split overfitting diagnostic."""
    t0 = time.time()
    cache_dir = _default_cache(cache_dir)
    cfg, qg, xs, ys, nu_ref = _wq_oracle(cache_dir, verbose)
    X, wid = _langevin_dataset(cache_dir, seed=seed, verbose=verbose)
    N = X.shape[0]
    in_A, in_B = cfg.in_A(X), cfg.in_B(X)
    q_ref = sy.bilinear_interp(qg, xs, ys, X)

    # integrated autocorrelation time of x[0], t02b's estimator (diagnostic)
    NW = int(wid.max()) + 1
    per = N // NW
    ac = np.zeros(200)
    for w in range(NW):
        v = X[w * per:(w + 1) * per, 0]
        v = v - v.mean()
        f = np.correlate(v, v, 'full')[len(v) - 1:len(v) - 1 + 200]
        ac += f / max(f[0], 1e-30)
    ac /= NW
    tau = 1 + 2 * np.sum(ac[1:np.argmax(ac < 0.05) if np.any(ac < 0.05) else 200])

    th = isotropic_directions(256, 2, rng=np.random.default_rng(5 + seed))
    Ms = [64, 256] if fast else [8, 16, 32, 64, 128, 256]

    splits = [
        ('random', iid_split(N, rng=np.random.default_rng(9 + seed))),
        ('block20', block_split(N, 20, group=wid)),
        ('block100', block_split(N, 100, group=wid)),
        ('walker', (np.flatnonzero(wid % 2 == 0), np.flatnonzero(wid % 2 == 1))),
    ]
    table = []
    overfit = {}
    for label, split in splits:
        rows = nested_M_curve(th, X, in_A, in_B, split, Ms, basis_kw=_BKW,
                              q_ref=q_ref, Xq=X)
        M, htr, hte, _, _, rm = rows[-1]      # the M=256 row
        # overfit percent exactly as t02b prints it: 100*(h_train-h_test)/h_train
        ov = float(100.0 * (htr - hte) / htr)
        overfit[label] = ov
        table.append(dict(split=label, M=int(M), overfit_pct=ov,
                          nu_train_ratio=float(1.0 / htr / nu_ref),
                          nu_test_ratio=float(1.0 / hte / nu_ref) if hte > 0 else np.nan,
                          rmse=float(rm)))
    proto = dict(random=-2.7, block20=31.0, block100=59.0, walker=64.0)
    for row in table:
        row['overfit_proto'] = proto[row['split']]
    _print_table(table, verbose)
    if verbose:
        print(f"  tau_int ~ {tau:.0f} thinned frames (prototype ~55)")

    checks = dict(
        random_small=overfit['random'] < BLOCKS_RANDOM_MAX,
        block100_large=overfit['block100'] > BLOCKS_B100_MIN,
        walker_large=overfit['walker'] > BLOCKS_WALKER_MIN,
        monotone=(overfit['block20'] <= overfit['block100'] + BLOCKS_MONOTONE_SLACK
                  and overfit['block100'] <= overfit['walker'] + BLOCKS_MONOTONE_SLACK),
    )
    details = dict(tau_int=float(tau), overfit=overfit,
                   overfit_prototype=proto, checks=checks,
                   bands=dict(random_max=BLOCKS_RANDOM_MAX,
                              block100_min=BLOCKS_B100_MIN,
                              walker_min=BLOCKS_WALKER_MIN,
                              monotone_slack=BLOCKS_MONOTONE_SLACK))
    return _result('blocks', all(checks.values()), table, details, t0, verbose)


# ===========================================================================
# Gate 4: bandwidth (t06b_bandwidth.py)
# ===========================================================================

def gate_bandwidth(fast=False, seed=0, cache_dir=None, verbose=True):
    """Adaptive profile/derivative-CV bandwidth vs tuned global bandwidth."""
    t0 = time.time()
    cache_dir = _default_cache(cache_dir)
    cfg, qg, xs, ys, nu_ref = _wq_oracle(cache_dir, verbose)
    M = 256
    BWS = [0.5, 1.0, 2.0, 4.0, 8.0, 16.0]

    table = []
    ratios = {}
    for N in [6000, 15000, 60000]:
        X = sy.grid_boltzmann_samples(cfg, N, n_grid=500,
                                      rng=np.random.default_rng(51 + seed))
        in_A, in_B = cfg.in_A(X), cfg.in_B(X)
        q_ref = sy.bilinear_interp(qg, xs, ys, X)
        th = isotropic_directions(M, 2, rng=np.random.default_rng(52 + seed))

        def score(bkw):
            # t06b's fit: full-data basis + halfset-eigen regularization
            # from the even/odd frame halves.
            bs = build_basis(th, X, in_A, in_B, n_bins=200, **bkw)
            G, a, b = assemble(bs, X, in_A, in_B)
            Ga, *_ = assemble(bs, X[::2], in_A[::2], in_B[::2])
            Gb, *_ = assemble(bs, X[1::2], in_A[1::2], in_B[1::2])
            Gr, _ = halfset_eigen_regularize(Ga, Gb, n_bands=12)
            sol = solve_dual(Gr, b - a, G_is_regularised=True)
            qh = clip01(evaluate(bs, sol['w'], float(-a @ sol['w']), X))
            return (transition_rmse(qh, q_ref, in_A, in_B),
                    sol['nu_hat'] / nu_ref)

        res = {}
        for bw in BWS:
            res[f'global bw={bw}'] = score(dict(bw_bins=bw))
        res['density ISE'] = score(dict(
            bw_fn=make_adaptive_bw_fn(tuple(BWS), 6, 7), bw_bins=1.5))
        res['profile CV'] = score(dict(
            bw_fn=make_halfset_bw_fn(tuple(BWS), 6, 7, 'profile'), bw_bins=1.5))
        res['deriv CV'] = score(dict(
            bw_fn=make_halfset_bw_fn(tuple(BWS), 6, 7, 'deriv'), bw_bins=1.5))

        glob = {k: v for k, v in res.items() if k.startswith('global')}
        best_key = min(glob, key=lambda k: glob[k][0])
        best = glob[best_key][0]
        pr = PROTO_BW_ROWS[N]
        ratios[N] = dict(deriv=res['deriv CV'][0] / best,
                         profile=res['profile CV'][0] / best,
                         ise=res['density ISE'][0] / best)
        table.append(dict(N=N, best_global=best, best_bw=best_key,
                          profile_cv=res['profile CV'][0],
                          deriv_cv=res['deriv CV'][0],
                          density_ise=res['density ISE'][0],
                          best_proto=pr['best'], deriv_proto=pr['deriv'],
                          ise_proto=pr['ise']))
    _print_table(table, verbose)

    checks = {}
    for N in [6000, 15000, 60000]:
        checks[f'deriv_cv_N{N}'] = ratios[N]['deriv'] <= BW_DERIV_VS_BEST_MAX
    checks['ise_worse_N6000'] = ratios[6000]['ise'] >= BW_ISE_VS_BEST_MIN
    details = dict(ratios={str(k): v for k, v in ratios.items()},
                   checks=checks,
                   bands=dict(deriv_vs_best_max=BW_DERIV_VS_BEST_MAX,
                              ise_vs_best_min=BW_ISE_VS_BEST_MIN))
    return _result('bandwidth', all(checks.values()), table, details, t0, verbose)


# ===========================================================================
# Gate 5: ids (t21_final.py, rotated block)
# ===========================================================================

def gate_ids(fast=False, seed=7, cache_dir=None, verbose=True):
    """Exact linear-in-d IDS vs single-pass isotropic at d=96/192/384."""
    t0 = time.time()
    cache_dir = _default_cache(cache_dir)
    cfg, qg, xs, ys, nu_ref = _wq_oracle(cache_dir, verbose)
    N, M = 20000, 256

    dims = [96, 192] if fast else [96, 192, 384]
    table = []
    checks = {}
    for dim in dims:
        X, in_A, in_B, q_ref = _nd_dataset(dim, seed, qg, xs, ys, n=N)
        split = iid_split(N, rng=np.random.default_rng(3))
        th = isotropic_directions(M, dim, rng=np.random.default_rng(11))
        iso_rmse, iso_nu = _score_sol(
            _fit_pass_w(th, X, in_A, in_B, _BKW, split, None),
            X, in_A, in_B, q_ref, nu_ref)
        res = ids_path_exact2(X, in_A, in_B, M=M, n_passes=3, basis_kw=_BKW,
                              rng=np.random.default_rng(14), split=split,
                              n_sub=None, n_floor_sub=None, tol=1e-6,
                              floor_tol=1e-4)
        ids_rmse, ids_nu = _score_sol(res['final'], X, in_A, in_B, q_ref, nu_ref)
        ranks = [int(h['r']) for h in res['ids_history']]
        pr = PROTO_IDS_ROWS[dim]
        table.append(dict(d=dim, iso_rmse=iso_rmse, ids_rmse=ids_rmse,
                          nu_ratio=ids_nu, ranks=str(ranks),
                          ids_proto=pr['ids'], iso_proto=pr['iso'],
                          ranks_proto=str(pr['ranks'])))
        checks[f'ids_rmse_d{dim}'] = ids_rmse <= IDS_RMSE_MAX[dim]
        checks[f'nu_d{dim}'] = IDS_NU_BAND[0] <= ids_nu <= IDS_NU_BAND[1]
        checks[f'iso_bad_d{dim}'] = iso_rmse >= IDS_ISO_RMSE_MIN
        checks[f'ranks_d{dim}'] = all(
            IDS_RANK_BAND[0] <= r <= IDS_RANK_BAND[1] for r in ranks)
        if verbose:
            _print_table(table[-1:], verbose)
    details = dict(checks=checks, fast_skipped_d384=bool(fast),
                   bands=dict(ids_rmse_max=IDS_RMSE_MAX, nu=IDS_NU_BAND,
                              iso_rmse_min=IDS_ISO_RMSE_MIN,
                              rank_band=IDS_RANK_BAND))
    return _result('ids', all(checks.values()), table, details, t0, verbose)


# ===========================================================================
# Gate 6: ids-multiseed (t19_multiseed.py)
# ===========================================================================

def gate_ids_multiseed(fast=False, seed=0, cache_dir=None, verbose=True):
    """Port exact-linear IDS vs the VENDORED dense reference (ids_path2)."""
    t0 = time.time()
    cache_dir = _default_cache(cache_dir)
    cfg, qg, xs, ys, nu_ref = _wq_oracle(cache_dir, verbose)
    ref_co, ref_rc = _import_reference()
    N, M = 20000, 256
    seeds = [7 + seed, 17 + seed, 27 + seed]

    def score_ref(sol, X, in_A, in_B, q_ref):
        qh = ref_co.clip01(ref_co.evaluate(sol['basis'], sol['w'], sol['c'], X))
        return ref_co.transition_rmse(qh, q_ref, in_A, in_B), sol['nu_hat'] / nu_ref

    table = []
    checks = {}
    for dim in [96, 192]:
        dense_r, exact_r, dense_nu, exact_nu = [], [], [], []
        for sd in seeds:
            X, in_A, in_B, q_ref = _nd_dataset(dim, sd, qg, xs, ys, n=N)
            split = iid_split(N, rng=np.random.default_rng(3))
            # dense reference arm: the vendored numpy prototype's ids_path2
            # (the dense path-scatter pipeline was not ported; t19's dense
            # arm IS this call)
            a = ref_rc.ids_path2(X, in_A, in_B, M=M, n_passes=3,
                                 basis_kw=_BKW, rng=np.random.default_rng(14),
                                 split=split, rank_rule='permutation')
            ra, na = score_ref(a['final'], X, in_A, in_B, q_ref)
            # port exact-linear arm, validated t21 settings
            c = ids_path_exact2(X, in_A, in_B, M=M, n_passes=3, basis_kw=_BKW,
                                rng=np.random.default_rng(14), split=split,
                                n_sub=None, n_floor_sub=None, tol=1e-6,
                                floor_tol=1e-4)
            rc_, nc = _score_sol(c['final'], X, in_A, in_B, q_ref, nu_ref)
            dense_r.append(ra); exact_r.append(rc_)
            dense_nu.append(na); exact_nu.append(nc)
            if verbose:
                print(f"    d={dim} seed={sd}: dense {ra:.4f} ({na:.2f}) "
                      f"exact {rc_:.4f} ({nc:.2f})")
        ratio = float(np.mean(exact_r) / np.mean(dense_r))
        pr = PROTO_IDS_MS[dim]
        table.append(dict(d=dim, dense_mean=float(np.mean(dense_r)),
                          dense_sd=float(np.std(dense_r)),
                          exact_mean=float(np.mean(exact_r)),
                          exact_sd=float(np.std(exact_r)), ratio=ratio,
                          dense_proto=pr['dense'], exact_proto=pr['exact'],
                          ratio_proto=pr['ratio']))
        checks[f'ratio_d{dim}'] = (IDS_MS_RATIO_BAND[0] <= ratio
                                   <= IDS_MS_RATIO_BAND[1])
    _print_table(table, verbose)
    details = dict(seeds=seeds, checks=checks,
                   bands=dict(ratio=IDS_MS_RATIO_BAND))
    return _result('ids-multiseed', all(checks.values()), table, details,
                   t0, verbose)


# ===========================================================================
# Gate 7: ids-consistency (fast shared-seed port vs reference)
# ===========================================================================

def gate_ids_consistency(fast=False, seed=7, cache_dir=None, verbose=True):
    """Port vs vendored-reference ids_path_exact2 on shared seeds (d=24)."""
    t0 = time.time()
    cache_dir = _default_cache(cache_dir)
    cfg, qg, xs, ys, nu_ref = _wq_oracle(cache_dir, verbose)
    ref_co, ref_rc = _import_reference()
    N, M, dim, n_passes = 6000, 64, 24, 2

    X, in_A, in_B, q_ref = _nd_dataset(dim, seed, qg, xs, ys, n=N)
    split = iid_split(N, rng=np.random.default_rng(3))

    th = isotropic_directions(M, dim, rng=np.random.default_rng(11))
    iso_rmse, _ = _score_sol(_fit_pass_w(th, X, in_A, in_B, _BKW, split, None),
                             X, in_A, in_B, q_ref, nu_ref)

    # Explicit kwargs on BOTH calls: the vendored prototype's DEFAULTS
    # (n_floor_sub=6000, floor_tol=1e-3) are not the validated settings;
    # t21/t19 ran n_floor_sub=None, floor_tol=1e-4 and so does the port.
    kw = dict(M=M, n_passes=n_passes, basis_kw=_BKW, split=split,
              n_sub=None, n_floor_sub=None, tol=1e-6, floor_tol=1e-4)
    res_port = ids_path_exact2(X, in_A, in_B,
                               rng=np.random.default_rng(14), **kw)
    port_rmse, port_nu = _score_sol(res_port['final'], X, in_A, in_B,
                                    q_ref, nu_ref)

    res_ref = ref_rc.ids_path_exact2(X, in_A, in_B,
                                     rng=np.random.default_rng(14), **kw)
    sol = res_ref['final']
    qh = ref_co.clip01(ref_co.evaluate(sol['basis'], sol['w'], sol['c'], X))
    ref_rmse = ref_co.transition_rmse(qh, q_ref, in_A, in_B)
    ref_nu = sol['nu_hat'] / nu_ref

    ratio = port_rmse / ref_rmse
    ranks_port = [int(h['r']) for h in res_port['ids_history']]
    ranks_ref = [int(h['r']) for h in res_ref['history'][:-1]]
    table = [dict(arm='isotropic (port)', rmse=iso_rmse),
             dict(arm='ids port', rmse=port_rmse, nu_ratio=port_nu,
                  ranks=str(ranks_port)),
             dict(arm='ids reference', rmse=ref_rmse, nu_ratio=ref_nu,
                  ranks=str(ranks_ref))]
    _print_table(table, verbose)

    checks = dict(
        ratio=IDS_CONS_RATIO_BAND[0] <= ratio <= IDS_CONS_RATIO_BAND[1],
        port_beats_iso=port_rmse <= IDS_CONS_VS_ISO_MAX * iso_rmse,
        ref_beats_iso=ref_rmse <= IDS_CONS_VS_ISO_MAX * iso_rmse,
    )
    details = dict(ratio=float(ratio), iso_rmse=float(iso_rmse),
                   port_rmse=float(port_rmse), ref_rmse=float(ref_rmse),
                   ranks_port=ranks_port, ranks_ref=ranks_ref, checks=checks,
                   bands=dict(ratio=IDS_CONS_RATIO_BAND,
                              vs_iso_max=IDS_CONS_VS_ISO_MAX))
    return _result('ids-consistency', all(checks.values()), table, details,
                   t0, verbose)
