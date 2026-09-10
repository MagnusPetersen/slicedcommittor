"""Freeze the 1.0 golden references FROM THE v0.6.0 CODE.  Ran once, at tag v0.6.0.

This script cannot run against the 1.0 library: it imports the pre-1.0 API
(``compute_enriched_basin_moment_weights``, ``evaluate_committor``) and the
``recovar`` research package plus its vendored numpy prototype (``slicedcv``),
all of which the 1.0 cut removed.  It is kept for provenance only: it documents
exactly how ``ebmc_golden.npz`` and ``halfset_reference.npz`` were produced.

    git -C lib worktree add ../lib-0.6 v0.6.0
    cd ../lib-0.6 && JAX_PLATFORMS=cpu python sliced_committor/tests/golden/freeze_1_0_references.py
"""

# ruff: noqa
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
LIB = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, LIB)
sys.path.insert(0, os.path.join(LIB, "recovar", "_reference"))

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

import recovar
import sliced_committor as sc
from recovar import systems
from slicedcv import recovar as r_rec  # (numpy prototype)
from sliced_committor.core.gram import _compute_derivative_matrix
from sliced_committor.core.solver import evaluate_committor, make_weighting_context

OUT_EBMC = os.path.join(HERE, "ebmc_golden.npz")
OUT_HALF = os.path.join(HERE, "halfset_reference.npz")


def _f(x):
    return np.asarray(x, dtype=np.float64)


# --------------------------------------------------------------------------- #
# 1. EBMC on the beta-invariance golden fixture (N=?, M=32, seed=0)
# --------------------------------------------------------------------------- #
g = np.load(os.path.join(HERE, "beta_invariance_golden.npz"))
X = jnp.asarray(g["samples"])
in_A, in_B = jnp.asarray(g["in_A"]), jnp.asarray(g["in_B"])
print("golden fixture N, dim =", X.shape, "nA, nB =", int(in_A.sum()), int(in_B.sum()))
res = sc.compute_sliced_committor(
    X, in_A=in_A, in_B=in_B, n_directions=32, seed=0, store_projected_samples=True
)
out = {"N_DIRECTIONS": 32, "SEED": 0}
pts = np.asarray(X)[:: max(1, X.shape[0] // 200)][:200]  # a fixed evaluation batch
out["eval_points"] = pts
for tik in ("auto", "halfset_eigen"):
    try:
        w = sc.compute_enriched_basin_moment_weights(
            res, X, tikhonov=tik, gram_dtype="float64", raise_on_degenerate=False
        )
    except Exception as e:
        print(f"  ebmc[{tik}] FAILED on the golden fixture: {e!r}")
        continue
    k = f"ebmc_{tik}"
    wv = _f(w["w"])
    out[f"{k}::w"] = wv
    out[f"{k}::c"] = float(w["c"])
    out[f"{k}::M_gap"] = float(w["M_gap"])
    out[f"{k}::eta_used"] = float(w["eta_used"])
    out[f"{k}::energy"] = float(wv @ _f(w["G"]) @ wv)
    out[f"{k}::a"] = _f(w["a"])
    out[f"{k}::b"] = _f(w["b"])
    q = sc.build_committor(res, w)
    out[f"{k}::q_at_samples_snapped"] = _f(q(X, in_A=in_A, in_B=in_B))
    out[f"{k}::q_at_points_free"] = _f(q(jnp.asarray(pts)))
    out[f"{k}::q_ref_eval_points_free"] = _f(
        evaluate_committor(res, jnp.asarray(pts), w, enforce_boundary_conditions=False)
    )
    if tik == "halfset_eigen":
        out[f"{k}::G_reg"] = _f(w["G_reg"])
        out[f"{k}::band_ssnr"] = _f(w["halfset_eigen"]["band_ssnr"])
    print(f"  ebmc[{tik}]: M_gap={float(w['M_gap']):.6g} energy={out[f'{k}::energy']:.6g}")
# the gate that build_committor == the (deleted) reference evaluator today
for tik in ("auto", "halfset_eigen"):
    k = f"ebmc_{tik}"
    if f"{k}::q_at_points_free" in out:
        d = np.max(np.abs(out[f"{k}::q_at_points_free"] - out[f"{k}::q_ref_eval_points_free"]))
        print(f"  build vs evaluate ({tik}): max|dq| = {d:.3e}")
try:
    hc = sc.compute_enriched_basin_moment_weights(
        res,
        X,
        None,
        "halfset_eigen",
        gram_dtype="float64",
        raise_on_degenerate=False,
        heldout_cap=True,
    )["heldout_cap"]
    out["ebmc_halfset_eigen::heldout_cap"] = float(hc["cap"])
    out["ebmc_halfset_eigen::heldout_per_fold"] = _f(hc["per_fold"])
    out["ebmc_halfset_eigen::heldout_se"] = float(hc["se"])
    print(f"  heldout cap on golden fixture: {hc['cap']:.6g} +- {hc['se']:.2g} (n_ok={hc['n_ok']})")
except Exception as e:
    print("  heldout_cap on golden fixture FAILED:", repr(e))
np.savez(OUT_EBMC, **out)
print("wrote", OUT_EBMC, "keys:", len(out))


# --------------------------------------------------------------------------- #
# 2. halfset_eigen_regularize reference (numpy prototype) on random SPD pairs
#    -- the five cases of recovar/tests/test_port_regression.py
# --------------------------------------------------------------------------- #
def _spd_pair(M, seed):  # == tests/_helpers.spd_pair, kept verbatim: it made the reference
    rng = np.random.default_rng(seed)
    B = rng.standard_normal((M, M))
    base = B @ B.T / M
    E1 = rng.standard_normal((M, M))
    E2 = rng.standard_normal((M, M))
    return base + 0.05 * (E1 @ E1.T) / M, base + 0.05 * (E2 @ E2.T) / M


out2 = {}
cases = [(8, 2, 10), (12, 3, 11), (16, 4, 12), (24, 8, 13), (33, 12, 14)]
out2["spd_cases"] = np.asarray(cases)
for M, nb, seed in cases:
    G1, G2 = _spd_pair(M, seed)
    R, info = r_rec.halfset_eigen_regularize(G1, G2, n_bands=nb)
    k = f"spd::{M}_{nb}_{seed}"
    out2[f"{k}::G1"], out2[f"{k}::G2"], out2[f"{k}::G_reg"] = G1, G2, _f(R)
    out2[f"{k}::band_ssnr"], out2[f"{k}::lam"], out2[f"{k}::lam_reg"] = (
        _f(info["band_ssnr"]),
        _f(info["lam"]),
        _f(info["lam_reg"]),
    )

# --------------------------------------------------------------------------- #
# 3. Wolfe-Quapp: the samples (to gate the ported generator) and the EXP-A
#    reference path (recovar.halfset_grams_from_arrays -> regularize -> solve)
#    on the lib's extracted (F, cos, a, b); plus the lib's own current output.
# --------------------------------------------------------------------------- #
cfg = systems.wolfe_quapp_config(beta=1.0, dim=2)
for n, seed in ((4000, 0), (3000, 2), (6000, 0), (20000, 0)):
    Xw = systems.grid_boltzmann_samples(cfg, n, rng=seed)
    out2[f"wq::X_{n}_{seed}"] = _f(Xw)
    out2[f"wq::in_A_{n}_{seed}"] = np.asarray(cfg.in_A(Xw), bool)
    out2[f"wq::in_B_{n}_{seed}"] = np.asarray(cfg.in_B(Xw), bool)
out2["wq::center_A"], out2["wq::center_B"] = _f(cfg.center_A), _f(cfg.center_B)
out2["wq::radius"], out2["wq::beta"] = float(cfg.radius), float(cfg.beta)
out2["wq::domain"] = _f(cfg.domain)

Xw = jnp.asarray(out2["wq::X_4000_0"])
iA, iB = out2["wq::in_A_4000_0"], out2["wq::in_B_4000_0"]
q, fit = sc.fit_committor(
    Xw,
    in_A=jnp.asarray(iA),
    in_B=jnp.asarray(iB),
    n_directions=64,
    seed=0,
    return_details=True,
    n_bins=50,
    n_min=10,
    binning_method="quantile",
    weights="ebmc",
    weight_kwargs={"tikhonov": "halfset_eigen"},
)
r = fit.result
ctx = make_weighting_context(r)
F = _compute_derivative_matrix(ctx, r.projected_samples)
G1, G2 = recovar.halfset_grams_from_arrays(F, None, _f(ctx.cos_matrix), iA, iB, n_folds=10)
valid = np.asarray(r.valid_mask, bool)
iv = np.flatnonzero(valid)
G_reg_ref, info = recovar.halfset_eigen_regularize(
    _f(G1)[np.ix_(iv, iv)], _f(G2)[np.ix_(iv, iv)], n_bands=12
)
a, b = _f(fit.weights["a"]), _f(fit.weights["b"])
sol = recovar.solve_with_G(G_reg_ref, a[iv], b[iv])
w_ref = np.zeros(len(valid))
w_ref[iv] = sol["w"]
out2["wq_fit::directions"] = _f(r.directions)
out2["wq_fit::committors_1d"] = _f(r.committors_1d)
out2["wq_fit::slice_coords"] = _f(r.slice_coords)
out2["wq_fit::valid_mask"] = valid
out2["wq_fit::G1"], out2["wq_fit::G2"] = _f(G1), _f(G2)
out2["wq_fit::G_reg_ref"], out2["wq_fit::band_ssnr_ref"] = _f(G_reg_ref), _f(info["band_ssnr"])
out2["wq_fit::a"], out2["wq_fit::b"] = a, b
out2["wq_fit::w_ref"], out2["wq_fit::M_gap_ref"], out2["wq_fit::c_ref"] = (
    w_ref,
    float(sol["M_gap"]),
    float(sol["c"]),
)
out2["wq_fit::w_lib"], out2["wq_fit::M_gap_lib"], out2["wq_fit::c_lib"] = (
    _f(fit.weights["w"]),
    float(fit.weights["M_gap"]),
    float(fit.weights["c"]),
)
out2["wq_fit::G_reg_lib"] = _f(fit.weights["G_reg"])
out2["wq_fit::energy_lib"] = float(fit.dirichlet_energy)
grid = np.stack(
    np.meshgrid(np.linspace(-2.5, 2.5, 41), np.linspace(-2.5, 2.5, 41), indexing="ij"), -1
).reshape(-1, 2)
out2["wq_fit::grid"] = grid
out2["wq_fit::q_grid_lib"] = _f(q(jnp.asarray(grid)))
print(
    f"  WQ parity today: max|w_lib-w_ref| = {np.max(np.abs(out2['wq_fit::w_lib'] - w_ref)):.3e}, "
    f"M_gap lib/ref = {out2['wq_fit::M_gap_lib']:.10g}/{out2['wq_fit::M_gap_ref']:.10g}"
)

# the held-out cap on the heldout-test fixture (n=6000, equal_width, n_bins=50, m=64)
X6 = jnp.asarray(out2["wq::X_6000_0"])
res6 = sc.compute_sliced_committor(
    X6,
    in_A=jnp.asarray(out2["wq::in_A_6000_0"]),
    in_B=jnp.asarray(out2["wq::in_B_6000_0"]),
    n_directions=64,
    seed=0,
    n_bins=50,
    n_min=10,
    binning_method="equal_width",
    store_projected_samples=True,
)
w6 = sc.compute_enriched_basin_moment_weights(
    res6,
    X6,
    None,
    "halfset_eigen",
    gram_dtype="float64",
    raise_on_degenerate=False,
    heldout_cap=True,
)
out2["wq_cap::cap"] = float(w6["heldout_cap"]["cap"])
out2["wq_cap::per_fold"] = _f(w6["heldout_cap"]["per_fold"])
out2["wq_cap::se"] = float(w6["heldout_cap"]["se"])
out2["wq_cap::w"], out2["wq_cap::M_gap"] = _f(w6["w"]), float(w6["M_gap"])
print(f"  heldout cap (n=6000 fixture): {out2['wq_cap::cap']:.8g} +- {out2['wq_cap::se']:.2g}")
np.savez(OUT_HALF, **out2)
print("wrote", OUT_HALF, "keys:", len(out2))
