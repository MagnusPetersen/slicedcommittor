"""Freeze the golden references of ``test_golden.py`` from the current library.

Rerun after a deliberate change of the numerics. It rewrites every entry that
depends on the slice basis and keeps the rest:

* ``beta_invariance_golden.npz``: the 500-sample fixture (samples, labels and
  its metadata are kept) and its slice basis (``n_directions=32, seed=0``).
* ``ebmc_golden.npz``: the weight solve on that basis (``auto`` and
  ``halfset_eigen``), the held-out cap, and the committor at the stored
  ``eval_points`` and at the samples. The script checks the library's
  committor against a separate numpy evaluation (interpolate, clip, combine)
  of the same basis and weights before it writes.
* ``halfset_reference.npz``: the Wolfe-Quapp samples and the inputs of the
  regulariser cases are kept (``freeze_1_0_references.py`` made them); the
  regulariser outputs are recomputed by ``FILTER_SPEC``, the band filter
  written from its definition (numpy's eigensolver, the uncentred FSC
  correlation), which is independent of the library's implementation. The
  Wolfe-Quapp fit and its held-out cap are refit, and the fit's independent
  reference path runs on the library's own slopes and basin moments in a
  subprocess on the ``lib-0.6`` worktree: the half-Grams of the numpy
  prototype the library was ported from (``halfset_grams_from_arrays`` of the
  v0.6.0 ``recovar`` package), then ``FILTER_SPEC``'s filter and the
  closed-form solve. The prototype's own filter
  and solve are not used: its band correlation was centred, so it depended
  on the signs of the eigenvectors, and its solve factorised the rebuilt
  filtered Gram, which turns rounding into ``1e-5`` of the weights.

    git -C lib worktree add ../lib-0.6 v0.6.0   # once
    cd lib && JAX_PLATFORMS=cpu python sliced_committor/tests/golden/freeze_references.py

Run it where the strict tier runs (jax ``test_golden.FROZEN_JAX``, the
freezing machine), so that tier stays bit-exact there.
"""

import os
import subprocess
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
LIB = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
LIB06 = os.environ.get("SLICED_COMMITTOR_LIB06", os.path.join(os.path.dirname(LIB), "lib-0.6"))
sys.path.insert(0, LIB)

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

import sliced_committor as sc
from sliced_committor.core import _halfset as hs
from sliced_committor.core._ebmc import _basis

BASIS = os.path.join(HERE, "beta_invariance_golden.npz")
EBMC = os.path.join(HERE, "ebmc_golden.npz")
HALF = os.path.join(HERE, "halfset_reference.npz")

FILTER_SPEC = r"""
import numpy as np


def spec_band_filter(G1, G2, n_bands):
    # eigenbands of the half-set mean, uncentred FSC correlation per band,
    # SSNR = 2c / (1 - c), lam_reg = lam (1 + 1/SSNR) floored at 1e-12 max
    G1, G2 = np.asarray(G1, np.float64), np.asarray(G2, np.float64)
    lam, V = np.linalg.eigh(0.5 * (G1 + G2))
    order = np.argsort(-lam)
    lam, V = lam[order], V[:, order]
    A1, A2 = V.T @ G1 @ V, V.T @ G2 @ V
    edges = np.linspace(0, lam.size, n_bands + 1).astype(int)
    band, ssnr = np.zeros(n_bands), np.zeros(lam.size)
    for k in range(n_bands):
        v1, v2 = A1[edges[k]:edges[k + 1]].ravel(), A2[edges[k]:edges[k + 1]].ravel()
        if v1.size < 3 or v1.std() == 0 or v2.std() == 0:
            sk = 1e6
        else:
            c = np.clip(v1 @ v2 / np.sqrt((v1 @ v1) * (v2 @ v2)), 0.0, 0.999999)
            sk = 2.0 * c / (1.0 - c)
        band[k] = sk
        ssnr[edges[k]:edges[k + 1]] = sk
    lam_reg = np.maximum(lam, 0.0) * (1.0 + 1.0 / np.maximum(ssnr, 1e-12))
    lam_reg = np.maximum(lam_reg, 1e-12 * max(lam_reg.max(), 1e-300))
    return V, lam, lam_reg, band


def spec_solve(G1, G2, a, b, n_bands):
    # the band filter, then the closed form w = G_reg^-1 (b - a) / R,
    # R = (b - a)^T G_reg^-1 (b - a), c = -a^T w, with
    # G_reg^-1 = V diag(1/lam_reg) V^T applied factor by factor
    V, _, lam_reg, band = spec_band_filter(G1, G2, n_bands)
    u = V @ ((V.T @ (b - a)) / lam_reg)
    gap = float((b - a) @ u)
    w = u / gap
    return (V * lam_reg) @ V.T, band, w, gap, float(-(a @ w))
"""

REFERENCE_BODY = r"""
import sys
import recovar

d = np.load(sys.argv[1])
G1, G2 = recovar.halfset_grams_from_arrays(d["F"], None, d["cos"], d["in_A"], d["in_B"], n_folds=10)
G1, G2 = np.asarray(G1, np.float64), np.asarray(G2, np.float64)
iv = np.flatnonzero(d["valid"])
G_reg, band_ssnr, w_v, gap, c = spec_solve(G1[np.ix_(iv, iv)], G2[np.ix_(iv, iv)], d["a"][iv], d["b"][iv], 12)
w = np.zeros(d["valid"].size)
w[iv] = w_v
np.savez(sys.argv[2], G1=G1, G2=G2, G_reg=G_reg, band_ssnr=band_ssnr, w=w, M_gap=gap, c=c)
"""
REFERENCE_PATH = FILTER_SPEC + REFERENCE_BODY
_spec = {}
exec(FILTER_SPEC, _spec)


def _f(x):
    return np.asarray(x, dtype=np.float64)


def _load(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def _numpy_committor(result, weights, points):
    """``clip(c + sum_j w_j clip(q_j(theta_j . x), 0, 1), 0, 1)`` with numpy interpolation."""
    proj = np.asarray(points, np.float64) @ _f(result.directions).T
    s, q1d = _f(result.slice_coords), np.nan_to_num(_f(result.committors_1d))
    valid, w = np.asarray(result.valid_mask, bool), _f(weights.w)
    q = np.full(proj.shape[0], float(weights.c))
    for j in np.flatnonzero(valid):
        q += w[j] * np.clip(np.interp(proj[:, j], s[j], q1d[j]), 0.0, 1.0)
    return np.clip(q, 0.0, 1.0)


def _change(name, old, new):
    if old is None:
        return
    old, new = _f(old), _f(new)
    if old.shape != new.shape:
        print(f"    {name}: shape {old.shape} -> {new.shape}")
        return
    scale = max(float(np.max(np.abs(old))), 1e-300)
    print(f"    {name}: max |new - old| = {float(np.max(np.abs(new - old))):.3e} (of {scale:.3e})")


# --------------------------------------------------------------------------- #
# 1. the slice basis of the 500-sample fixture
# --------------------------------------------------------------------------- #
g = _load(BASIS)
X = jnp.asarray(g["samples"])
in_A, in_B = jnp.asarray(g["in_A"]), jnp.asarray(g["in_B"])
res = sc.compute_sliced_committor(X, in_A=in_A, in_B=in_B, n_directions=32, seed=0)
basis = {
    k: g[k]
    for k in ("n_samples", "n_directions", "sampling_beta", "beta_ref", "samples", "in_A", "in_B")
}
basis.update(
    directions=_f(res.directions),
    slice_coords=_f(res.slice_coords),
    committors_1d=_f(res.committors_1d),
    valid_mask=np.asarray(res.valid_mask, bool),
)
basis["log_density (= -β·F)"] = _f(res.free_energies)
print("basis fixture:", X.shape)
for k in ("directions", "slice_coords", "committors_1d", "log_density (= -β·F)"):
    _change(k, g.get(k), basis[k])
np.savez(BASIS, **basis)

# --------------------------------------------------------------------------- #
# 2. the weight solve on it
# --------------------------------------------------------------------------- #
e_old = _load(EBMC)
e = {"N_DIRECTIONS": 32, "SEED": 0, "eval_points": e_old["eval_points"]}
pts = jnp.asarray(e["eval_points"])
for tik in ("auto", "halfset_eigen"):
    w = sc.solve_weights(res, tikhonov=tik, raise_on_degenerate=False)
    q = sc.build_committor(res, w)
    k = f"ebmc_{tik}"
    e[f"{k}::w"] = _f(w.w)
    e[f"{k}::c"] = float(w.c)
    e[f"{k}::M_gap"] = float(w.moment_gap)
    e[f"{k}::eta_used"] = float(w.ridge)
    e[f"{k}::energy"] = float(w.dirichlet_energy)
    e[f"{k}::a"] = _f(w.diagnostics["a"])
    e[f"{k}::b"] = _f(w.diagnostics["b"])
    e[f"{k}::q_at_samples_snapped"] = _f(q(X, in_A=in_A, in_B=in_B))
    e[f"{k}::q_at_points_free"] = _f(q(pts))
    e[f"{k}::q_ref_eval_points_free"] = e[f"{k}::q_at_points_free"]
    if tik == "halfset_eigen":
        e[f"{k}::G_reg"] = _f(w.diagnostics["G_reg"])
        e[f"{k}::band_ssnr"] = _f(w.diagnostics["band_ssnr"])
    d = np.max(np.abs(e[f"{k}::q_at_points_free"] - _numpy_committor(res, w, e["eval_points"])))
    assert d < 1e-12, d
    print(
        f"  {tik}: energy {e[f'{k}::energy']:.6g}, M_gap {e[f'{k}::M_gap']:.6g}, library vs numpy committor {d:.2e}"
    )
    for name in ("energy", "M_gap", "q_at_samples_snapped"):
        _change(f"{k}::{name}", e_old.get(f"{k}::{name}"), e[f"{k}::{name}"])
hc = sc.solve_weights(res, tikhonov="halfset_eigen", heldout_cap=True).heldout_cap
e["ebmc_halfset_eigen::heldout_cap"] = float(hc["cap"])
e["ebmc_halfset_eigen::heldout_per_fold"] = _f(hc["per_fold"])
e["ebmc_halfset_eigen::heldout_se"] = float(hc["se"])
print(f"  held-out cap {hc['cap']:.6g} +- {hc['se']:.2g}")
np.savez(EBMC, **e)

# --------------------------------------------------------------------------- #
# 3. Wolfe-Quapp: the fit, its independent reference path, the held-out cap
# --------------------------------------------------------------------------- #
h_old = _load(HALF)
h = {k: v for k, v in h_old.items() if k == "spd_cases" or k.startswith(("spd::", "wq::"))}
for M, nb, seed in h["spd_cases"]:
    k = f"spd::{M}_{nb}_{seed}"
    V, lam, lam_reg, band = _spec["spec_band_filter"](h[k + "::G1"], h[k + "::G2"], int(nb))
    R = (V * lam_reg) @ V.T
    lib_R, _ = hs.halfset_eigen_regularize(h[k + "::G1"], h[k + "::G2"], n_bands=int(nb))
    dev = np.max(np.abs(lib_R - R)) / np.max(np.abs(R))
    print(f"  regulariser case {k}: library vs spec {dev:.2e}")
    _change(f"{k}::G_reg", h_old.get(k + "::G_reg"), R)
    h.update(
        {k + "::G_reg": R, k + "::band_ssnr": band, k + "::lam": lam, k + "::lam_reg": lam_reg}
    )
Xw, iA, iB = h["wq::X_4000_0"], h["wq::in_A_4000_0"], h["wq::in_B_4000_0"]
q, fit = sc.fit_committor(
    jnp.asarray(Xw),
    in_A=jnp.asarray(iA),
    in_B=jnp.asarray(iB),
    n_directions=64,
    seed=0,
    n_bins=50,
    n_min=10,
    binning_method="quantile",
    tikhonov="halfset_eigen",
    return_details=True,
)
r, wts = fit.result, fit.weights
valid = np.asarray(r.valid_mask, bool)
b4 = _basis(r)
a, b = _f(wts.diagnostics["a"]), _f(wts.diagnostics["b"])
with tempfile.TemporaryDirectory() as td:
    src, dst = os.path.join(td, "in.npz"), os.path.join(td, "out.npz")
    np.savez(src, F=_f(b4.F), cos=_f(b4.cos_matrix), a=a, b=b, in_A=iA, in_B=iB, valid=valid)
    env = dict(os.environ, PYTHONPATH=LIB06, JAX_PLATFORMS="cpu")
    subprocess.run([sys.executable, "-c", REFERENCE_PATH, src, dst], env=env, check=True, cwd=td)
    ref = _load(dst)
h.update(
    {
        "wq_fit::directions": _f(r.directions),
        "wq_fit::committors_1d": _f(r.committors_1d),
        "wq_fit::slice_coords": _f(r.slice_coords),
        "wq_fit::valid_mask": valid,
        "wq_fit::G1": ref["G1"],
        "wq_fit::G2": ref["G2"],
        "wq_fit::G_reg_ref": ref["G_reg"],
        "wq_fit::band_ssnr_ref": ref["band_ssnr"],
        "wq_fit::a": a,
        "wq_fit::b": b,
        "wq_fit::w_ref": ref["w"],
        "wq_fit::M_gap_ref": float(ref["M_gap"]),
        "wq_fit::c_ref": float(ref["c"]),
        "wq_fit::w_lib": _f(wts.w),
        "wq_fit::M_gap_lib": float(wts.moment_gap),
        "wq_fit::c_lib": float(wts.c),
        "wq_fit::G_reg_lib": _f(wts.diagnostics["G_reg"]),
        "wq_fit::energy_lib": float(wts.dirichlet_energy),
        "wq_fit::grid": h_old["wq_fit::grid"],
        "wq_fit::q_grid_lib": _f(q(jnp.asarray(h_old["wq_fit::grid"]))),
    }
)
print(
    f"Wolfe-Quapp fit: max|w_lib - w_ref| = {np.max(np.abs(h['wq_fit::w_lib'] - h['wq_fit::w_ref'])):.3e}, "
    f"M_gap lib/ref = {h['wq_fit::M_gap_lib']:.10g}/{h['wq_fit::M_gap_ref']:.10g}"
)
for k in ("wq_fit::committors_1d", "wq_fit::energy_lib", "wq_fit::q_grid_lib"):
    _change(k, h_old.get(k), h[k])

X6 = jnp.asarray(h["wq::X_6000_0"])
res6 = sc.compute_sliced_committor(
    X6,
    in_A=jnp.asarray(h["wq::in_A_6000_0"]),
    in_B=jnp.asarray(h["wq::in_B_6000_0"]),
    n_directions=64,
    seed=0,
    n_bins=50,
    n_min=10,
    binning_method="equal_width",
)
w6 = sc.solve_weights(res6, tikhonov="halfset_eigen", heldout_cap=True)
h["wq_cap::cap"] = float(w6.heldout_cap["cap"])
h["wq_cap::per_fold"] = _f(w6.heldout_cap["per_fold"])
h["wq_cap::se"] = float(w6.heldout_cap["se"])
h["wq_cap::w"] = _f(w6.w)
h["wq_cap::M_gap"] = float(w6.moment_gap)
print(
    f"Wolfe-Quapp held-out cap (n=6000, equal_width): {h['wq_cap::cap']:.8g} +- {h['wq_cap::se']:.2g}"
)
_change("wq_cap::cap", h_old.get("wq_cap::cap"), h["wq_cap::cap"])
np.savez(HALF, **h)
print("wrote", BASIS, EBMC, HALF, sep="\n  ")
