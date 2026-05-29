"""Sliced committor walkthrough on the rotated Wolfe-Quapp 2D potential.

Single end-to-end tour of the public API:

    1. Input contract: an (N, dim) sample array + two boolean basin masks.
    2. Solve: ``compute_sliced_committor`` returns a SlicedCommittorResult.
    3. Inspect: ``result.summary()`` / ``why_masked`` for diagnostics.
    4. Weight: four solvers (diagonal RD, full Gram, BMC, EBMC) with
       ``summarize_gram_diagnostics`` for the constrained ones.
    5. Evaluate: ``evaluate_committor`` on a 2D contour grid and at a few
       individual query points.
    6. Cross-check: against a converged 300x300 Jacobi PDE baseline.

Headline settings match paper figure 1 (Petersen et al., 2026) on the
rotated Wolfe-Quapp benchmark of Bonati, Zhang & Parrinello (PNAS 2019):
    U(x', y') = x'^4 + y'^4 - 2 x'^2 - 4 y'^2 + x' y' + 0.3 x' + 0.1 y'
at coordinates rotated by theta = -0.15 pi. Two deepest minima
A = (-1.717, 0.783) and B = (1.676, -0.813) serve as basins (radius 0.3,
beta = 1, N = 100 000 samples, M = 256 directions).

Runtime on CPU: ~2-3 minutes (sliced solves ~60 s, PDE baseline ~90 s).

Run:
    python 01_wolfe_quapp_2d.py
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Circle
from scipy.interpolate import RegularGridInterpolator
from scipy.special import expit

# Float64 is required for the EBMC / BMC / full-Gram constrained solves.
jax.config.update("jax_enable_x64", True)

import sliced_committor as sc


# ============================================================================
# Paper PRL plotting style (matches paper figure 1)
# ============================================================================
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 8,
    "axes.titlesize": 9,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.04,
    "text.usetex": False,
    "mathtext.fontset": "cm",
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 3,
    "ytick.major.size": 3,
    "lines.linewidth": 1.0,
})

DOUBLE_COL = 7.0
HIGHLIGHT = "#d62728"


# ============================================================================
# 0. Settings (paper figure 1)
# ============================================================================
BETA = 1.0
N_SAMPLES = 100_000
N_SAMPLE_GRID = 250
N_DIRECTIONS = 256
N_BINS = 200
SEED = 42
STATE_RADIUS = 0.3
CENTRE_A = np.array([-1.717,  0.783])
CENTRE_B = np.array([ 1.676, -0.813])
DOMAIN = (-2.5, 2.5)

SOLVER_KWARGS = dict(
    n_min=1,
    binning_method="equal_width",
    density_floor=1e-3,
    rd_kappa=1e24,
)

PDE_N_GRID = 300
PDE_TOL = 1e-9
PDE_MAX_ITERS = 10_000_000

CONTOUR_LEVELS = list(expit(np.linspace(
    np.log(0.01 / 0.99), np.log(0.99 / 0.01), 13)))


# ============================================================================
# 1. The potential and a Boltzmann sampler (stand-in for your data)
# ============================================================================
# In real use, ``samples`` comes from MD, MCMC, or replica exchange. Here we
# importance-sample it from the analytic Boltzmann density on a grid.
def potential(x, y):
    theta = -0.15 * np.pi
    c, s = np.cos(theta), np.sin(theta)
    xr = x * c - y * s
    yr = x * s + y * c
    return (xr ** 4 + yr ** 4
            - 2.0 * xr ** 2 - 4.0 * yr ** 2
            + xr * yr
            + 0.3 * xr + 0.1 * yr)


def boltzmann_samples(n_samples=N_SAMPLES, n_grid=N_SAMPLE_GRID,
                      beta=BETA, seed=SEED):
    rng = np.random.default_rng(seed)
    xs = np.linspace(*DOMAIN, n_grid)
    ys = np.linspace(*DOMAIN, n_grid)
    X, Y = np.meshgrid(xs, ys, indexing="xy")
    log_rho = -beta * potential(X, Y)
    log_rho -= log_rho.max()
    p = np.exp(log_rho)
    p /= p.sum()
    idx = rng.choice(p.size, size=n_samples, replace=True, p=p.ravel())
    rows, cols = np.divmod(idx, n_grid)
    dx = xs[1] - xs[0]; dy = ys[1] - ys[0]
    sx = xs[cols] + (rng.random(n_samples) - 0.5) * dx
    sy = ys[rows] + (rng.random(n_samples) - 0.5) * dy
    return np.stack([sx, sy], axis=1)


# ============================================================================
# 2. Converged 2D PDE baseline (weighted Jacobi)
# ============================================================================
def jacobi_committor_2d(n_grid=PDE_N_GRID, beta=BETA, radius=STATE_RADIUS,
                        max_iters=PDE_MAX_ITERS, tol=PDE_TOL,
                        report_every=20_000):
    """Weighted Jacobi solve of the committor PDE on a regular grid.

    Returns (q, X, Y, iters, final_delta). Raises if the iteration budget
    is exhausted before reaching ``tol`` (the baseline must be converged
    for the cross-check below to be meaningful).
    """
    xs = np.linspace(*DOMAIN, n_grid)
    ys = np.linspace(*DOMAIN, n_grid)
    X, Y = np.meshgrid(xs, ys, indexing="xy")
    rho = np.exp(-beta * potential(X, Y))
    Wx = 0.5 * (rho[:, :-1] + rho[:, 1:])
    Wy = 0.5 * (rho[:-1, :] + rho[1:, :])
    coef = np.zeros_like(rho)
    coef[:, :-1] += Wx; coef[:, 1:] += Wx
    coef[:-1, :] += Wy; coef[1:, :] += Wy
    in_A = ((X - CENTRE_A[0]) ** 2 + (Y - CENTRE_A[1]) ** 2) < radius ** 2
    in_B = ((X - CENTRE_B[0]) ** 2 + (Y - CENTRE_B[1]) ** 2) < radius ** 2
    q = np.where(in_B, 1.0, 0.0)
    inv_coef = np.where(coef > 0, 1.0 / np.maximum(coef, 1e-30), 0.0)
    delta = np.inf
    for it in range(max_iters):
        flux = np.zeros_like(q)
        flux[:, 1:] += Wx * q[:, :-1]
        flux[:, :-1] += Wx * q[:, 1:]
        flux[1:, :] += Wy * q[:-1, :]
        flux[:-1, :] += Wy * q[1:, :]
        q_new = flux * inv_coef
        q_new[in_A] = 0.0; q_new[in_B] = 1.0
        delta = float(np.max(np.abs(q_new - q)))
        q = q_new
        if delta < tol:
            return q, X, Y, it + 1, delta
        if (it + 1) % report_every == 0:
            print(f"    PDE iter {it + 1:>7d}: max|dq| = {delta:.2e}")
    raise RuntimeError(
        f"Jacobi PDE did not converge in {max_iters} iters; "
        f"final max|dq|={delta:.2e}, tol={tol:.2e}. Increase max_iters."
    )


# ============================================================================
# Main walkthrough
# ============================================================================
def main():
    # ------------------------------------------------------------------------
    # Step 1: get your data into (N, dim) shape + define basins as bool masks
    # ------------------------------------------------------------------------
    print("[1] Generating samples...")
    samples = boltzmann_samples()
    in_A = np.linalg.norm(samples - CENTRE_A, axis=1) < STATE_RADIUS
    in_B = np.linalg.norm(samples - CENTRE_B, axis=1) < STATE_RADIUS
    print(f"    samples.shape = {samples.shape}, dtype = {samples.dtype}")
    print(f"    |A| = {in_A.sum():,}, |B| = {in_B.sum():,}, "
          f"|A & B| = {(in_A & in_B).sum()}")

    samples_j = jnp.asarray(samples)
    in_A_j = jnp.asarray(in_A)
    in_B_j = jnp.asarray(in_B)

    # ------------------------------------------------------------------------
    # Step 2: compute the sliced committor
    # ------------------------------------------------------------------------
    print(f"\n[2] compute_sliced_committor(M={N_DIRECTIONS}, n_bins={N_BINS})...")
    result = sc.compute_sliced_committor(
        samples_j, in_A=in_A_j, in_B=in_B_j,
        n_directions=N_DIRECTIONS, n_bins=N_BINS, seed=SEED,
        **SOLVER_KWARGS,
    )

    # ------------------------------------------------------------------------
    # Step 3: inspect the result. SlicedCommittorResult is a NamedTuple.
    # ------------------------------------------------------------------------
    print("\n[3] result.summary():")
    print(result.summary())
    print("\n    SlicedCommittorResult fields:")
    print(f"      directions      : {tuple(result.directions.shape)}")
    print(f"      slice_coords    : {tuple(result.slice_coords.shape)}  "
          "(1D grid per slice)")
    print(f"      free_energies   : {tuple(result.free_energies.shape)}  "
          "(-log rho per slice)")
    print(f"      committors_1d   : {tuple(result.committors_1d.shape)}  "
          "(1D RD committor per slice)")
    print(f"      valid_mask      : "
          f"{int(result.valid_mask.sum())}/{result.valid_mask.shape[0]} valid")
    if int((~result.valid_mask).sum()):
        bad = np.flatnonzero(np.asarray(~result.valid_mask))[:3]
        for k in bad:
            print(f"        why_masked(result, {k:d}): "
                  f"{sc.why_masked(result, int(k))}")

    # ------------------------------------------------------------------------
    # Step 4: pick weights. Four solvers, ordered cheap-diagonal -> constraint-rich.
    #   diagonal RD : cheapest, no constraints
    #   full Gram   : lifts the four diagonal approximations
    #   BMC         : pins basin moments
    #   EBMC        : BMC + slice-basis enrichment (production default)
    # ------------------------------------------------------------------------
    print("\n[4] Weight solvers:")
    w_diag = sc.compute_weights_multi(
        result, sc.get_default_weight_functions(),
    )["corrected_dirichlet_inv_rd"]
    w_full = sc.compute_full_gram_weights(result, samples_j)
    w_bmc  = sc.compute_basin_moment_weights(result, samples_j)
    w_ebmc = sc.compute_enriched_basin_moment_weights(result, samples_j)

    for name, w in [("diag", w_diag), ("full_gram", w_full),
                    ("bmc",  w_bmc), ("ebmc", w_ebmc)]:
        arr = np.asarray(w["w"] if isinstance(w, dict) else w)
        print(f"    {name:>10s}: sum={arr.sum():+.4f}, "
              f"#neg={int((arr < 0).sum()):>3d}, "
              f"|w|_max={np.abs(arr).max():.4f}")

    print("\n    summarize_gram_diagnostics(full_gram):")
    for line in sc.summarize_gram_diagnostics(w_full).splitlines():
        print(f"      {line}")

    # ------------------------------------------------------------------------
    # Step 5: evaluate the committor anywhere. evaluate_committor takes the
    # result + weight (array or dict) + query points (P, dim) and returns
    # (P,) values. Boundary enforcement + transition rescaling are paper
    # defaults for clean visuals.
    # ------------------------------------------------------------------------
    print("\n[5] evaluate_committor:")
    # (a) at three named points (saddle region, near A, near B).
    probe = jnp.asarray([
        [0.0, 0.0],
        CENTRE_A + np.array([0.5, 0.0]),
        CENTRE_B + np.array([-0.5, 0.0]),
    ])
    q_probe = np.asarray(sc.evaluate_committor(
        result, probe, w_full,
        enforce_boundary_conditions=True, rescale_transition=True,
    ))
    for pt, q in zip(np.asarray(probe), q_probe):
        print(f"      q({pt[0]:+.2f}, {pt[1]:+.2f}) = {q:.4f}")

    # (b) on a 250x250 grid for plotting.
    eval_xs = np.linspace(*DOMAIN, N_SAMPLE_GRID)
    eval_ys = np.linspace(*DOMAIN, N_SAMPLE_GRID)
    EX, EY = np.meshgrid(eval_xs, eval_ys, indexing="xy")
    pts = jnp.asarray(np.stack([EX.ravel(), EY.ravel()], axis=1))
    q_full = np.asarray(sc.evaluate_committor(
        result, pts, w_full,
        enforce_boundary_conditions=True, rescale_transition=True,
    )).reshape(EX.shape)
    q_ebmc = np.asarray(sc.evaluate_committor(
        result, pts, w_ebmc,
        enforce_boundary_conditions=True, rescale_transition=True,
    )).reshape(EX.shape)

    # ------------------------------------------------------------------------
    # Step 6: cross-check against a converged 2D PDE baseline.
    # ------------------------------------------------------------------------
    print(f"\n[6] PDE baseline ({PDE_N_GRID}x{PDE_N_GRID}, tol={PDE_TOL:.0e})...")
    q_pde_raw, X_pde, Y_pde, pde_iters, pde_delta = jacobi_committor_2d()
    print(f"    converged in {pde_iters:,} iters; final max|dq| = {pde_delta:.2e}")

    interp = RegularGridInterpolator(
        (Y_pde[:, 0], X_pde[0, :]), q_pde_raw,
        method="linear", bounds_error=False, fill_value=np.nan,
    )
    q_pde = interp(np.stack([EY.ravel(), EX.ravel()], axis=1)).reshape(EX.shape)

    rho_eval = np.exp(-BETA * potential(EX, EY))
    transition_mask = (rho_eval > 1e-3 * rho_eval.max()) & np.isfinite(q_pde)
    rmse_full = float(np.sqrt(np.nanmean((q_full - q_pde)[transition_mask] ** 2)))
    rmse_ebmc = float(np.sqrt(np.nanmean((q_ebmc - q_pde)[transition_mask] ** 2)))
    print(f"    RMSE on transition region: full_gram = {rmse_full:.4f}, "
          f"ebmc = {rmse_ebmc:.4f}")

    # ------------------------------------------------------------------------
    # Step 7: plot (paper figure-1 style). Three single-row panels:
    #   (a) samples + basins on a greyscale free-energy backdrop,
    #   (b) iso-committor overlay: PDE (solid black) vs sliced full Gram (dashed red),
    #   (c) iso-committor overlay: PDE (solid black) vs sliced EBMC (dashed red).
    # ------------------------------------------------------------------------
    print("\n[7] Plotting (paper style)...")
    F_eval = BETA * potential(EX, EY)
    F_eval -= F_eval[transition_mask].min()
    F_vmax = np.percentile(F_eval[transition_mask], 92)
    F_plot = np.where(transition_mask, F_eval, np.nan)
    q_full_m = np.where(transition_mask, q_full, np.nan)
    q_ebmc_m = np.where(transition_mask, q_ebmc, np.nan)
    q_pde_m = np.where(transition_mask, q_pde, np.nan)

    fig, axes = plt.subplots(1, 3, figsize=(DOUBLE_COL, 2.40),
                              gridspec_kw=dict(wspace=0.32))
    ax_a, ax_b, ax_c = axes

    def _backdrop(ax, alpha):
        # F_plot is xy-indexed (meshgrid indexing='xy'): row -> y, col -> x.
        # imshow with origin='lower' wants exactly that orientation, no transpose.
        ax.imshow(F_plot, origin="lower",
                  extent=[*DOMAIN, *DOMAIN], cmap="Greys",
                  vmin=0, vmax=F_vmax, aspect="auto", alpha=alpha)
        ax.contour(EX, EY, transition_mask.astype(float), levels=[0.5],
                   colors="grey", linewidths=0.5, linestyles=":", zorder=2)

    def _basins(ax):
        for centre, lab in zip((CENTRE_A, CENTRE_B), ("A", "B")):
            ax.add_patch(Circle(centre, STATE_RADIUS, fill=False,
                                edgecolor="black", linewidth=1.0, zorder=5))
            ax.text(centre[0], centre[1], lab, ha="center", va="center",
                    fontsize=8, fontweight="bold", color="black", zorder=6)

    def _frame(ax):
        ax.set_xlim(DOMAIN); ax.set_ylim(DOMAIN); ax.set_aspect("equal")
        ax.set_xticks([-2, 0, 2]); ax.set_yticks([-2, 0, 2])
        ax.set_xlabel(r"$x_1$", labelpad=1)

    # Panel (a): samples on F backdrop.
    _backdrop(ax_a, alpha=0.85)
    rng = np.random.default_rng(0)
    show = rng.choice(N_SAMPLES, size=4000, replace=False)
    ax_a.scatter(samples[show, 0], samples[show, 1], s=1, c="0.25",
                 alpha=0.45, zorder=3, lw=0)
    _basins(ax_a)
    _frame(ax_a)
    ax_a.set_ylabel(r"$x_2$", labelpad=2)
    ax_a.set_title(r"(a) Equilibrium samples", pad=4)

    def _iso_overlay(ax, q_sliced_m, rmse, title):
        _backdrop(ax, alpha=0.55)
        ax.contour(EX, EY, q_pde_m, levels=CONTOUR_LEVELS,
                   colors="black", linewidths=0.9, zorder=3)
        ax.contour(EX, EY, q_sliced_m, levels=CONTOUR_LEVELS,
                   colors=HIGHLIGHT, linewidths=0.9, linestyles="--", zorder=4)
        _basins(ax)
        ax.legend(handles=[
            Line2D([0], [0], color="black", lw=0.9, label="PDE reference"),
            Line2D([0], [0], color=HIGHLIGHT, lw=0.9, ls="--",
                   label=r"Sliced $\bar q$"),
        ], loc="upper right", fontsize=6, framealpha=0.85, handlelength=1.8)
        ax.text(0.03, 0.03, f"RMSE = {rmse:.3f}", transform=ax.transAxes,
                fontsize=7, ha="left", va="bottom",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                          alpha=0.85, edgecolor="none"))
        _frame(ax)
        ax.set_title(title, pad=4)

    _iso_overlay(ax_b, q_full_m, rmse_full,
                 r"(b) Full Gram weights")
    _iso_overlay(ax_c, q_ebmc_m, rmse_ebmc,
                 r"(c) EBMC weights")
    ax_b.tick_params(labelleft=False)
    ax_c.tick_params(labelleft=False)

    out = "01_wolfe_quapp_2d.png"
    fig.savefig(out)
    fig.savefig(out.replace(".png", ".pdf"))
    print(f"saved plot to {out} (+ .pdf)")


if __name__ == "__main__":
    main()
