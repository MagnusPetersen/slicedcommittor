"""Sliced committor walkthrough on the rotated Wolfe-Quapp 2D potential.

Single end-to-end tour of the public API:

    1. Input contract: an (N, dim) sample array + two boolean basin masks.
    2. Solve: ``compute_sliced_committor`` returns a SlicedCommittorResult.
    3. Inspect: ``result.summary()`` / ``why_masked`` for diagnostics.
    4. Weight: four solvers (diagonal RD, full Gram, BMC, EBMC) with
       ``summarize_gram_diagnostics`` for the constrained ones.
    5. Evaluate: ``build_committor`` returns a callable committor, evaluated
       on a 2D contour grid and at a few individual query points.
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

# Shared Wolfe-Quapp scaffolding (potential, paper settings, Boltzmann sampler,
# PDE baseline, plotting style) lives in _wolfe_quapp so example 02 (rates)
# reuses it without duplication.
from _wolfe_quapp import (
    BETA,
    CENTRE_A,
    CENTRE_B,
    DOMAIN,
    DOUBLE_COL,
    HIGHLIGHT,
    N_BINS,
    N_DIRECTIONS,
    N_SAMPLE_GRID,
    N_SAMPLES,
    PDE_N_GRID,
    PDE_TOL,
    SEED,
    SOLVER_KWARGS,
    STATE_RADIUS,
    apply_paper_style,
    boltzmann_samples,
    jacobi_committor_2d,
    potential,
)

import sliced_committor as sc

apply_paper_style()


# 01-specific: log-spaced iso-committor levels for the overlay contour plot.
CONTOUR_LEVELS = list(expit(np.linspace(np.log(0.01 / 0.99), np.log(0.99 / 0.01), 13)))


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
    print(f"    |A| = {in_A.sum():,}, |B| = {in_B.sum():,}, |A & B| = {(in_A & in_B).sum()}")

    samples_j = jnp.asarray(samples)
    in_A_j = jnp.asarray(in_A)
    in_B_j = jnp.asarray(in_B)

    # ------------------------------------------------------------------------
    # Step 2: compute the sliced committor
    # ------------------------------------------------------------------------
    print(f"\n[2] compute_sliced_committor(M={N_DIRECTIONS}, n_bins={N_BINS})...")
    result = sc.compute_sliced_committor(
        samples_j,
        in_A=in_A_j,
        in_B=in_B_j,
        n_directions=N_DIRECTIONS,
        n_bins=N_BINS,
        seed=SEED,
        **SOLVER_KWARGS,
    )

    # ------------------------------------------------------------------------
    # Step 3: inspect the result. SlicedCommittorResult is a NamedTuple.
    # ------------------------------------------------------------------------
    print("\n[3] result.summary():")
    print(result.summary())
    print("\n    SlicedCommittorResult fields:")
    print(f"      directions      : {tuple(result.directions.shape)}")
    print(f"      slice_coords    : {tuple(result.slice_coords.shape)}  (1D grid per slice)")
    print(f"      free_energies   : {tuple(result.free_energies.shape)}  (-log rho per slice)")
    print(
        f"      committors_1d   : {tuple(result.committors_1d.shape)}  (1D RD committor per slice)"
    )
    print(
        f"      valid_mask      : {int(result.valid_mask.sum())}/{result.valid_mask.shape[0]} valid"
    )
    if int((~result.valid_mask).sum()):
        bad = np.flatnonzero(np.asarray(~result.valid_mask))[:3]
        for k in bad:
            print(f"        why_masked(result, {k:d}): {sc.why_masked(result, int(k))}")

    # ------------------------------------------------------------------------
    # Step 4: pick weights. Four solvers, ordered cheap-diagonal -> constraint-rich.
    #   diagonal RD : cheapest, no constraints
    #   full Gram   : lifts the four diagonal approximations
    #   BMC         : pins basin moments
    #   EBMC        : BMC + slice-basis enrichment (production default)
    # ------------------------------------------------------------------------
    print("\n[4] Weight solvers:")
    w_diag = sc.compute_weights_multi(
        result,
        sc.get_default_weight_functions(),
    )["corrected_dirichlet_inv_rd"]
    w_full = sc.compute_full_gram_weights(result, samples_j)
    w_bmc = sc.compute_basin_moment_weights(result, samples_j)
    w_ebmc = sc.compute_enriched_basin_moment_weights(result, samples_j)

    for name, w in [("diag", w_diag), ("full_gram", w_full), ("bmc", w_bmc), ("ebmc", w_ebmc)]:
        arr = np.asarray(w["w"] if isinstance(w, dict) else w)
        print(
            f"    {name:>10s}: sum={arr.sum():+.4f}, "
            f"#neg={int((arr < 0).sum()):>3d}, "
            f"|w|_max={np.abs(arr).max():.4f}"
        )

    print("\n    summarize_gram_diagnostics(full_gram):")
    for line in sc.summarize_gram_diagnostics(w_full).splitlines():
        print(f"      {line}")

    # ------------------------------------------------------------------------
    # Step 5: build the committor once, then call it anywhere. build_committor
    # takes the result + weight (array or dict) and returns a callable q; the
    # callable takes query points (P, dim) and returns (P,) values. Boundary
    # enforcement + transition rescaling are paper defaults for clean visuals.
    # ------------------------------------------------------------------------
    print("\n[5] build_committor:")
    q = sc.build_committor(
        result,
        w_full,
        enforce_boundary_conditions=True,
        rescale_transition=True,
    )
    # (a) at three named points (saddle region, near A, near B).
    probe = jnp.asarray(
        [
            [0.0, 0.0],
            CENTRE_A + np.array([0.5, 0.0]),
            CENTRE_B + np.array([-0.5, 0.0]),
        ]
    )
    q_probe = np.asarray(q(probe))
    for pt, qv in zip(np.asarray(probe), q_probe):
        print(f"      q({pt[0]:+.2f}, {pt[1]:+.2f}) = {qv:.4f}")

    # (b) on a 250x250 grid for plotting.
    eval_xs = np.linspace(*DOMAIN, N_SAMPLE_GRID)
    eval_ys = np.linspace(*DOMAIN, N_SAMPLE_GRID)
    EX, EY = np.meshgrid(eval_xs, eval_ys, indexing="xy")
    pts = jnp.asarray(np.stack([EX.ravel(), EY.ravel()], axis=1))
    q_full = np.asarray(q(pts)).reshape(EX.shape)
    q_ebmc = np.asarray(
        sc.build_committor(
            result,
            w_ebmc,
            enforce_boundary_conditions=True,
            rescale_transition=True,
        )(pts)
    ).reshape(EX.shape)

    # ------------------------------------------------------------------------
    # Step 6: cross-check against a converged 2D PDE baseline.
    # ------------------------------------------------------------------------
    print(f"\n[6] PDE baseline ({PDE_N_GRID}x{PDE_N_GRID}, tol={PDE_TOL:.0e})...")
    q_pde_raw, X_pde, Y_pde, pde_iters, pde_delta = jacobi_committor_2d()
    print(f"    converged in {pde_iters:,} iters; final max|dq| = {pde_delta:.2e}")

    interp = RegularGridInterpolator(
        (Y_pde[:, 0], X_pde[0, :]),
        q_pde_raw,
        method="linear",
        bounds_error=False,
        fill_value=np.nan,
    )
    q_pde = interp(np.stack([EY.ravel(), EX.ravel()], axis=1)).reshape(EX.shape)

    rho_eval = np.exp(-BETA * potential(EX, EY))
    transition_mask = (rho_eval > 1e-3 * rho_eval.max()) & np.isfinite(q_pde)
    rmse_full = float(np.sqrt(np.nanmean((q_full - q_pde)[transition_mask] ** 2)))
    rmse_ebmc = float(np.sqrt(np.nanmean((q_ebmc - q_pde)[transition_mask] ** 2)))
    print(f"    RMSE on transition region: full_gram = {rmse_full:.4f}, ebmc = {rmse_ebmc:.4f}")

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

    fig, axes = plt.subplots(1, 3, figsize=(DOUBLE_COL, 2.40), gridspec_kw=dict(wspace=0.32))
    ax_a, ax_b, ax_c = axes

    def _backdrop(ax, alpha):
        # F_plot is xy-indexed (meshgrid indexing='xy'): row -> y, col -> x.
        # imshow with origin='lower' wants exactly that orientation, no transpose.
        ax.imshow(
            F_plot,
            origin="lower",
            extent=[*DOMAIN, *DOMAIN],
            cmap="Greys",
            vmin=0,
            vmax=F_vmax,
            aspect="auto",
            alpha=alpha,
        )
        ax.contour(
            EX,
            EY,
            transition_mask.astype(float),
            levels=[0.5],
            colors="grey",
            linewidths=0.5,
            linestyles=":",
            zorder=2,
        )

    def _basins(ax):
        for centre, lab in zip((CENTRE_A, CENTRE_B), ("A", "B")):
            ax.add_patch(
                Circle(centre, STATE_RADIUS, fill=False, edgecolor="black", linewidth=1.0, zorder=5)
            )
            ax.text(
                centre[0],
                centre[1],
                lab,
                ha="center",
                va="center",
                fontsize=8,
                fontweight="bold",
                color="black",
                zorder=6,
            )

    def _frame(ax):
        ax.set_xlim(DOMAIN)
        ax.set_ylim(DOMAIN)
        ax.set_aspect("equal")
        ax.set_xticks([-2, 0, 2])
        ax.set_yticks([-2, 0, 2])
        ax.set_xlabel(r"$x_1$", labelpad=1)

    # Panel (a): samples on F backdrop.
    _backdrop(ax_a, alpha=0.85)
    rng = np.random.default_rng(0)
    show = rng.choice(N_SAMPLES, size=4000, replace=False)
    ax_a.scatter(samples[show, 0], samples[show, 1], s=1, c="0.25", alpha=0.45, zorder=3, lw=0)
    _basins(ax_a)
    _frame(ax_a)
    ax_a.set_ylabel(r"$x_2$", labelpad=2)
    ax_a.set_title(r"(a) Equilibrium samples", pad=4)

    def _iso_overlay(ax, q_sliced_m, rmse, title):
        _backdrop(ax, alpha=0.55)
        ax.contour(EX, EY, q_pde_m, levels=CONTOUR_LEVELS, colors="black", linewidths=0.9, zorder=3)
        ax.contour(
            EX,
            EY,
            q_sliced_m,
            levels=CONTOUR_LEVELS,
            colors=HIGHLIGHT,
            linewidths=0.9,
            linestyles="--",
            zorder=4,
        )
        _basins(ax)
        ax.legend(
            handles=[
                Line2D([0], [0], color="black", lw=0.9, label="PDE reference"),
                Line2D([0], [0], color=HIGHLIGHT, lw=0.9, ls="--", label=r"Sliced $\bar q$"),
            ],
            loc="upper right",
            fontsize=6,
            framealpha=0.85,
            handlelength=1.8,
        )
        ax.text(
            0.03,
            0.03,
            f"RMSE = {rmse:.3f}",
            transform=ax.transAxes,
            fontsize=7,
            ha="left",
            va="bottom",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.85, edgecolor="none"),
        )
        _frame(ax)
        ax.set_title(title, pad=4)

    _iso_overlay(ax_b, q_full_m, rmse_full, r"(b) Full Gram weights")
    _iso_overlay(ax_c, q_ebmc_m, rmse_ebmc, r"(c) EBMC weights")
    ax_b.tick_params(labelleft=False)
    ax_c.tick_params(labelleft=False)

    out = "01_wolfe_quapp_2d.png"
    fig.savefig(out)
    fig.savefig(out.replace(".png", ".pdf"))
    print(f"saved plot to {out} (+ .pdf)")


if __name__ == "__main__":
    main()
