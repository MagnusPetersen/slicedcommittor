"""The sliced committor on the rotated Wolfe-Quapp 2D potential, step by step.

    1. Input contract: an (N, dim) sample array and two boolean basin masks.
    2. Slices: ``compute_sliced_committor`` returns the slice basis, a
       ``SlicedCommittorResult`` (``result.summary()`` describes it).
    3. Weights: ``solve_weights`` under the two ridge rules, the half-set
       spectral filter (the default) and the closed-form scalar ridge, with
       the Dirichlet energy and the moment gap of each.
    4. The callable: ``build_committor`` returns ``q(x)``; ``rescale_transition``
       is the affine post-processor the paper's figures use.
    5. Cross-check against a converged 300 x 300 Jacobi PDE committor.
    6. The paper's figure-1 layout.

Settings match paper figure 1 (Petersen et al., 2026) on the benchmark of
Bonati, Zhang and Parrinello (PNAS 2019):
    U(x', y') = x'^4 + y'^4 - 2 x'^2 - 4 y'^2 + x' y' + 0.3 x' + 0.1 y'
at coordinates rotated by theta = -0.15 pi, basins of radius 0.3 around
the two deepest minima, beta = 1, N = 100 000 samples, M = 256 directions.

Runtime on CPU: about 2-3 minutes (the PDE baseline is most of it).

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

jax.config.update("jax_enable_x64", True)  # the weight solve needs float64

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

# log-spaced iso-committor levels for the overlay
CONTOUR_LEVELS = list(expit(np.linspace(np.log(0.01 / 0.99), np.log(0.99 / 0.01), 13)))


def main():
    # ------------------------------------------------------------------------
    # 1. (N, dim) samples and two boolean basin masks
    # ------------------------------------------------------------------------
    print("[1] Generating samples...")
    samples = boltzmann_samples()
    in_A = np.linalg.norm(samples - CENTRE_A, axis=1) < STATE_RADIUS
    in_B = np.linalg.norm(samples - CENTRE_B, axis=1) < STATE_RADIUS
    print(f"    samples.shape = {samples.shape}, |A| = {in_A.sum():,}, |B| = {in_B.sum():,}")
    X, mA, mB = jnp.asarray(samples), jnp.asarray(in_A), jnp.asarray(in_B)

    # ------------------------------------------------------------------------
    # 2. the slice basis
    # ------------------------------------------------------------------------
    print(f"\n[2] compute_sliced_committor(M={N_DIRECTIONS}, n_bins={N_BINS})...")
    result = sc.compute_sliced_committor(
        X, in_A=mA, in_B=mB, n_directions=N_DIRECTIONS, n_bins=N_BINS, seed=SEED, **SOLVER_KWARGS
    )
    print(result.summary())
    print(
        f"    directions {tuple(result.directions.shape)}, 1D committors "
        f"{tuple(result.committors_1d.shape)}, valid slices "
        f"{int(result.valid_mask.sum())}/{result.valid_mask.shape[0]}"
    )

    # ------------------------------------------------------------------------
    # 3. the weights under the two ridge rules
    # ------------------------------------------------------------------------
    print("\n[3] solve_weights:")
    weights = {
        "halfset": sc.solve_weights(result),  # tikhonov="halfset_eigen", the default
        "auto": sc.solve_weights(result, tikhonov="auto"),  # the closed-form scalar ridge
    }
    for name, w in weights.items():
        arr = np.asarray(w.w)
        print(
            f"    {name:>8s}: energy w^T G w = {w.dirichlet_energy:.5f}, moment gap R = "
            f"{w.moment_gap:.4f}, cond = {w.cond:.3f}, ridge = {w.ridge:.2e}, "
            f"#negative weights = {int((arr < 0).sum())}, c = {w.c:+.4f}"
        )
    print("    (lower energy is closer to the true committor; no reference needed)")

    # ------------------------------------------------------------------------
    # 4. the callable committor
    # ------------------------------------------------------------------------
    print("\n[4] build_committor:")
    q = {name: sc.build_committor(result, w) for name, w in weights.items()}
    probe = jnp.asarray(
        [[0.0, 0.0], CENTRE_A + np.array([0.5, 0.0]), CENTRE_B - np.array([0.5, 0.0])]
    )
    for pt, qv in zip(np.asarray(probe), np.asarray(q["halfset"](probe))):
        print(f"    q({pt[0]:+.2f}, {pt[1]:+.2f}) = {qv:.4f}")

    # the display grid; basin masks snap q to 0 / 1, rescale_transition stretches
    # the transition region so the boundary values are exactly 0 and 1 (figures only)
    xs = np.linspace(*DOMAIN, N_SAMPLE_GRID)
    EX, EY = np.meshgrid(xs, xs, indexing="xy")
    pts = jnp.asarray(np.stack([EX.ravel(), EY.ravel()], axis=1))
    grid_A = np.linalg.norm(np.asarray(pts) - CENTRE_A, axis=1) < STATE_RADIUS
    grid_B = np.linalg.norm(np.asarray(pts) - CENTRE_B, axis=1) < STATE_RADIUS
    q_grid = {
        name: np.asarray(
            sc.rescale_transition(fn(pts, in_A=grid_A, in_B=grid_B), grid_A, grid_B)
        ).reshape(EX.shape)
        for name, fn in q.items()
    }

    # ------------------------------------------------------------------------
    # 5. the PDE cross-check
    # ------------------------------------------------------------------------
    print(f"\n[5] PDE baseline ({PDE_N_GRID}x{PDE_N_GRID}, tol={PDE_TOL:.0e})...")
    q_pde_raw, X_pde, Y_pde, pde_iters, pde_delta = jacobi_committor_2d()
    print(f"    converged in {pde_iters:,} iterations; final max|dq| = {pde_delta:.2e}")
    interp = RegularGridInterpolator(
        (Y_pde[:, 0], X_pde[0, :]),
        q_pde_raw,
        method="linear",
        bounds_error=False,
        fill_value=np.nan,
    )
    q_pde = interp(np.stack([EY.ravel(), EX.ravel()], axis=1)).reshape(EX.shape)
    rho = np.exp(-BETA * potential(EX, EY))
    mask = (rho > 1e-3 * rho.max()) & np.isfinite(q_pde)
    rmse = {
        name: float(np.sqrt(np.nanmean((qg - q_pde)[mask] ** 2))) for name, qg in q_grid.items()
    }
    print(
        "    RMSE on the density-masked transition region: "
        + ", ".join(f"{k} = {v:.4f}" for k, v in rmse.items())
    )

    # ------------------------------------------------------------------------
    # 6. the figure
    # ------------------------------------------------------------------------
    print("\n[6] Plotting...")
    F = BETA * potential(EX, EY)
    F -= F[mask].min()
    F_vmax = np.percentile(F[mask], 92)
    F_plot = np.where(mask, F, np.nan)
    q_pde_m = np.where(mask, q_pde, np.nan)

    fig, (ax_a, ax_b, ax_c) = plt.subplots(
        1, 3, figsize=(DOUBLE_COL, 2.40), gridspec_kw=dict(wspace=0.32)
    )

    def backdrop(ax, alpha):
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
            mask.astype(float),
            levels=[0.5],
            colors="grey",
            linewidths=0.5,
            linestyles=":",
            zorder=2,
        )

    def basins(ax):
        for centre, lab in zip((CENTRE_A, CENTRE_B), ("A", "B")):
            ax.add_patch(
                Circle(centre, STATE_RADIUS, fill=False, edgecolor="black", linewidth=1.0, zorder=5)
            )
            ax.text(*centre, lab, ha="center", va="center", fontsize=8, fontweight="bold", zorder=6)

    def frame(ax):
        ax.set_xlim(DOMAIN)
        ax.set_ylim(DOMAIN)
        ax.set_aspect("equal")
        ax.set_xticks([-2, 0, 2])
        ax.set_yticks([-2, 0, 2])
        ax.set_xlabel(r"$x_1$", labelpad=1)

    backdrop(ax_a, alpha=0.85)
    show = np.random.default_rng(0).choice(N_SAMPLES, size=4000, replace=False)
    ax_a.scatter(samples[show, 0], samples[show, 1], s=1, c="0.25", alpha=0.45, zorder=3, lw=0)
    basins(ax_a)
    frame(ax_a)
    ax_a.set_ylabel(r"$x_2$", labelpad=2)
    ax_a.set_title("(a) Equilibrium samples", pad=4)

    def overlay(ax, name, title):
        backdrop(ax, alpha=0.55)
        ax.contour(EX, EY, q_pde_m, levels=CONTOUR_LEVELS, colors="black", linewidths=0.9, zorder=3)
        ax.contour(
            EX,
            EY,
            np.where(mask, q_grid[name], np.nan),
            levels=CONTOUR_LEVELS,
            colors=HIGHLIGHT,
            linewidths=0.9,
            linestyles="--",
            zorder=4,
        )
        basins(ax)
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
            f"RMSE = {rmse[name]:.3f}",
            transform=ax.transAxes,
            fontsize=7,
            ha="left",
            va="bottom",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.85, edgecolor="none"),
        )
        frame(ax)
        ax.set_title(title, pad=4)

    overlay(ax_b, "halfset", "(b) Half-set filter (default)")
    overlay(ax_c, "auto", "(c) Scalar ridge")
    ax_b.tick_params(labelleft=False)
    ax_c.tick_params(labelleft=False)

    out = "01_wolfe_quapp_2d.png"
    fig.savefig(out)
    fig.savefig(out.replace(".png", ".pdf"))
    print(f"saved plot to {out} (+ .pdf)")


if __name__ == "__main__":
    main()
