"""Reaction rates on the rotated Wolfe-Quapp 2D potential, against an exact reference.

Companion to ``01_wolfe_quapp_2d.py``. The rate is a reduction of the
committor-coordinate flux ``nu_R(q) = D_q(q) pi(q)``, and this script builds
the pair ``{D_q, pi}`` every way the library offers, on one committor:

    1. The committor in one call, ``fit_committor``.
    2. Dynamics: a window-stratified overdamped-Langevin swarm at a known
       mobility D0 (short independent segments seeded everywhere, since a
       single unbiased trajectory almost never visits the ~5.5 kT barrier).
    3. The static ensemble: the density pi(q), the populations, and the
       iso-committor mean squared gradient <|grad q|^2>_q (the co-area profile).
    4. Three constructors of D_q:
         (a) measured on the committor: Kramers-Moyal at an explicit lag,
             after a lag scan;
         (b) mapped from the collective variable x through the Jacobian:
             D measured along x (which recovers D0 exactly, the estimator
             check), times <|grad q|^2>_q / <|grad x|^2>;
         (c) the assumed configurational D0, the same map with cv_grad_sq = 1.
    5. The four reductions of each, the flatness of each flux, the
       committor-free Kramers baseline along x, and the exact
       transition-path-theory rate from the converged PDE committor.
    6. A printed table with PASS/FAIL self-checks, and a four-panel figure.

Runtime on CPU: about 2-3 minutes (committor and swarm ~30-60 s, PDE ~90 s).

Run:
    python 02_rates_wolfe_quapp.py
"""

from __future__ import annotations

import jax
import matplotlib.pyplot as plt
import numpy as np

jax.config.update("jax_enable_x64", True)

from _wolfe_quapp import (
    BETA,
    CENTRE_A,
    CENTRE_B,
    DOUBLE_COL,
    HIGHLIGHT,
    N_DIRECTIONS,
    SEED,
    SOLVER_KWARGS,
    STATE_RADIUS,
    apply_paper_style,
    boltzmann_samples,
    jacobi_committor_2d,
    potential,
    potential_grad,
)

import sliced_committor as sc
from sliced_committor.rates.baselines import pmf_kramers_rate

apply_paper_style()

N_SAMPLES = 100_000  # the flux integrand peaks at the barrier, where pi is tiny
D0 = 0.05  # the mobility of the overdamped dynamics (beta = 1)
SWARM_DT = 5e-3
SWARM_SEG_LEN = 80
SWARM_N_BASIN = 2200  # seeds drawn from the equilibrium ensemble
SWARM_N_PATH = 800  # seeds along the A -> B line (the barrier tube)
N_BINS = 30  # the committor grid of the profiles and the reductions
BAND = (0.3, 0.7)  # the plateau band
REDUCTIONS = ("plateau", "arithmetic", "harmonic", "local")


def langevin_swarm(seeds, *, dt=SWARM_DT, seg_len=SWARM_SEG_LEN, seed=SEED):
    """Euler-Maruyama, one short independent segment per seed; segment-major frames.

    x <- x - D0 grad U(x) dt + sqrt(2 D0 dt) eta: the stationary density is exp(-U)
    and the diffusion coefficient of every coordinate is D0.
    """
    rng = np.random.default_rng(seed)
    state = np.asarray(seeds, dtype=np.float64).copy()
    n_seg = state.shape[0]
    noise = np.sqrt(2.0 * D0 * dt)
    frames = np.empty((n_seg, seg_len, 2))
    for t in range(seg_len):
        frames[:, t] = state
        state = state - D0 * potential_grad(state) * dt + noise * rng.standard_normal((n_seg, 2))
    return frames.reshape(n_seg * seg_len, 2), np.repeat(np.arange(n_seg), seg_len)


def exact_reference_rate():
    """The exact TPT rate of the overdamped dynamics: nu_R* = D0 <|grad q|^2>_pi on the PDE committor."""
    q_pde, X, Y, iters, delta = jacobi_committor_2d()
    rho = np.exp(-BETA * potential(X, Y))
    Z = float(rho.sum())
    gy, gx = np.gradient(q_pde, float(Y[1, 0] - Y[0, 0]), float(X[0, 1] - X[0, 0]))
    nu_R = D0 * float(np.sum(rho * (gx**2 + gy**2))) / Z
    rho_B = float(np.sum(rho * q_pde)) / Z
    return {
        "nu_R": nu_R,
        "rho_A": 1.0 - rho_B,
        "rho_B": rho_B,
        "k_AB": nu_R / (1.0 - rho_B),
        "pde_iters": iters,
        "pde_delta": delta,
    }


def rel(k, k_ref):
    return abs(k - k_ref) / max(abs(k_ref), 1e-30)


def main():
    # ------------------------------------------------------------------------
    # 1. the committor
    # ------------------------------------------------------------------------
    print(f"[1] fit_committor on {N_SAMPLES:,} samples (M={N_DIRECTIONS})...")
    samples = boltzmann_samples(n_samples=N_SAMPLES)
    in_A = np.linalg.norm(samples - CENTRE_A, axis=1) < STATE_RADIUS
    in_B = np.linalg.norm(samples - CENTRE_B, axis=1) < STATE_RADIUS
    q, fit = sc.fit_committor(
        samples,
        in_A=in_A,
        in_B=in_B,
        n_directions=N_DIRECTIONS,
        seed=SEED,
        return_details=True,
        **SOLVER_KWARGS,
    )
    print(f"    Dirichlet energy {fit.dirichlet_energy:.5f}; q(0, 0) = {float(q(np.zeros(2))):.3f}")

    # ------------------------------------------------------------------------
    # 2. dynamics with a known mobility
    # ------------------------------------------------------------------------
    print(
        f"\n[2] Langevin swarm (D0={D0}): {SWARM_N_BASIN}+{SWARM_N_PATH} seeds x {SWARM_SEG_LEN} steps..."
    )
    rng = np.random.default_rng(SEED)
    seeds = np.concatenate(
        [
            samples[rng.choice(N_SAMPLES, size=SWARM_N_BASIN, replace=False)],
            (1.0 - np.linspace(0.0, 1.0, SWARM_N_PATH)[:, None]) * CENTRE_A
            + np.linspace(0.0, 1.0, SWARM_N_PATH)[:, None] * CENTRE_B,
        ]
    )
    traj, wids = langevin_swarm(seeds)
    print(f"    trajectory: {traj.shape[0]:,} frames in {int(wids.max()) + 1} windows")

    # ------------------------------------------------------------------------
    # 3. the static ensemble
    # ------------------------------------------------------------------------
    print("\n[3] The static ensemble:")
    qx = np.asarray(q(samples))
    pi = sc.density(qx, n_bins=N_BINS)
    rho_A, rho_B = sc.basin_populations(qx, in_A=in_A, in_B=in_B)
    g_q = sc.committor_grad_sq(q, samples, n_bins=N_BINS)
    print(
        f"    rho_A = {rho_A:.3f}, rho_B = {rho_B:.3f}; "
        f"<|grad q|^2>_q spans {np.nanmin(g_q.values):.3g} .. {np.nanmax(g_q.values):.3g}"
    )

    # ------------------------------------------------------------------------
    # 4. three constructors of D_q
    # ------------------------------------------------------------------------
    print("\n[4] The committor-coordinate diffusion, three ways:")
    qt = np.asarray(q(traj))
    scan = sc.lag_scan(
        qt, dt=SWARM_DT, lags=(1, 2, 5, 10), window_ids=wids, n_bins=N_BINS, band=BAND
    )
    print(
        "    (a) lag scan of D_q on the band: "
        + ", ".join(f"lag {int(L)}: {v:.4f}" for L, v in zip(scan.levels, scan.values))
    )
    D_q = {"measured": sc.diffusion_profile(qt, dt=SWARM_DT, lag=1, window_ids=wids, n_bins=N_BINS)}

    x_t = traj[:, 0]
    D_x = sc.diffusion_profile(
        x_t, dt=SWARM_DT, lag=1, window_ids=wids, n_bins=N_BINS, span=(x_t.min(), x_t.max())
    )
    D_s = float(np.nanmedian(D_x.values))
    g_s = sc.linear_response_grad_sq(samples[:, 0], samples)  # x is a feature: exactly 1
    print(
        f"    (b) D along x: median {D_s:.4f} (input D0 = {D0}, {rel(D_s, D0):.1%} off); <|grad x|^2> = {g_s:.4f}"
    )
    D_q["mapped"] = sc.committor_diffusion_from_cv(g_q, D_s=D_s, cv_grad_sq=g_s)
    D_q["assumed"] = sc.committor_diffusion_from_cv(g_q, D_s=D0, cv_grad_sq=1.0)
    print("    (c) the assumed D0 through the same map")

    # ------------------------------------------------------------------------
    # 5. the reductions, the baseline, and the exact rate
    # ------------------------------------------------------------------------
    print("\n[5] Exact reference from the PDE committor...")
    ref = exact_reference_rate()
    k_ref = ref["k_AB"]
    print(
        f"    PDE converged in {ref['pde_iters']:,} iterations; nu_R* = {ref['nu_R']:.4g}, k_AB* = {k_ref:.4g}"
    )

    rates = {
        name: {red: sc.rate_from_profiles(pi, D, rho_A, rho_B, reduction=red) for red in REDUCTIONS}
        for name, D in D_q.items()
    }
    x_s = samples[:, 0]
    pi_x = sc.density(x_s, n_bins=60, span=(x_s.min(), x_s.max()))
    kramers = pmf_kramers_rate(pi_x, D_x)

    print(f"\n[6] k_AB per constructor and reduction (exact k_AB* = {k_ref:.4g}):\n")
    print(
        f"  {'D_q':<10s}"
        + "".join(f"{r:>14s}" for r in REDUCTIONS)
        + f"{'flatness':>10s}{'flux_cv':>9s}"
    )
    print("  " + "-" * 79)
    for name, by_red in rates.items():
        cells = "".join(
            f"{by_red[r]['k_AB']:>8.4g} {rel(by_red[r]['k_AB'], k_ref):>5.0%}" for r in REDUCTIONS
        )
        flat = by_red["plateau"]["flatness"]
        cv = sc.flux_flatness(by_red["plateau"]["nu"], (0.2, 0.8))
        print(f"  {name:<10s}{cells}{flat:>10.3f}{cv:>9.3f}")
    print(
        f"  {'Kramers (x)':<10s}{kramers['k_AB']:>8.4g} {rel(kramers['k_AB'], k_ref):>5.0%}"
        f"   (delta F = {kramers['delta_F_AB']:.2f} kT, D at the barrier {kramers['D_barrier']:.4f})"
    )
    print("\n    All reductions coincide for the exact committor; their spread and the")
    print("    flatness of the flux measure how far the fitted committor is from it.")

    k_map = rates["mapped"]["plateau"]["k_AB"]
    k_ass = rates["assumed"]["plateau"]["k_AB"]
    checks = [
        (f"D along x recovers D0            ({rel(D_s, D0):>5.1%} <= 15%)", rel(D_s, D0) <= 0.15),
        (
            f"mapped plateau k_AB ~ exact      ({rel(k_map, k_ref):>5.1%} <= 30%)",
            rel(k_map, k_ref) <= 0.30,
        ),
        (
            f"assumed-D0 plateau k_AB ~ exact  ({rel(k_ass, k_ref):>5.1%} <= 30%)",
            rel(k_ass, k_ref) <= 0.30,
        ),
        (
            f"flux flat on the band            ({rates['assumed']['plateau']['flatness']:>5.3f} <= 0.35)",
            rates["assumed"]["plateau"]["flatness"] <= 0.35,
        ),
    ]
    print("\n[7] Self-checks:")
    for label, ok in checks:
        print(f"    {'PASS' if ok else 'FAIL'}  {label}")

    # ------------------------------------------------------------------------
    # 8. the figure
    # ------------------------------------------------------------------------
    print("\n[8] Plotting...")
    fig, axes = plt.subplots(
        2, 2, figsize=(DOUBLE_COL, 4.8), gridspec_kw=dict(wspace=0.30, hspace=0.42)
    )
    ax_pi, ax_D, ax_nu, ax_k = axes.ravel()

    good = pi.counts > 0
    ax_pi.plot(pi.levels[good], pi.values[good], color="0.15", marker="o", ms=2.0, lw=1.0)
    ax_pi.set_title(r"(a) Density $\pi(\bar q)$", pad=4)
    ax_pi.set_xlabel(r"committor $\bar q$", labelpad=1)
    ax_pi.set_ylabel(r"$\pi$", labelpad=2)
    ax_pi.set_xlim(0, 1)

    goodD = D_x.counts > 0
    ax_D.plot(D_x.levels[goodD], D_x.values[goodD], color="0.15", marker="o", ms=2.0, lw=1.0)
    ax_D.axhline(D0, color=HIGHLIGHT, ls="--", lw=1.0, label=rf"$D_0 = {D0}$")
    ax_D.set_title(r"(b) $D$ along $x$ recovers $D_0$", pad=4)
    ax_D.set_xlabel(r"CV $x$", labelpad=1)
    ax_D.set_ylabel(r"$D$", labelpad=2)
    ax_D.set_ylim(0, 3.0 * D0)
    ax_D.legend(loc="best", fontsize=6, framealpha=0.85)

    for name, ls in (("measured", ":"), ("mapped", "--"), ("assumed", "-")):
        nu = rates[name]["plateau"]["nu"]
        ok = np.isfinite(nu.values) & (nu.values > 0)
        ax_nu.plot(
            nu.levels[ok],
            nu.values[ok],
            color="0.15",
            ls=ls,
            lw=1.0,
            marker="o",
            ms=1.8,
            label=name,
        )
    ax_nu.axhline(ref["nu_R"], color=HIGHLIGHT, ls="--", lw=1.0, label=r"$\nu_R^{*}$ (PDE)")
    ax_nu.axvspan(*BAND, color="0.85", zorder=0)
    ax_nu.set_title(r"(c) Flux $\nu_R(\bar q) = D_q \pi$", pad=4)
    ax_nu.set_xlabel(r"committor $\bar q$", labelpad=1)
    ax_nu.set_ylabel(r"$\nu_R$", labelpad=2)
    ax_nu.set_xlim(0, 1)
    ax_nu.set_yscale("log")
    ax_nu.legend(loc="best", fontsize=6, framealpha=0.85)

    names = [f"{n}\n{r}" for n in D_q for r in ("plateau", "harmonic")] + ["Kramers\n(x)"]
    kvals = [rates[n][r]["k_AB"] for n in D_q for r in ("plateau", "harmonic")] + [kramers["k_AB"]]
    xpos = np.arange(len(names))
    ax_k.bar(xpos, kvals, color="0.55", width=0.62, zorder=2)
    ax_k.axhline(
        k_ref, color=HIGHLIGHT, ls="--", lw=1.0, label=rf"$k_{{AB}}^{{*}} = {k_ref:.3g}$", zorder=3
    )
    ax_k.set_yscale("log")
    ax_k.set_xticks(xpos)
    ax_k.set_xticklabels(names, fontsize=5.5)
    ax_k.set_ylabel(r"$k_{AB}$", labelpad=2)
    ax_k.set_title("(d) Rate versus the exact reference", pad=4)
    ax_k.legend(loc="best", fontsize=6, framealpha=0.85)

    out = "02_rates_wolfe_quapp.png"
    fig.savefig(out)
    fig.savefig(out.replace(".png", ".pdf"))
    print(f"saved plot to {out} (+ .pdf)")


if __name__ == "__main__":
    main()
