"""Reaction-rate computations on the rotated Wolfe-Quapp 2D potential.

Companion to ``01_wolfe_quapp_2d.py`` (which builds the committor field). This
script shows the full ``sliced_committor`` rate API on the same benchmark and
cross-checks every method against an EXACT reference rate, so it doubles as an
end-to-end test of the rate code:

    1. Committor in one call: ``q = fit_committor(samples, in_A=, in_B=)``.
    2. Diffusion data: a window-stratified overdamped-Langevin swarm at a known
       constant mobility D0 (β=1) seeded across the domain.
    3. Three quantity primitives -- density π(q), diffusion D(q), reactive flux
       Φ(q) -- each profiled along the committor coordinate.
    4. Exact reference rate from the converged PDE committor:
       ν_R* = D0·⟨|∇q|²⟩_π, k_AB* = ν_R*/ρ_A*.
    5. The rate formulas vs that reference: the feature-space Dirichlet form and
       TPT flux (which use the known D0), the coordinate-invariant committor_rate
       (its harmonic / local reductions ARE Berezhkovskii-Szabo MFPT / local),
       and a Kramers harmonic estimate.
    6. A printed comparison table + a four-panel figure.

Why a swarm rather than one long trajectory: the WQ barrier is ~5.5 kT above
A, so a single unbiased trajectory almost never visits the saddle. The mobility
is constant, so the drift-corrected Kramers-Moyal estimator recovers D0 from
short independent segments seeded everywhere (the realistic "swarm / umbrella"
case) -- ``window_ids`` keeps frame pairs within a segment.

The rate functions themselves are jax+numpy only; this example additionally
uses matplotlib + scipy (install with ``pip install -e .[examples]``).

Runtime on CPU: ~2-3 minutes (committor + swarm ~30-60 s, PDE baseline ~90 s).

Run:
    python 02_rates_wolfe_quapp.py
"""

from __future__ import annotations

import jax
import matplotlib.pyplot as plt
import numpy as np

# Float64 is required for the EBMC / full-Gram constrained solves.
jax.config.update("jax_enable_x64", True)

# Shared Wolfe-Quapp scaffolding (see 01_wolfe_quapp_2d.py / _wolfe_quapp.py).
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

apply_paper_style()


# ============================================================================
# 0. Settings
# ============================================================================
# Many samples: the flux integrand D|∇q̄|²·π peaks at the barrier where π is tiny.
N_SAMPLES = 100_000
D0 = 0.05  # constant mobility of the overdamped dynamics (β = 1)

# Window-stratified Langevin swarm for D(q). Short independent segments seeded
# across the domain so every committor bin fills.
SWARM_DT = 5e-3
SWARM_SEG_LEN = 80
SWARM_N_BASIN = 2200  # seeds drawn from the equilibrium ensemble (basins)
SWARM_N_PATH = 800  # seeds along the A->B line (the barrier tube)
RATE_N_BINS = 60  # coordinate bins for D(x) / π in the rate formulas
FLUX_N_BINS = 30  # iso-committor bins for Φ(q̄) (coarser -> smoother)
PLATEAU = (0.3, 0.7)  # iso-committor plateau for TPT / range Dirichlet


# ============================================================================
# 1. Overdamped Langevin swarm  ->  (trajectory, window_ids)
# ============================================================================
def langevin_swarm(seeds, *, dt=SWARM_DT, seg_len=SWARM_SEG_LEN, seed=SEED):
    """Vectorised Euler-Maruyama: one short independent segment per seed.

    Overdamped Langevin at β = 1 and mobility D0:
        x <- x - D0 ∇U(x) dt + sqrt(2 D0 dt) η ,   η ~ N(0, I).
    Stationary density ∝ exp(-U) (matches the Boltzmann sampler); the constant
    diffusion coefficient is D0. Returns ``(traj, window_ids)`` with the frames
    laid out segment-major (each segment contiguous) so within-window
    Kramers-Moyal pairs are adjacent.
    """
    rng = np.random.default_rng(seed)
    seeds = np.asarray(seeds, dtype=np.float64)
    n_seg = seeds.shape[0]
    noise = np.sqrt(2.0 * D0 * dt)
    state = seeds.copy()
    frames = np.empty((n_seg, seg_len, 2), dtype=np.float64)
    for t in range(seg_len):
        frames[:, t] = state
        state = state - D0 * potential_grad(state) * dt + noise * rng.standard_normal((n_seg, 2))
    traj = frames.reshape(n_seg * seg_len, 2)  # segment-major
    window_ids = np.repeat(np.arange(n_seg), seg_len)
    return traj, window_ids


# ============================================================================
# 2. Exact reference rate from the converged PDE committor
# ============================================================================
def exact_reference_rate():
    """ν_R* = D0·⟨|∇q|²⟩_π and k_AB* from the weighted-Jacobi PDE committor.

    This is the gold standard: the exact transition-path-theory rate of the
    overdamped dynamics at mobility D0, computed on the grid (no sampling).
    """
    q_pde, X, Y, iters, delta = jacobi_committor_2d()
    rho = np.exp(-BETA * potential(X, Y))
    Z = float(rho.sum())
    dx = float(X[0, 1] - X[0, 0])
    dy = float(Y[1, 0] - Y[0, 0])
    gy, gx = np.gradient(q_pde, dy, dx)  # axis0 = y, axis1 = x
    grad2 = gx**2 + gy**2
    nu_R = D0 * float(np.sum(rho * grad2)) / Z
    rho_B = float(np.sum(rho * q_pde)) / Z
    rho_A = 1.0 - rho_B
    return {
        "nu_R": nu_R,
        "rho_A": rho_A,
        "rho_B": rho_B,
        "k_AB": nu_R / rho_A,
        "k_BA": nu_R / rho_B,
        "pde_iters": iters,
        "pde_delta": delta,
    }


# ============================================================================
# Helpers for the printed table
# ============================================================================
def _rel(k, k_ref):
    return abs(k - k_ref) / max(abs(k_ref), 1e-30)


def _row(name, d, k_ref):
    nu = d.get("nu_R", float("nan"))
    return (
        f"  {name:<26s} {nu:>10.4g} {d['rho_A']:>8.3f} "
        f"{d['k_AB']:>11.4g} {_rel(d['k_AB'], k_ref):>9.1%}"
    )


# ============================================================================
# Main
# ============================================================================
def main():
    # ------------------------------------------------------------------------
    # Step 1: samples + basin masks, then the committor in ONE call.
    # ------------------------------------------------------------------------
    print(f"[1] fit_committor on {N_SAMPLES:,} samples (M={N_DIRECTIONS}, EBMC)...")
    samples = boltzmann_samples(n_samples=N_SAMPLES)
    in_A = np.linalg.norm(samples - CENTRE_A, axis=1) < STATE_RADIUS
    in_B = np.linalg.norm(samples - CENTRE_B, axis=1) < STATE_RADIUS
    q = sc.fit_committor(
        samples,
        in_A=in_A,
        in_B=in_B,
        n_directions=N_DIRECTIONS,
        seed=SEED,
        **SOLVER_KWARGS,
    )
    print(
        f"    callable committor ready; q(centreA)={float(q(CENTRE_A)):.3f}, "
        f"q(centreB)={float(q(CENTRE_B)):.3f}, q(0,0)={float(q(np.zeros(2))):.3f}"
    )

    # ------------------------------------------------------------------------
    # Step 2: window-stratified Langevin swarm for the diffusion estimate.
    # ------------------------------------------------------------------------
    print(
        f"\n[2] Langevin swarm (D0={D0}): "
        f"{SWARM_N_BASIN}+{SWARM_N_PATH} seeds x {SWARM_SEG_LEN} steps..."
    )
    rng = np.random.default_rng(SEED)
    basin_seeds = samples[rng.choice(N_SAMPLES, size=SWARM_N_BASIN, replace=False)]
    s = np.linspace(0.0, 1.0, SWARM_N_PATH)[:, None]
    path_seeds = (1.0 - s) * CENTRE_A[None, :] + s * CENTRE_B[None, :]  # A -> B line
    seeds = np.concatenate([basin_seeds, path_seeds], axis=0)
    traj, wids = langevin_swarm(seeds)
    print(f"    trajectory: {traj.shape[0]:,} frames in {int(wids.max()) + 1} windows")

    # ------------------------------------------------------------------------
    # Step 3: the three quantity primitives. density / flux are profiled along
    # the committor; diffusion is profiled along the physical CV x (see below).
    # ------------------------------------------------------------------------
    print("\n[3] Quantity primitives:")
    cv_s = np.asarray(samples)[:, 0]  # collective variable: the x coordinate
    cv_t = traj[:, 0]
    pi_prof = sc.density(q, samples, n_bins=RATE_N_BINS)
    # Diffusion along x: the x-marginal of the overdamped dynamics has diffusion
    # D0 for ANY potential, so D(x) recovers the known input -- the estimator
    # test. (Profiled along the committor instead it would give the q*-dependent
    # D(q̄) = D0·|∇q̄|², which is what Berezhkovskii-Szabo consumes internally.)
    D_prof = sc.diffusion_coefficient(
        q,
        traj,
        dt=SWARM_DT,
        coordinate=cv_t,
        window_ids=wids,
        lag=1,
        n_bins=RATE_N_BINS,
    )
    phi_prof = sc.reactive_flux(q, samples, D=D0, n_bins=FLUX_N_BINS)
    # density along the same CV, to show coordinate= on a second primitive.
    pi_cv = sc.density(q, samples, coordinate=cv_s, n_bins=RATE_N_BINS)
    print(
        f"    density π(q̄): {int((pi_prof.counts > 0).sum())}/{RATE_N_BINS} bins, "
        f"∫π dq ≈ {float(np.nansum(pi_prof.values) * (pi_prof.levels[1] - pi_prof.levels[0])):.3f}"
    )
    print(f"    density π(x):  {int((pi_cv.counts > 0).sum())}/{RATE_N_BINS} bins (coordinate=x)")

    # diffusion-recovery self-check: D(x) recovers the constant input mobility D0.
    good = (D_prof.counts > 0) & np.isfinite(D_prof.values)
    D_med = float(np.nanmedian(D_prof.values[good])) if np.any(good) else float("nan")
    print(
        f"    diffusion D(x): median = {D_med:.4f}  (input D0 = {D0}; "
        f"rel.err {_rel(D_med, D0):.1%}) [KM estimator recovers the mobility]"
    )

    # reactive-flux flatness self-check (TPT flux conservation on the plateau).
    sel = (phi_prof.levels >= PLATEAU[0]) & (phi_prof.levels <= PLATEAU[1]) & (phi_prof.counts > 0)
    phi_flat = float(np.std(phi_prof.values[sel]) / max(abs(np.mean(phi_prof.values[sel])), 1e-30))
    print(f"    reactive flux Φ(q): plateau flatness on {PLATEAU} = {phi_flat:.3f}")

    # ------------------------------------------------------------------------
    # Step 4: exact reference rate from the PDE committor.
    # ------------------------------------------------------------------------
    print("\n[4] Exact reference rate (converged PDE committor)...")
    ref = exact_reference_rate()
    print(f"    PDE converged in {ref['pde_iters']:,} iters (max|dq|={ref['pde_delta']:.1e})")
    print(f"    ν_R* = {ref['nu_R']:.4g}, ρ_A* = {ref['rho_A']:.3f}, k_AB* = {ref['k_AB']:.4g}")

    # ------------------------------------------------------------------------
    # Step 5: the rate methods.
    # ------------------------------------------------------------------------
    print("\n[5] Rate methods...")
    r_dir = sc.dirichlet_rate(q, samples, D=D0, in_A=in_A, in_B=in_B)
    r_dir_band = sc.dirichlet_rate(q, samples, D=D0, at=PLATEAU, in_A=in_A, in_B=in_B)
    r_tpt = sc.tpt_rate(q, samples, D=D0, at=PLATEAU, in_A=in_A, in_B=in_B, n_bins=FLUX_N_BINS)
    r_bs_loc = sc.berezhkovskii_szabo_rate(
        q,
        samples,
        traj,
        dt=SWARM_DT,
        window_ids=wids,
        mode="local",
        at=0.5,
        n_bins=RATE_N_BINS,
        lag=1,
        in_A=in_A,
        in_B=in_B,
    )
    r_bs_mfpt = sc.berezhkovskii_szabo_rate(
        q,
        samples,
        traj,
        dt=SWARM_DT,
        window_ids=wids,
        mode="mfpt",
        at=None,
        n_bins=RATE_N_BINS,
        lag=1,
        in_A=in_A,
        in_B=in_B,
    )
    # Kramers needs interior free-energy wells: use the physical CV x (the two
    # WQ minima sit near x = ±1.7, a genuine double well), not the bare committor.
    r_kram = sc.kramers_rate(
        q,
        samples,
        traj,
        dt=SWARM_DT,
        window_ids=wids,
        n_bins=RATE_N_BINS,
        coordinate=cv_s,
        traj_coordinate=cv_t,
        lag=1,
        in_A=in_A,
        in_B=in_B,
    )
    # committor_rate is the coordinate-invariant UNIFIED interface: it reduces the
    # same {D_q(q), π(q)} pair four ways and needs NO length-scale D (so it works
    # in any feature space, unlike Dirichlet / TPT). On the committor coordinate
    # its "harmonic" / "local" reductions ARE Berezhkovskii-Szabo mfpt / local;
    # the spread across reductions is a committor-quality diagnostic (they all
    # coincide for the exact committor). For real systems with no trusted D this
    # is the recommended path. (find_plateau / saddle_bridge_D extend it; see
    # docs/recipes.md.)
    cr = {
        red: sc.committor_rate(
            q, samples, traj, dt=SWARM_DT, window_ids=wids, reduction=red,
            n_bins=RATE_N_BINS, lag=1, in_A=in_A, in_B=in_B,
        )["k_AB"]
        for red in ("arithmetic", "plateau", "local", "harmonic")
    }

    # ------------------------------------------------------------------------
    # Step 6: comparison table.
    # ------------------------------------------------------------------------
    k_ref = ref["k_AB"]
    print("\n[6] Rate comparison (k_AB* = exact PDE reference):\n")
    print(f"  {'method':<26s} {'nu_R':>10s} {'rho_A':>8s} {'k_AB':>11s} {'rel.err':>9s}")
    print("  " + "-" * 67)
    print(
        f"  {'exact (PDE)':<26s} {ref['nu_R']:>10.4g} {ref['rho_A']:>8.3f} "
        f"{k_ref:>11.4g} {'--':>9s}"
    )
    print(_row("Dirichlet (full)", r_dir, k_ref))
    print(_row("Dirichlet (0.3-0.7)", r_dir_band, k_ref))
    print(_row("TPT flux (plateau)", r_tpt, k_ref))
    print(_row("Berezhkovskii-Szabo loc", r_bs_loc, k_ref))
    print(_row("Berezhkovskii-Szabo mfpt", r_bs_mfpt, k_ref))
    print(_row("Kramers (CV=x)", r_kram, k_ref))
    print(
        f"\n    TPT plateau flatness = {r_tpt.get('plateau_flatness', float('nan')):.3f}; "
        f"Kramers ΔF_AB = {r_kram['delta_F_AB']:.2f} kT, "
        f"D_barrier = {r_kram['D_barrier']:.4f}"
    )
    print("    Variational flux (Dirichlet/TPT) and Berezhkovskii-Szabo local land within")
    print("    ~5-25% of k_AB*; the BS-MFPT integral and the 1D Kramers harmonic estimate")
    print("    are tail / CV-sensitive -> a factor of ~2-6 (order of magnitude).")
    print("\n    committor_rate reductions (coordinate-invariant, no D supplied):")
    print("      " + "   ".join(f"{r}={v:.4g}" for r, v in cr.items()))
    print("      harmonic == B-Szabo mfpt, local == B-Szabo local; the reduction")
    print("      spread is a committor-quality diagnostic (all equal for exact q).")

    # ------------------------------------------------------------------------
    # Step 7: self-checks (the example doubles as an integration test of the
    # rate code; the unit tests in sliced_committor/tests cover the rest).
    # ------------------------------------------------------------------------
    checks = [
        (f"D(x) recovers D0          ({_rel(D_med, D0):>5.1%} <= 15%)", _rel(D_med, D0) <= 0.15),
        (f"reactive-flux plateau flat ({phi_flat:>5.3f} <= 0.35)", phi_flat <= 0.35),
        (
            f"Dirichlet k_AB ~ exact    ({_rel(r_dir['k_AB'], k_ref):>5.1%} <= 30%)",
            _rel(r_dir["k_AB"], k_ref) <= 0.30,
        ),
        (
            f"TPT k_AB ~ exact          ({_rel(r_tpt['k_AB'], k_ref):>5.1%} <= 30%)",
            _rel(r_tpt["k_AB"], k_ref) <= 0.30,
        ),
        (
            "committor_rate harmonic == B-Szabo mfpt (unified interface)",
            _rel(cr["harmonic"], r_bs_mfpt["k_AB"]) <= 1e-9,
        ),
    ]
    print("\n[7] Self-checks:")
    for label, ok in checks:
        print(f"    {'PASS' if ok else 'FAIL'}  {label}")

    # ------------------------------------------------------------------------
    # Step 8: figure -- three quantity profiles + a rate bar chart.
    # ------------------------------------------------------------------------
    print("\n[8] Plotting...")
    fig, axes = plt.subplots(
        2, 2, figsize=(DOUBLE_COL, 4.8), gridspec_kw=dict(wspace=0.30, hspace=0.42)
    )
    ax_pi, ax_D, ax_phi, ax_k = axes.ravel()

    def _profile_plot(
        ax,
        prof,
        title,
        ylabel,
        *,
        xlabel=r"committor $\bar q$",
        xlim=(0, 1),
        ref_val=None,
        ref_lab=None,
    ):
        good = prof.counts > 0
        ax.plot(prof.levels[good], prof.values[good], color="0.15", marker="o", ms=2.0, lw=1.0)
        if ref_val is not None:
            ax.axhline(ref_val, color=HIGHLIGHT, ls="--", lw=1.0, label=ref_lab)
            ax.legend(loc="best", fontsize=6, framealpha=0.85)
        ax.set_title(title, pad=4)
        ax.set_xlabel(xlabel, labelpad=1)
        ax.set_ylabel(ylabel, labelpad=2)
        ax.set_xlim(*xlim)

    _profile_plot(ax_pi, pi_prof, r"(a) Density $\pi(\bar q)$", r"$\pi$")
    _profile_plot(
        ax_D,
        D_prof,
        r"(b) Diffusion $D(x)$ recovers $D_0$",
        r"$D$",
        xlabel=r"CV $x$",
        xlim=(cv_s.min(), cv_s.max()),
        ref_val=D0,
        ref_lab=rf"$D_0={D0}$",
    )
    ax_D.set_ylim(0, 3.0 * D0)
    _profile_plot(
        ax_phi,
        phi_prof,
        r"(c) Reactive flux $\Phi(\bar q)$",
        r"$\Phi$",
        ref_val=ref["nu_R"],
        ref_lab=r"$\nu_R^{*}$ (PDE)",
    )
    ax_phi.axvspan(*PLATEAU, color="0.85", zorder=0)

    # (d) k_AB per method vs the exact reference.
    names = [
        "Dirichlet\n(full)",
        "TPT\n(plateau)",
        "B-Szabo\n(local)",
        "B-Szabo\n(mfpt)",
        "Kramers\n(CV=x)",
    ]
    kvals = [r_dir["k_AB"], r_tpt["k_AB"], r_bs_loc["k_AB"], r_bs_mfpt["k_AB"], r_kram["k_AB"]]
    xpos = np.arange(len(names))
    # Log scale: the estimates span a decade (accurate methods hug k_AB*, the
    # crude ones overshoot), unreadable on a linear axis dominated by Kramers.
    ax_k.bar(xpos, kvals, color="0.55", width=0.62, zorder=2)
    ax_k.axhline(
        k_ref, color=HIGHLIGHT, ls="--", lw=1.0, label=rf"$k_{{AB}}^{{*}}={k_ref:.3g}$", zorder=3
    )
    ax_k.set_yscale("log")
    ax_k.set_ylim(3e-4, 1.2 * max(kvals))
    ax_k.set_xticks(xpos)
    ax_k.set_xticklabels(names, fontsize=6)
    ax_k.set_ylabel(r"$k_{AB}$", labelpad=2)
    ax_k.set_title("(d) Rate vs exact reference", pad=4)
    ax_k.legend(loc="best", fontsize=6, framealpha=0.85)

    out = "02_rates_wolfe_quapp.png"
    fig.savefig(out)
    fig.savefig(out.replace(".png", ".pdf"))
    print(f"saved plot to {out} (+ .pdf)")


if __name__ == "__main__":
    main()
