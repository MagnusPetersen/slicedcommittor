"""Shared scaffolding for the Wolfe-Quapp 2D examples (01 committor, 02 rates).

Holds the bits both example scripts need so neither duplicates them:

* the rotated Wolfe-Quapp potential ``potential`` (+ analytic gradient
  ``potential_grad`` used by the overdamped Langevin swarm in example 02),
* the paper figure-1 settings (basins, beta, domain, solver kwargs),
* a Boltzmann importance sampler ``boltzmann_samples`` (stand-in for MD data),
* a converged Jacobi PDE committor ``jacobi_committor_2d`` (the cross-check
  baseline; also the source of the exact reference rate in example 02),
* ``apply_paper_style`` for the shared PRL matplotlib rcParams.

Headline settings reproduce paper figure 1 (Petersen et al., 2026) on the
rotated Wolfe-Quapp benchmark of Bonati, Zhang & Parrinello (PNAS 2019):

    U(x', y') = x'^4 + y'^4 - 2 x'^2 - 4 y'^2 + x' y' + 0.3 x' + 0.1 y'

at coordinates rotated by theta = -0.15 pi. The two deepest minima
A = (-1.717, 0.783) and B = (1.676, -0.813) serve as basins (radius 0.3,
beta = 1).

This module is example-internal (leading underscore): the examples run as
scripts, so ``from _wolfe_quapp import ...`` resolves via the examples dir on
sys.path[0]. It is not part of the importable ``sliced_committor`` package.
"""

from __future__ import annotations

import numpy as np

# ============================================================================
# Settings (paper figure 1)
# ============================================================================
BETA = 1.0
N_SAMPLES = 100_000
N_SAMPLE_GRID = 250
N_DIRECTIONS = 256
N_BINS = 200
SEED = 42
STATE_RADIUS = 0.3
CENTRE_A = np.array([-1.717, 0.783])
CENTRE_B = np.array([1.676, -0.813])
DOMAIN = (-2.5, 2.5)
THETA = -0.15 * np.pi

SOLVER_KWARGS = dict(
    n_min=1,
    binning_method="equal_width",
    density_floor=1e-3,
    rd_kappa=1e24,
)

PDE_N_GRID = 300
PDE_TOL = 1e-9
PDE_MAX_ITERS = 10_000_000


# ============================================================================
# The potential and its gradient
# ============================================================================
def potential(x, y):
    """Rotated Wolfe-Quapp energy U(x, y) (paper figure 1)."""
    c, s = np.cos(THETA), np.sin(THETA)
    xr = x * c - y * s
    yr = x * s + y * c
    return xr**4 + yr**4 - 2.0 * xr**2 - 4.0 * yr**2 + xr * yr + 0.3 * xr + 0.1 * yr


def potential_grad(xy):
    """Analytic gradient ∇U at points ``xy`` of shape ``(..., 2)`` -> same shape.

    Used by the overdamped Langevin swarm in example 02. Chain rule through the
    rigid rotation (x, y) -> (xr, yr): ∂U/∂x = c·U_xr + s·U_yr,
    ∂U/∂y = -s·U_xr + c·U_yr.
    """
    xy = np.asarray(xy, dtype=np.float64)
    x, y = xy[..., 0], xy[..., 1]
    c, s = np.cos(THETA), np.sin(THETA)
    xr = x * c - y * s
    yr = x * s + y * c
    dU_dxr = 4.0 * xr**3 - 4.0 * xr + yr + 0.3
    dU_dyr = 4.0 * yr**3 - 8.0 * yr + xr + 0.1
    gx = c * dU_dxr + s * dU_dyr
    gy = -s * dU_dxr + c * dU_dyr
    return np.stack([gx, gy], axis=-1)


# ============================================================================
# Boltzmann sampler (stand-in for your MD / MCMC data)
# ============================================================================
def boltzmann_samples(n_samples=N_SAMPLES, n_grid=N_SAMPLE_GRID, beta=BETA, seed=SEED):
    """Importance-sample the analytic Boltzmann density on a grid -> ``(N, 2)``.

    In real use ``samples`` comes from MD, MCMC, or replica exchange; here we
    draw from exp(-beta U) so the library stays usage-focused.
    """
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
    dx = xs[1] - xs[0]
    dy = ys[1] - ys[0]
    sx = xs[cols] + (rng.random(n_samples) - 0.5) * dx
    sy = ys[rows] + (rng.random(n_samples) - 0.5) * dy
    return np.stack([sx, sy], axis=1)


# ============================================================================
# Converged 2D PDE baseline (weighted Jacobi)
# ============================================================================
def jacobi_committor_2d(
    n_grid=PDE_N_GRID,
    beta=BETA,
    radius=STATE_RADIUS,
    max_iters=PDE_MAX_ITERS,
    tol=PDE_TOL,
    report_every=20_000,
):
    """Weighted Jacobi solve of the committor PDE on a regular grid.

    Returns (q, X, Y, iters, final_delta). Raises if the iteration budget
    is exhausted before reaching ``tol`` (the baseline must be converged
    for the cross-check / reference rate to be meaningful).
    """
    xs = np.linspace(*DOMAIN, n_grid)
    ys = np.linspace(*DOMAIN, n_grid)
    X, Y = np.meshgrid(xs, ys, indexing="xy")
    rho = np.exp(-beta * potential(X, Y))
    Wx = 0.5 * (rho[:, :-1] + rho[:, 1:])
    Wy = 0.5 * (rho[:-1, :] + rho[1:, :])
    coef = np.zeros_like(rho)
    coef[:, :-1] += Wx
    coef[:, 1:] += Wx
    coef[:-1, :] += Wy
    coef[1:, :] += Wy
    in_A = ((X - CENTRE_A[0]) ** 2 + (Y - CENTRE_A[1]) ** 2) < radius**2
    in_B = ((X - CENTRE_B[0]) ** 2 + (Y - CENTRE_B[1]) ** 2) < radius**2
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
        q_new[in_A] = 0.0
        q_new[in_B] = 1.0
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
# Shared PRL plotting style (matches paper figure 1)
# ============================================================================
DOUBLE_COL = 7.0
HIGHLIGHT = "#d62728"


def apply_paper_style():
    """Apply the paper figure-1 matplotlib rcParams (imports matplotlib lazily)."""
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
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
        }
    )
