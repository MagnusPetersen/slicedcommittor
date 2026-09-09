"""Shared test helpers: samplers with known answers, and the named tolerance.

* ``two_basin_samples``: a Gaussian cloud with two spherical basins (geometry only).
* ``wolfe_quapp_samples``: exact Boltzmann samples of the rotated Wolfe-Quapp
  potential at beta = 1 (the paper's Figure 1 system); ported verbatim from the
  research prototype so the frozen references in ``golden/`` reproduce bit for bit.
* ``double_well_samples`` + ``double_well_pde``: a 2D double well with a Jacobi
  finite-difference committor as the reference.
"""

import jax.numpy as jnp
import numpy as np

# "the solver satisfied its constraint" tolerance, shared across files
TOL_SOLVER = 1e-6


def two_basin_samples(n=500, *, dim=2, seed=0, radius=0.7, sep=2.0):
    """Standard-normal cloud with disjoint spherical basins at ``-sep e_0`` / ``+sep e_0``."""
    rng = np.random.default_rng(seed)
    samples = jnp.asarray(rng.standard_normal((n, dim)))
    center = jnp.zeros(dim).at[0].set(sep)
    in_A = jnp.linalg.norm(samples + center, axis=1) < radius
    in_B = jnp.linalg.norm(samples - center, axis=1) < radius
    in_A = in_A & ~in_B
    return samples, in_A, in_B


# ---------------------------------------------------------------------------
# Rotated Wolfe-Quapp (Bonati et al. PNAS 2019), beta = 1
# ---------------------------------------------------------------------------
WQ_CENTER_A = np.array([-1.717, 0.783])
WQ_CENTER_B = np.array([1.676, -0.813])
WQ_RADIUS = 0.3
WQ_DOMAIN = ((-2.5, 2.5), (-2.5, 2.5))


def rotated_wolfe_quapp(x):
    """Wolfe-Quapp rotated by ``theta = -0.15 pi``; ``x`` is ``(..., 2)``."""
    xx, yy = x[..., 0], x[..., 1]
    th = -0.15 * np.pi
    c, s = np.cos(th), np.sin(th)
    xr = xx * c - yy * s
    yr = xx * s + yy * c
    return xr**4 + yr**4 - 2.0 * xr**2 - 4.0 * yr**2 + xr * yr + 0.3 * xr + 0.1 * yr


def wolfe_quapp_samples(n, seed=0, *, beta=1.0, n_grid=400):
    """Exact i.i.d. Boltzmann samples via a fine grid plus in-cell jitter.

    Returns ``(X, in_A, in_B)`` as numpy arrays; the basins are discs of radius
    ``WQ_RADIUS`` around the two minima. Same RNG stream as the research
    prototype, so ``golden/halfset_reference.npz`` pins this generator.
    """
    rng = np.random.default_rng(seed)
    (x0, x1), (y0, y1) = WQ_DOMAIN
    xs = np.linspace(x0, x1, n_grid)
    ys = np.linspace(y0, y1, n_grid)
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    pts = np.stack([X.ravel(), Y.ravel()], axis=-1)
    logp = -beta * rotated_wolfe_quapp(pts)
    logp -= logp.max()
    p = np.exp(logp)
    p /= p.sum()
    idx = rng.choice(pts.shape[0], size=n, p=p)
    out = pts[idx].copy()
    dx, dy = xs[1] - xs[0], ys[1] - ys[0]
    out[:, 0] += rng.uniform(-dx / 2, dx / 2, n)
    out[:, 1] += rng.uniform(-dy / 2, dy / 2, n)
    in_A = np.sum((out - WQ_CENTER_A) ** 2, axis=-1) < WQ_RADIUS**2
    in_B = np.sum((out - WQ_CENTER_B) ** 2, axis=-1) < WQ_RADIUS**2
    return out, in_A, in_B


# ---------------------------------------------------------------------------
# 2D double well with a finite-difference committor reference
# ---------------------------------------------------------------------------
DW_CENTER_A = (-1.0, 0.0)
DW_CENTER_B = (1.0, 0.0)
DW_RADIUS = 0.25
DW_BOX = (-1.8, 1.8, -1.5, 1.5)


def double_well_potential(x, y):
    return (x**2 - 1.0) ** 2 + 0.5 * y**2


def double_well_pde(n_grid=64, *, tol=1e-7, max_iters=30_000):
    """Jacobi solve of ``div(exp(-V) grad q) = 0`` with q = 0 on A, q = 1 on B.

    Returns ``(q, X, Y)`` on the regular grid (zero-flux outer boundary).
    """
    x_min, x_max, y_min, y_max = DW_BOX
    xs = np.linspace(x_min, x_max, n_grid)
    ys = np.linspace(y_min, y_max, n_grid)
    X, Y = np.meshgrid(xs, ys, indexing="xy")
    rho = np.exp(-double_well_potential(X, Y))
    Wx = 0.5 * (rho[:, :-1] + rho[:, 1:])
    Wy = 0.5 * (rho[:-1, :] + rho[1:, :])
    coef = np.zeros_like(rho)
    coef[:, :-1] += Wx
    coef[:, 1:] += Wx
    coef[:-1, :] += Wy
    coef[1:, :] += Wy
    in_A = ((X - DW_CENTER_A[0]) ** 2 + (Y - DW_CENTER_A[1]) ** 2) < DW_RADIUS**2
    in_B = ((X - DW_CENTER_B[0]) ** 2 + (Y - DW_CENTER_B[1]) ** 2) < DW_RADIUS**2
    q = np.where(in_B, 1.0, 0.0)
    inv_coef = np.where(coef > 0, 1.0 / np.maximum(coef, 1e-30), 0.0)
    for _ in range(max_iters):
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
            break
    return q, X, Y


def double_well_samples(n=2000, seed=0, n_grid=200):
    """Boltzmann samples of the 2D double well with in-cell jitter; ``(samples, in_A, in_B)``."""
    rng = np.random.default_rng(seed)
    x_min, x_max, y_min, y_max = DW_BOX
    xs = np.linspace(x_min, x_max, n_grid)
    ys = np.linspace(y_min, y_max, n_grid)
    X, Y = np.meshgrid(xs, ys, indexing="xy")
    p = np.exp(-double_well_potential(X, Y))
    p /= p.sum()
    idx = rng.choice(p.size, size=n, replace=True, p=p.ravel())
    dx, dy = xs[1] - xs[0], ys[1] - ys[0]
    rows, cols = np.divmod(idx, n_grid)
    sx = xs[cols] + (rng.random(n) - 0.5) * dx
    sy = ys[rows] + (rng.random(n) - 0.5) * dy
    samples = np.stack([sx, sy], axis=1)
    in_A = (samples[:, 0] - DW_CENTER_A[0]) ** 2 + (
        samples[:, 1] - DW_CENTER_A[1]
    ) ** 2 < DW_RADIUS**2
    in_B = (samples[:, 0] - DW_CENTER_B[0]) ** 2 + (
        samples[:, 1] - DW_CENTER_B[1]
    ) ** 2 < DW_RADIUS**2
    return samples, in_A, in_B


def interpolate_grid(q_grid, X, Y, points):
    """Bilinear interpolation of a gridded field at scattered points."""
    xs, ys = X[0, :], Y[:, 0]
    fx = (points[:, 0] - xs[0]) / (xs[1] - xs[0])
    fy = (points[:, 1] - ys[0]) / (ys[1] - ys[0])
    i0 = np.clip(np.floor(fx).astype(int), 0, len(xs) - 2)
    j0 = np.clip(np.floor(fy).astype(int), 0, len(ys) - 2)
    ax, ay = fx - i0, fy - j0
    return (
        (1 - ax) * (1 - ay) * q_grid[j0, i0]
        + ax * (1 - ay) * q_grid[j0, i0 + 1]
        + (1 - ax) * ay * q_grid[j0 + 1, i0]
        + ax * ay * q_grid[j0 + 1, i0 + 1]
    )


def double_well_eval_points(nx=25, ny=15, pad=0.30):
    """A coarse interior grid of the double well, excluding the basins."""
    EX, EY = np.meshgrid(np.linspace(-1.4, 1.4, nx), np.linspace(-0.9, 0.9, ny), indexing="xy")
    pts = np.stack([EX.ravel(), EY.ravel()], axis=1)
    keep = ~(
        ((pts[:, 0] - DW_CENTER_A[0]) ** 2 + pts[:, 1] ** 2 < pad**2)
        | ((pts[:, 0] - DW_CENTER_B[0]) ** 2 + pts[:, 1] ** 2 < pad**2)
    )
    return pts[keep]
