"""Validation against an inlined 2D Jacobi PDE committor on a double-well.

The PDE solver is intentionally kept inside the test file: it is a
reference for validation, not a public library feature. The math is the
same finite-difference Jacobi sweep used in the main-monorepo baseline
(``src/core/baseline.py``); copied here so the library has no test-time
dependency on the upstream research repo.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

import sliced_committor as sc

# ---------------------------------------------------------------------------
# Test-internal reference: 2D Jacobi committor solver.
# ---------------------------------------------------------------------------


def _double_well_potential(x, y):
    return (x**2 - 1.0) ** 2 + 0.5 * y**2


def _solve_committor_2d_jacobi(
    n_grid: int = 64,
    x_min: float = -1.8,
    x_max: float = 1.8,
    y_min: float = -1.5,
    y_max: float = 1.5,
    centre_A=(-1.0, 0.0),
    centre_B=(1.0, 0.0),
    radius: float = 0.25,
    beta: float = 1.0,
    tol: float = 1e-7,
    max_iters: int = 30_000,
):
    """Solve ``div(e^{-beta V} grad q) = 0`` with q=0 on A, q=1 on B.

    Returns ``(q, X, Y)`` on the regular grid.
    """
    xs = np.linspace(x_min, x_max, n_grid)
    ys = np.linspace(y_min, y_max, n_grid)
    xs[1] - xs[0]
    ys[1] - ys[0]
    X, Y = np.meshgrid(xs, ys, indexing="xy")

    V = _double_well_potential(X, Y)
    rho = np.exp(-beta * V)

    # Edge weights (arithmetic mean of density on neighbouring cells).
    Wx = 0.5 * (rho[:, :-1] + rho[:, 1:])  # horizontal edges, shape (n, n-1)
    Wy = 0.5 * (rho[:-1, :] + rho[1:, :])  # vertical edges, shape (n-1, n)

    # Each cell's coefficient is the sum of all incident edge weights. The
    # boundary cells therefore only sum the edges that touch the interior,
    # which yields a zero-Neumann (no-flux) outer boundary automatically.
    coef = np.zeros_like(rho)
    coef[:, :-1] += Wx  # left edge contribution
    coef[:, 1:] += Wx  # right edge contribution
    coef[:-1, :] += Wy  # bottom edge contribution
    coef[1:, :] += Wy  # top edge contribution

    # Boundary masks.
    in_A_mask = ((X - centre_A[0]) ** 2 + (Y - centre_A[1]) ** 2) < radius**2
    in_B_mask = ((X - centre_B[0]) ** 2 + (Y - centre_B[1]) ** 2) < radius**2

    q = np.where(in_B_mask, 1.0, 0.0)
    q[in_A_mask] = 0.0
    q[in_B_mask] = 1.0

    for it in range(max_iters):
        q_new = q.copy()
        flux = np.zeros_like(q)
        flux[:, 1:] += Wx * q[:, :-1]
        flux[:, :-1] += Wx * q[:, 1:]
        flux[1:, :] += Wy * q[:-1, :]
        flux[:-1, :] += Wy * q[1:, :]

        inv_coef = np.where(coef > 0, 1.0 / np.maximum(coef, 1e-30), 0.0)
        q_new = flux * inv_coef
        # Domain-boundary Neumann via copying nearest interior already
        # handled because edge weights are zero at the outer grid frame.
        q_new[in_A_mask] = 0.0
        q_new[in_B_mask] = 1.0

        delta = float(np.max(np.abs(q_new - q)))
        q = q_new
        if delta < tol:
            break

    return q, X, Y, in_A_mask, in_B_mask


# ---------------------------------------------------------------------------
# Sampling for the sliced solver.
# ---------------------------------------------------------------------------


def _equilibrium_samples(
    n_samples: int = 2000,
    seed: int = 0,
    n_grid: int = 200,
    x_min=-1.8,
    x_max=1.8,
    y_min=-1.5,
    y_max=1.5,
    centre_A=(-1.0, 0.0),
    centre_B=(1.0, 0.0),
    radius=0.25,
):
    rng = np.random.default_rng(seed)
    xs = np.linspace(x_min, x_max, n_grid)
    ys = np.linspace(y_min, y_max, n_grid)
    X, Y = np.meshgrid(xs, ys, indexing="xy")
    V = _double_well_potential(X, Y)
    p = np.exp(-V)
    p /= p.sum()
    flat = p.ravel()
    idx = rng.choice(flat.size, size=n_samples, replace=True, p=flat)
    # Add a small jitter within each cell.
    dx = xs[1] - xs[0]
    dy = ys[1] - ys[0]
    rows, cols = np.divmod(idx, n_grid)
    sx = xs[cols] + (rng.random(n_samples) - 0.5) * dx
    sy = ys[rows] + (rng.random(n_samples) - 0.5) * dy
    samples = np.stack([sx, sy], axis=1)
    in_A = (samples[:, 0] - centre_A[0]) ** 2 + (samples[:, 1] - centre_A[1]) ** 2 < radius**2
    in_B = (samples[:, 0] - centre_B[0]) ** 2 + (samples[:, 1] - centre_B[1]) ** 2 < radius**2
    return samples, in_A, in_B


@pytest.fixture(scope="module")
def pde_reference():
    return _solve_committor_2d_jacobi(n_grid=64)


@pytest.fixture(scope="module")
def sliced_samples():
    return _equilibrium_samples(n_samples=2500)


def _interpolate_pde(q_grid, X, Y, points):
    """Bilinear interpolation of the gridded committor at scattered points."""
    xs = X[0, :]
    ys = Y[:, 0]
    dx = xs[1] - xs[0]
    dy = ys[1] - ys[0]
    fx = (points[:, 0] - xs[0]) / dx
    fy = (points[:, 1] - ys[0]) / dy
    i0 = np.clip(np.floor(fx).astype(int), 0, len(xs) - 2)
    j0 = np.clip(np.floor(fy).astype(int), 0, len(ys) - 2)
    ax = fx - i0
    ay = fy - j0
    q00 = q_grid[j0, i0]
    q10 = q_grid[j0, i0 + 1]
    q01 = q_grid[j0 + 1, i0]
    q11 = q_grid[j0 + 1, i0 + 1]
    return (1 - ax) * (1 - ay) * q00 + ax * (1 - ay) * q10 + (1 - ax) * ay * q01 + ax * ay * q11


def test_sliced_matches_pde_double_well(pde_reference, sliced_samples):
    """End-to-end: sliced EBMC committor agrees with the 2D PDE solution."""
    q_grid, X, Y, in_A_mask, in_B_mask = pde_reference
    samples, in_A, in_B = sliced_samples

    result = sc.compute_sliced_committor(
        jnp.asarray(samples),
        in_A=jnp.asarray(in_A),
        in_B=jnp.asarray(in_B),
        n_directions=256,
        n_bins=120,
        seed=42,
    )
    ebmc = sc.compute_enriched_basin_moment_weights(result, jnp.asarray(samples))

    # Evaluate sliced + PDE on a coarse interior grid (avoid the basins).
    eval_xs = np.linspace(-1.4, 1.4, 25)
    eval_ys = np.linspace(-0.9, 0.9, 15)
    EX, EY = np.meshgrid(eval_xs, eval_ys, indexing="xy")
    pts = np.stack([EX.ravel(), EY.ravel()], axis=1)

    interior_mask = ~(
        ((pts[:, 0] - (-1.0)) ** 2 + pts[:, 1] ** 2 < 0.30**2)
        | ((pts[:, 0] - 1.0) ** 2 + pts[:, 1] ** 2 < 0.30**2)
    )
    pts_int = pts[interior_mask]

    q_pred = np.asarray(sc.build_committor(result, ebmc)(jnp.asarray(pts_int)))
    q_true = _interpolate_pde(q_grid, X, Y, pts_int)

    rmse = float(np.sqrt(np.mean((q_pred - q_true) ** 2)))
    # 2D double-well with 256 directions converges very fast; tolerance is
    # set well above the float-precision noise floor so the test is stable
    # across JAX minor versions.
    assert rmse < 0.10, f"RMSE={rmse:.3f}"
