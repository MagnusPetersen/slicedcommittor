"""Shared test helpers: samplers with known answers, and the named tolerance.

* ``two_basin_samples``: a Gaussian cloud with two spherical basins (geometry only).
* ``wolfe_quapp_samples``: exact Boltzmann samples of the rotated Wolfe-Quapp
  potential at beta = 1 (the paper's Figure 1 system); ported verbatim from the
  research prototype so the frozen references in ``golden/`` reproduce bit for bit.
* ``double_well_samples`` + ``double_well_pde``: a 2D double well with a Jacobi
  finite-difference committor as the reference.
* ``ou_trajectory`` / ``windowed_ou`` / ``overdamped_double_well_trajectory``:
  time series with a known diffusion coefficient, for the rate estimators.
* ``double_well_1d_profiles``: the analytic ``{pi(q), D_q(q)}`` of the exact
  committor of a 1D double well, whose flux is constant by construction.
* ``umbrella_double_well_dataset``: umbrella windows on a 2D double well with a
  known mobility, as a ``USDataset`` (``umbrella_double_well`` is its generator).
* ``unit_directions``, ``spd_pair``, ``write_colvar``: small fixtures shared
  by several test files.
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


# ---------------------------------------------------------------------------
# Time series with a known diffusion coefficient
# ---------------------------------------------------------------------------
def ou_trajectory(n, *, dt=0.01, k=1.0, D=0.05, seed=0, rng=None):
    """Euler-Maruyama Ornstein-Uhlenbeck ``dx = -k x dt + sqrt(2 D) dW`` from 0; ``(n,)``."""
    rng = np.random.default_rng(seed) if rng is None else rng
    x = np.empty(n)
    xi = 0.0
    noise = np.sqrt(2.0 * D * dt)
    for t in range(n):
        xi = xi - k * xi * dt + noise * rng.standard_normal()
        x[t] = xi
    return x


def windowed_ou(n_win=6, n_per=20000, *, dt=0.01, k=1.0, D=0.05, seed=3, centers=None):
    """``n_win`` independent confined OU runs laid end to end; ``(x, window_ids)``.

    ``centers[w]`` shifts window ``w`` (all at 0 by default), for tests of a
    selection by window mean.
    """
    rng = np.random.default_rng(seed)
    xs, wid = [], []
    for w in range(n_win):
        x = ou_trajectory(n_per, dt=dt, k=k, D=D, rng=rng)
        xs.append(x + (0.0 if centers is None else float(centers[w])))
        wid.append(np.full(n_per, w))
    return np.concatenate(xs), np.concatenate(wid)


def overdamped_double_well_trajectory(T=60000, *, dt=0.01, D0=0.05, seed=1):
    """Overdamped Langevin on the 2D double well with mobility ``D0``; ``(T, 2)``."""
    rng = np.random.default_rng(seed)
    x = np.array([-1.0, 0.0])
    out = np.empty((T, 2))
    noise = np.sqrt(2.0 * D0 * dt)
    for t in range(T):
        grad = np.array([4.0 * x[0] * (x[0] ** 2 - 1.0), x[1]])
        x = x - D0 * grad * dt + noise * rng.standard_normal(2)
        out[t] = x
    return out


def double_well_1d_profiles(n_bins=2000, *, D=0.05):
    """Analytic ``{pi(q), D_q(q)}`` of 1D diffusion in ``V = (x^2 - 1)^2`` between the minima.

    With the minima as the states the exact committor is ``q(x) = int_{-1}^x
    e^V / int_{-1}^1 e^V``, so ``pi(q) = Z_q e^{-2V(x(q))} / Z_pi`` (normalised
    on the transition region) and ``D_q(q) = D q'(x(q))^2``, whose product is
    the constant ``D / (Z_q Z_pi)``. Returns ``(pi, D_q, rho_A, rho_B,
    nu_exact)`` on a midpoint grid of ``n_bins`` bins; ``rho_A`` is the same
    discrete sum the library uses, so every reduction must reproduce
    ``nu_exact / rho`` to rounding.
    """
    from sliced_committor import Profile

    x = np.linspace(-1.0, 1.0, 200_001)
    dx = x[1] - x[0]
    V = (x**2 - 1.0) ** 2
    eV = np.exp(V)
    cum = np.concatenate([[0.0], np.cumsum(0.5 * (eV[1:] + eV[:-1]) * dx)])
    Z_q = float(cum[-1])
    emV = np.exp(-V)
    Z_pi = float(np.sum(0.5 * (emV[1:] + emV[:-1]) * dx))
    centers = (np.arange(n_bins) + 0.5) / n_bins
    x_c = np.interp(centers, cum / Z_q, x)
    V_c = (x_c**2 - 1.0) ** 2
    pi = Z_q * np.exp(-2.0 * V_c) / Z_pi
    D_q = D * np.exp(2.0 * V_c) / Z_q**2
    rho_A = float(np.sum((1.0 - centers) * pi) / n_bins)
    ones = np.ones(n_bins, dtype=np.int64)
    return (
        Profile(centers, pi, ones),
        Profile(centers, D_q, ones),
        rho_A,
        1.0 - rho_A,
        D / (Z_q * Z_pi),
    )


def umbrella_double_well(n_windows=8, n_per=20000, *, dt=0.01, D0=0.05, kappa=15.0, seed=0):
    """Overdamped Langevin umbrella windows on ``V = (x^2 - 1)^2 + y^2 / 2``, restrained in ``x``.

    Returns ``(x, y, window_ids, centers, kappa)`` in per-window time order,
    all windows integrated at once. The restraint ``0.5 kappa (x - c_k)^2``
    acts on ``x`` only and ``y`` is a decoupled nuisance coordinate, so the
    exact committor is a function of ``x`` alone and the mobility ``D0`` is
    the diffusion coefficient of both coordinates.
    """
    rng = np.random.default_rng(seed)
    centers = np.linspace(-1.4, 1.4, n_windows)
    noise = np.sqrt(2.0 * D0 * dt)
    x = centers.copy()
    y = np.zeros(n_windows)
    xs = np.empty((n_windows, n_per))
    ys = np.empty((n_windows, n_per))
    for t in range(n_per):
        fx = 4.0 * x * (x * x - 1.0) + kappa * (x - centers)
        x = x - D0 * fx * dt + noise * rng.standard_normal(n_windows)
        y = y - D0 * y * dt + noise * rng.standard_normal(n_windows)
        xs[:, t], ys[:, t] = x, y
    window_ids = np.repeat(np.arange(n_windows), n_per)
    return xs.ravel(), ys.ravel(), window_ids, centers, kappa


def umbrella_double_well_dataset(
    n_windows=8, n_per=20000, *, dt=0.01, D0=0.05, kappa=15.0, seed=0, basin_edge=0.9
):
    """:func:`umbrella_double_well` wrapped as a ``USDataset`` with basins ``|x| > basin_edge``."""
    from sliced_committor.umbrella import USDataset

    x, y, wid, centers, kappa = umbrella_double_well(
        n_windows, n_per, dt=dt, D0=D0, kappa=kappa, seed=seed
    )
    return USDataset(
        features=np.stack([x, y], axis=1),
        cvs=x[:, None],
        window_ids=wid,
        window_centers=centers[:, None],
        window_kappa=np.full((centers.size, 1), kappa),
        beta=1.0,
        dt=dt,
        in_A=x < -basin_edge,
        in_B=x > basin_edge,
        cv_periodic=(None,),
        meta={},
    )


def unit_directions(dim, m, seed=11):
    """``(m, dim)`` random unit vectors (numpy RNG, so independent of the solver's draw)."""
    rng = np.random.default_rng(seed)
    raw = rng.standard_normal((m, dim))
    return jnp.asarray(raw / np.linalg.norm(raw, axis=1, keepdims=True))


def spd_pair(M, seed):
    """Two noisy SPD matrices sharing a base: a half-Gram pair for the filter tests."""
    rng = np.random.default_rng(seed)
    B = rng.standard_normal((M, M))
    base = B @ B.T / M
    E1, E2 = rng.standard_normal((M, M)), rng.standard_normal((M, M))
    return base + 0.05 * (E1 @ E1.T) / M, base + 0.05 * (E2 @ E2.T) / M


def write_colvar(path, time, columns, fields, periodic=None):
    """Write a minimal PLUMED COLVAR file; ``columns`` maps field name -> array."""
    lines = [f"#! FIELDS {' '.join(fields)}"]
    for name, (lo, hi) in (periodic or {}).items():
        lines.append(f"#! SET min_{name} {lo}")
        lines.append(f"#! SET max_{name} {hi}")
    data = np.column_stack([time] + [columns[f] for f in fields if f != "time"])
    body = "\n".join(" ".join(f"{v:.6f}" for v in row) for row in data)
    path.write_text("\n".join(lines) + "\n" + body + "\n")
    return path
