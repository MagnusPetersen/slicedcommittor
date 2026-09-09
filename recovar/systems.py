"""
Test systems: potentials, states, exact Boltzmann sampling, Langevin sampling,
and a sparse finite-difference committor reference.

Self-contained numpy/scipy port of the project's potentials.py + baseline.py.

Note on temperature: all validated prototype results use beta=1.0 for the
Wolfe-Quapp systems. At beta=2.0 the barrier is unsampled (only 0.02% of
frames have q in (0.1, 0.9)), so every Dirichlet-energy quantity is
meaningless and the RMSE is trivially small for any method; see
_reference/REPORT.md 'Setup'. The config factories keep the historical
default beta=2.0; callers pass beta=1.0 explicitly.
"""
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from dataclasses import dataclass
from typing import Callable, Tuple, Optional


# ---------------------------------------------------------------------------
# Potentials
# ---------------------------------------------------------------------------

def rotated_wolfe_quapp(x):
    """Wolfe-Quapp rotated by theta = -0.15 pi (Bonati et al. PNAS 2019)."""
    xx, yy = x[..., 0], x[..., 1]
    th = -0.15 * np.pi
    c, s = np.cos(th), np.sin(th)
    xr = xx * c - yy * s
    yr = xx * s + yy * c
    return (xr**4 + yr**4 - 2.0 * xr**2 - 4.0 * yr**2
            + xr * yr + 0.3 * xr + 0.1 * yr)


def double_well(x, barrier=2.0, well=1.0, sep=2.0):
    scale = sep / 2.0
    V = barrier * ((x[..., 0] / scale)**2 - 1)**2
    if x.shape[-1] > 1:
        V = V + well * np.sum(x[..., 1:]**2, axis=-1)
    return V


def rotated_wolfe_quapp_nd(x, harmonic_k=1.0):
    """2D WQ in the leading plane + harmonic nuisance dimensions."""
    V = rotated_wolfe_quapp(x[..., :2])
    if x.shape[-1] > 2:
        V = V + 0.5 * harmonic_k * np.sum(x[..., 2:]**2, axis=-1)
    return V


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class Config:
    potential: Callable
    domain: Tuple[Tuple[float, float], ...]
    center_A: np.ndarray
    center_B: np.ndarray
    radius: float
    beta: float
    name: str = ""
    state_dims: int = 2       # states defined on the first `state_dims` coords

    @property
    def dim(self):
        return len(self.domain)

    def in_A(self, x):
        d = x[..., :self.state_dims] - self.center_A[:self.state_dims]
        return np.sum(d**2, axis=-1) < self.radius**2

    def in_B(self, x):
        d = x[..., :self.state_dims] - self.center_B[:self.state_dims]
        return np.sum(d**2, axis=-1) < self.radius**2


def wolfe_quapp_config(beta=2.0, dim=2, harmonic_k=1.0, radius=0.3):
    cA = np.array([-1.717, 0.783])
    cB = np.array([1.676, -0.813])
    dom = ((-2.5, 2.5), (-2.5, 2.5)) + tuple((-3.0, 3.0) for _ in range(dim - 2))
    if dim == 2:
        pot = rotated_wolfe_quapp
    else:
        pot = lambda x: rotated_wolfe_quapp_nd(x, harmonic_k)
    return Config(potential=pot, domain=dom,
                  center_A=np.concatenate([cA, np.zeros(dim - 2)]),
                  center_B=np.concatenate([cB, np.zeros(dim - 2)]),
                  radius=radius, beta=beta,
                  name=f"Rotated Wolfe-Quapp {dim}D", state_dims=2)


def double_well_config(beta=2.0, dim=2):
    dom = ((-1.8, 1.8),) + tuple((-1.5, 1.5) for _ in range(dim - 1))
    return Config(potential=double_well, domain=dom,
                  center_A=np.concatenate([[-1.0], np.zeros(dim - 1)]),
                  center_B=np.concatenate([[1.0], np.zeros(dim - 1)]),
                  radius=0.3, beta=beta, name=f"Double Well {dim}D",
                  state_dims=dim)


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------

def grid_boltzmann_samples(cfg, n_samples, n_grid=400, rng=None,
                           harmonic_k=1.0):
    """Exact i.i.d. Boltzmann samples via a fine 2D grid (+ Gaussian nuisance).

    Adds uniform jitter within each grid cell so the samples are continuous.
    """
    rng = np.random.default_rng(rng)
    (x0, x1), (y0, y1) = cfg.domain[0], cfg.domain[1]
    xs = np.linspace(x0, x1, n_grid)
    ys = np.linspace(y0, y1, n_grid)
    X, Y = np.meshgrid(xs, ys, indexing='ij')
    pts = np.stack([X.ravel(), Y.ravel()], axis=-1)
    if cfg.dim == 2:
        V = cfg.potential(pts)
    else:
        pad = np.concatenate([pts, np.zeros((pts.shape[0], cfg.dim - 2))], axis=1)
        V = cfg.potential(pad)
    logp = -cfg.beta * V
    logp -= logp.max()
    p = np.exp(logp)
    p /= p.sum()
    idx = rng.choice(pts.shape[0], size=n_samples, p=p)
    out = pts[idx].copy()
    dx = xs[1] - xs[0]
    dy = ys[1] - ys[0]
    out[:, 0] += rng.uniform(-dx / 2, dx / 2, n_samples)
    out[:, 1] += rng.uniform(-dy / 2, dy / 2, n_samples)
    if cfg.dim == 2:
        return out
    sigma = 1.0 / np.sqrt(cfg.beta * harmonic_k)
    nuis = sigma * rng.standard_normal((n_samples, cfg.dim - 2))
    return np.concatenate([out, nuis], axis=1)


def grad_rotated_wolfe_quapp(x):
    """Analytic gradient of the rotated Wolfe-Quapp potential."""
    th = -0.15 * np.pi
    c, s = np.cos(th), np.sin(th)
    xx, yy = x[..., 0], x[..., 1]
    xr = xx * c - yy * s
    yr = xx * s + yy * c
    dxr = 4 * xr**3 - 4 * xr + yr + 0.3
    dyr = 4 * yr**3 - 8 * yr + xr + 0.1
    g = np.empty_like(x)
    g[..., 0] = dxr * c + dyr * s
    g[..., 1] = -dxr * s + dyr * c
    return g


def _grad_numeric(pot, x, h=1e-4):
    g = np.zeros_like(x)
    for k in range(x.shape[-1]):
        e = np.zeros(x.shape[-1])
        e[k] = h
        g[:, k] = (pot(x + e) - pot(x - e)) / (2 * h)
    return g


def langevin_samples(cfg, n_steps, dt=2e-4, D=1.0, n_walkers=8, rng=None,
                     burn=20000, thin=1, x0=None, grad=None):
    """Overdamped Langevin trajectory: dx = -D beta grad V dt + sqrt(2 D dt) dW.

    Returns (frames, walker_id) with frames in trajectory order per walker,
    so contiguous blocks are genuinely time-correlated.
    """
    rng = np.random.default_rng(rng)
    d = cfg.dim
    if x0 is None:
        x = np.tile((cfg.center_A + cfg.center_B) / 2.0, (n_walkers, 1))
        x = x + 0.1 * rng.standard_normal((n_walkers, d))
    else:
        x = x0.copy()
    sq = np.sqrt(2.0 * D * dt)
    lo = np.array([b[0] for b in cfg.domain])
    hi = np.array([b[1] for b in cfg.domain])
    gfun = grad if grad is not None else (lambda z: _grad_numeric(cfg.potential, z))

    for _ in range(burn):
        g = gfun(x)
        x = x - D * cfg.beta * g * dt + sq * rng.standard_normal(x.shape)
        x = np.clip(x, lo, hi)

    keep = n_steps // thin
    out = np.empty((keep, n_walkers, d))
    for i in range(n_steps):
        g = gfun(x)
        x = x - D * cfg.beta * g * dt + sq * rng.standard_normal(x.shape)
        x = np.clip(x, lo, hi)
        if i % thin == 0 and i // thin < keep:
            out[i // thin] = x
    frames = out.transpose(1, 0, 2).reshape(-1, d)          # walker-major
    wid = np.repeat(np.arange(n_walkers), keep)
    return frames, wid


# ---------------------------------------------------------------------------
# Finite-difference committor reference (sparse direct solve, 2D)
# ---------------------------------------------------------------------------

def pde_committor_2d(cfg, n_grid=301, domain=None):
    """Solve div(rho D grad q)=0, q|A=0, q|B=1, Neumann on the box edge.

    Sparse direct solve (no Jacobi iteration): exact to machine precision.
    Returns q (n_grid,n_grid), xs, ys.
    """
    dom = domain if domain is not None else cfg.domain[:2]
    (x0, x1), (y0, y1) = dom
    xs = np.linspace(x0, x1, n_grid)
    ys = np.linspace(y0, y1, n_grid)
    hx = xs[1] - xs[0]
    hy = ys[1] - ys[0]
    X, Y = np.meshgrid(xs, ys, indexing='ij')
    pts = np.stack([X, Y], axis=-1).reshape(-1, 2)
    V = cfg.potential(pts).reshape(n_grid, n_grid)
    V = V - V.min()
    rho = np.exp(-cfg.beta * V)

    inA = cfg.in_A(pts).reshape(n_grid, n_grid)
    inB = cfg.in_B(pts).reshape(n_grid, n_grid)
    fixed = inA | inB

    N = n_grid * n_grid
    idx = np.arange(N).reshape(n_grid, n_grid)

    rows, cols, vals = [], [], []
    rhs = np.zeros(N)

    # arithmetic-mean face conductances (the reference prototype's validated
    # choice; its comment said "harmonic-mean" but the code averages)
    def face(a, b):
        return 0.5 * (a + b)

    cxp = np.zeros_like(rho); cxm = np.zeros_like(rho)
    cyp = np.zeros_like(rho); cym = np.zeros_like(rho)
    cxp[:-1, :] = face(rho[:-1, :], rho[1:, :]) / hx**2
    cxm[1:, :] = face(rho[1:, :], rho[:-1, :]) / hx**2
    cyp[:, :-1] = face(rho[:, :-1], rho[:, 1:]) / hy**2
    cym[:, 1:] = face(rho[:, 1:], rho[:, :-1]) / hy**2

    ii = idx.ravel()
    fx = fixed.ravel()

    # Dirichlet rows
    rows.append(ii[fx]); cols.append(ii[fx]); vals.append(np.ones(fx.sum()))
    rhs[ii[fx]] = inB.ravel()[fx].astype(float)

    free = ~fx
    diag = -(cxp + cxm + cyp + cym).ravel()
    rows.append(ii[free]); cols.append(ii[free]); vals.append(diag[free])

    for (c, shift, axis) in ((cxp, -1, 0), (cxm, 1, 0), (cyp, -1, 1), (cym, 1, 1)):
        nb = np.roll(idx, shift, axis=axis)
        cf = c.ravel()
        m = free & (cf != 0)
        rows.append(ii[m]); cols.append(nb.ravel()[m]); vals.append(cf[m])

    A = sp.csr_matrix((np.concatenate(vals),
                       (np.concatenate(rows), np.concatenate(cols))),
                      shape=(N, N))
    q = spla.spsolve(A.tocsc(), rhs)
    return q.reshape(n_grid, n_grid), xs, ys


def bilinear_interp(q_grid, xs, ys, pts):
    """Interpolate a 2D grid field at arbitrary points (clamped at edges)."""
    x = np.clip(pts[:, 0], xs[0], xs[-1])
    y = np.clip(pts[:, 1], ys[0], ys[-1])
    hx = xs[1] - xs[0]; hy = ys[1] - ys[0]
    i = np.clip(((x - xs[0]) / hx).astype(int), 0, len(xs) - 2)
    j = np.clip(((y - ys[0]) / hy).astype(int), 0, len(ys) - 2)
    tx = (x - xs[i]) / hx
    ty = (y - ys[j]) / hy
    return ((1 - tx) * (1 - ty) * q_grid[i, j]
            + tx * (1 - ty) * q_grid[i + 1, j]
            + (1 - tx) * ty * q_grid[i, j + 1]
            + tx * ty * q_grid[i + 1, j + 1])


def pde_reactive_flux(cfg, q_grid, xs, ys):
    """nu_AB = D[q] = int rho |grad q|^2 dx, normalised so int rho dx = 1."""
    hx = xs[1] - xs[0]; hy = ys[1] - ys[0]
    X, Y = np.meshgrid(xs, ys, indexing='ij')
    pts = np.stack([X, Y], axis=-1).reshape(-1, 2)
    V = cfg.potential(pts).reshape(q_grid.shape)
    V = V - V.min()
    rho = np.exp(-cfg.beta * V)
    Z = rho.sum() * hx * hy
    gx, gy = np.gradient(q_grid, hx, hy)
    return float((rho * (gx**2 + gy**2)).sum() * hx * hy / Z)


def nd_separable_dataset(n_samples, dim, beta=1.0, rng=None, n_grid=500,
                         nuisance_sigma=None, transform=None):
    """Separable ND Wolfe-Quapp dataset with an arbitrary observed basis.

    V(x) = V_2d(x[:2]) + sum_i x_i^2 / (2 beta sigma_i^2)   (D = I)

    The committor depends only on x[:2], so the 2D PDE solution is exact
    ground truth in any dimension and for any *orthogonal* change of observed
    basis (an orthogonal map preserves D = I).

    nuisance_sigma : per-dimension equilibrium std of the nuisance
                     coordinates (varying it makes the features badly scaled
                     while keeping D = I).
    transform      : (d,d) orthogonal matrix; the method sees X @ transform.T
                     while the labels and reference come from the latent X.

    Returns (X_observed, in_A, in_B, base_2d).
    """
    rng = np.random.default_rng(rng)
    cfg2 = wolfe_quapp_config(beta=beta, dim=2)
    base = grid_boltzmann_samples(cfg2, n_samples, n_grid=n_grid, rng=rng)
    if dim > 2:
        if nuisance_sigma is None:
            nuisance_sigma = np.ones(dim - 2)
        nuisance_sigma = np.asarray(nuisance_sigma, dtype=float)
        nz = rng.standard_normal((n_samples, dim - 2)) * nuisance_sigma[None, :]
        X = np.concatenate([base, nz], axis=1)
    else:
        X = base
    inA = cfg2.in_A(base)
    inB = cfg2.in_B(base)
    Xobs = X if transform is None else X @ transform.T
    return Xobs, inA, inB, base


def random_rotation(d, rng):
    rng = np.random.default_rng(rng)
    Q, R = np.linalg.qr(rng.standard_normal((d, d)))
    return Q * np.sign(np.diag(R))[None, :]
