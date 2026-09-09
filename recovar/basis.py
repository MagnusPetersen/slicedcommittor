"""
Slice basis for the sliced committor (JAX-aware port of ``_reference/slicedcv/core.py``).

    qbar(x) = c + sum_j w_j q_j(theta_j . x)

Per-direction 1D profiles q_j(s) are RD (reaction-diffusion) committor
solutions on a uniform grid, built from smoothed histograms of the projected
samples. D = I in feature space throughout (as in the paper's molecular
examples).

Dtype contract: direction matrices (thetas) float64; basis arrays (grids, q,
qp) float32. The RD solve and histogram pipeline run in float64 and are
rounded to float32 only at storage. The heavy eval_at gather/interpolation is
a jitted JAX kernel; everything else is numpy/scipy (histograms, banded
solves, small linear algebra), where JAX adds nothing.
"""
from typing import NamedTuple

import numpy as np
from scipy.linalg import solve_banded

import jax
import jax.numpy as jnp


# ---------------------------------------------------------------------------
# Directions
# ---------------------------------------------------------------------------

def isotropic_directions(M, d, rng):
    rng = np.random.default_rng(rng)
    v = rng.standard_normal((M, d))
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def lda_directions(M, X, in_A, in_B, rng, kappa=20.0):
    """Power-spherical-like sampler concentrated on the Fisher LDA axis.

    mu = normalised (mean_B - mean_A) whitened by the pooled within-class
    covariance; directions are drawn as mu tilted by an isotropic
    perturbation whose size is set by `kappa` (larger = tighter).
    """
    rng = np.random.default_rng(rng)
    d = X.shape[1]
    mA = X[in_A].mean(0)
    mB = X[in_B].mean(0)
    Sw = np.cov(X[in_A].T) * in_A.sum() + np.cov(X[in_B].T) * in_B.sum()
    Sw = np.atleast_2d(Sw) / (in_A.sum() + in_B.sum())
    Sw += 1e-6 * np.trace(Sw) / d * np.eye(d)
    mu = np.linalg.solve(Sw, mB - mA)
    mu /= np.linalg.norm(mu)
    pert = rng.standard_normal((M, d)) / np.sqrt(kappa)
    v = mu[None, :] + pert
    return v / np.linalg.norm(v, axis=1, keepdims=True)


# ---------------------------------------------------------------------------
# Histogram smoothing with (optionally) position-dependent bandwidth
# ---------------------------------------------------------------------------

def _smooth_matrix(n_bins, bw_bins):
    """Mass-conserving Gaussian smoothing matrix.

    bw_bins: scalar or (n_bins,) per-source-bin bandwidth in units of bins.
    Returns S with smoothed = S @ counts.
    """
    bw = np.broadcast_to(np.asarray(bw_bins, dtype=float), (n_bins,))
    i = np.arange(n_bins)
    D = i[:, None] - i[None, :]                     # (target, source)
    sig = np.maximum(bw[None, :], 1e-6)
    K = np.exp(-0.5 * (D / sig) ** 2)
    K /= K.sum(axis=0, keepdims=True)               # conserve source mass
    return K


def _histograms(s, in_A, in_B, edges, bw_bins, w=None):
    n_bins = len(edges) - 1
    if w is None:
        h_all, _ = np.histogram(s, bins=edges)
        h_A, _ = np.histogram(s[in_A], bins=edges)
        h_B, _ = np.histogram(s[in_B], bins=edges)
    else:
        h_all, _ = np.histogram(s, bins=edges, weights=w)
        h_A, _ = np.histogram(s[in_A], bins=edges, weights=w[in_A])
        h_B, _ = np.histogram(s[in_B], bins=edges, weights=w[in_B])
    if bw_bins is not None:
        S = _smooth_matrix(n_bins, bw_bins)
        h_all = S @ h_all.astype(float)
        h_A = S @ h_A.astype(float)
        h_B = S @ h_B.astype(float)
    return h_all.astype(float), h_A.astype(float), h_B.astype(float)


# ---------------------------------------------------------------------------
# 1D reaction-diffusion slice profile
# ---------------------------------------------------------------------------

def rd_profile(rho, rho_A, rho_B, h, kappa_rel=1e6):
    """Solve d/ds[rho q'] = kappa [rho_A q - rho_B (1-q)] with Neumann ends.

    kappa is set relative to the diffusion scale so the stiff limit is
    reached in a scale-free way:  kappa = kappa_rel * max(rho)/h^2 / max(rho_A+rho_B).
    """
    n = len(rho)
    rho = np.maximum(rho, 1e-300)
    face = 0.5 * (rho[:-1] + rho[1:]) / h**2        # (n-1,)

    absorb = rho_A + rho_B
    scale_diff = face.max() if face.size else 1.0
    scale_abs = absorb.max() if absorb.max() > 0 else 1.0
    kappa = kappa_rel * scale_diff / scale_abs

    lower = np.zeros(n); upper = np.zeros(n)
    lower[1:] = face
    upper[:-1] = face
    diag = -(lower + upper) - kappa * absorb
    rhs = -kappa * rho_B

    # Guard: with no absorption anywhere the pure-Neumann operator is
    # singular (constant null space). This happens for random directions in
    # high d that see no state density. Add a tiny well-scaled ridge and
    # fall back to a constant profile if it is still singular.
    dead = np.abs(diag) < 1e-300
    diag[dead] = -1.0
    rhs[dead] = 0.0
    # Only the absorption-free case is singular (pure Neumann Laplacian has a
    # constant null space). Scale the ridge to the DIFFUSION scale -- scaling
    # it to max|diag| would be dominated by the stiff absorption term and
    # would swamp the diffusion-scale entries.
    # DO NOT 'fix' this guard: a max|diag|-scaled ridge silently degraded the
    # committor ~20x (see _reference/REPORT.md, 'A correction found by
    # verifying').
    if absorb.max() <= 0:
        diag = diag - 1e-8 * max(scale_diff, 1e-300)

    ab = np.zeros((3, n))
    ab[0, 1:] = upper[:-1]
    ab[1, :] = diag
    ab[2, :-1] = lower[1:]
    try:
        q = solve_banded((1, 1), ab, rhs)
    except Exception:
        return np.full(n, 0.5)
    if not np.all(np.isfinite(q)):
        return np.full(n, 0.5)
    return np.clip(q, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Slice basis
# ---------------------------------------------------------------------------

@jax.jit
def _eval_at_kernel(grids, q, qp, S):
    """Vectorised uniform-grid linear interpolation, jitted over (M, P).

    Same algorithm as the numpy reference eval_at: clamped q outside the
    grid, qp = 0 outside the grid, float32 output. Arithmetic is float64;
    the grid spacing is recovered from the full span (grids are stored
    float32, and (g[-1]-g[0])/(nb-1) carries less rounding than g[1]-g[0],
    keeping the result within float32 rounding of the float64 reference).
    """
    nb = q.shape[1]
    grids = grids.astype(jnp.float64)
    S = S.astype(jnp.float64)
    g0 = grids[:, :1]
    h = (grids[:, -1:] - grids[:, :1]) / (nb - 1)
    t = (S - g0) / h
    i = jnp.clip(jnp.floor(t), 0, nb - 2).astype(jnp.int32)
    f = jnp.clip(t - i, 0.0, 1.0)
    lo = t < 0
    hi = t > nb - 1

    def gather(Y):
        Y = Y.astype(jnp.float64)
        y0 = jnp.take_along_axis(Y, i, axis=1)
        y1 = jnp.take_along_axis(Y, i + 1, axis=1)
        return Y, y0 * (1 - f) + y1 * f

    q64, qv = gather(q)
    qv = jnp.where(lo, q64[:, :1], jnp.where(hi, q64[:, -1:], qv))
    _, qpv = gather(qp)
    qpv = jnp.where(lo | hi, 0.0, qpv)
    return qv.astype(jnp.float32), qpv.astype(jnp.float32)


class SliceBasis(NamedTuple):
    """Per-direction 1D profiles q_j(s) and derivatives q_j'(s) on a grid."""

    thetas: np.ndarray              # (M, d) float64
    grids: np.ndarray               # (M, n_bins) float32
    q: np.ndarray                   # (M, n_bins) float32
    qp: np.ndarray                  # (M, n_bins) float32

    @property
    def M(self):
        return self.q.shape[0]

    @property
    def n_bins(self):
        return self.q.shape[1]

    def subset(self, idx):
        return SliceBasis(self.thetas[idx], self.grids[idx],
                          self.q[idx], self.qp[idx])

    def eval_at(self, S):
        """Interpolate q and q' at projections S (M, P).

        Grids are uniform per direction (linspace), so this is fully
        vectorised: no Python loop over directions. Returns float32 (M, P).
        """
        qv, qpv = _eval_at_kernel(self.grids, self.q, self.qp,
                                  np.asarray(S))
        return np.asarray(qv), np.asarray(qpv)


def build_basis(thetas, X, in_A, in_B, n_bins=200, bw_bins=1.5,
                kappa_rel=1e6, bw_fn=None, pad=0.02, sample_weights=None):
    """Build slice profiles for every direction.

    bw_fn: optional callable(j, edges, s, in_A, in_B) -> per-bin bandwidth
           (scalar or (n_bins,) per-source-bin vector, as accepted by
           _smooth_matrix), used by the adaptive-bandwidth experiment.
    """
    M = thetas.shape[0]
    grids = np.empty((M, n_bins))
    Q = np.empty((M, n_bins))
    QP = np.empty((M, n_bins))
    S_all = X @ thetas.T                            # (N, M)

    for j in range(M):
        s = S_all[:, j]
        lo, hi = s.min(), s.max()
        span = hi - lo
        lo -= pad * span; hi += pad * span
        edges = np.linspace(lo, hi, n_bins + 1)
        centers = 0.5 * (edges[:-1] + edges[1:])
        h = centers[1] - centers[0]

        bw = bw_bins
        if bw_fn is not None:
            bw = bw_fn(j, edges, s, in_A, in_B)
        h_all, h_A, h_B = _histograms(s, in_A, in_B, edges, bw,
                                      w=sample_weights)

        rho = h_all / (h_all.sum() * h + 1e-300)
        rA = h_A / (h_A.sum() * h + 1e-300) if h_A.sum() > 0 else h_A
        rB = h_B / (h_B.sum() * h + 1e-300) if h_B.sum() > 0 else h_B

        q = rd_profile(rho, rA, rB, h, kappa_rel=kappa_rel)
        qp = np.gradient(q, h)

        grids[j] = centers
        Q[j] = q
        QP[j] = qp

    return SliceBasis(np.asarray(thetas, dtype=np.float64),
                      grids.astype(np.float32),
                      Q.astype(np.float32),
                      QP.astype(np.float32))
