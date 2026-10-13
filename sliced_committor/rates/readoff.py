"""The flux read-off: the variational principle applied to the fit's own level sets.

A fitted committor ``u`` is a coordinate, and every function ``f(u)`` of it is
a trial function with the same level sets and a different spacing between
them. By the co-area formula,

    E[f(u)] = int f'(z)^2 Phi(z) dz,   Phi(z) = flux of rho D grad u through {u = z},
    F[f(u)] = int f'(z) g(z) dz,       g(z) = P_B(u > z) - P_A(u > z),

with ``F`` the empirical fidelity (the difference of the basin averages), so
``g(z)`` is how well the level ``z`` separates the labelled states.
Cauchy-Schwarz in ``z`` minimises the quotient ``E / F^2`` over every ``f`` in
closed form,

    nu = [ int g(z)^2 / Phi(z) dz ]^-1 ,    attained at  f' ~ g / Phi .

For the exact committor ``g = 1`` and ``Phi = nu_AB`` on ``(0, 1)``, so ``nu``
is the reactive flux, and it stays the flux under any monotone recalibration
of the committor. For a fit, ``nu`` is the variational upper bound on the flux
over the fit's reparametrisations (up to the fidelity substitution), so it
never exceeds the fit's own ``E / F^2``, and ``E / nu - 1`` bounds the
relative Dirichlet error at unit fidelity from below. No band is chosen: a
level that cuts through a state's own samples has ``g < 1`` and loses weight
in proportion.

On the samples ``f`` is piecewise linear on equal-count bins of ``u``. The
ramps ``psi_i(u) = clip((u - z_i) / dz_i, 0, 1)`` have a diagonal Gram
``C_i / dz_i^2`` (``C_i`` the weighted ``grad_sq`` of the samples in bin
``i``), and the quotient's optimum is the constrained solve of the weights
with that Gram: one sort and one pass, exact on the samples.
"""

import numpy as np

from ._coordinate import Profile
from ._numerics import normalize_weights, per_frame


def _equal_count_edges(values, n_bins):
    """Edges of ``n_bins`` equal-count bins covering every value (duplicates merged)."""
    edges = np.quantile(values, np.linspace(0.0, 1.0, n_bins + 1))
    edges = np.unique(edges)
    if edges.size < 2:
        raise ValueError("variational_flux: the coordinate takes a single value")
    # a pad at the top puts the largest value inside the last bin
    edges[-1] += 1e-12 * max(1.0, abs(float(edges[-1])))
    return edges


def variational_flux(values, grad_sq, *, in_A, in_B, sample_weights=None, n_bins=100) -> dict:
    """The reactive flux read off a fitted committor's level sets, with no band.

    Minimises the quotient ``E[f(u)] / F[f(u)]^2`` over all functions ``f`` of
    the coordinate ``u`` (see the module docstring). The flux comes out in the
    units of ``grad_sq`` times the weights: multiply by the configurational
    diffusion scale ``D0`` for the physical flux, then divide by the
    populations for the rate constants.

    Args:
        values: ``(N,)`` the coordinate at the samples, normally the fitted
            committor **unclipped** (``build_committor(..., clip=False)``).
            A clipped committor restricts ``f`` to be constant outside
            ``[0, 1]`` and reads a larger flux.
        grad_sq: ``(N,)`` ``grad u^T M grad u`` at the samples, in the metric of
            the fit's Dirichlet form, so that ``sum_n W_n grad_sq_n`` is its
            energy.
        in_A, in_B: ``(N,)`` bool basin labels.
        sample_weights: ``(N,)`` weights of the static ensemble (MBAR or WHAM
            for biased data); None is uniform. They weight the energy. The
            basin averages are taken over the frames without weights, as the
            fit's basin moments are.
        n_bins: number of equal-count bins of ``values`` on which ``f`` is
            piecewise linear. The read-off moves by about a percent between
            25 and 400 bins.

    Returns:
        dict with ``nu`` (the flux), ``energy`` (``E = sum_n W_n grad_sq_n``),
        ``calibration_error`` (``energy / nu - 1``), ``q_tilde`` (the optimal
        reparametrisation ``f(u)`` at the samples, offset to mean zero over A
        and of unit empirical fidelity), ``edges`` and the per-bin
        :class:`Profile` s ``flux`` (``Phi``) and ``separation`` (``g``, the
        ramp average of ``P_B(u > z) - P_A(u > z)`` over the bin).

    Raises:
        ValueError: when no level of ``u`` separates the labelled states.
    """
    u = np.asarray(values, dtype=np.float64).reshape(-1)
    n = u.shape[0]
    g2 = per_frame("grad_sq", grad_sq, n, np.float64)
    in_A = per_frame("in_A", in_A, n, bool)
    in_B = per_frame("in_B", in_B, n, bool)
    if not in_A.any() or not in_B.any():
        raise ValueError("variational_flux: both basins need samples")
    W = normalize_weights(sample_weights, n)

    edges = _equal_count_edges(u, int(n_bins))
    K = edges.size - 1
    dz = np.diff(edges)
    k = np.clip(np.searchsorted(edges, u, side="right") - 1, 0, K - 1)
    t = np.clip((u - edges[k]) / dz[k], 0.0, 1.0)  # each sample's position inside its bin

    C = np.bincount(k, weights=W * g2, minlength=K)
    counts = np.bincount(k, minlength=K)

    def ramp_mean(mask):
        # the basin average of every ramp: 1 above its bin, t inside it, 0 below
        m = mask.astype(np.float64)
        inside = np.bincount(k, weights=m * t, minlength=K)
        above = m.sum() - np.cumsum(np.bincount(k, weights=m, minlength=K))
        return (above + inside) / m.sum()

    phi = ramp_mean(in_B) - ramp_mean(in_A)
    G = C / dz**2
    stiff = G > 0  # a bin no gradient crosses takes no part: f is flat across it
    R = float(np.sum(phi[stiff] ** 2 / G[stiff]))
    if not R > 0:
        raise ValueError("variational_flux: no level of the coordinate separates the basins")
    s = np.zeros(K)
    s[stiff] = phi[stiff] / G[stiff] / R  # unit fidelity: phi @ s = 1

    f = np.concatenate(([0.0], np.cumsum(s)))[k] + s[k] * t
    q_tilde = f - f[in_A].mean()
    energy = float(W @ g2)
    nu = 1.0 / R
    centers = 0.5 * (edges[:-1] + edges[1:])
    return {
        "nu": nu,
        "energy": energy,
        "calibration_error": energy / nu - 1.0,
        "q_tilde": q_tilde,
        "edges": edges,
        "flux": Profile(centers, C / dz, counts),
        "separation": Profile(centers, phi, counts),
        "n_bins": K,
    }
