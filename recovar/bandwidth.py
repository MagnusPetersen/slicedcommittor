"""Adaptive per-region profile bandwidth for the 1D slice histograms (RECOVAR sec. 5).

Two bandwidth rules, both returned as a ``bw_fn`` for ``basis.build_basis``
(called per direction j as ``bw_fn(j, edges, s, in_A, in_B)`` and returning a
per-bin bandwidth vector in units of bins):

* ``make_halfset_bw_fn`` -- the VALIDATED rule. Halfset CV risk on the
  committor PROFILE (or, with ``target='deriv'``, on its derivative q',
  which is what actually enters the Gram). Measured
  (``_reference/REPORT.md`` sec. 5): it beats the best hand-tuned global
  bandwidth without seeing the truth (derivative target: 0.81-0.93x the
  tuned-on-truth global optimum across N = 6k..60k) and removes
  n_bins/bandwidth as a tuned parameter.
* ``make_adaptive_bw_fn`` -- the REFUTED negative control (density-ISE).
  Kept for the record; do not use it for production fits.

Bandwidth matters most where data is thin: RMSE varies 4x across the
bandwidth grid at N = 3000.
"""
import numpy as np

from .basis import _histograms, _smooth_matrix, rd_profile


def make_halfset_bw_fn(bw_grid=(0.5, 1.0, 2.0, 4.0, 8.0, 16.0), n_regions=6,
                       split_seed=0, target='profile'):
    """Per-region bandwidth chosen by halfset risk on the *committor profile*
    (not on the density).

    For each candidate bandwidth b, build the profile on each half of the
    frames and score it against the *other* half's finest-bandwidth profile:

        R(b) = mean_region[ (q_1^b - q_2^fine)^2 + (q_2^b - q_1^fine)^2 ] / 2

    which is the standard CV risk against a noisy but unbiased target, so it
    trades bias against variance with the right minimiser. target='deriv'
    scores q' instead of q (q' is what enters the Gram).
    """
    def bw_fn(j, edges, s, in_A, in_B):
        n_bins = len(edges) - 1
        h = edges[1] - edges[0]
        rng = np.random.default_rng(split_seed + j)
        m = rng.random(len(s)) < 0.5
        halves = []
        for sel in (m, ~m):
            halves.append(_histograms(s[sel], in_A[sel], in_B[sel],
                                      edges, None))

        def prof(hist, b):
            S = _smooth_matrix(n_bins, b)
            ha, hA, hB = (S @ hist[0], S @ hist[1], S @ hist[2])
            rho = ha / max(ha.sum() * h, 1e-300)
            rA = hA / max(hA.sum() * h, 1e-300)
            rB = hB / max(hB.sum() * h, 1e-300)
            q = rd_profile(rho, rA, rB, h)
            return np.gradient(q, h) if target == 'deriv' else q

        bfine = min(bw_grid)
        fine = [prof(halves[0], bfine), prof(halves[1], bfine)]
        cand = {b: [prof(halves[0], b), prof(halves[1], b)] for b in bw_grid}

        redges = np.linspace(0, n_bins, n_regions + 1).astype(int)
        bw = np.empty(n_bins)
        for r in range(n_regions):
            sl = slice(redges[r], redges[r + 1])
            best, best_b = np.inf, bw_grid[0]
            for b in bw_grid:
                q1, q2 = cand[b]
                risk = 0.5 * (np.mean((q1[sl] - fine[1][sl]) ** 2)
                              + np.mean((q2[sl] - fine[0][sl]) ** 2))
                if risk < best:
                    best, best_b = risk, b
            bw[sl] = best_b
        return bw
    return bw_fn


def make_adaptive_bw_fn(bw_grid=(0.5, 1.0, 2.0, 4.0, 8.0, 16.0), n_regions=6,
                        split_seed=0):
    """REFUTED negative control: per-region bandwidth by density-ISE CV.

    Chooses, per s-region, the bandwidth minimising the halfset
    Rudemo/Bowman integrated-squared-error risk of the smoothed DENSITY.
    Standard KDE cross-validation in spirit -- and the wrong objective for
    this method: measured 1.36-2.8x WORSE than the best global bandwidth
    (``_reference/REPORT.md`` sec. 5), because the density is smoothest
    exactly where the committor profile is steepest. Kept only as the
    negative control against ``make_halfset_bw_fn``, which scores the
    profile (or its derivative) instead and does beat the tuned global
    bandwidth. Do not use this rule for production fits.
    """
    def bw_fn(j, edges, s, in_A, in_B):
        n_bins = len(edges) - 1
        rng = np.random.default_rng(split_seed + j)
        m = rng.random(len(s)) < 0.5
        hs = [_histograms(s[sel], in_A[sel], in_B[sel], edges, None)[0]
              for sel in (m, ~m)]
        hs = [h / max(h.sum(), 1) for h in hs]
        redges = np.linspace(0, n_bins, n_regions + 1).astype(int)
        bw = np.empty(n_bins)
        for r in range(n_regions):
            sl = slice(redges[r], redges[r + 1])
            best, best_bw = np.inf, bw_grid[0]
            for b in bw_grid:
                S = _smooth_matrix(n_bins, b)
                f1, f2 = S @ hs[0], S @ hs[1]
                # CV risk: ||f||^2 - 2 <f1, f2>  (Rudemo/Bowman ISE estimator)
                risk = (np.mean(((f1 + f2) / 2)[sl] ** 2)
                        - 2 * np.mean((f1 * f2)[sl]))
                if risk < best:
                    best, best_bw = risk, b
            bw[sl] = best_bw
        return bw
    return bw_fn
