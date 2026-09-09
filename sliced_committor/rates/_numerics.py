"""Host-side numpy helpers shared by the rate quantities.

The published rates are checked against reference values at 2e-3 relative, and
the density histogram defines the binning every one of them reads. Keeping
these two small routines under this package's control means a change in a
third-party histogram or normalisation convention cannot move them.
"""

import numpy as np


def normalize_weights(sample_weights, n: int) -> np.ndarray:
    """``(n,)`` weights summing to one; uniform ``1/n`` when None."""
    if sample_weights is None:
        return np.full(n, 1.0 / n)
    w = np.asarray(sample_weights, dtype=np.float64).reshape(-1)
    if w.shape[0] != n:
        raise ValueError(f"sample_weights length {w.shape[0]} != number of points {n}")
    s = float(w.sum())
    if s <= 0:
        raise ValueError("sample_weights must have a positive sum")
    return w / s


def density_histogram(levels, edges, weights=None):
    """Empirical density on ``edges`` (integrating to one) and the raw bin counts.

    With ``weights`` (MBAR or WHAM) the histogram is reweighted to the target
    ensemble; ``counts`` are always the unweighted per-bin sample counts.
    """
    levels = np.asarray(levels, dtype=np.float64)
    dq = float(edges[1] - edges[0])
    if weights is None:
        hist, _ = np.histogram(levels, bins=edges)
        counts = hist
    else:
        w = np.asarray(weights, dtype=np.float64).reshape(-1)
        if w.shape[0] != levels.shape[0]:
            raise ValueError(f"weights length {w.shape[0]} != number of levels {levels.shape[0]}")
        hist, _ = np.histogram(levels, bins=edges, weights=w)
        counts, _ = np.histogram(levels, bins=edges)
    pi = hist / max(float(np.sum(hist)) * dq, 1e-30)
    return pi, counts
