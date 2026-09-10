"""Host-side numpy helpers shared by the rate quantities.

The published rates are checked against reference values at 2e-3 relative, and
the bin grid and the density histogram define the binning every one of them
reads. Keeping these small routines under this package's control means a
change in a third-party histogram or normalisation convention cannot move them.
"""

import numpy as np


def bin_grid(n_bins: int, span=(0.0, 1.0)):
    """``(edges, centers)`` of ``n_bins`` equal-width bins over ``span``: the grid every profile uses."""
    lo, hi = float(span[0]), float(span[1])
    edges = np.linspace(lo, hi, n_bins + 1)
    return edges, 0.5 * (edges[:-1] + edges[1:])


def per_frame(name, values, n: int, dtype=None):
    """``values`` as a ``(n,)`` array (None passes through); any other length is an error naming ``name``."""
    if values is None:
        return None
    arr = np.asarray(values, dtype=dtype).reshape(-1)
    if arr.shape[0] != n:
        raise ValueError(f"{name} length {arr.shape[0]} != number of frames {n}")
    return arr


def normalize_weights(sample_weights, n: int) -> np.ndarray:
    """``(n,)`` weights summing to one; uniform ``1/n`` when None."""
    if sample_weights is None:
        return np.full(n, 1.0 / n)
    w = per_frame("sample_weights", sample_weights, n, np.float64)
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
        w = per_frame("weights", weights, levels.shape[0], np.float64)
        hist, _ = np.histogram(levels, bins=edges, weights=w)
        counts, _ = np.histogram(levels, bins=edges)
    pi = hist / max(float(np.sum(hist)) * dq, 1e-30)
    return pi, counts
