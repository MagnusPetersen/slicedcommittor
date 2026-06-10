"""scipy-free numeric helpers for the rate computations (host-side numpy).

The library depends only on jax + numpy. These small helpers replace the
``scipy.integrate`` calls used in the research code so no scipy dependency is
pulled in.
"""

import numpy as np


def cumulative_trapezoid(y, dx: float = 1.0, initial: float = 0.0) -> np.ndarray:
    """Cumulative trapezoidal integral, matching ``scipy.integrate``'s
    ``cumulative_trapezoid(y, dx=dx, initial=initial)``.

    Output has the same length as ``y``; element 0 equals ``initial`` and
    element k equals ``initial + ∫_0^k y``.
    """
    y = np.asarray(y, dtype=np.float64)
    seg = 0.5 * (y[1:] + y[:-1]) * dx
    return np.concatenate([[float(initial)], float(initial) + np.cumsum(seg)])


def normalize_weights(sample_weights, n: int) -> np.ndarray:
    """Return ``(n,)`` weights summing to 1; uniform ``1/n`` when None."""
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
    """Empirical density on ``edges`` (integrates to 1), plus raw bin counts.

    With ``weights`` (e.g. MBAR), the histogram is reweighted to the target
    ensemble; ``counts`` are always the unweighted per-bin sample counts (used
    for undersampling guards).
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
