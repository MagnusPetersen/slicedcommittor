"""Simple PMF-based Kramers rate along the umbrella CV (comparison baseline).

A committor-free reference: build the free energy ``F(s) = -log pi(s)`` and the
diffusion ``D(s)`` along the bias CV (or, for a 2D bias, along the projection
onto the inactive->active axis), then apply the overdamped Kramers barrier-
crossing formula. This reuses :func:`sliced_committor.kramers_rate` with an
explicit ``coordinate=`` (so the profiles are along the physical CV); a monotone
ramp on that coordinate supplies the basin populations. It is deliberately
independent of the sliced-committor fit so the bar plot has an honest baseline.
"""

from __future__ import annotations

import numpy as np

from ._containers import USDataset


def progress_coordinate(dataset: USDataset) -> np.ndarray:
    """1D progress coordinate ``s`` increasing from basin A to basin B.

    For a 1D bias this is the CV itself; for a 2D bias it is the projection onto
    the (normalised) A->B axis estimated from the basin sample means.

    Args:
        dataset: umbrella dataset with basin masks and CVs.

    Returns:
        ``(N,)`` progress coordinate, oriented so ``mean(s[A]) < mean(s[B])``.
    """
    cvs = np.asarray(dataset.cvs, dtype=float)
    in_A = np.asarray(dataset.in_A, dtype=bool)
    in_B = np.asarray(dataset.in_B, dtype=bool)
    if dataset.n_cv == 1:
        s = cvs[:, 0].copy()
    else:
        cA = cvs[in_A].mean(axis=0)
        cB = cvs[in_B].mean(axis=0)
        axis = cB - cA
        norm = np.linalg.norm(axis)
        axis = axis / norm if norm > 0 else np.eye(dataset.n_cv)[0]
        s = (cvs - cA[None, :]) @ axis
    if in_A.any() and in_B.any() and s[in_A].mean() > s[in_B].mean():
        s = -s
    return s


def pmf_kramers_rate(
    dataset: USDataset,
    sample_weights: np.ndarray,
    *,
    diffusion_method: str = "hummer",
    n_bins: int = 120,
    per_window: bool = True,
) -> dict:
    """PMF-based Kramers rate along the umbrella CV.

    Args:
        dataset: umbrella dataset.
        sample_weights: ``(N,)`` MBAR/WHAM weights.
        diffusion_method: ``"hummer"`` (default) or ``"kramers_moyal"``.
        n_bins: PMF / diffusion bin count along the CV.
        per_window: estimate the CV diffusion per window (one D per window),
            matching the per-window rate processing of the rest of the pipeline.
            Only affects ``"kramers_moyal"`` (``"hummer"`` is per-window already).

    Returns:
        the :func:`sliced_committor.kramers_rate` dict (``k_AB``, ``k_BA``,
        ``delta_F_AB``, ``D_barrier``, ...).
    """
    import jax

    jax.config.update("jax_enable_x64", True)
    from sliced_committor import kramers_rate

    s = progress_coordinate(dataset)
    in_A = np.asarray(dataset.in_A, dtype=bool)
    in_B = np.asarray(dataset.in_B, dtype=bool)
    lo = float(s[in_A].mean()) if in_A.any() else float(np.percentile(s, 5))
    hi = float(s[in_B].mean()) if in_B.any() else float(np.percentile(s, 95))
    span = hi - lo if hi != lo else 1.0

    def q_ramp(x):
        import jax.numpy as jnp

        x = jnp.asarray(x)
        val = (x[..., 0] - lo) / span
        return jnp.clip(val, 0.0, 1.0)

    samples1d = s[:, None]
    return kramers_rate(
        q_ramp,
        samples1d,
        samples1d,
        dt=dataset.dt,
        sample_weights=np.asarray(sample_weights, dtype=float),
        in_A=in_A,
        in_B=in_B,
        coordinate=s,
        traj_coordinate=s,
        window_ids=np.asarray(dataset.window_ids),
        n_bins=n_bins,
        diffusion_method=diffusion_method,
        per_window=per_window,
    )
