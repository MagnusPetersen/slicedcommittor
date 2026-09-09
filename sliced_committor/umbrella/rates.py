"""Fit the committor of an umbrella-sampling dataset and read off its rates.

:func:`fit_and_rate` is the estimator half of the umbrella workflow: one
committor fit on the reweighted static ensemble, then the rate bundle from the
``{D_q, pi}`` pair for every requested diffusion constructor and reduction,
with the committor-free Kramers baseline beside them. It orchestrates nothing
else; building the :class:`USDataset` is the caller's job.

The diffusion constructors (``diffusion=``):

* ``"cvmap"``, the paper's route: ``D_s`` along the progress coordinate from
  :func:`sliced_committor.pooled_acf_diffusion` (or the value passed as
  ``D_s``), mapped into committor space through the Jacobian by
  :func:`sliced_committor.committor_diffusion_from_cv`, with the CV's mean
  squared gradient from :func:`sliced_committor.linear_response_grad_sq`.
* ``"hummer_q"``: Hummer's ``Var / tau_int`` of the committor itself, per
  window. Rigorous only where the restraint confines the coordinate it is
  measured on, which the committor is not; a diagnostic.
* ``"km_q"``: the Kramers-Moyal (mean squared displacement) estimate on the
  committor at ``lag``. It needs a diffusive regime on the committor at that
  lag, which slow systems lack; see :func:`sliced_committor.lag_scan`.

Rates come out in inverse units of ``dataset.dt``;
:mod:`sliced_committor.rates.units` converts them.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from ..core.committor import fit_committor
from ..rates import (
    basin_populations,
    committor_diffusion_from_cv,
    committor_grad_sq,
    density,
    diffusion_profile,
    flux_flatness,
    hummer_diffusion,
    linear_response_grad_sq,
    pooled_acf_diffusion,
    rate_from_profiles,
)
from ..rates.baselines import pmf_kramers_rate as _kramers
from .dataset import USDataset

logger = logging.getLogger(__name__)

DIFFUSIONS = ("cvmap", "hummer_q", "km_q")
REDUCTIONS = ("plateau", "harmonic", "arithmetic", "local")
#: the fixed band of the label-free committor-quality score ``flux_cv``
QUALITY_BAND = (0.2, 0.8)


def progress_coordinate(dataset: USDataset) -> np.ndarray:
    """A 1D progress coordinate ``s`` increasing from basin A to basin B.

    The biased CV itself for a 1D bias; for a 2D bias the projection onto the
    normalised A -> B axis between the basin means. Oriented so that
    ``mean(s[A]) < mean(s[B])``.
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


def pmf_kramers_rate(dataset: USDataset, sample_weights, *, run_ids=None, n_bins=120) -> dict:
    """The committor-free Kramers baseline along the progress coordinate.

    ``F(s) = -log pi(s)`` from the reweighted density and the per-window
    Hummer diffusion along ``s`` go into
    :func:`sliced_committor.rates.baselines.pmf_kramers_rate`. A comparison
    row only: it needs interior wells and a barrier well above ``kT``.
    """
    s = progress_coordinate(dataset)
    span = (float(s.min()), float(s.max()))
    pi = density(s, sample_weights=sample_weights, n_bins=n_bins, span=span)
    D = hummer_diffusion(s, dataset.window_ids, dt=float(dataset.dt), run_ids=run_ids)
    return _kramers(pi, D)


def _plain(rate: dict) -> dict:
    """A rate dict without its profile, with plain Python scalars (JSON-ready)."""
    out = {}
    for k, v in rate.items():
        if k == "nu":
            continue
        if isinstance(v, tuple):
            out[k] = tuple(float(x) for x in v)
        elif isinstance(v, (bool, str)):
            out[k] = v
        else:
            out[k] = float(v)
    return out


def fit_and_rate(
    dataset: USDataset,
    features,
    sample_weights,
    *,
    n_directions: int = 256,
    n_bins: int = 200,
    seed: int = 0,
    tikhonov="halfset_eigen",
    direction_sampling=None,
    directions=None,
    feature_metric=None,
    bridge_metric=None,
    D_s: float | None = None,
    run_ids=None,
    diffusion=("cvmap",),
    reductions=("plateau", "harmonic", "arithmetic"),
    lag: int = 1,
    n_diff_bins: int = 25,
    strict: bool = True,
    **solver_kwargs,
) -> dict[str, Any]:
    """Fit the committor of an umbrella dataset and return its rate bundle.

    Args:
        dataset: the umbrella dataset (basins, windows, ``dt``, the biased CV).
        features: ``(N, d)`` committor features, row-aligned with the dataset.
        sample_weights: ``(N,)`` MBAR/WHAM weights (:func:`reweight`).
        n_directions, n_bins, seed, tikhonov, direction_sampling, directions,
            feature_metric, **solver_kwargs: forwarded to
            :func:`sliced_committor.fit_committor`. Binning defaults to
            ``equal_width``: quantile bins put too few bins in the sparsely
            sampled barrier and inflate the rate (about 3x on Wolfe-Quapp).
        bridge_metric: the metric of the CV -> committor map, applied to BOTH
            mean squared gradients (:func:`sliced_committor.committor_grad_sq`
            and :func:`sliced_committor.linear_response_grad_sq`); None is
            the identity.
        D_s: a known diffusion along the progress coordinate; None measures it
            with :func:`sliced_committor.pooled_acf_diffusion` over all
            windows (call that yourself to select windows).
        run_ids: ``(N,)`` labels of independent contiguous runs (replicates)
            for the autocorrelation-based estimators, or None.
        diffusion: which constructors of ``D_q`` to run, from
            ``("cvmap", "hummer_q", "km_q")``.
        reductions: which reductions of ``{D_q, pi}`` to report, from
            ``("plateau", "harmonic", "arithmetic", "local")``.
        lag: the Kramers-Moyal lag of ``"km_q"``, in frames.
        n_diff_bins: the committor grid of the profiles and the reductions.
        strict: raise on the first failing estimator (default); False records
            the failure under ``errors`` and carries on.

    Returns:
        dict with ``committor`` (fit diagnostics), ``q_samples``, ``D_s``,
        ``cv_grad_sq``, ``profiles`` (``levels``, ``pi``, ``grad_sq``, and per
        constructor ``D_q`` and ``flux``), ``rates`` (per constructor, per
        reduction: :func:`sliced_committor.rate_from_profiles` without its
        profile), ``flux_cv`` (per constructor: the constancy of the flux over
        ``QUALITY_BAND``), ``kramers`` (the baseline) and ``errors``.
    """
    import jax
    import jax.numpy as jnp

    jax.config.update("jax_enable_x64", True)
    unknown = sorted(set(diffusion) - set(DIFFUSIONS))
    if unknown:
        raise ValueError(f"unknown diffusion constructor(s) {unknown}; choose from {DIFFUSIONS}")
    unknown = sorted(set(reductions) - set(REDUCTIONS))
    if unknown:
        raise ValueError(f"unknown reduction(s) {unknown}; choose from {REDUCTIONS}")

    feats = jnp.asarray(np.asarray(features, dtype=float))
    in_A = np.asarray(dataset.in_A, dtype=bool)
    in_B = np.asarray(dataset.in_B, dtype=bool)
    if not in_A.any() or not in_B.any():
        raise ValueError(f"empty basin(s): |A|={int(in_A.sum())}, |B|={int(in_B.sum())}")
    sw = np.asarray(sample_weights, dtype=float)
    solver: dict[str, Any] = {
        "n_bins": n_bins,
        "binning_method": "equal_width",
        "sample_weights": jnp.asarray(sw),
    }
    solver.update(solver_kwargs)
    if directions is not None:
        solver["directions"] = jnp.asarray(directions)
        n_directions = int(solver["directions"].shape[0])
    if direction_sampling is not None:
        solver["direction_sampling"] = direction_sampling
    if feature_metric is not None:
        solver["feature_metric"] = feature_metric

    q, detail = fit_committor(
        feats,
        in_A=jnp.asarray(in_A),
        in_B=jnp.asarray(in_B),
        n_directions=n_directions,
        seed=seed,
        tikhonov=tikhonov,
        return_details=True,
        **solver,
    )
    qv = np.asarray(q(feats), dtype=float)
    wid = np.asarray(dataset.window_ids)
    dt = float(dataset.dt)
    s = progress_coordinate(dataset)

    out: dict[str, Any] = {
        "committor": {
            "n_directions": int(np.asarray(detail.result.directions).shape[0]),
            "n_features": int(feats.shape[1]),
            "valid_fraction": float(np.mean(np.asarray(detail.result.valid_mask))),
            "dirichlet_energy": float(detail.weights.dirichlet_energy),
            "moment_gap": float(detail.weights.moment_gap),
            "tikhonov": str(tikhonov),
        },
        "q_samples": qv,
        "errors": {},
    }

    def attempt(key, fn):
        try:
            return fn()
        except Exception as exc:
            if strict:
                raise
            logger.warning("%s failed: %s", key, exc)
            out["errors"][key] = f"{type(exc).__name__}: {exc}"
            return None

    pi = density(qv, sample_weights=sw, n_bins=n_diff_bins)
    rho_A, rho_B = basin_populations(qv, sample_weights=sw, in_A=in_A, in_B=in_B)
    g_q = committor_grad_sq(q, feats, sample_weights=sw, n_bins=n_diff_bins, metric=bridge_metric)
    pi_v = np.asarray(pi.values)
    out["profiles"] = {
        "levels": np.asarray(pi.levels),
        "pi": pi_v,
        "grad_sq": np.asarray(g_q.values),
        "D_q": {},
        "flux": {},
    }

    # the diffusion along the progress coordinate, and the CV's gradient in feature space
    if D_s is None:
        pooled = attempt("D_s", lambda: pooled_acf_diffusion(s, wid, dt=dt, run_ids=run_ids))
        out["D_s"] = (
            None
            if pooled is None
            else {
                "value": pooled.D,
                "tau_int": pooled.tau_int,
                "n_windows": pooled.n_windows,
                "n_runs": pooled.n_runs,
                "ci": pooled.ci,
                "source": "pooled_acf",
            }
        )
    else:
        out["D_s"] = {"value": float(D_s), "source": "given"}
    out["cv_grad_sq"] = attempt(
        "cv_grad_sq",
        lambda: linear_response_grad_sq(
            s, np.asarray(features, dtype=float), sw, metric=bridge_metric
        ),
    )

    constructors = {
        "cvmap": lambda: committor_diffusion_from_cv(
            g_q, D_s=out["D_s"]["value"], cv_grad_sq=out["cv_grad_sq"]
        ),
        "hummer_q": lambda: hummer_diffusion(qv, wid, dt=dt, run_ids=run_ids),
        "km_q": lambda: diffusion_profile(qv, dt=dt, lag=lag, window_ids=wid, n_bins=n_diff_bins),
    }
    out["rates"], out["flux_cv"] = {}, {}
    for name in diffusion:
        if name == "cvmap" and (out["D_s"] is None or out["cv_grad_sq"] is None):
            out["errors"]["cvmap"] = "no D_s or CV gradient to map (see the errors above)"
            continue
        D_q = attempt(name, constructors[name])
        if D_q is None:
            continue
        by_reduction, nu = {}, None
        for reduction in reductions:
            rate = attempt(
                f"{name}/{reduction}",
                lambda r=reduction, D=D_q: rate_from_profiles(pi, D, rho_A, rho_B, reduction=r),
            )
            if rate is not None:
                nu = rate["nu"]
                by_reduction[reduction] = _plain(rate)
        if nu is None:
            continue
        out["rates"][name] = by_reduction
        out["flux_cv"][name] = flux_flatness(nu, QUALITY_BAND)
        nu_v = np.asarray(nu.values)
        out["profiles"]["flux"][name] = nu_v
        out["profiles"]["D_q"][name] = np.where(
            pi_v > 0, nu_v / np.where(pi_v > 0, pi_v, 1.0), np.nan
        )

    out["kramers"] = attempt("kramers", lambda: pmf_kramers_rate(dataset, sw, run_ids=run_ids))
    return out
