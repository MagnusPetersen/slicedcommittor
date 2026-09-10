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
    Profile,
    basin_populations,
    committor_diffusion_from_cv,
    density,
    diffusion_profile,
    flux_flatness,
    hummer_diffusion,
    linear_response_grad_sq,
    pooled_acf_diffusion,
    rate_from_profiles,
)
from ..rates.baselines import pmf_kramers_rate
from ..rates.formulas import REDUCTIONS, flux_profiles
from ..rates.quantities import _grad_sq_profile, _values_and_grad_sq
from .dataset import USDataset

logger = logging.getLogger(__name__)

DIFFUSIONS = ("cvmap", "hummer_q", "km_q")
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


def kramers_baseline(dataset: USDataset, sample_weights, *, run_ids=None, n_bins=120) -> dict:
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
    return pmf_kramers_rate(pi, D)


def _plain(rate: dict) -> dict:
    """A rate dict without its profiles, with plain Python scalars (JSON-ready)."""
    out = {}
    for k, v in rate.items():
        if isinstance(v, Profile):
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
        sample_weights: ``(N,)`` MBAR/WHAM weights (:func:`reweight`), summing
            to one.
        n_directions, n_bins, seed, tikhonov, solver_kwargs: forwarded to
            :func:`sliced_committor.fit_committor` (``direction_sampling``,
            ``directions``, ``feature_metric``, ``boundary_quantile``, ... ride
            along as the remaining keyword arguments, ``solver_kwargs``).
            Binning defaults to ``equal_width``:
            quantile bins put too few bins in the sparsely sampled barrier and
            inflate the rate (about 3x on Wolfe-Quapp).
        bridge_metric: the metric of the CV -> committor map, applied to BOTH
            mean squared gradients (:func:`sliced_committor.committor_grad_sq`
            and :func:`sliced_committor.linear_response_grad_sq`); None is
            the identity.
        D_s: a known diffusion along the progress coordinate; None measures it
            with :func:`sliced_committor.pooled_acf_diffusion` over all
            windows (call that yourself to select windows).
        run_ids: ``(N,)`` labels of independent contiguous runs (replicates),
            which no diffusion estimator reads across; None when every window
            is one run.
        diffusion: which constructors of ``D_q`` to run, from
            ``("cvmap", "hummer_q", "km_q")``.
        reductions: which reductions of ``{D_q, pi}`` to report, from
            ``("plateau", "harmonic", "arithmetic", "local")``.
        lag: the Kramers-Moyal lag of ``"km_q"``, in frames.
        n_diff_bins: the committor grid of the profiles and the reductions.
        strict: raise on the first failing estimator (default); False records
            an estimator's ``ValueError``/``RuntimeError`` under ``errors`` and
            carries on with the rest of the bundle (the committor fit itself
            always raises).

    Returns:
        dict with ``committor`` (fit diagnostics), ``q_samples``, ``D_s``,
        ``cv_grad_sq``, ``profiles`` (``levels``, ``counts``, ``pi``,
        ``grad_sq``, and per constructor ``D_q`` and ``flux``), ``rates`` (per
        constructor, per reduction: :func:`sliced_committor.rate_from_profiles`
        without its profiles), ``flux_cv`` (per constructor: the constancy of
        the flux over ``QUALITY_BAND``), ``kramers`` (the baseline) and
        ``errors``.
    """
    unknown = sorted(set(diffusion) - set(DIFFUSIONS))
    if unknown:
        raise ValueError(f"unknown diffusion constructor(s) {unknown}; choose from {DIFFUSIONS}")
    unknown = sorted(set(reductions) - set(REDUCTIONS))
    if unknown:
        raise ValueError(f"unknown reduction(s) {unknown}; choose from {REDUCTIONS}")

    X = np.asarray(features, dtype=np.float64)
    in_A = np.asarray(dataset.in_A, dtype=bool)
    in_B = np.asarray(dataset.in_B, dtype=bool)
    sw = np.asarray(sample_weights, dtype=np.float64)
    solver: dict[str, Any] = {
        "n_bins": n_bins,
        "binning_method": "equal_width",
        "sample_weights": sw,
        **solver_kwargs,
    }
    q, detail = fit_committor(
        X,
        in_A=in_A,
        in_B=in_B,
        n_directions=n_directions,
        seed=seed,
        tikhonov=tikhonov,
        return_details=True,
        **solver,
    )
    out: dict[str, Any] = {
        "committor": {
            "n_directions": int(np.asarray(detail.result.directions).shape[0]),
            "n_features": int(X.shape[1]),
            "valid_fraction": float(np.mean(np.asarray(detail.result.valid_mask))),
            "dirichlet_energy": float(detail.weights.dirichlet_energy),
            "moment_gap": float(detail.weights.moment_gap),
            "tikhonov": str(tikhonov),
        },
        "errors": {},
    }
    del detail  # the (M, N) projected samples are not needed past this point

    # one autodiff pass gives the per-sample committor AND its squared gradient
    qv, grad_sq = _values_and_grad_sq(q, X, bridge_metric)
    out["q_samples"] = qv
    wid = np.asarray(dataset.window_ids)
    dt = float(dataset.dt)
    s = progress_coordinate(dataset)

    def attempt(key, fn):
        try:
            return fn()
        except (ValueError, RuntimeError) as exc:
            if strict:
                raise
            logger.warning("%s failed: %s", key, exc)
            out["errors"][key] = f"{type(exc).__name__}: {exc}"
            return None

    pi = density(qv, sample_weights=sw, n_bins=n_diff_bins)
    rho_A, rho_B = basin_populations(qv, sample_weights=sw, in_A=in_A, in_B=in_B)
    g_q = _grad_sq_profile(qv, grad_sq, sample_weights=sw, n_bins=n_diff_bins)
    out["profiles"] = {
        "levels": np.asarray(pi.levels),
        "counts": np.asarray(pi.counts),
        "pi": np.asarray(pi.values),
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
        "cv_grad_sq", lambda: linear_response_grad_sq(s, X, sw, metric=bridge_metric)
    )

    constructors = {
        "cvmap": lambda: committor_diffusion_from_cv(
            g_q, D_s=out["D_s"]["value"], cv_grad_sq=out["cv_grad_sq"]
        ),
        "hummer_q": lambda: hummer_diffusion(qv, wid, dt=dt, run_ids=run_ids),
        "km_q": lambda: diffusion_profile(
            qv, dt=dt, lag=lag, window_ids=wid, run_ids=run_ids, n_bins=n_diff_bins
        ),
    }
    out["rates"], out["flux_cv"] = {}, {}
    for name in diffusion:
        if name == "cvmap" and (out["D_s"] is None or out["cv_grad_sq"] is None):
            out["errors"]["cvmap"] = "no D_s or CV gradient to map (see the errors above)"
            continue
        D_q = attempt(name, constructors[name])
        if D_q is None:
            continue
        nu, D_q_grid = flux_profiles(pi, D_q)
        out["profiles"]["flux"][name] = np.asarray(nu.values)
        out["profiles"]["D_q"][name] = np.asarray(D_q_grid.values)
        out["flux_cv"][name] = flux_flatness(nu, QUALITY_BAND)
        out["rates"][name] = {}
        for reduction in reductions:
            rate = attempt(
                f"{name}/{reduction}",
                lambda r=reduction, D=D_q: rate_from_profiles(pi, D, rho_A, rho_B, reduction=r),
            )
            if rate is not None:
                out["rates"][name][reduction] = _plain(rate)

    out["kramers"] = attempt("kramers", lambda: kramers_baseline(dataset, sw, run_ids=run_ids))
    return out
