"""The committor as a callable JAX function.

The sliced committor is just a pure function ``q(x) -> committor``: project x
onto each direction, interpolate the 1D slice committors, recombine with the
solved weights. Expressing it as a closure makes it the single, intuitive
object a user works with -- and because it is a plain JAX function,
``jax.grad(q)`` (wrapped as :func:`committor_gradient`) yields the spatial
gradient ``∇q`` for free.

Two entry points:

* :func:`fit_committor` -- one shot: samples + basin labels in, callable out.
* :func:`build_committor` -- from an already-computed result + weights.

The recombination here is a direct (autodiff-friendly) linear/centered/
smoothstep sum that reproduces the solver's evaluation to float64 precision;
it deliberately avoids the log-space normalisation path so the gradient is
numerically clean.
"""

from collections.abc import Callable
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp

from .solver import (
    SlicedCommittorResult,
    _apply_boundary_conditions,
    _evaluate_centered_from_qall,
    _evaluate_powered_smoothstep_from_qall,
    _qall_onthefly,
    compute_basin_moment_weights,
    compute_enriched_basin_moment_weights,
    compute_enriched_basin_moment_weights_power,
    compute_full_gram_weights,
    compute_sliced_committor,
    compute_weights_multi,
)
from .weights import corrected_dirichlet_inv_rd

Weights = jnp.ndarray | dict


class CommittorFit(NamedTuple):
    """Bundle returned by :func:`fit_committor` when ``return_details=True``.

    ``committor`` is the callable ``q(x)``; ``result`` and ``weights`` are the
    underlying :class:`SlicedCommittorResult` and weight array/dict, kept for
    diagnostics (``result.summary()``, ``why_masked``,
    ``summarize_gram_diagnostics``). ``dirichlet_energy`` is the variational
    objective 𝓓[q̂] of the recombined committor (see
    :func:`committor_dirichlet_energy`), a label-free relative quality ranker
    (lower = closer to the true committor). Anything else a caller might want
    (per-sample committor, masked-slice reasons) is derivable from ``result``.
    """

    committor: Callable
    result: SlicedCommittorResult
    weights: Weights
    dirichlet_energy: float | None = None


# Weight-solver registry for the ``weights=`` string shortcut in fit_committor.
_WEIGHT_SOLVERS = {
    "ebmc": compute_enriched_basin_moment_weights,
    "pesb": compute_enriched_basin_moment_weights_power,
    "bmc": compute_basin_moment_weights,
    "full_gram": compute_full_gram_weights,
}
_DIAGONAL_ALIASES = {"diagonal", "corrected_dirichlet_inv_rd", "rd"}


def _resolve_combiner(weights: Weights):
    """Resolve a weight array/dict into ``(kind, payload)`` for recombination.

    Classifies the weights into the recombination ansatz, in precedence order:
    PESB smoothstep > centered-basis (EBMC) > plain array. The array path uses
    the linear normalised combiner ``(Σ wⱼ qⱼ) / Σ wⱼ``, which equals the
    solver's signed-logsumexp evaluator up to floating-point round-off.
    """
    if isinstance(weights, dict):
        if "w_by_power" in weights and "n_values" in weights:
            return "pesb", (
                jnp.asarray(weights["w_by_power"]),
                jnp.asarray(weights["n_values"]),
                float(weights.get("c", 0.0)),
            )
        if "c" in weights or weights.get("q_bar") is not None:
            w = jnp.asarray(weights["w"])
            qb = weights.get("q_bar")
            q_bar = jnp.asarray(qb) if qb is not None else jnp.zeros_like(w)
            return "centered", (w, float(weights.get("c", 0.0)), q_bar)
        if "w" in weights:
            return "array", jnp.asarray(weights["w"])
        if "sign_w" in weights and "log_abs_w" in weights:
            w = jnp.asarray(weights["sign_w"]) * jnp.exp(jnp.asarray(weights["log_abs_w"]))
            return "array", w
        raise ValueError(
            "weights dict missing recognised keys; expected one of "
            "'w', 'c'/'q_bar', 'w_by_power'+'n_values', or 'sign_w'+'log_abs_w'."
        )
    return "array", jnp.asarray(weights)


def _energy_basis(weights) -> str:
    """Which formula :func:`committor_dirichlet_energy` uses for these weights.

    ``"gram"`` for the Gram-family solvers (a dict carrying ``G``: ebmc / pesb /
    bmc / full_gram), whose energy is the true ``wᵀG w``; ``"diagonal"`` for the
    diagonal RD / bare-array solver, whose energy is the smaller-scale diagonal
    approximation. Ranking the ``"dirichlet"`` energy across the two is not
    valid. Shared by :func:`committor_dirichlet_energy` and the sweep selector.
    """
    return "gram" if isinstance(weights, dict) and "G" in weights else "diagonal"


def build_committor(
    result: SlicedCommittorResult,
    weights: Weights,
    *,
    clip: bool = True,
    enforce_boundary_conditions: bool = True,
    rescale_transition: bool = False,
) -> Callable:
    """Build the callable committor ``q(points, *, in_A=None, in_B=None)``.

    Args:
        result: a :class:`SlicedCommittorResult` from
            :func:`compute_sliced_committor`.
        weights: a ``(M,)`` array or a weight-solver dict (raw ``w``,
            centered-basis EBMC, PESB smoothstep, or signed log-space). The
            form is resolved once here, so the returned closure is monomorphic
            (jittable and differentiable).
        clip: clip the output to ``[0, 1]`` (default True).
        enforce_boundary_conditions: when True (default) and the caller passes
            ``in_A`` / ``in_B`` masks for the query points, snap those points to
            q=0 / q=1. The gradient path (no masks) is never snapped.
        rescale_transition: affine-rescale the non-basin region to span
            ``[0, 1]`` before snapping (only with boundary masks).

    Returns:
        ``committor``: a function mapping ``(P, dim)`` -> ``(P,)`` (or a single
        ``(dim,)`` point -> scalar). Pure, jittable, and differentiable via
        :func:`committor_gradient` / ``jax.grad``.
    """
    directions = result.directions
    s_coords = result.slice_coords
    q_1d = jnp.where(jnp.isnan(result.committors_1d), 0.0, result.committors_1d)
    valid = result.valid_mask
    kind, payload = _resolve_combiner(weights)

    def _q_bar(points_flat, original_shape):
        q_all = _qall_onthefly(directions, s_coords, q_1d, points_flat)  # (M, P)
        if kind == "array":
            w = payload
            w_eff = w * valid.astype(w.dtype)
            Z = jnp.sum(w_eff)
            Z_safe = jnp.where(jnp.abs(Z) > 0, Z, 1.0)
            return ((w_eff @ q_all) / Z_safe).reshape(original_shape)
        if kind == "centered":
            w, c, q_bar_off = payload
            return _evaluate_centered_from_qall(q_all, w, c, q_bar_off, valid, original_shape)
        w_by_power, n_values, c = payload
        return _evaluate_powered_smoothstep_from_qall(
            q_all, w_by_power, n_values, c, valid, original_shape
        )

    def committor(points, *, in_A=None, in_B=None):
        points = jnp.asarray(points)
        original_shape = points.shape[:-1]
        points_flat = points.reshape(-1, points.shape[-1])
        q_res = _q_bar(points_flat, original_shape)
        if clip:
            q_res = jnp.clip(q_res, 0.0, 1.0)
        if enforce_boundary_conditions and (in_A is not None or in_B is not None):
            q_res = _apply_boundary_conditions(
                q_res, in_A, in_B, original_shape, rescale_transition
            )
        return q_res

    return committor


def committor_gradient(committor: Callable, points: jnp.ndarray) -> jnp.ndarray:
    """Spatial gradient ``∇q(x)`` of a callable committor via autodiff.

    Differentiates the *smooth* committor (no boundary snapping): autodiff
    through the piecewise-linear slice interpolation reproduces the analytic
    slice slope ``∂q/∂s = Δq/Δs``, so ``∇q = Σ_m w_m q'_m(θ_m·x) θ_m``.

    Args:
        committor: a callable from :func:`build_committor` / :func:`fit_committor`.
        points: ``(P, dim)`` (or a single ``(dim,)`` point).

    Returns:
        ``(P, dim)`` gradients (or ``(dim,)`` for a single point).
    """
    points = jnp.asarray(points)
    dim = points.shape[-1]
    original_shape = points.shape[:-1]
    points_flat = points.reshape(-1, dim)

    def _scalar_q(x):
        return committor(x)

    grads = jax.vmap(jax.grad(_scalar_q))(points_flat)  # (P, dim)
    return grads.reshape((*original_shape, dim))


def committor_dirichlet_energy(
    result: SlicedCommittorResult,
    weights: Weights,
    *,
    mode: str = "auto",
) -> float:
    """Dirichlet energy 𝓓[q̂] = ∫ (∇q̂)ᵀ D ∇q̂ ρ of the recombined committor.

    This is the variational objective the weight solve minimises, expressed in
    slice space as the quadratic form ``wᵀG w`` (the RD / β=1, D=1 convention).
    By the variational principle a lower value is closer to the true committor,
    so it is a label-free *relative* quality ranker for model selection. It
    needs no diffusion D, no lag, and no trajectory: it is the static
    ``<D|∇q̄|²>`` object, distinct from the rate-calibrated config-space
    :func:`~sliced_committor.rates.dirichlet_rate`.

    The energy is the *physical* gradient energy ``wᵀG w`` (the true
    ``∫(∇q̂)²ρ``), computed consistently with :func:`build_committor`'s combiner
    so it is the energy of the committor actually built. It deliberately
    excludes the solver's Tikhonov ridge, so it is comparable across the
    Gram-family solvers (ebmc / pesb / bmc / full_gram all share the same Gram):

    * EBMC / PESB (centered / smoothstep ansatz): ``wᵀG w``. For EBMC the ansatz
      is ``c + Σ_j w_j q_j`` against the ``(M, M)`` Gram; for PESB it is
      ``c + Σ_{j,k} w_{j,k} Ψ_k(q_j)`` against the ``(M·P, M·P)`` Kronecker-lifted
      augmented Gram, with ``w`` the flattened ``(M·P,)`` weight vector. (The
      solver's reported ``optimal_dirichlet_energy = 1/M_gap`` is the
      *regularised* metric ``wᵀ(G+ηI)w`` and differs by the small ridge term.)
    * full_gram / bmc (normalised array combiner ``(Σ w_j q_j)/Σw``):
      ``w̃ᵀG w̃`` with the same effective weights ``w̃ = w·valid / Σ(w·valid)``.
    * diagonal / raw ``(M,)`` array (no Gram returned): the diagonal
      approximation ``Σ_j w̃_j² D_j^RD`` from ``result.log_dirichlet`` (the
      diagonal solver's own model; off-diagonal coupling and the per-slice
      relative normalisation are not captured, so it is only loosely comparable
      to the Gram-family energies).

    Args:
        result: the :class:`SlicedCommittorResult` from the fit.
        weights: the weight array/dict returned by the weight solver.
        mode: ``"auto"`` (default: Gram form when available, else the diagonal
            fallback), ``"gram"`` (require a Gram / reported energy; raise if
            absent), or ``"diagonal"`` (always use the ``log_dirichlet`` form).

    Returns:
        the Dirichlet energy as a float (0.0 if no valid slice contributes).
    """
    if mode not in ("auto", "gram", "diagonal"):
        raise ValueError(f"mode must be 'auto', 'gram', or 'diagonal'; got {mode!r}.")

    valid = jnp.asarray(result.valid_mask)
    kind, payload = _resolve_combiner(weights)
    G = weights["G"] if _energy_basis(weights) == "gram" else None

    # Effective combination weights, matching build_committor exactly. Invalid
    # slices are masked out so the energy is the gradient energy of the committor
    # actually built (idempotent for the library solvers, whose w is already
    # valid-masked; the guard makes the contract hold for hand-built dicts too).
    if kind == "centered":
        w = jnp.asarray(payload[0])  # constant bias drops from the gradient
        w_eff = w * valid.astype(w.dtype)
    elif kind == "pesb":
        wbp = jnp.asarray(payload[0])  # (M, P) enriched weights
        w_eff = (wbp * valid.astype(wbp.dtype)[:, None]).reshape(-1)  # (M*P,)
    else:  # "array": full_gram / bmc / diagonal -> normalised combiner
        w = jnp.asarray(payload)
        w_eff = w * valid.astype(w.dtype)
        Z = jnp.sum(w_eff)
        Z_safe = jnp.where(jnp.abs(Z) > 0, Z, 1.0)
        w_eff = w_eff / Z_safe

    # Physical gradient energy wᵀG w (excludes the solver's Tikhonov ridge).
    if mode != "diagonal" and G is not None:
        Gm = jnp.asarray(G)
        return float(w_eff @ (Gm @ w_eff))

    if mode == "gram":
        raise ValueError(
            "mode='gram' requires the weight solver to return a Gram matrix 'G' "
            "(ebmc/pesb/bmc/full_gram); got a bare weight array. Use mode='auto' "
            "or 'diagonal'."
        )

    # Diagonal fallback: Σ_j w̃_j² D_j^RD from per-slice log Dirichlet energy.
    if kind != "array":
        raise ValueError(
            f"the {kind!r} ansatz needs its Gram matrix; the diagonal fallback "
            "applies only to the array combiner."
        )
    if result.log_dirichlet is None:
        raise ValueError(
            "diagonal Dirichlet energy needs result.log_dirichlet, which is None. "
            "Recompute the sliced committor (it is populated by default)."
        )
    log_D = jnp.asarray(result.log_dirichlet)
    # Contribute only finite-energy, non-zero-weight slices. Masked slices carry
    # log_D = +inf; guarding the product (not just D_j) also avoids the
    # 0 * exp(huge) -> NaN that a zero-weight slice with an overflowing log_D
    # would otherwise poison the sum with.
    contributes = jnp.isfinite(log_D) & (w_eff != 0)
    safe_log_D = jnp.where(contributes, log_D, 0.0)
    term = jnp.where(contributes, (w_eff**2) * jnp.exp(safe_log_D), 0.0)
    return float(jnp.sum(term))


def _solve_weights(result, samples, weights, weight_kwargs):
    """Resolve the ``weights=`` argument of fit_committor to an array/dict."""
    if isinstance(weights, str):
        key = weights.lower()
        if key in _DIAGONAL_ALIASES:
            return compute_weights_multi(result, [corrected_dirichlet_inv_rd], samples=samples)[
                "corrected_dirichlet_inv_rd"
            ]
        if key not in _WEIGHT_SOLVERS:
            choices = sorted([*_WEIGHT_SOLVERS, "diagonal"])
            raise ValueError(
                f"unknown weights={weights!r}; choose from {choices} "
                "or pass a precomputed array/dict or a callable(result, samples)."
            )
        return _WEIGHT_SOLVERS[key](result, samples, **weight_kwargs)
    if callable(weights):
        return weights(result, samples, **weight_kwargs)
    return weights  # precomputed array or dict


def fit_committor(
    samples: jnp.ndarray,
    *,
    in_A: jnp.ndarray,
    in_B: jnp.ndarray,
    weights: str | Weights | Callable = "ebmc",
    n_directions: int = 256,
    return_details: bool = False,
    seed: int = 42,
    weight_kwargs: dict | None = None,
    build_kwargs: dict | None = None,
    **solver_kwargs: Any,
):
    """Fit a sliced committor in one call and return the callable ``q(x)``.

    Wraps :func:`compute_sliced_committor` + a weight solve +
    :func:`build_committor`. The internals (the result, the weights) are hidden
    by default; pass ``return_details=True`` to get them back for diagnostics.

    Args:
        samples: ``(N, dim)`` configurations.
        in_A, in_B: ``(N,)`` bool basin-membership masks.
        weights: which weight solver to use. A string
            (``"ebmc"`` default, ``"pesb"``, ``"bmc"``, ``"full_gram"``,
            ``"diagonal"``), a callable ``solver(result, samples, **weight_kwargs)``,
            or a precomputed weight array/dict.
        n_directions: number of projection directions.
        return_details: if True, return ``(q, CommittorFit)`` instead of ``q``.
        seed: random seed for direction sampling.
        weight_kwargs: extra kwargs forwarded to the weight solver.
        build_kwargs: extra kwargs forwarded to :func:`build_committor`
            (e.g. ``enforce_boundary_conditions``, ``clip``).
        **solver_kwargs: extra kwargs forwarded to
            :func:`compute_sliced_committor` (e.g. ``n_bins``, ``rd_kappa``,
            ``boundary_quantile``, ``sample_weights``, ``directions``).

    Returns:
        ``q`` (callable) by default, or ``(q, CommittorFit)`` when
        ``return_details=True``.
    """
    samples = jnp.asarray(samples)
    result = compute_sliced_committor(
        samples, in_A=in_A, in_B=in_B, n_directions=n_directions, seed=seed, **solver_kwargs
    )
    w = _solve_weights(result, samples, weights, weight_kwargs or {})
    q = build_committor(result, w, **(build_kwargs or {}))
    if return_details:
        energy = committor_dirichlet_energy(result, w)
        return q, CommittorFit(committor=q, result=result, weights=w, dirichlet_energy=energy)
    return q
