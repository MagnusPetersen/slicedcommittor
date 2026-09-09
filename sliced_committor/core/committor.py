"""The committor as a callable JAX function.

``q(x) = c + sum_j w_j q_j(theta_j . x)``: project ``x`` onto each direction,
interpolate the 1D slice committors, recombine with the solved weights. As a
pure JAX function it differentiates, so :func:`committor_gradient` is
``jax.grad``. Two entry points: :func:`fit_committor` (samples and labels in,
callable out) and :func:`build_committor` (from a slice basis and weights).
"""

from collections.abc import Callable
from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import jit

from ._ebmc import Weights, solve_weights
from .solver import SlicedCommittorResult, _qall_onthefly, compute_sliced_committor


class CommittorFit(NamedTuple):
    """What :func:`fit_committor` returns with ``return_details=True``: the
    callable, the slice basis and the solved :class:`Weights`."""

    committor: Callable
    result: SlicedCommittorResult
    weights: Weights

    @property
    def dirichlet_energy(self) -> float:
        """The variational objective ``w^T G w`` of this fit (lower is better)."""
        return self.weights.dirichlet_energy


@partial(jit, static_argnames=("original_shape",))
def _combine(q_all, w, c, valid_mask, original_shape):
    """``c + sum_j w_j clip(q_j(x))`` over the valid slices."""
    w_eff = w * valid_mask.astype(w.dtype)
    q_clip = jnp.clip(q_all, 0.0, 1.0)
    return (c + jnp.sum(w_eff[:, None] * q_clip, axis=0)).reshape(original_shape)


def build_committor(
    result: SlicedCommittorResult, weights: Weights, *, clip: bool = True
) -> Callable:
    """The callable committor ``q(points, *, in_A=None, in_B=None)``.

    A pure function of the points: ``(P, dim) -> (P,)`` (a single ``(dim,)``
    point gives a scalar), clipped to ``[0, 1]`` unless ``clip=False``. Passing
    basin masks for the points snaps them to ``q = 0`` in A and ``q = 1`` in B;
    the gradient path never snaps. For a display-ready field see
    :func:`rescale_transition`.
    """
    directions = result.directions
    s_coords = result.slice_coords
    q_1d = jnp.where(jnp.isnan(result.committors_1d), 0.0, result.committors_1d)
    valid = jnp.asarray(result.valid_mask)
    w = jnp.asarray(weights.w)
    c = float(weights.c)

    def committor(points, *, in_A=None, in_B=None):
        points = jnp.asarray(points)
        original_shape = points.shape[:-1]
        points_flat = points.reshape(-1, points.shape[-1])
        q = _combine(
            _qall_onthefly(directions, s_coords, q_1d, points_flat), w, c, valid, original_shape
        )
        if clip:
            q = jnp.clip(q, 0.0, 1.0)
        if in_A is not None:
            q = jnp.where(jnp.asarray(in_A).reshape(original_shape), 0.0, q)
        if in_B is not None:
            q = jnp.where(jnp.asarray(in_B).reshape(original_shape), 1.0, q)
        return q

    return committor


def rescale_transition(q, in_A, in_B):
    """Affine-rescale the transition region of an evaluated field to span ``[0, 1]``.

    A weighted average of slice committors can compress the range between the
    basins. This maps the non-basin values of ``q`` to ``[0, 1]`` by their
    minimum and maximum over the batch, then snaps A to 0 and B to 1. It is a
    property of the batch, not of the committor function, which is why it is
    a post-processing step and not an option of the callable.
    """
    q = jnp.asarray(q)
    mask_A = jnp.asarray(in_A).reshape(q.shape)
    mask_B = jnp.asarray(in_B).reshape(q.shape)
    in_transition = ~mask_A & ~mask_B
    q_trans = jnp.where(in_transition, q, jnp.nan)
    q_min = jnp.nanmin(q_trans)
    span = jnp.maximum(jnp.nanmax(q_trans) - q_min, 1e-10)
    q = jnp.where(in_transition, (q - q_min) / span, q)
    q = jnp.where(mask_A, 0.0, q)
    return jnp.where(mask_B, 1.0, q)


def committor_gradient(committor: Callable, points) -> jnp.ndarray:
    """``grad q(x)`` by autodiff; ``(P, dim)`` for ``(P, dim)`` points.

    Differentiates the smooth committor (no snapping). Autodiff through the
    piecewise-linear slice interpolation gives the analytic slice slope, so
    ``grad q = sum_j w_j q_j'(theta_j . x) theta_j``.
    """
    points = jnp.asarray(points)
    dim = points.shape[-1]
    original_shape = points.shape[:-1]
    grads = jax.vmap(jax.grad(lambda x: committor(x)))(points.reshape(-1, dim))
    return grads.reshape((*original_shape, dim))


def fit_committor(
    samples,
    *,
    in_A,
    in_B,
    n_directions: int = 256,
    seed: int = 42,
    tikhonov="halfset_eigen",
    heldout_cap: bool = False,
    return_details: bool = False,
    **solver_kwargs,
):
    """Fit a sliced committor in one call and return the callable ``q(x)``.

    :func:`compute_sliced_committor` (the slice basis), :func:`solve_weights`
    (the weights) and :func:`build_committor` (the callable), in that order.

    Args:
        samples: ``(N, dim)`` configurations.
        in_A, in_B: ``(N,)`` bool basin labels.
        n_directions: number of slices ``M``.
        seed: PRNG seed for the direction draw.
        tikhonov: ``'halfset_eigen'`` (default) | ``'auto'`` | absolute ridge.
        heldout_cap: also read the out-of-sample Dirichlet cap
            (``fit.weights.heldout_cap``), for ranking settings without a
            reference committor.
        return_details: return ``(q, CommittorFit)`` instead of ``q``.
        **solver_kwargs: forwarded to :func:`compute_sliced_committor`
            (``n_bins``, ``binning_method``, ``boundary_quantile``,
            ``sample_weights``, ``directions``, ``direction_sampling``,
            ``feature_metric``, ...).
    """
    result = compute_sliced_committor(
        jnp.asarray(samples),
        in_A=in_A,
        in_B=in_B,
        n_directions=n_directions,
        seed=seed,
        **solver_kwargs,
    )
    weights = solve_weights(result, tikhonov=tikhonov, heldout_cap=heldout_cap)
    q = build_committor(result, weights)
    if return_details:
        return q, CommittorFit(committor=q, result=result, weights=weights)
    return q
