"""Direction sampling for the slice basis.

Two samplers. ``directions_uniform`` draws unit vectors uniformly on the
sphere. The LDA cone concentrates part of the draw around the Fisher
discriminant axis between the basins: a stratified mixture

    p = alpha p_unif + (1 - alpha) p_PS(. ; v, kappa(mu, d)),

with ``p_PS(x; v, kappa) ~ (1 + v . x)^kappa`` the power-spherical density
around the axis ``v`` and ``kappa(mu, d) = mu (d - 1) / (1 - mu)`` fixed by the
dimension-invariant mean cosine ``mu``. The uniform part is a pointwise
coverage floor ``p >= alpha p_unif``.

:class:`DirectionSamplingConfig` drives the draw inside
:func:`sliced_committor.compute_sliced_committor`. The pieces are also public
(:func:`compute_lda_axis`, :func:`sample_power_spherical_mixture`) so the
axis can be inspected, replaced by a coordinate of your own, or the draw
reproduced outside the solver and passed back as ``directions=``.
"""

import warnings
from functools import partial
from typing import NamedTuple

import jax.numpy as jnp
from jax import jit, random


class DirectionSamplingConfig(NamedTuple):
    """How :func:`~sliced_committor.compute_sliced_committor` draws its directions.

    Attributes:
        mode: ``'uniform'`` (default) or ``'lda'`` (the Fisher cone).
        lda_shrinkage: ridge on the pooled within-class covariance, as a
            fraction of ``trace / dim``.
        mu: mean cosine of the power-spherical component, in ``[0, 1)``;
            0 is uniform. The paper selects ``mu`` per system by the held-out cap.
        alpha: uniform mixture weight in ``[0, 1]``; 1 is pure uniform, 0 has
            no coverage floor (warned below 0.1).
        axis: optional user-supplied ``(dim,)`` axis, skipping the LDA.
    """

    mode: str = "uniform"
    lda_shrinkage: float = 1e-2
    mu: float = 0.4
    alpha: float = 0.5
    axis: jnp.ndarray | None = None


def _normalize_rows(u, eps=1e-30):
    norms = jnp.linalg.norm(u, axis=-1, keepdims=True)
    return u / jnp.maximum(norms, eps)


@partial(jit, static_argnums=(1, 2))
def directions_uniform(key, n_directions: int, dim: int):
    """``(n_directions, dim)`` unit vectors uniform on the sphere."""
    raw = random.normal(key, shape=(n_directions, dim))
    norms = jnp.linalg.norm(raw, axis=-1, keepdims=True)
    return raw / jnp.maximum(norms, 1e-10)


def _class_mean(samples, mask):
    w = mask.astype(samples.dtype)
    n = jnp.sum(w)
    return n, (w[:, None] * samples).sum(axis=0) / jnp.maximum(n, 1.0)


def compute_lda_axis(samples, in_A, in_B, *, shrinkage: float = 1e-2):
    """The Fisher discriminant axis between the basins, as a unit vector.

    ``Sigma_w^-1 (mu_B - mu_A)`` with the pooled within-class covariance
    ridged by ``shrinkage * trace(Sigma_w) / dim``. Returns ``(axis, info)``;
    ``info['mean_mahalanobis']`` is the basin separation in the whitened metric.
    """
    samples = jnp.asarray(samples)
    in_A = jnp.asarray(in_A).astype(bool)
    in_B = jnp.asarray(in_B).astype(bool)
    n_A, mu_A = _class_mean(samples, in_A)
    n_B, mu_B = _class_mean(samples, in_B)
    n_A_i, n_B_i = int(n_A), int(n_B)
    if n_A_i == 0 or n_B_i == 0:
        raise ValueError(
            f"compute_lda_axis: need at least one sample in each state (n_A={n_A_i}, n_B={n_B_i})."
        )
    delta = mu_B - mu_A
    dim = samples.shape[1]
    diffs_A = samples[in_A] - mu_A
    diffs_B = samples[in_B] - mu_B
    n_pool = jnp.maximum(n_A + n_B - 2.0, 1.0)
    Sigma_w = (diffs_A.T @ diffs_A + diffs_B.T @ diffs_B) / n_pool
    ridge = shrinkage * jnp.trace(Sigma_w) / jnp.maximum(dim, 1)
    raw_axis = jnp.linalg.solve(Sigma_w + ridge * jnp.eye(dim, dtype=Sigma_w.dtype), delta)
    mahal = jnp.sqrt(jnp.sum(delta * raw_axis))
    norm = jnp.linalg.norm(raw_axis)
    if float(norm) < 1e-12:
        raise ValueError(
            "compute_lda_axis: the discriminant direction has near-zero norm (mu_A ~ mu_B)."
        )
    axis = raw_axis / norm
    info = {
        "method": "fisher",
        "n_A": n_A_i,
        "n_B": n_B_i,
        "ridge_used": float(ridge),
        "mean_mahalanobis": float(mahal),
        "axis": axis,
    }
    return axis, info


@partial(jit, static_argnums=(2, 3))
def _sample_power_spherical(key, axis, n: int, dim: int, kappa):
    """``n`` unit vectors from the power-spherical density around ``axis``.

    ``u ~ Beta(kappa + (d-1)/2, (d-1)/2)``, ``t = 2u - 1`` is the cosine to the
    axis; the orthogonal part is uniform on the axis' orthogonal complement.
    """
    key_t, key_w = random.split(key)
    a1 = kappa + (dim - 1.0) / 2.0
    a2 = (dim - 1.0) / 2.0
    u = random.beta(key_t, a1, a2, shape=(n,))
    t = jnp.clip(2.0 * u - 1.0, -1.0 + 1e-7, 1.0 - 1e-7)
    z = random.normal(key_w, shape=(n, dim))
    z_perp = z - (z @ axis)[:, None] * axis[None, :]
    w = _normalize_rows(z_perp, eps=1e-10)
    sqrt_1mt2 = jnp.sqrt(jnp.maximum(1.0 - t * t, 0.0))
    return t[:, None] * axis[None, :] + sqrt_1mt2[:, None] * w


def sample_power_spherical_mixture(key, axis, n_directions: int, dim: int, *, mu=0.0, alpha=1.0):
    """Stratified mixture of uniform and power-spherical directions on the sphere.

    Args:
        key: JAX PRNG key.
        axis: ``(dim,)`` unit axis of the cone, or None for pure uniform.
        n_directions: total number of directions.
        dim: ambient dimension (>= 2).
        mu: mean cosine of the power-spherical component, in ``[0, 1)``.
        alpha: uniform mixture weight in ``[0, 1]``.

    Stratification is exact: ``round((1 - alpha) N)`` directions in the cone
    and the rest uniform. The output order (uniform rows first) is an
    implementation detail.
    """
    if dim < 2:
        raise ValueError(f"dim must be >= 2, got {dim}")
    if not (0.0 <= mu < 1.0):
        raise ValueError(f"mu must be in [0, 1), got {mu}")
    if not (0.0 <= alpha <= 1.0):
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")
    if axis is None or alpha >= 1.0:
        return directions_uniform(key, n_directions, dim)

    axis = jnp.asarray(axis)
    if axis.shape != (dim,):
        raise ValueError(f"axis must have shape (dim={dim},), got {axis.shape}")
    if bool(jnp.abs(jnp.linalg.norm(axis) - 1.0) > 1e-4):
        warnings.warn("sample_power_spherical_mixture: renormalizing the axis.", stacklevel=2)
    axis = axis / jnp.maximum(jnp.linalg.norm(axis), 1e-12)

    n_cone = int(round((1.0 - alpha) * n_directions))
    n_uniform = n_directions - n_cone
    key_uniform, key_cone = random.split(key)
    kappa = jnp.asarray(mu * (dim - 1.0) / (1.0 - mu), dtype=axis.dtype)
    parts = []
    if n_uniform > 0:
        parts.append(directions_uniform(key_uniform, n_uniform, dim))
    if n_cone > 0:
        parts.append(_sample_power_spherical(key_cone, axis, n_cone, dim, kappa))
    return parts[0] if len(parts) == 1 else jnp.concatenate(parts, axis=0)


def _sample_directions(key, n_directions, dim, samples, in_A, in_B, cfg: DirectionSamplingConfig):
    """The draw :func:`compute_sliced_committor` makes for a config: ``(directions, info)``."""
    if cfg.mode == "uniform":
        return directions_uniform(key, n_directions, dim), None
    if cfg.mode != "lda":
        raise ValueError(
            f"DirectionSamplingConfig.mode must be 'uniform' or 'lda'; got {cfg.mode!r}"
        )
    if cfg.axis is not None:
        axis = jnp.asarray(cfg.axis)
        axis = axis / jnp.maximum(jnp.linalg.norm(axis), 1e-12)
        info = {
            "method": "user_supplied",
            "n_A": int(jnp.sum(in_A)),
            "n_B": int(jnp.sum(in_B)),
            "ridge_used": 0.0,
            "mean_mahalanobis": float("nan"),
            "axis": axis,
        }
    else:
        axis, info = compute_lda_axis(samples, in_A, in_B, shrinkage=cfg.lda_shrinkage)
    if cfg.alpha < 0.1:
        warnings.warn(
            f"DirectionSamplingConfig: alpha={cfg.alpha} < 0.1 leaves almost no uniform coverage floor.",
            stacklevel=2,
        )
    directions = sample_power_spherical_mixture(
        key, axis, n_directions, dim, mu=float(cfg.mu), alpha=float(cfg.alpha)
    )
    info["mu"] = float(cfg.mu)
    info["alpha"] = float(cfg.alpha)
    return directions, info
