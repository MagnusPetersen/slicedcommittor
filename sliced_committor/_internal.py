"""Shared low-level helpers used by solver, weights, and calibration."""

from functools import partial

import jax
import jax.numpy as jnp
from jax import jit, random


@partial(jit, static_argnums=(1, 2))
def sample_random_directions(key: random.PRNGKey, n_directions: int, dim: int) -> jnp.ndarray:
    """Sample ``n_directions`` unit vectors uniformly on the (dim-1)-sphere.

    Each direction is an independent draw; in high dimensions, near-duplicates
    are extremely unlikely.
    """
    raw = random.normal(key, shape=(n_directions, dim))
    norms = jnp.linalg.norm(raw, axis=-1, keepdims=True)
    return raw / jnp.maximum(norms, 1e-10)


def resolve_basin_labels(result, in_A, in_B, caller: str):
    """Return ``(in_A, in_B)``, falling back to ``result.in_A`` / ``result.in_B``.

    Raises ValueError with a ``caller``-tagged message when neither the
    override nor the fallback is available.
    """
    if in_A is None:
        in_A = result.in_A
    if in_B is None:
        in_B = result.in_B
    if in_A is None or in_B is None:
        raise ValueError(
            f"{caller} needs basin labels. Pass in_A / in_B explicitly, "
            "or ensure they were stored on the result via "
            "compute_sliced_committor."
        )
    return in_A, in_B


def to_log_abs_sign(w):
    """Convert a signed array to its (sign, log|·|) representation.

    Zero maps to (sign=0, log_abs=-inf), which is the additive identity
    under ``signed_logsumexp``.
    """
    abs_w = jnp.abs(w)
    sign_w = jnp.sign(w)
    log_abs_w = jnp.where(abs_w > 0, jnp.log(abs_w), -jnp.inf)
    return sign_w, log_abs_w


def signed_logsumexp(sign, log_abs, axis=None, keepdims=False):
    """Compute (sgn, log|·|) of Σ sign·exp(log_abs) along ``axis``.

    Splits the sum into positive and negative parts, logsumexps each,
    then combines via log|e^lp − e^ln|. Exact cancellation returns
    (sign=0, log_abs=-inf) without NaN.
    """
    log_pos = jnp.where(sign > 0, log_abs, -jnp.inf)
    log_neg = jnp.where(sign < 0, log_abs, -jnp.inf)
    lp = jax.scipy.special.logsumexp(log_pos, axis=axis, keepdims=keepdims)
    ln = jax.scipy.special.logsumexp(log_neg, axis=axis, keepdims=keepdims)
    m = jnp.maximum(lp, ln)
    both_neg_inf = jnp.isneginf(lp) & jnp.isneginf(ln)
    safe_lp = jnp.where(both_neg_inf, 0.0, lp)
    safe_ln = jnp.where(both_neg_inf, 0.0, ln)
    diff = jnp.abs(safe_lp - safe_ln)
    out_log_abs = jnp.where(both_neg_inf, -jnp.inf, m + jnp.log1p(-jnp.exp(-diff)))
    out_sign = jnp.where(both_neg_inf, 0.0, jnp.sign(safe_lp - safe_ln))
    return out_sign, out_log_abs
