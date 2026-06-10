"""
Direction sampling for the sliced committor.

Default sampler: uniform on the unit sphere (``sample_directions`` with the
default ``DirectionSamplingConfig``).

Alternative samplers (factories that return ``(M, dim)`` direction matrices):

  * ``directions_uniform``: baseline, no bias.
  * ``sample_power_spherical_mixture``: power-spherical + uniform mixture
    with a user-supplied informed axis (LDA, dominant CV, etc.). The
    mixture is

        p_mix = α · p_unif + (1 − α) · Σ_k β_k · p_PS(·; v_k, κ(μ, d))

    where ``p_PS(x; v, κ) ∝ (1 + v·x)^κ`` and κ is reparameterised by the
    dimension-invariant mean cosine ``μ = E[v·x]``,
    ``κ(μ, d) = μ(d − 1)/(1 − μ)``. Stratified sampling gives a pointwise
    coverage floor ``p_mix(x) ≥ α · p_unif(x)``.
  * ``directions_tica_ema``: EMA-integrated TICA covariance basis with
    linear, no-truncation coloring. Single hyperparameter ``alpha_ratio``
    (default 30). Accepts a single time-ordered trajectory or a sequence
    of independent trajectories of possibly different lengths.
  * ``directions_tica_ema_decomposed``: bias-aware TICA-EMA for
    umbrella-sampling data (per-window features + MBAR weights). Splits
    the slow-mode estimator into a Q-orthogonal within-window EMA
    (Bartlett bandwidth 1/√N_k) and a Q-aligned between-window mean
    covariance, combined via trace-matching. Window-relabeling
    invariant. Single hyperparameter ``ridge``.

Helper: ``compute_lda_axis`` for the LDA bias direction.
"""

import warnings
from collections.abc import Sequence
from functools import partial
from typing import Any, Dict, List, NamedTuple, Optional, Tuple, Union

import jax
import jax.numpy as jnp
import numpy as np
from jax import jit, random

from ._internal import sample_random_directions

# Geometric modes that compute K canonical bias axes and feed them through
# the power-spherical + uniform mixture sampler. The mixture sampler is
# unchanged; only the source of ``bias_axes`` differs.
GEOMETRIC_MODES = ("pca", "gcpca")


class DirectionSamplingConfig(NamedTuple):
    """
    Configuration for direction sampling in ``compute_sliced_committor``.

    Fields:
        mode: One of:
            ``'uniform'``  : baseline, no bias (default).
            ``'lda'``      : single-axis LDA bias; see ``lda_method``.
            ``'pca'``      : top-/bottom-K PCA bias axes (label-free).
            ``'gcpca'``    : generalised contrastive PCA v4.1 bias axes
                             (target = ``A ∪ B``, background by
                             ``gcpca_background``).
        lda_method: ``'mean_diff'`` | ``'fisher'``. ``'fisher'`` is the
            full within-class covariance with shrinkage and is the
            recommended default; ``'mean_diff'`` is the no-whitening
            baseline.
        lda_shrinkage: Ridge coefficient for ``'fisher'`` (fraction of
            ``trace(Σ_w)/dim``).
        mu: Mean cosine of the power-spherical component, in [0, 1).
            Dimension-invariant. μ=0 ⇒ PS component is uniform. Default
            0.4 (villin (μ, α) sweep optimum at ε=equilibrium).
        alpha: Uniform mixture weight, in [0, 1]. α=1 is pure uniform;
            α=0 is pure informed (no coverage floor, dispatcher warns).
            Default 0.5 (sweep-validated; near-identical RMSE for α∈[0.3, 0.7]
            but α=0.5 is the safest middle ground for varied sample sizes).
        n_bias_axes: K, number of bias axes for the geometric modes
            (``'pca'``, ``'gcpca'``); ignored by ``'uniform'`` and
            ``'lda'``. Default 4.
        pca_variant: ``'top'`` (largest variance) | ``'bottom'`` (smallest
            variance; unsupervised slow-mode analogue).
        gcpca_background: ``'bulk'`` = samples outside ``A ∪ B``;
            ``'all'`` = all samples.
        gcpca_ridge: numerical ridge on ``C_t + C_bg``.
        axis: Optional user-supplied unit axis; skips axis computation
            (only honoured by ``mode='lda'``).
        axis_weights: Optional (K,) mixture weights over informed axes
            for multi-axis bias. For the geometric modes the default is
            eigenvalue-magnitude-weighted (more informative axes get more
            power-spherical samples).
    """

    mode: str = "uniform"
    lda_method: str = "fisher"
    lda_shrinkage: float = 1e-2
    mu: float = 0.4
    alpha: float = 0.5
    # Geometric modes ---------------------------------------------------------
    n_bias_axes: int = 4
    pca_variant: str = "top"
    gcpca_background: str = "bulk"
    gcpca_ridge: float = 1e-6
    # User-supplied overrides -------------------------------------------------
    axis: jnp.ndarray | None = None
    axis_weights: jnp.ndarray | None = None


def _weighted_class_stats(
    samples: jnp.ndarray, mask: jnp.ndarray
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Return (N, μ, Σ_diag) for the masked subset of ``samples``."""
    w = mask.astype(samples.dtype)
    n = jnp.sum(w)
    denom = jnp.maximum(n, 1.0)
    mu = (w[:, None] * samples).sum(axis=0) / denom
    diffs = samples - mu
    var_diag = (w[:, None] * diffs * diffs).sum(axis=0) / denom
    return n, mu, var_diag


def compute_lda_axis(
    samples: jnp.ndarray,
    in_A: jnp.ndarray,
    in_B: jnp.ndarray,
    method: str = "fisher",
    shrinkage: float = 1e-2,
) -> tuple[jnp.ndarray, dict]:
    """
    Compute a unit-norm discriminant axis between state A and state B samples.

    Args:
        samples: (N, dim) sample array (any feature space).
        in_A, in_B: (N,) boolean masks.
        method: ``'mean_diff'`` | ``'fisher'``.  ``'fisher'`` (default)
            uses the full pooled within-class covariance with ridge
            ``η = shrinkage · trace(Σ_w)/dim``; ``'mean_diff'`` is the
            no-whitening baseline.
        shrinkage: Ridge for ``'fisher'``. ``η = shrinkage · trace(Σ_w)/dim``.

    Returns:
        (axis, info) where ``axis`` is a (dim,) unit vector and ``info`` is
        a dict with diagnostic fields: ``method``, ``n_A``, ``n_B``,
        ``ridge_used``, ``mean_mahalanobis``.
    """
    samples = jnp.asarray(samples)
    in_A = jnp.asarray(in_A).astype(bool)
    in_B = jnp.asarray(in_B).astype(bool)

    n_A, mu_A, _ = _weighted_class_stats(samples, in_A)
    n_B, mu_B, _ = _weighted_class_stats(samples, in_B)

    n_A_i = int(n_A)
    n_B_i = int(n_B)
    if n_A_i == 0 or n_B_i == 0:
        raise ValueError(
            f"compute_lda_axis: need at least one sample in each state (n_A={n_A_i}, n_B={n_B_i})."
        )

    delta = mu_B - mu_A
    dim = samples.shape[1]

    ridge_used = 0.0

    if method == "mean_diff":
        raw_axis = delta
        mahal = jnp.linalg.norm(delta)
    elif method == "fisher":
        samples_A = samples[in_A]
        samples_B = samples[in_B]
        diffs_A = samples_A - mu_A
        diffs_B = samples_B - mu_B
        n_pool = jnp.maximum(n_A + n_B - 2.0, 1.0)
        Sigma_w = (diffs_A.T @ diffs_A + diffs_B.T @ diffs_B) / n_pool
        tr = jnp.trace(Sigma_w)
        ridge = shrinkage * tr / jnp.maximum(dim, 1)
        ridge_used = float(ridge)
        Sigma_reg = Sigma_w + ridge * jnp.eye(dim, dtype=Sigma_w.dtype)
        raw_axis = jnp.linalg.solve(Sigma_reg, delta)
        mahal = jnp.sqrt(jnp.sum(delta * raw_axis))
    else:
        raise ValueError(f"Unknown lda_method '{method}'. Expected 'mean_diff' or 'fisher'.")

    norm = jnp.linalg.norm(raw_axis)
    if float(norm) < 1e-12:
        raise ValueError(
            "compute_lda_axis: discriminant direction has near-zero norm "
            "(μ_A ≈ μ_B). Check state definitions."
        )
    axis = raw_axis / norm

    info = {
        "method": method,
        "n_A": n_A_i,
        "n_B": n_B_i,
        "ridge_used": ridge_used,
        "mean_mahalanobis": float(mahal),
        "axis": axis,
    }
    return axis, info


# =============================================================================
# Power-spherical + uniform mixture sampler
# =============================================================================


@partial(jit, static_argnums=(2, 3))
def _sample_power_spherical(
    key: random.PRNGKey, axis: jnp.ndarray, n: int, dim: int, kappa: jnp.ndarray
) -> jnp.ndarray:
    """Sample n unit vectors from PS(axis; κ) on S^{dim-1}.

    Recipe:
      u ~ Beta(κ + (d-1)/2, (d-1)/2);  t = 2u - 1           # cosine to axis
      z ~ N(0, I_d); w = (z - (z·axis) axis) / ‖…‖           # uniform on axis⊥
      x = t·axis + √(1 − t²)·w
    """
    key_t, key_w = random.split(key)

    a1 = kappa + (dim - 1.0) / 2.0
    a2 = (dim - 1.0) / 2.0
    u = random.beta(key_t, a1, a2, shape=(n,))
    t = 2.0 * u - 1.0
    t = jnp.clip(t, -1.0 + 1e-7, 1.0 - 1e-7)

    z = random.normal(key_w, shape=(n, dim))
    z_perp = z - (z @ axis)[:, None] * axis[None, :]
    z_perp_norm = jnp.linalg.norm(z_perp, axis=-1, keepdims=True)
    w = z_perp / jnp.maximum(z_perp_norm, 1e-10)

    sqrt_1mt2 = jnp.sqrt(jnp.maximum(1.0 - t * t, 0.0))
    return t[:, None] * axis[None, :] + sqrt_1mt2[:, None] * w


def sample_power_spherical_mixture(
    key: random.PRNGKey,
    bias_axes: jnp.ndarray | None,
    n_directions: int,
    dim: int,
    *,
    mu: float = 0.0,
    alpha: float = 1.0,
    axis_weights: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """Stratified mixture of uniform and power-spherical components on S^{d-1}.

    Args:
        key: JAX PRNG key.
        bias_axes: (K, dim) unit vectors, or None for pure uniform.
        n_directions: total number of directions to return.
        dim: ambient dimension; must be ≥ 2.
        mu: mean cosine of the PS component, in [0, 1). Dimension-invariant.
        alpha: uniform mixture weight, in [0, 1]. α=1 ⇒ pure uniform.
        axis_weights: (K,) mixture weights over informed axes;
            defaults to uniform over K.

    Returns:
        (n_directions, dim) array of unit vectors. Output ordering is
        ``[uniform rows, PS rows grouped by axis]``; this is an
        implementation detail, callers should not rely on it.

    Stratification is exact: per-stratum counts are computed as
    ``round((1-α) β_k N)`` for the PS components, with rounding drift
    absorbed into the uniform bucket so the total is exactly N.
    """
    if dim < 2:
        raise ValueError(f"dim must be >= 2, got {dim}")
    if not (0.0 <= mu < 1.0):
        raise ValueError(f"mu must be in [0, 1), got {mu}")
    if not (0.0 <= alpha <= 1.0):
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")

    if bias_axes is None or alpha >= 1.0:
        return sample_random_directions(key, n_directions, dim)

    bias_axes = jnp.asarray(bias_axes)
    if bias_axes.ndim != 2 or bias_axes.shape[1] != dim:
        raise ValueError(f"bias_axes must have shape (K, dim={dim}), got {bias_axes.shape}")

    norms = jnp.linalg.norm(bias_axes, axis=-1)
    if bool(jnp.any(jnp.abs(norms - 1.0) > 1e-4)):
        warnings.warn(
            "sample_power_spherical_mixture: renormalizing bias_axes rows (norm deviation > 1e-4).",
            stacklevel=2,
        )
    bias_axes = bias_axes / jnp.maximum(norms[:, None], 1e-12)

    K = int(bias_axes.shape[0])

    if axis_weights is None:
        beta_w = np.ones(K, dtype=np.float64) / K
    else:
        beta_w = np.asarray(axis_weights, dtype=np.float64)
        if beta_w.shape != (K,):
            raise ValueError(f"axis_weights must have shape ({K},), got {beta_w.shape}")
        s = float(beta_w.sum())
        if not np.isclose(s, 1.0, atol=1e-6):
            raise ValueError(f"axis_weights must sum to 1, got {s}")

    # Stratification: Python-side so inner samplers can be jitted with
    # static shapes.
    n_per_axis = [int(round((1.0 - alpha) * float(beta_w[k]) * n_directions)) for k in range(K)]
    n_uniform = int(n_directions - sum(n_per_axis))
    if n_uniform < 0:
        # Rare: rounding pushed PS sum over N. Trim the largest axis.
        k_max = int(np.argmax(n_per_axis))
        n_per_axis[k_max] += n_uniform  # subtract overflow
        n_uniform = 0

    # Key splitting: one key per component.
    keys = random.split(key, K + 1)
    key_uniform = keys[0]
    keys_ps = keys[1:]

    # κ from μ: dimension-invariant reparameterisation.
    kappa = jnp.asarray(mu * (dim - 1.0) / (1.0 - mu), dtype=bias_axes.dtype)

    parts = []
    if n_uniform > 0:
        parts.append(sample_random_directions(key_uniform, n_uniform, dim))
    for k in range(K):
        if n_per_axis[k] > 0:
            parts.append(
                _sample_power_spherical(
                    keys_ps[k],
                    bias_axes[k],
                    n_per_axis[k],
                    dim,
                    kappa,
                )
            )

    if len(parts) == 1:
        return parts[0]
    return jnp.concatenate(parts, axis=0)


# =============================================================================
# Dispatcher
# =============================================================================


def sample_directions(
    key: random.PRNGKey,
    n_directions: int,
    dim: int,
    samples: jnp.ndarray,
    in_A: jnp.ndarray,
    in_B: jnp.ndarray,
    cfg: DirectionSamplingConfig,
) -> tuple[jnp.ndarray, dict | None]:
    """
    Top-level dispatcher used by ``compute_sliced_committor``.

    For ``mode='uniform'`` returns ``(directions, None)``.
    For ``mode='lda'`` computes the axis via ``compute_lda_axis`` (unless
    ``cfg.axis`` is supplied), then draws from the power-spherical +
    uniform mixture parameterised by ``(cfg.mu, cfg.alpha)``.
    For the geometric modes (``'pca'``, ``'gcpca'``) computes
    ``cfg.n_bias_axes`` axes via the corresponding factory and feeds them
    through the same mixture; per-axis weights default to the
    eigenvalue-magnitude simplex unless ``cfg.axis_weights`` is supplied.
    """
    if cfg.mode == "uniform":
        return sample_random_directions(key, n_directions, dim), None

    if cfg.mode == "lda":
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
            axis, info = compute_lda_axis(
                samples,
                in_A,
                in_B,
                method=cfg.lda_method,
                shrinkage=cfg.lda_shrinkage,
            )

        if cfg.alpha < 0.1:
            warnings.warn(
                f"DirectionSamplingConfig: alpha={cfg.alpha} < 0.1 is a very "
                "thin coverage floor; the full-Gram solver may have no "
                "transverse probes on S^{d-1}. Consider alpha >= 0.2.",
                stacklevel=2,
            )

        directions = sample_power_spherical_mixture(
            key,
            axis[None, :],
            n_directions,
            dim,
            mu=float(cfg.mu),
            alpha=float(cfg.alpha),
            axis_weights=cfg.axis_weights,
        )
        info["mu"] = float(cfg.mu)
        info["alpha"] = float(cfg.alpha)
        return directions, info

    if cfg.mode in GEOMETRIC_MODES:
        K = int(cfg.n_bias_axes)
        if K < 1:
            raise ValueError(f"DirectionSamplingConfig.n_bias_axes must be ≥ 1; got {K}.")

        if cfg.mode == "pca":
            axes, eigvals, info = pca_basis(
                samples,
                K,
                variant=cfg.pca_variant,
            )
        else:  # 'gcpca'
            axes, eigvals, info = gcpca_basis(
                samples,
                in_A,
                in_B,
                K,
                background=cfg.gcpca_background,
                ridge=float(cfg.gcpca_ridge),
            )

        if cfg.alpha < 0.1:
            warnings.warn(
                f"DirectionSamplingConfig: alpha={cfg.alpha} < 0.1 is a very "
                "thin coverage floor; the full-Gram solver may have no "
                "transverse probes on S^{d-1}. Consider alpha >= 0.2.",
                stacklevel=2,
            )

        if cfg.axis_weights is None:
            # Eigenvalue-magnitude simplex: more informative axes get more
            # power-spherical samples. ``|λ|`` keeps sign-ambiguous spectra
            # (gcPCA can return negative λ) valid.
            abs_eig = jnp.abs(eigvals)
            total = float(jnp.sum(abs_eig))
            if total > 0:
                aw = abs_eig / total
            else:
                aw = jnp.ones(int(axes.shape[0]), dtype=axes.dtype) / float(axes.shape[0])
        else:
            aw = cfg.axis_weights

        directions = sample_power_spherical_mixture(
            key,
            axes,
            n_directions,
            dim,
            mu=float(cfg.mu),
            alpha=float(cfg.alpha),
            axis_weights=aw,
        )
        info["mu"] = float(cfg.mu)
        info["alpha"] = float(cfg.alpha)
        info["n_bias_axes"] = int(cfg.n_bias_axes)
        return directions, info

    raise ValueError(
        f"Unknown direction_sampling mode '{cfg.mode}'. Expected one of: "
        f"'uniform', 'lda', {', '.join(repr(m) for m in GEOMETRIC_MODES)}."
    )


def select_endpoint_windows(in_A, in_B, max_frames: int) -> tuple[np.ndarray, int, int]:
    """Pick a contiguous data window starting at the first A-frame and at the
    first B-frame, take the next ``max_frames`` frames from each, then merge.

    Used to study data efficiency: the sliced committor needs both endpoints
    to be present in the input, so a naive "first N frames" subsample is
    biased. This protocol guarantees both endpoints are visited regardless
    of how small ``max_frames`` is.

    Args:
        in_A: (N,) bool, frame-wise membership in state A. NumPy or JAX.
        in_B: (N,) bool, frame-wise membership in state B. NumPy or JAX.
        max_frames: per-endpoint window size.

    Returns:
        idx: sorted, deduplicated NumPy int array of selected frame indices
            (length is at most ``2 * max_frames``; less if the windows overlap
            or run past the trajectory end).
        idx_A_start: first frame in state A.
        idx_B_start: first frame in state B.

    Raises:
        IndexError: if either state is never visited.
    """
    in_A_np = np.asarray(in_A)
    in_B_np = np.asarray(in_B)
    idx_A_start = int(np.flatnonzero(in_A_np)[0])
    idx_B_start = int(np.flatnonzero(in_B_np)[0])
    n_total = in_A_np.shape[0]
    win_A = np.arange(idx_A_start, min(idx_A_start + max_frames, n_total))
    win_B = np.arange(idx_B_start, min(idx_B_start + max_frames, n_total))
    idx = np.unique(np.concatenate([win_A, win_B]))
    return idx, idx_A_start, idx_B_start


# ---------------------------------------------------------------------------
# Direction-sampling factories
# ---------------------------------------------------------------------------


def _normalize_rows(u: jnp.ndarray, eps: float = 1e-30) -> jnp.ndarray:
    norms = jnp.linalg.norm(u, axis=-1, keepdims=True)
    return u / jnp.maximum(norms, eps)


# ---------------------------------------------------------------------------
# Geometric-basis factories (PCA, gcPCA).
# Output contract: ``<name>_basis(samples, K, ...) -> (axes, eigvals, info)``
# where ``axes`` is ``(K, dim)`` unit-norm, ready to feed into
# :func:`sample_power_spherical_mixture` as ``bias_axes``.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# PCA: top-K (or bottom-K) eigenvectors of the sample covariance.
# Label-free, pure geometry.
# ---------------------------------------------------------------------------


def pca_basis(
    samples: jnp.ndarray,
    K: int,
    *,
    variant: str = "top",
    center: bool = True,
) -> tuple[jnp.ndarray, jnp.ndarray, dict[str, Any]]:
    """Top-K or bottom-K principal-component axes of ``samples``.

    Args:
        samples: ``(N, dim)`` feature matrix.
        K: number of axes to return.
        variant: ``'top'`` (default) returns the K eigenvectors with the
            largest variance, the standard PCA basis. ``'bottom'`` returns
            the K eigenvectors with the smallest variance (unsupervised
            analogue of TICA's slow modes in the zero-lag limit; the
            directions along which the sample cloud is tightest).
        center: subtract the sample mean before forming the covariance
            (default True). Set False when ``samples`` is already centered
            or when you want raw second-moment eigenvectors.

    Returns:
        ``(axes, eigvals, info)`` where ``axes`` is ``(K, dim)`` unit-norm,
        ``eigvals`` is ``(K,)`` per-axis variance, and ``info`` carries
        the full eigenvalue spectrum.
    """
    X = jnp.asarray(samples)
    N, dim = X.shape
    Xc = X - X.mean(axis=0, keepdims=True) if center else X

    _, S, Vt = jnp.linalg.svd(Xc, full_matrices=False)
    eigvals_full = (S * S) / max(N - 1, 1)
    V_cols = Vt.T

    n_avail = V_cols.shape[1]
    K_eff = min(int(K), n_avail)
    if K_eff < int(K):
        warnings.warn(
            f"pca_basis: requested K={int(K)} but only {n_avail} components "
            f"are available (limited by min(N, dim)). Returning K={K_eff}.",
            stacklevel=2,
        )

    if variant == "top":
        axes = V_cols[:, :K_eff].T
        eigvals = eigvals_full[:K_eff]
    elif variant == "bottom":
        axes = V_cols[:, n_avail - K_eff :].T[::-1]
        eigvals = eigvals_full[n_avail - K_eff :][::-1]
    else:
        raise ValueError(f"variant must be 'top' or 'bottom'; got {variant!r}.")

    axes = _normalize_rows(axes)
    info: dict[str, Any] = {
        "method": "pca",
        "variant": variant,
        "eigenvalues": eigvals,
        "eigenvalues_full": eigvals_full,
        "n_samples": int(N),
        "dim": int(dim),
    }
    return axes, eigvals, info


def directions_pca(samples: jnp.ndarray, K: int, *, variant: str = "top") -> jnp.ndarray:
    """Convenience: return only the ``(K, dim)`` unit-norm PCA axes."""
    axes, _, _ = pca_basis(samples, K, variant=variant)
    return axes


# ---------------------------------------------------------------------------
# gcPCA v4.1 (symmetric, orthogonal generalised contrastive PCA).
#
# Reference: Schissler et al. 2025 (PLoS Comput. Biol.). Hyperparameter-free
# replacement for cPCA: leading axes carry second-moment contrasts that
# distinguish target (``A ∪ B``) from background, invisible to LDA's
# mean-shift direction.
# ---------------------------------------------------------------------------


def gcpca_basis(
    samples: jnp.ndarray,
    in_A: jnp.ndarray,
    in_B: jnp.ndarray,
    K: int,
    *,
    background: str = "bulk",
    ridge: float = 1e-6,
    center: bool = True,
) -> tuple[jnp.ndarray, jnp.ndarray, dict[str, Any]]:
    """Top-K generalised contrastive PCA axes of ``A ∪ B`` against background.

    Solves the v4.1 symmetric pencil
    ``(C_target − C_bg) v = λ (C_target + C_bg) v`` via Cholesky whitening
    of the sum ``C_target + C_bg + η I``.

    Args:
        samples: ``(N, dim)`` feature matrix.
        in_A, in_B: ``(N,)`` boolean masks.
        K: number of axes to return.
        background: ``'bulk'`` (default) = samples outside ``A ∪ B``;
            ``'all'`` = all samples (standard cPCA convention).
        ridge: numerical stabiliser; ``η = ridge · trace(C_t + C_bg) / dim``.
        center: subtract per-class mean before forming covariances.

    Returns:
        ``(axes, eigvals, info)``. ``axes`` is ``(K, dim)`` unit-norm.
        ``eigvals[k]`` is the Rayleigh quotient ``λ_k ∈ [-1, 1]``: positive
        means target-enriched, negative means background-enriched.
    """
    X = jnp.asarray(samples)
    in_A = jnp.asarray(in_A).astype(bool)
    in_B = jnp.asarray(in_B).astype(bool)
    is_target = in_A | in_B
    if background == "bulk":
        is_bg = ~is_target
    elif background == "all":
        is_bg = jnp.ones_like(is_target)
    else:
        raise ValueError(f"background must be 'bulk' or 'all'; got {background!r}.")

    X_t = X[is_target]
    X_b = X[is_bg]
    n_t = int(X_t.shape[0])
    n_b = int(X_b.shape[0])
    if n_t < 2 or n_b < 2:
        raise ValueError(
            f"gcpca_basis: need ≥2 target and ≥2 background samples; "
            f"got n_target={n_t}, n_background={n_b}."
        )

    if center:
        X_t = X_t - X_t.mean(axis=0, keepdims=True)
        X_b = X_b - X_b.mean(axis=0, keepdims=True)

    dim = X.shape[1]
    C_t = (X_t.T @ X_t) / max(n_t - 1, 1)
    C_b = (X_b.T @ X_b) / max(n_b - 1, 1)
    S_sum = C_t + C_b
    S_dif = C_t - C_b
    eta = float(ridge) * float(jnp.trace(S_sum)) / max(dim, 1)
    eye = jnp.eye(dim, dtype=X.dtype)
    B = S_sum + eta * eye

    L = jnp.linalg.cholesky(B)
    Linv = jnp.linalg.solve(L, eye)
    A_white = Linv @ S_dif @ Linv.T
    A_white = 0.5 * (A_white + A_white.T)
    lam, U = jnp.linalg.eigh(A_white)
    lam = lam[::-1]
    U = U[:, ::-1]
    V = Linv.T @ U

    n_avail = V.shape[1]
    K_eff = min(int(K), n_avail)
    axes = _normalize_rows(V[:, :K_eff].T)
    eigvals = lam[:K_eff]

    info: dict[str, Any] = {
        "method": "gcpca",
        "variant": "v4.1",
        "background": background,
        "n_target": n_t,
        "n_background": n_b,
        "ridge_used": eta,
        "eigenvalues": eigvals,
        "eigenvalues_full": lam,
    }
    return axes, eigvals, info


def directions_gcpca(
    samples: jnp.ndarray, in_A: jnp.ndarray, in_B: jnp.ndarray, K: int, *, background: str = "bulk"
) -> jnp.ndarray:
    """Convenience: return only the ``(K, dim)`` unit-norm gcPCA axes."""
    axes, _, _ = gcpca_basis(samples, in_A, in_B, K, background=background)
    return axes


# ---------------------------------------------------------------------------
# Public factories
# ---------------------------------------------------------------------------


def directions_uniform(
    key: random.PRNGKey,
    M: int,
    dim: int,
) -> jnp.ndarray:
    """Uniform on the (dim-1)-sphere."""
    raw = random.normal(key, shape=(M, dim))
    return _normalize_rows(raw)


def _color_normalize_directions(
    key: random.PRNGKey,
    M: int,
    V: jnp.ndarray,
    mu: jnp.ndarray,
) -> jnp.ndarray:
    """Linear coloring + sphere normalization shared by the TICA-EMA factories.

    ``w_k = max(μ_k, 0)``; draw ``q ~ N(0, I_d)``; return unit-norm rows of
    ``(q ⊙ w) Vᵀ``. The unit-sphere normalization is the implicit truncation
    of negative-eigenvalue modes.
    """
    d = V.shape[0]
    w = jnp.maximum(mu, 0.0)
    q = random.normal(key, shape=(M, d), dtype=V.dtype)
    return _normalize_rows((q * w[None, :]) @ V.T)


def _backward_ema_scan(phi: jnp.ndarray, decay: jnp.ndarray) -> jnp.ndarray:
    """Backward EMA over time axis of a (N, d) trajectory.

    phi_bar_t = phi_t + decay * phi_bar_{t+1},   phi_bar_{N-1} = phi_{N-1}.

    Single-pass O(N·d) scan. The integrated estimator
    M_K ≈ (phi.T @ phi_bar) / N is a (d, d) accumulation done outside.
    """

    def step(carry, x):
        new = x + decay * carry
        return new, new

    rev = phi[::-1]
    init = jnp.zeros(phi.shape[1], dtype=phi.dtype)
    _, out_rev = jax.lax.scan(step, init, rev)
    return out_rev[::-1]


# ---------------------------------------------------------------------------
# Multi-trajectory EMA covariance helpers
# ---------------------------------------------------------------------------


def _normalize_traj_input(
    samples: jnp.ndarray | Sequence[jnp.ndarray],
) -> list[jnp.ndarray]:
    """Coerce ``samples`` into a list of (N_k, d) jnp arrays.

    Accepts either a single (N, d) array or a Python sequence of (N_k, d)
    arrays. All trajectories must share the second dimension and dtype;
    lengths may differ.
    """
    if isinstance(samples, (list, tuple)):
        if len(samples) == 0:
            raise ValueError("samples must contain at least one trajectory")
        trajs = [jnp.asarray(t) for t in samples]
    else:
        trajs = [jnp.asarray(samples)]

    d = trajs[0].shape[-1]
    dtype = trajs[0].dtype
    for t in trajs:
        if t.ndim != 2:
            raise ValueError(f"each trajectory must be 2D (N_k, d); got shape {t.shape}")
        if t.shape[-1] != d:
            raise ValueError(f"all trajectories must share dim={d}; got {t.shape[-1]}")
        if t.dtype != dtype:
            raise ValueError(f"all trajectories must share dtype={dtype}; got {t.dtype}")
        if t.shape[0] < 1:
            raise ValueError("each trajectory must have at least 1 frame")
    return trajs


def _ema_covariances_batched(
    padded: jnp.ndarray,
    decay: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Aggregated EMA covariances for a (K, T_max, d) batch of trajectories.

    Trailing-zero padding is supported: for each row k, valid frames occupy
    indices [0, N_k); padded slots [N_k, T_max) must be zero. The valid
    portion must already be mean-centered. Padded entries contribute exactly
    zero to both sums, and the backward EMA scan's zero initial carry
    propagates correctly through trailing zero rows until reaching each
    trajectory's actual last valid frame at t = N_k - 1.

    When all N_k are equal, T_max = N_k and there is no padding overhead.

    Returns un-normalized ``(Σ_k Φ_kᵀ Φ_k, Σ_k Φ_kᵀ Φ̄_k)``.
    """
    bar = jax.vmap(_backward_ema_scan, in_axes=(0, None))(padded, decay)
    C0_sum = jnp.einsum("knd,kne->de", padded, padded)
    Ma_sum = jnp.einsum("knd,kne->de", padded, bar)
    return C0_sum, Ma_sum


def _ema_covariances(
    samples: jnp.ndarray | Sequence[jnp.ndarray],
    alpha_ratio: float,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Aggregated EMA covariances (C0, M_a) across K ≥ 1 trajectories.

    Decay is shared across trajectories: ``α = alpha_ratio / N_mean`` with
    ``N_mean = N_total / K`` (the arithmetic mean of trajectory lengths).
    This treats α as a property of the question (the timescale being
    probed) rather than the amount of data: adding more replicas of the
    same length improves variance without sliding the kernel toward slower
    modes that the individual trajectories cannot resolve. For K = 1 this
    reduces to ``α = alpha_ratio / N`` and matches the single-trajectory
    behaviour bitwise. The backward EMA carry is reset at every trajectory
    boundary, eliminating cross-trajectory leakage.

    K == 1 keeps the legacy single-trajectory code path. K > 1 builds a
    (K, T_max, d) trailing-padded batch and delegates to
    ``_ema_covariances_batched``. For trajectories of highly skewed lengths
    the padded batch is wasteful in memory; if that becomes a bottleneck,
    bucket the input by length and call this helper once per bucket.
    """
    trajs = _normalize_traj_input(samples)
    K = len(trajs)
    d = trajs[0].shape[-1]
    dtype = trajs[0].dtype

    Ns = [int(t.shape[0]) for t in trajs]
    N_total = int(sum(Ns))
    N_mean = N_total / K

    alpha = jnp.asarray(alpha_ratio / N_mean, dtype=dtype)
    decay = jnp.exp(-alpha)

    if K == 1:
        phi = trajs[0] - trajs[0].mean(axis=0, keepdims=True)
        phi_bar = _backward_ema_scan(phi, decay)
        C0_sum = phi.T @ phi
        Ma_sum = phi.T @ phi_bar
    else:
        sum_acc = jnp.sum(jnp.stack([t.sum(axis=0) for t in trajs], axis=0), axis=0)
        global_mean = (sum_acc / jnp.asarray(N_total, dtype=dtype))[None, :]

        T_max = max(Ns)
        padded = jnp.stack(
            [jnp.pad(t - global_mean, ((0, T_max - int(t.shape[0])), (0, 0))) for t in trajs],
            axis=0,
        )
        C0_sum, Ma_sum = _ema_covariances_batched(padded, decay)

    N_norm = jnp.asarray(N_total, dtype=dtype)
    return C0_sum / N_norm, Ma_sum / N_norm


def _solve_ema_gep(
    C0: jnp.ndarray,
    M_a: jnp.ndarray,
    ridge: float,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Solve the generalized eigenproblem ``sym(M_a) v = μ C₀ v`` via
    Cholesky whitening.

    Returns ``(V, mu)`` where ``V[:, k]`` is the k-th generalized eigenvector
    (columns ordered by decreasing μ_k) and ``mu`` are the eigenvalues.
    """
    d = C0.shape[0]
    dtype = C0.dtype
    M_sym = 0.5 * (M_a + M_a.T)
    eye_d = jnp.eye(d, dtype=dtype)
    L = jnp.linalg.cholesky(C0 + ridge * eye_d)
    Linv = jnp.linalg.solve(L, eye_d)
    Mw = Linv @ M_sym @ Linv.T
    mu, U = jnp.linalg.eigh(0.5 * (Mw + Mw.T))
    mu = mu[::-1]
    U = U[:, ::-1]
    V = Linv.T @ U
    return V, mu


def _tica_ema_basis(
    samples: jnp.ndarray | Sequence[jnp.ndarray],
    alpha_ratio: float,
    ridge: float,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """EMA-integrated covariance basis: solve M_sym v = μ C0 v, slowest-first.

    Accepts either a single ``(N, d)`` trajectory or a sequence of
    ``(N_k, d)`` independent trajectories. The backward EMA is reset at
    every trajectory boundary; the aggregated estimators use a shared
    decay ``α = alpha_ratio / N_mean`` with ``N_mean = N_total / K``.

    Returns ``(V, mu)`` where ``V[:, k]`` is the k-th generalized eigenvector
    (columns ordered by decreasing μ_k) and ``mu`` are the eigenvalues.
    Shared by ``directions_tica_ema`` and downstream diagnostics.
    """
    C0, M_a = _ema_covariances(samples, alpha_ratio)
    return _solve_ema_gep(C0, M_a, ridge)


def directions_tica_ema(
    key: random.PRNGKey,
    M: int,
    samples: jnp.ndarray | Sequence[jnp.ndarray],
    alpha_ratio: float = 30.0,
    ridge: float = 1e-6,
) -> jnp.ndarray:
    """EMA-integrated covariance basis with linear, no-truncation coloring.

    Solves the generalized eigenproblem M_sym v = μ C0 v with
    M_sym = sym((Σ_k Φ_kᵀ Φ̄_k) / N_total) and Φ̄_k = backward_EMA(Φ_k)
    per trajectory. Linear per-mode weights w_k = max(μ_k, 0); no hard
    truncation: the unit-sphere normalization at the end is the implicit
    truncation.

    One hyperparameter beyond the numerical ridge: ``alpha_ratio``.
    Empirically robust band [20, 80]; default 30 ties or beats the prior
    decade-rule recipe across d ∈ {10, 20, 30} on rotated Wolfe-Quapp.

    Args:
        key: JAX PRNG key.
        M: number of directions.
        samples: either a single ``(N, dim)`` time-ordered feature matrix or
            a sequence of ``(N_k, dim)`` independent trajectories. Frames
            within a trajectory must be in time order; multiple trajectories
            are treated as independent runs whose backward EMA carry resets
            at every boundary. All trajectories share dim and dtype;
            lengths may differ. Passing a 1-element list is equivalent to
            passing the bare array.
        alpha_ratio: kernel rate α = alpha_ratio / N_mean where
            N_mean = N_total / K is the arithmetic mean of trajectory
            lengths. Sets the EMA integration window to roughly
            N_mean / alpha_ratio frames. The α is a property of the
            timescale being probed, not of the dataset size, so adding
            more replicas of the same length reduces variance without
            sliding the kernel toward slower modes.
        ridge: stabilising ridge on C(0) for the Cholesky.

    Returns:
        ``(M, dim)`` unit-norm rows.
    """
    V, mu = _tica_ema_basis(samples, alpha_ratio, ridge)
    return _color_normalize_directions(key, M, V, mu)


# ---------------------------------------------------------------------------
# Decomposed TICA-EMA: bias-aware sampler for umbrella-sampling data
# ---------------------------------------------------------------------------
#
# On umbrella-sampling data, ``directions_tica_ema`` requires concatenating
# windows in Q-order so the backward EMA picks up the artificial monotone
# Q-ramp as the "slow mode": that's window-relabeling-dependent, has no
# physical estimand, and needs per-system ``alpha_ratio`` tuning.
#
# The decomposed estimator below replaces the trick with two terms, each of
# which estimates a separate physical quantity under MBAR. Let ``w_n`` be
# MBAR weights (Σ_n w_n = 1) and ``W_k = Σ_{n ∈ k} w_n`` the per-window
# weight. With ``μ_k = W_k^{-1} Σ_{n ∈ k} w_n φ_n`` and
# ``μ_glob = Σ_k W_k μ_k``, the law of total variance gives
#
#     C₀_MBAR = C_within + C_between
#     C_within  = Σ_n w_n δφ_n δφ_nᵀ,            δφ_n = φ_n − μ_{k(n)}
#     C_between = Σ_k W_k (μ_k − μ_glob)(μ_k − μ_glob)ᵀ                (rank ≤ K-1)
#
# Slow-mode estimator (separated by subspace):
#
#     M_within  = Σ_k W_k · (1/N_k) Σ_t δφ_t^(k) · backward_EMA(δφ^(k); α_w^(k))_tᵀ
#                 with Bartlett-style α_w^(k) = 1/√N_k.
#     M_slow    = M_within + β · C_between,    β = tr(M_within) / tr(C_between)
#
# Physics: in the ∇Q-orthogonal subspace the harmonic bias is invisible, so
# the per-window EMA estimates honest lagged covariance there. C_between
# carries the Q-aligned signal directly (variance of MBAR-reweighted window
# means). β has units of time (scale-matching by trace). Scope: stationary
# harmonic biases; time-dependent metadynamics biases are out of scope.


def _normalize_window_inputs(
    samples: Sequence[jnp.ndarray],
    mbar_weights: Sequence[jnp.ndarray],
) -> tuple[list[jnp.ndarray], list[jnp.ndarray]]:
    """Validate parallel per-window samples + MBAR-weight lists.

    Samples validation reuses ``_normalize_traj_input`` (length ≥ 1, shared
    dim/dtype, 2D, ≥ 1 frame each); this helper enforces K ≥ 2, the
    matching weight list length, and per-window weight shapes. MBAR weights
    are cast to the samples' dtype.
    """
    if not isinstance(samples, (list, tuple)):
        raise TypeError(
            "samples must be a sequence of (N_k, dim) arrays; "
            "the decomposed scheme requires K>=2 windows."
        )
    if not isinstance(mbar_weights, (list, tuple)):
        raise TypeError("mbar_weights must be a sequence of (N_k,) arrays parallel to samples.")
    if len(samples) < 2:
        raise ValueError(
            f"directions_tica_ema_decomposed requires at least 2 windows "
            f"(C_between is rank 0 / β is undefined for K=1); got K={len(samples)}."
        )
    if len(mbar_weights) != len(samples):
        raise ValueError(
            f"samples and mbar_weights must have the same length; got "
            f"{len(samples)} vs {len(mbar_weights)}."
        )

    samples_j = _normalize_traj_input(samples)
    dtype = samples_j[0].dtype
    weights_j: list[jnp.ndarray] = []
    for k, (s, w) in enumerate(zip(samples_j, mbar_weights)):
        w_j = jnp.asarray(w, dtype=dtype)
        if w_j.ndim != 1 or w_j.shape[0] != s.shape[0]:
            raise ValueError(
                f"mbar_weights[{k}] must have shape ({s.shape[0]},); got {tuple(w_j.shape)}"
            )
        weights_j.append(w_j)
    return samples_j, weights_j


def _decomposed_estimators(
    samples_j: list[jnp.ndarray],
    weights_j: list[jnp.ndarray],
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Compute (C_between, M_within, C0_MBAR) via MBAR variance decomposition.

    Pads K mixed-length per-window centered trajectories to ``T_max`` with
    trailing zeros and vmaps the backward EMA scan over a per-window decay
    ``decay_k = exp(-1/√N_k)``. Padded slots contribute exactly zero to all
    sums (the EMA carry is zero-initialized and decays back to zero through
    trailing zero rows). Empty / unsampled windows (W_k = 0) are silently
    skipped: their per-window scalar (W_k or W_k/N_k) zeros their
    contribution to every term.
    """
    K = len(samples_j)
    d = samples_j[0].shape[-1]
    dtype = samples_j[0].dtype

    # ---- Per-window aggregates (Python K-loop; K is small for US data) ----
    Ns_list = [int(s.shape[0]) for s in samples_j]
    W_k_list: list[jnp.ndarray] = []
    mu_k_list: list[jnp.ndarray] = []
    eps = jnp.asarray(1e-30, dtype=dtype)
    for s, w in zip(samples_j, weights_j):
        Wk = w.sum()
        sum_wphi = w @ s  # (d,)
        mu = jnp.where(Wk > 0, sum_wphi / jnp.maximum(Wk, eps), jnp.zeros((d,), dtype=dtype))
        W_k_list.append(Wk)
        mu_k_list.append(mu)
    W_k = jnp.stack(W_k_list)  # (K,)
    mu_k = jnp.stack(mu_k_list)  # (K, d)
    mu_glob = (W_k[:, None] * mu_k).sum(axis=0)  # (d,)

    # ---- C_between (Q-aligned signal, rank ≤ K-1) ----
    diff = mu_k - mu_glob[None, :]  # (K, d)
    C_between = (W_k[:, None] * diff).T @ diff

    # ---- Pad centered samples + weights to T_max ----
    T_max = max(Ns_list)
    padded_delta = jnp.stack(
        [jnp.pad(samples_j[k] - mu_k[k], ((0, T_max - Ns_list[k]), (0, 0))) for k in range(K)],
        axis=0,
    )  # (K, T_max, d)
    padded_w = jnp.stack(
        [jnp.pad(weights_j[k], (0, T_max - Ns_list[k])) for k in range(K)],
        axis=0,
    )  # (K, T_max)

    # ---- C_within = Σ_n w_n δφ_n δφ_nᵀ via padded einsum ----
    weighted_delta = padded_w[..., None] * padded_delta
    C_within = jnp.einsum("ktd,kte->de", weighted_delta, padded_delta)
    C0_MBAR = C_within + C_between

    # ---- M_within: per-window backward EMA, Bartlett bandwidth 1/√N_k ----
    Ns_arr = jnp.asarray(Ns_list, dtype=dtype)
    decay = jnp.exp(-1.0 / jnp.sqrt(Ns_arr))  # (K,)
    bar = jax.vmap(_backward_ema_scan, in_axes=(0, 0))(padded_delta, decay)
    c_k = jnp.where(W_k > 0, W_k / Ns_arr, jnp.zeros_like(W_k))  # (K,)
    M_within = jnp.einsum("k,ktd,kte->de", c_k, padded_delta, bar)

    return C_between, M_within, C0_MBAR


def _tica_ema_decomposed_basis(
    samples: Sequence[jnp.ndarray],
    mbar_weights: Sequence[jnp.ndarray],
    ridge: float = 2.0,
) -> tuple[jnp.ndarray, jnp.ndarray, dict[str, jnp.ndarray]]:
    """Decomposed TICA-EMA basis with diagnostic accessors.

    Solves ``sym(M_slow) v = μ (C0_MBAR + ρI) v`` where
    ``M_slow = M_within + β · C_between`` and ``β = tr(M_within)/tr(C_between)``.

    Returns ``(V, mu, diagnostics)``. ``V[:, k]`` is the k-th generalized
    eigenvector (columns ordered by decreasing μ_k). ``diagnostics`` carries:
        ``beta``         : trace-matched scalar.
        ``tr_M_within``  : raw trace (before β scaling).
        ``tr_C_between`` : raw trace (before β scaling).
        ``M_within``, ``C_between``, ``C0_MBAR`` : the (d, d) blocks
            (consumed by section-9 validation: f_Q-aligned per-mode
            decomposition, β-sensitivity sweeps, relabeling-invariance
            checks).
    """
    samples_j, weights_j = _normalize_window_inputs(samples, mbar_weights)
    C_between, M_within, C0_MBAR = _decomposed_estimators(samples_j, weights_j)

    tr_Mw = jnp.trace(M_within)
    tr_Cb = jnp.trace(C_between)
    eps = jnp.asarray(1e-30, dtype=C_between.dtype)
    beta = tr_Mw / jnp.maximum(tr_Cb, eps)
    M_slow = M_within + beta * C_between

    V, mu = _solve_ema_gep(C0_MBAR, M_slow, ridge)
    diagnostics: dict[str, jnp.ndarray] = {
        "beta": beta,
        "tr_M_within": tr_Mw,
        "tr_C_between": tr_Cb,
        "M_within": M_within,
        "C_between": C_between,
        "C0_MBAR": C0_MBAR,
    }
    return V, mu, diagnostics


def directions_tica_ema_decomposed(
    key: random.PRNGKey,
    M: int,
    samples: Sequence[jnp.ndarray],
    mbar_weights: Sequence[jnp.ndarray],
    ridge: float = 2.0,
) -> jnp.ndarray:
    """Bias-aware TICA-EMA via MBAR variance decomposition. One ridge knob.

    For umbrella-sampling data, splits the slow-mode estimator into a
    Q-orthogonal within-window EMA (Bartlett bandwidth ``α_w^(k) = 1/√N_k``)
    and a Q-aligned between-window mean covariance, combined via trace
    matching ``β = tr(M_within)/tr(C_between)``. Inputs are parallel
    per-window lists (mirroring ``ChignolinDataset.per_window_features``).

    The ``ridge`` default of 2.0 was tuned on chignolin cartesian_heavy
    (276-D, K=17, leading C0_MBAR eigenvalue ≈ 5): 5-seed snap RMSE
    0.169±0.003 / ρ_snap 0.938±0.010 (vs 0.262 / 0.614 at ridge=1e-6).
    Other system regimes (flat C0_MBAR spectra in particular, e.g. torsion
    features) prefer smaller ridges; tune empirically. Dropping whitening
    entirely (ρ → ∞) was disconfirmed: the cart_heavy plateau at ρ ∈ {1, 2}
    is *partial* whitening (squashes the trailing-noise tail without
    flattening the leading mode), not the asymptotic limit.

    Args:
        key: JAX PRNG key.
        M: number of directions.
        samples: length-K sequence of ``(N_k, dim)`` per-window feature
            matrices. Frames within a window must be time-ordered.
        mbar_weights: length-K sequence of ``(N_k,)`` MBAR weights,
            parallel to ``samples``. The full collection should sum to 1
            (the MBAR normalization); per-window sums ``W_k`` need not be
            uniform. Windows with ``W_k = 0`` (unsampled) are silently
            skipped.
        ridge: stabilising ridge on ``C0_MBAR`` for the Cholesky whitening.

    Returns:
        ``(M, dim)`` unit-norm rows. Linear coloring with
        ``w_k = max(μ_k, 0)``; unit-sphere normalization is the only
        truncation.
    """
    V, mu, _ = _tica_ema_decomposed_basis(samples, mbar_weights, ridge)
    return _color_normalize_directions(key, M, V, mu)
