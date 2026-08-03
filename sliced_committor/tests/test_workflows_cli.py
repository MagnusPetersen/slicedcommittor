"""Unit tests for US-workflow CLI helpers and direction-mode dispatch.

These cover the pure/logic pieces added for the standalone CLI: the --reference
parser, and build_directions' handling of informed modes (incl. the 1D no-op
guard so cv_only does not trip the power-spherical dim>=2 requirement).
"""

import numpy as np
import pytest


def test_parse_references_basic():
    from sliced_committor.workflows.cli import _parse_references

    assert _parse_references(None) is None
    assert _parse_references([]) is None
    refs = _parse_references(["expt=0.0105:1/us", "msm=4.5e5"])
    assert refs["expt"] == {"k": 0.0105, "units": "1/us", "source": "--reference"}
    assert refs["msm"]["k"] == 4.5e5
    assert refs["msm"]["units"] == "1/s"  # UNITS defaults to 1/s


def test_parse_references_errors():
    from sliced_committor.workflows.cli import _parse_references

    with pytest.raises(SystemExit):
        _parse_references(["missing_equals"])  # no '='
    with pytest.raises(SystemExit):
        _parse_references(["k=not_a_number"])  # non-numeric value


def test_build_directions_uniform_is_none():
    from sliced_committor.workflows.committor_rates import build_directions

    feats = np.random.default_rng(0).normal(size=(40, 3))
    assert build_directions(None, feats, None, "uniform", 64, 0) is None


@pytest.mark.parametrize("mode", ["lda", "pca", "gcpca", "tica_ema", "tica_ema_decomposed"])
def test_build_directions_1d_is_uniform_noop(mode):
    # On a 1D feature (e.g. cv_only) every informed mode collapses to the single
    # axis up to sign, so build_directions returns None (uniform path) instead of
    # hitting the power-spherical mixture's dim>=2 requirement. Regression guard.
    from sliced_committor.workflows.committor_rates import build_directions

    feats_1d = np.linspace(0.0, 1.0, 50)[:, None]
    assert build_directions(None, feats_1d, None, mode, 64, 0) is None


def test_build_directions_lda_2d_returns_supervised_config():
    from sliced_committor import DirectionSamplingConfig
    from sliced_committor.workflows.committor_rates import build_directions

    feats_2d = np.random.default_rng(0).normal(size=(60, 2))
    cfg = build_directions(None, feats_2d, None, "lda", 64, 0)
    assert isinstance(cfg, DirectionSamplingConfig)
    assert cfg.mode == "lda"


def test_lda_is_the_fast_mode_informed_default():
    # The fast sweep should pair uniform with LDA (not TICA).
    from sliced_committor.workflows.config import get_config
    from sliced_committor.workflows.sweep import _sweep_axes

    cfg = get_config("chignolin")
    axes = _sweep_axes("fast", cfg, has_trajectory=True)
    assert axes["direction_modes"] == ("uniform", "lda")


def test_cvmap_plateau_is_feature_scaling_invariant():
    """TPT_cvmap (the CV-mapped flux plateau) is invariant to an isotropic feature
    rescaling f -> alpha*f, which is exactly why it replaced the feature-space
    TPT_cv: the configurational scale D0 = D_s/<|grad s|^2> transforms as alpha^2
    and cancels the gradient's 1/alpha^2, whereas TPT_cv's frozen scalar D_cv does
    not (so TPT_cv = D_cv * <|grad q|^2> scaled as 1/alpha^2).
    """
    import jax.numpy as jnp

    from sliced_committor import (
        committor_gradient,
        committor_rate,
        fit_committor,
        mapped_committor_diffusion,
    )
    from sliced_committor.workflows.committor_rates import _cv_feature_grad_sq

    rng = np.random.default_rng(0)
    n = 3000
    s = np.concatenate([rng.normal(-1.0, 0.35, n // 2), rng.normal(1.0, 0.35, n // 2)])
    F = np.stack([s, rng.normal(0.0, 1.0, n)], axis=1)  # CV + an orthogonal feature
    in_A, in_B = s < -0.6, s > 0.6
    w = np.full(n, 1.0 / n)

    def cvmap_rate_and_grad(scale):
        feats = jnp.asarray(F * scale)
        q = fit_committor(
            feats,
            in_A=jnp.asarray(in_A),
            in_B=jnp.asarray(in_B),
            weights="ebmc",
            n_directions=128,
            seed=0,
            sample_weights=jnp.asarray(w),
        )
        gs = _cv_feature_grad_sq(s, np.asarray(feats), w)
        m_dq = mapped_committor_diffusion(q, feats, D_s=0.05, cv_grad_sq=gs, n_bins=50)
        k = committor_rate(
            q,
            feats,
            feats,
            dt=1.0,
            reduction="plateau",
            at="auto",
            sample_weights=jnp.asarray(w),
            in_A=in_A,
            in_B=in_B,
            n_bins=50,
            D_profile=m_dq,
        )["k_AB"]
        grad_sq = float(np.sum(w * np.sum(np.asarray(committor_gradient(q, feats)) ** 2, axis=1)))
        return float(k), grad_sq

    k1, g1 = cvmap_rate_and_grad(1.0)
    k2, g2 = cvmap_rate_and_grad(2.0)
    # The CV-mapped plateau rate is invariant to the feature scaling ...
    assert abs(k2 - k1) / k1 < 0.02
    # ... precisely because <|grad q|^2> scales as 1/alpha^2 (a frozen-scalar
    # TPT_cv = D_cv * <|grad q|^2> would have changed by ~4x here).
    assert abs(g1 / g2 - 4.0) < 0.1
