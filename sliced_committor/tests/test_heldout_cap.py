"""Gates for the held-out Dirichlet cap read through the half-set solve.

``heldout_cap_halfset`` is the label-free criterion that ranks TRIAL SPACES:
direction sets, sampler parameters, M. It differs from the in-sample Dirichlet
energy in the one way that matters for that job, being read out of sample, and
from the ridge-selection cap in another, being read through the solve that is
actually deployed rather than through a scalar-ridge surrogate.

These gates fix the properties the selection depends on. They do not assert a
particular numeric value, which would only re-encode the implementation.
"""

import jax
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp

from sliced_committor import (
    compute_enriched_basin_moment_weights,
    compute_sliced_committor,
)

pytestmark = pytest.mark.filterwarnings("ignore::UserWarning")


def _fit(n=6000, m=64, seed=0, n_bins=50):
    systems = pytest.importorskip("recovar.systems")
    cfg = systems.wolfe_quapp_config(beta=1.0, dim=2)
    X = systems.grid_boltzmann_samples(cfg, n, rng=seed)
    in_A, in_B = cfg.in_A(X), cfg.in_B(X)
    assert in_A.sum() > 50 and in_B.sum() > 50, "basins under-populated"
    res = compute_sliced_committor(
        jnp.asarray(X), in_A=jnp.asarray(in_A), in_B=jnp.asarray(in_B),
        n_directions=m, seed=seed, n_bins=n_bins, n_min=10,
        binning_method="equal_width", store_projected_samples=True,
    )
    return res, jnp.asarray(X)


def _cap(res, X, **kw):
    w = compute_enriched_basin_moment_weights(
        res, X, None, "halfset_eigen", gram_dtype="float64",
        raise_on_degenerate=False, heldout_cap=True, **kw)
    return w


def test_cap_is_returned_only_when_asked():
    """A plain fit must not pay for the criterion."""
    res, X = _fit()
    plain = compute_enriched_basin_moment_weights(
        res, X, None, "halfset_eigen", gram_dtype="float64",
        raise_on_degenerate=False)
    assert "heldout_cap" not in plain
    assert "heldout_cap" in _cap(res, X)


def test_cap_is_finite_positive_and_uses_every_fold():
    res, X = _fit()
    hc = _cap(res, X)["heldout_cap"]
    assert np.isfinite(hc["cap"]) and hc["cap"] > 0
    # A fold that silently fails would bias the mean toward whatever the
    # surviving folds happen to say, so the count is part of the contract.
    assert hc["n_ok"] == hc["n_folds"] == 10
    assert np.isfinite(hc["per_fold"]).all()


def test_cap_does_not_change_the_weights():
    """The criterion is a read-out, not a different fit."""
    res, X = _fit()
    a = compute_enriched_basin_moment_weights(
        res, X, None, "halfset_eigen", gram_dtype="float64",
        raise_on_degenerate=False)
    b = _cap(res, X)
    np.testing.assert_array_equal(np.asarray(a["w"]), np.asarray(b["w"]))
    assert float(a["M_gap"]) == float(b["M_gap"])


def test_cap_is_deterministic():
    res, X = _fit()
    assert _cap(res, X)["heldout_cap"]["cap"] == _cap(res, X)["heldout_cap"]["cap"]


def _paired(alt, base):
    """Paired fold-wise comparison of two caps. Returns (mean, t, n_positive)."""
    d = np.asarray(alt["per_fold"]) - np.asarray(base["per_fold"])
    assert np.isfinite(d).all(), "the pairing requires every fold on both sides"
    se = float(np.std(d, ddof=1) / np.sqrt(d.size))
    return float(np.mean(d)), float(np.mean(d) / se), int((d > 0).sum()), d.size


@pytest.mark.parametrize("label,kw", [("fewer_directions", {"m": 4}),
                                      ("coarser_histograms", {"n_bins": 4})])
def test_cap_ranks_a_degraded_trial_space_worse(label, kw):
    """The criterion must rank, or it is of no use for selection.

    Both degradations shrink or blunt the trial space without touching anything
    else, so the cap has to rise. This is the property the (mu, alpha) selection
    rests on.

    The comparison is PAIRED across folds, which is the only correct way to make
    it. Both fits see the same sample, so the fold partition is the same and the
    fold-to-fold scatter in the cap LEVEL is a common offset: each fold
    estimates nu_AB slightly differently. An unpaired test against that scatter
    overstates the uncertainty of a trial-space-to-trial-space DIFFERENCE by
    about an order of magnitude, which is the reasoning ``select_ridge_cv``
    gives for using a paired standard error on its own ridge grid.

    N is 20000 rather than the 6000 the cheaper gates use. At 6000 the fold
    noise swallows the coarse-histogram contrast (measured: t = 1.6, 7 of 10
    folds agreeing), which is a fact about how much data the criterion needs to
    resolve a difference, not a defect in it.
    """
    base_res, X = _fit(n=20000)
    alt_res, _ = _fit(n=20000, **kw)
    base = _cap(base_res, X)["heldout_cap"]
    alt = _cap(alt_res, X)["heldout_cap"]
    mean_d, t, n_pos, n = _paired(alt, base)
    assert mean_d > 0, f"{label}: the cap fell for a worse trial space"
    assert t > 4.0, f"{label}: paired t = {t:.1f}, not separated from fold noise"
    assert n_pos == n, f"{label}: only {n_pos}/{n} folds agree on the sign"


def test_cap_is_read_out_of_sample():
    """It must not simply track the in-sample energy.

    If the cap were an in-sample quantity it would fall monotonically with M by
    construction and could not rank trial spaces at all. The gate is weak on
    purpose: what is asserted is that the two are distinguishable, not that they
    order any particular pair differently.
    """
    res, X = _fit()
    out = _cap(res, X)
    w = np.asarray(out["w"])
    energy = float(w @ np.asarray(out["G"]) @ w)
    assert not np.isclose(out["heldout_cap"]["cap"], energy, rtol=1e-9)
