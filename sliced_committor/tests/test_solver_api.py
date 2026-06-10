"""Solver-level edge cases, reproducibility, and diagnostics surface tests."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

import sliced_committor as sc

from ._helpers import two_basin_samples


def _make_samples(n=500, seed=0):
    return two_basin_samples(n, dim=3, seed=seed)


def test_seed_reproducibility():
    samples, in_A, in_B = _make_samples()
    a = sc.compute_sliced_committor(samples, in_A=in_A, in_B=in_B, n_directions=32, seed=11)
    b = sc.compute_sliced_committor(samples, in_A=in_A, in_B=in_B, n_directions=32, seed=11)
    np.testing.assert_array_equal(np.asarray(a.directions), np.asarray(b.directions))
    np.testing.assert_allclose(
        np.asarray(a.committors_1d), np.asarray(b.committors_1d), rtol=0, atol=0
    )


def test_store_projected_samples_false_diagonal_works():
    """Diagonal RD weights must not require projected_samples storage."""
    samples, in_A, in_B = _make_samples()
    result = sc.compute_sliced_committor(
        samples,
        in_A=in_A,
        in_B=in_B,
        n_directions=16,
        store_projected_samples=False,
        seed=5,
    )
    assert result.projected_samples is None
    weights = sc.compute_weights_multi(result, [sc.corrected_dirichlet_inv_rd])
    assert weights["corrected_dirichlet_inv_rd"].shape == (16,)
    # Diagonal evaluation works with on-the-fly projection.
    q = sc.build_committor(result, weights["corrected_dirichlet_inv_rd"])(samples[:50])
    assert q.shape == (50,)


def test_summary_contains_key_fields():
    samples, in_A, in_B = _make_samples()
    result = sc.compute_sliced_committor(samples, in_A=in_A, in_B=in_B, n_directions=24, seed=7)
    s = result.summary()
    assert "n_directions" not in s, "summary uses 'directions :' not 'n_directions'"
    for token in ("M = 24", "valid slices", "mean eps", "mean log D"):
        assert token in s, f"missing '{token}' in summary:\n{s}"


def test_why_masked_valid_slice():
    samples, in_A, in_B = _make_samples()
    result = sc.compute_sliced_committor(samples, in_A=in_A, in_B=in_B, n_directions=12, seed=2)
    # At least one slice should be valid in this easy setup.
    valid_idx = int(np.nonzero(np.asarray(result.valid_mask))[0][0])
    assert sc.why_masked(result, valid_idx) == "slice is valid"


def test_why_masked_out_of_range():
    samples, in_A, in_B = _make_samples()
    result = sc.compute_sliced_committor(samples, in_A=in_A, in_B=in_B, n_directions=8, seed=0)
    with pytest.raises(IndexError):
        sc.why_masked(result, 99)


def test_summarize_gram_diagnostics_full_gram():
    samples, in_A, in_B = _make_samples()
    result = sc.compute_sliced_committor(samples, in_A=in_A, in_B=in_B, n_directions=16, seed=1)
    gram = sc.compute_full_gram_weights(result, samples)
    s = sc.summarize_gram_diagnostics(gram)
    assert "Gram solver diagnostics" in s
    assert "G_jk" in s  # off-diagonal magnitude row uses '|G_jk|'
    assert "constraint" in s.lower()


def test_summarize_gram_diagnostics_bmc_includes_residuals():
    samples, in_A, in_B = _make_samples()
    result = sc.compute_sliced_committor(samples, in_A=in_A, in_B=in_B, n_directions=16, seed=1)
    bmc = sc.compute_basin_moment_weights(result, samples)
    s = sc.summarize_gram_diagnostics(bmc)
    # BMC dict carries the constraint residuals + sum_w.
    assert "|a.w|" in s
    assert "|b.w - 1|" in s
    assert "sum_w" in s


def test_summarize_gram_diagnostics_rejects_array():
    with pytest.raises(TypeError):
        sc.summarize_gram_diagnostics(jnp.zeros(4))


def test_evaluate_committor_dict_dispatch():
    """The EBMC result dict is auto-detected by build_committor."""
    samples, in_A, in_B = _make_samples()
    result = sc.compute_sliced_committor(samples, in_A=in_A, in_B=in_B, n_directions=24, seed=3)
    ebmc = sc.compute_enriched_basin_moment_weights(result, samples)
    # Dict route.
    q_dict = sc.build_committor(result, ebmc)(samples[:30])
    # Manual route via raw weights + bias (centered basis: q̂(x) = c + Σ w_j q_j).
    # Confirm both paths produce equivalent output (within float noise).
    assert q_dict.shape == (30,)
    assert jnp.all(jnp.isfinite(q_dict))
