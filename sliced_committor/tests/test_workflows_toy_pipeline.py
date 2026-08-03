"""End-to-end toy validation: reuse stored samples, recover the PDE rate.

Skipped when the prior pipeline's stored samples are unavailable (e.g. running
outside this workstation) or when pymbar is missing.
"""

import pytest

from sliced_committor.workflows.loaders import TOY_DATA_ROOT


def _toy_available(name):
    return (TOY_DATA_ROOT / name / "us_traj.npz").is_file()


@pytest.mark.skipif(not _toy_available("double_well"), reason="toy samples not present")
def test_double_well_recovers_pde_rate():
    pytest.importorskip("pymbar")
    from sliced_committor.workflows.pipeline import run_pipeline

    res = run_pipeline("double_well", mode="fast", n_directions=128, write=False)
    # No known-D cross-check any more (US-only processing); the coordinate-invariant
    # CV-mapped TPT flux plateau should land near the stored PDE reference for this
    # near-optimal CV.
    pde = res["references"]["PDE"]["k"]
    tpt = [
        v["k_AB"]
        for k, v in res["methods"].items()
        if k.endswith(":: TPT_cvmap")
        and isinstance(v, dict)
        and v.get("k_AB")
        and v["k_AB"] == v["k_AB"]
    ]
    assert tpt, "no TPT_cvmap estimator found"
    best = min(tpt, key=lambda x: abs(x - pde))
    assert pde / 3.0 < best < 3.0 * pde  # within a factor of 3 of the exact PDE rate


@pytest.mark.skipif(not _toy_available("double_well"), reason="toy samples not present")
def test_toy_report_artifacts(tmp_path):
    pytest.importorskip("pymbar")
    pytest.importorskip("matplotlib")
    from sliced_committor.workflows.pipeline import run_pipeline

    res = run_pipeline("double_well", mode="fast", n_directions=64, out_dir=tmp_path, write=True)
    out = res["output"]
    expected = {
        "json": "rates_by_method.json",
        "csv": "rates_by_method.csv",
        "diagnostics": "diagnostics.json",
        "plot_rates": "rate_comparison.png",
        "plot_scatter": "cv_q_scatter_grid.png",
        "plot_profiles": "profiles.png",
        "markdown": "report.md",
    }
    for key, fname in expected.items():
        assert (tmp_path / fname).is_file()
        assert key in out
