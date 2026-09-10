"""The 1.0 stability promise: the public surface is pinned, importable, and
nothing public leaks past ``__all__``; importing the package stays light."""

import importlib
import subprocess
import sys

import sliced_committor as sc

EXPECTED = sorted(
    [
        "fit_committor",
        "build_committor",
        "committor_gradient",
        "rescale_transition",
        "CommittorFit",
        "compute_sliced_committor",
        "SlicedCommittorResult",
        "solve_weights",
        "Weights",
        "bootstrap_weights",
        "Bootstrap",
        "RepresentationError",
        "DirectionSamplingConfig",
        "directions_uniform",
        "compute_lda_axis",
        "sample_power_spherical_mixture",
        "sincos_pullback_metric",
        "AngleSign",
        "density",
        "committor_grad_sq",
        "basin_populations",
        "diffusion_profile",
        "lag_scan",
        "hummer_diffusion",
        "pooled_acf_diffusion",
        "PooledDiffusion",
        "committor_diffusion_from_cv",
        "committor_diffusion_from_cv_reparam",
        "linear_response_grad_sq",
        "committor_rate",
        "rate_from_profiles",
        "Profile",
        "value_at",
        "flux_flatness",
        "find_plateau",
        "PlateauWindow",
    ]
)


def test_public_surface_is_pinned():
    assert sorted(sc.__all__) == EXPECTED


def test_every_public_name_is_importable():
    for name in sc.__all__:
        assert hasattr(sc, name), name


def test_no_undeclared_public_names_in_the_package_namespace():
    """Module-level names of the package are either public or private/modules."""
    allowed = set(sc.__all__) | {"core", "rates", "umbrella", "tests"}
    leaked = [
        n
        for n in vars(sc)
        if not n.startswith("_") and n not in allowed and not n.startswith("jax")
    ]
    assert not leaked, leaked


def test_import_is_light():
    """Neither the package nor the umbrella subpackage pulls an optional or slow
    dependency at import (mdtraj, pymbar and matplotlib are on-use only, and
    scipy.stats is only needed by the reparametrisation route)."""
    code = (
        "import sys, sliced_committor, sliced_committor.umbrella; "
        "heavy = {'mdtraj', 'matplotlib', 'pymbar', 'scipy.stats'} & set(sys.modules); "
        "print(sorted(heavy))"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "[]", out.stdout


def test_version_is_single_sourced():
    assert importlib.import_module("sliced_committor").__version__ == "1.0.0"
