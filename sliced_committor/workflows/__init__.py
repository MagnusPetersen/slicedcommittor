"""Umbrella-sampling -> sliced-committor -> kinetics workflows.

This subpackage turns umbrella-sampling MD data (PLUMED COLVAR + per-window
trajectories, or the analytic-toy ``.npz`` archives) into reaction-rate
estimates by many method combinations, with diagnostics and a comparison plot.
It builds on the core committor and rate machinery and adds only what that
machinery lacks: data ingestion (COLVAR, trajectories), MBAR/WHAM reweighting,
featurization, the sweep orchestrator, plotting, and a CLI.

The pipeline is functional: loaders return an immutable
:class:`~sliced_committor.workflows._containers.USDataset`, each step takes it
and returns plain values or new datasets.

Heavy optional dependencies (mdtraj, pymbar, matplotlib, pyyaml) are imported
lazily, so ``import sliced_committor`` stays light. Access the API via
``from sliced_committor.workflows import run_sweep`` (or the attributes below);
the relevant module is imported on first use. Install them with
``pip install -e .[workflows]``.

Programmatic entry points::

    from sliced_committor.workflows import load_us_dataset, load_toy_dataset, run_sweep

    ds = load_us_dataset("US_data/US_chignolin_gmx")        # auto-discovers + labels
    result = run_sweep(ds, mode="fast")                      # rates by method
    # or the analytic toys (reuse stored samples + references):
    ds, refs = load_toy_dataset("double_well")
"""

from __future__ import annotations

# Light, dependency-free exports are safe to bind eagerly.
from ._containers import Reweighting, USDataset, split_by_window
from .config import REGISTRY, Region, SystemConfig, compute_basins, get_config, match_config

__all__ = [
    # containers / config (light)
    "USDataset",
    "Reweighting",
    "split_by_window",
    "SystemConfig",
    "Region",
    "REGISTRY",
    "get_config",
    "match_config",
    "compute_basins",
    # lazy (require optional deps)
    "read_colvar",
    "parse_restraint",
    "load_trajectory",
    "featurize",
    "reweight",
    "mbar_weights",
    "wham_weights",
    "load_us_dataset",
    "load_toy_dataset",
    "fit_and_rate",
    "pmf_kramers_rate",
    "run_sweep",
    "plot_rate_comparison",
    "write_report",
    "regenerate_report",
    "replot_profiles",
    "run_pipeline",
]

# Map lazily-exported names to their defining module (relative to this package).
_LAZY = {
    "read_colvar": ".colvar",
    "parse_restraint": ".bias",
    "load_trajectory": ".trajectory",
    "featurize": ".featurize",
    "reweight": ".reweight",
    "mbar_weights": ".reweight",
    "wham_weights": ".reweight",
    "load_us_dataset": ".loaders",
    "load_toy_dataset": ".loaders",
    "fit_and_rate": ".committor_rates",
    "pmf_kramers_rate": ".pmf_kramers",
    "run_sweep": ".sweep",
    "plot_rate_comparison": ".plotting",
    "replot_profiles": ".plotting",
    "write_report": ".report",
    "regenerate_report": ".report",
    "run_pipeline": ".pipeline",
}


def __getattr__(name: str):  # PEP 562 lazy attribute access
    module_name = _LAZY.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    module = importlib.import_module(module_name, __name__)
    return getattr(module, name)


def __dir__():
    return sorted(__all__)
