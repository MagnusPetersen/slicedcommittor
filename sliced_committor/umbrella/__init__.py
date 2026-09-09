"""Umbrella sampling: the dataset contract, reweighting, IO, and the rate bundle.

Umbrella-sampling data reach the committor as a static ensemble with MBAR (or
WHAM) weights and reach the diffusion as per-window time series. The
:class:`USDataset` holds both views in one per-window, time-ordered layout;
:func:`reweight` supplies the weights; :func:`fit_and_rate` fits the committor
and reads off the rates for every diffusion constructor and reduction asked
for, with the committor-free Kramers baseline beside them. Building the
dataset from files is the caller's job, with :func:`read_colvar`,
:func:`parse_restraint`, :func:`load_trajectory` and :func:`align_colvar_traj`
as the pieces.

The heavy optional dependencies (pymbar for MBAR, mdtraj for trajectories and
:mod:`sliced_committor.umbrella.mdtraj_metric`) are imported on use only.
Install them with ``pip install sliced-committor[umbrella]``.
"""

from .dataset import Reweighting, USDataset, split_by_window
from .io import (
    ColvarData,
    RestraintSpec,
    align_colvar_traj,
    colvar_dt,
    load_trajectory,
    parse_restraint,
    read_colvar,
)
from .rates import fit_and_rate, pmf_kramers_rate, progress_coordinate
from .reweight import mbar_weights, reduced_harmonic, reweight, wham_weights

__all__ = [
    # the dataset
    "USDataset",
    "Reweighting",
    "split_by_window",
    # reweighting
    "reweight",
    "mbar_weights",
    "wham_weights",
    "reduced_harmonic",
    # IO
    "read_colvar",
    "colvar_dt",
    "ColvarData",
    "parse_restraint",
    "RestraintSpec",
    "load_trajectory",
    "align_colvar_traj",
    # the committor and its rates
    "fit_and_rate",
    "pmf_kramers_rate",
    "progress_coordinate",
]
