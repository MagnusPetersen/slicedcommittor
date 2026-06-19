"""High-level end-to-end driver: load -> sweep -> report.

:func:`run_pipeline` ties the pieces together for either an analytic toy (by
registry name) or a real umbrella-sampling folder (by path), and writes the
output artifacts. The CLI is a thin wrapper around it.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from .config import TOY_SYSTEMS
from .loaders import load_toy_dataset, load_us_dataset
from .report import write_report
from .sweep import run_sweep

logger = logging.getLogger(__name__)


def run_pipeline(
    target: str,
    *,
    mode: str = "fast",
    out_dir: str | Path | None = None,
    system: str | None = None,
    sidecar: dict | None = None,
    n_directions: int = 256,
    reweight_method: str = "auto",
    traj_stride: int = 1,
    max_frames: int | None = None,
    diffusion_mode: str = "bins_hummer",
    n_diff_bins: int = 25,
    committor_sweep_grid: dict | str | None = "auto",
    references: dict[str, dict[str, Any]] | None = None,
    seed: int = 0,
    plot: bool = True,
    write: bool = True,
) -> dict[str, Any]:
    """Run the full pipeline on a toy system name or a US folder path.

    Args:
        target: a toy system name (``"double_well"``, ...) or a path to a US
            campaign folder.
        mode: ``"fast"`` or ``"exhaustive"``.
        out_dir: output directory (defaults to ``./sc_us_out/<system>``).
        system: registry name override (US folders).
        sidecar: optional sidecar overrides (US folders).
        n_directions: projection directions per committor fit.
        reweight_method: ``"auto"`` | ``"mbar"`` | ``"wham"``.
        traj_stride: trajectory read stride (US folders).
        max_frames: frame cap after loading (US folders).
        diffusion_mode: ``"bins_hummer"`` (default, per-window Hummer on an
            ``n_diff_bins`` committor-bin rate grid), ``"bins"`` (per-bin
            Kramers-Moyal), or ``"per_window"`` (Hummer on a fine grid).
        n_diff_bins: committor bin count for ``diffusion_mode="bins"``.
        committor_sweep_grid: ``"auto"`` (default) auto-tunes each committor per
            feature space by Dirichlet energy (EBMC vs PESB); a dict sweeps a
            custom grid; ``None``/``{}`` disables tuning. See
            :func:`committor_rates.fit_and_rate`.
        references: optional ``{label: {"k": value, "units": str, ...}}`` reference
            rate(s) to plot. When given they OVERRIDE the registry rates; when
            ``None`` the known-system registry rates are used (and an unregistered
            folder simply gets no reference band).
        seed: RNG seed.
        plot: generate the comparison/scatter/profile PNGs (set ``False`` to skip).
        write: write artifacts to ``out_dir``.

    Returns:
        the sweep result dict (augmented with ``output`` paths when ``write``).
    """
    path = Path(target)
    # An explicit ``references=`` always wins; otherwise toys supply their own
    # analytic references and US folders fall back (in run_sweep) to the registry.
    if path.is_dir():
        dataset = load_us_dataset(
            path,
            system=system,
            sidecar=sidecar,
            traj_stride=traj_stride,
            max_frames=max_frames,
            seed=seed,
        )
        system_name = dataset.system_name
    elif target in TOY_SYSTEMS:
        dataset, toy_references = load_toy_dataset(target)
        system_name = dataset.system_name
        if references is None:
            references = toy_references
    else:
        raise ValueError(
            f"{target!r} is neither an existing folder nor a known toy system "
            f"{TOY_SYSTEMS}. Pass a US folder path or a toy name."
        )

    result = run_sweep(
        dataset,
        mode=mode,
        n_directions=n_directions,
        reweight_method=reweight_method,
        references=references,
        diffusion_mode=diffusion_mode,
        n_diff_bins=n_diff_bins,
        committor_sweep_grid=committor_sweep_grid,
        seed=seed,
    )

    if write:
        out = Path(out_dir) if out_dir is not None else Path.cwd() / "sc_us_out" / system_name
        written = write_report(result, out, plot=plot)
        result["output"] = written
    return result
