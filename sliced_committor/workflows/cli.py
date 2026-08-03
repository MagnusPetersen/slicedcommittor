"""Command-line entry point: ``sliced-committor-us``.

Launched on a folder of umbrella-sampling data (or a toy system name), it
discovers the per-window COLVAR + trajectory + bias files, reweights (MBAR with
WHAM fallback), fits sliced committors and rates across method combinations, and
writes a rate-by-method file, diagnostics, a comparison bar plot, and a markdown
report.

Examples::

    sliced-committor-us US_data/US_chignolin_gmx --mode fast
    sliced-committor-us US_data/US_cSrc_activation_gmx --reweight wham --max-frames 40000
    sliced-committor-us double_well --out runs/double_well
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path


def _load_sidecar(path: str | None) -> dict | None:
    if path is None:
        return None
    import yaml

    with open(path) as fh:
        return yaml.safe_load(fh)


def _parse_references(items: list[str] | None) -> dict | None:
    """Parse ``NAME=VALUE[:UNITS]`` CLI references into the plotting schema.

    ``--reference experiment=0.0105:1/us`` -> ``{"experiment": {"k": 0.0105,
    "units": "1/us", "source": "--reference"}}``. UNITS defaults to ``1/s`` (the
    usual literature unit for MD rates); it is ignored for reduced-unit toys.
    Returns ``None`` when nothing was passed, so the registry default is used.
    """
    if not items:
        return None
    refs: dict[str, dict] = {}
    for item in items:
        if "=" not in item:
            raise SystemExit(f"--reference must be NAME=VALUE[:UNITS]; got {item!r}")
        name, _, rhs = item.partition("=")
        value, _, units = rhs.partition(":")
        try:
            k = float(value)
        except ValueError as exc:
            raise SystemExit(f"--reference {item!r}: VALUE {value!r} is not a number") from exc
        refs[name.strip()] = {"k": k, "units": (units.strip() or "1/s"), "source": "--reference"}
    return refs


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="sliced-committor-us",
        description="Umbrella-sampling -> sliced committor -> kinetics pipeline. "
        "Point it at a folder of US data: it scans subfolders for PLUMED COLVAR "
        "files, trajectories (.xtc/.trr/...) + topology (.gro/.tpr/.pdb), and the "
        "restraint (plumed.dat or window-name decode), then fits sliced committors "
        "and writes the exhaustive estimator plots + a report.",
        epilog="If discovery cannot match the basins/CVs (an unregistered folder), "
        "pass --system NAME or a --sidecar YAML with keys: region_A, region_B "
        "(dicts: {kind, on, lo, hi} or {center, radius}), cv_columns (list[int]), "
        "kappa (float|list), beta (float), topology (path). Run with -v to see "
        "every discovery decision.",
    )
    p.add_argument("target", help="path to a US folder, or a toy system name (double_well, ...)")
    p.add_argument("--mode", choices=("fast", "exhaustive"), default="fast", help="sweep breadth")
    p.add_argument("--system", default=None, help="registry system name override (US folders)")
    p.add_argument(
        "--reweight", choices=("auto", "mbar", "wham"), default="auto", help="reweighting engine"
    )
    p.add_argument(
        "--n-directions", type=int, default=256, help="projection directions per committor fit"
    )
    p.add_argument("--out", default=None, help="output directory (default ./sc_us_out/<system>)")
    p.add_argument("--sidecar", default=None, help="path to a sidecar YAML config")
    p.add_argument(
        "--reference",
        nargs="*",
        metavar="NAME=VALUE[:UNITS]",
        default=None,
        help="optional reference rate(s) to plot, e.g. --reference expt=0.0105:1/us "
        "(UNITS default 1/s). Repeatable. Overrides registry references; if omitted, "
        "known systems use their registry reference and unregistered folders plot none.",
    )
    p.add_argument("--traj-stride", type=int, default=1, help="trajectory read stride (US folders)")
    p.add_argument("--max-frames", type=int, default=None, help="cap total frames after loading")
    p.add_argument(
        "--window-fraction",
        type=float,
        default=None,
        metavar="F",
        help="keep only the first fraction F (0<F<1) of each window's frames before "
        "reweighting, emulating a shorter simulation (e.g. 0.1 = first 10%% of every "
        "window). Distinct from --traj-stride/--max-frames (uniform thinning); use it "
        "to probe how the rate estimators degrade with less sampling per window.",
    )
    p.add_argument("--seed", type=int, default=0, help="RNG seed")
    p.add_argument("--no-plot", action="store_true", help="skip the comparison plot")
    p.add_argument("-v", "--verbose", action="store_true", help="verbose logging")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )
    # Importing here keeps `import sliced_committor` light and surfaces missing
    # optional deps only when the CLI actually runs.
    from .pipeline import run_pipeline

    sidecar = _load_sidecar(args.sidecar)
    references = _parse_references(args.reference)
    result = run_pipeline(
        args.target,
        mode=args.mode,
        out_dir=args.out,
        system=args.system,
        sidecar=sidecar,
        n_directions=args.n_directions,
        reweight_method=args.reweight,
        traj_stride=args.traj_stride,
        max_frames=args.max_frames,
        window_fraction=args.window_fraction,
        references=references,
        seed=args.seed,
        plot=not args.no_plot,
        write=True,
    )

    out = result.get("output", {})
    rw = result.get("reweighting", {})
    print(f"system     : {result.get('system')}")
    print(f"mode       : {result.get('mode')}")
    print(f"reweighting: {rw.get('method')} (n_eff={rw.get('n_eff', float('nan')):.0f})")
    n_methods = sum(
        1 for v in result.get("methods", {}).values() if isinstance(v, dict) and "k_AB" in v
    )
    print(f"methods    : {n_methods} rate estimates")
    if out:
        print("outputs:")
        for kind, path in out.items():
            print(f"  {kind:11s} {path}")
    if "report.md" in str(out.get("markdown", "")):
        print(f"\nopen {out.get('markdown')} for the summary.")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
