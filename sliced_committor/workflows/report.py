"""Write sweep results to disk: JSON, CSV, three plots, and a markdown report.

Outputs (under ``out_dir``):

* ``rates_by_method.json`` - methods + reweighting + references + diagnostics
  (the large per-combo scatter/profile arrays are dropped; they live in the PNGs).
* ``rates_by_method.csv`` - one row per ``feat/dir :: estimator`` with k_AB / k_BA.
* ``diagnostics.json`` - reweighting + sweep diagnostics + per-combo committor info.
* ``rate_comparison.png`` - the three estimators side-by-side.
* ``cv_q_scatter_grid.png`` - CV vs sliced-q, featurization x direction.
* ``profiles.png`` - D / flux / PMF profiles over CV and q.
* ``report.md`` - human-readable summary.
"""

from __future__ import annotations

import csv
import json
import logging
import pickle
from pathlib import Path
from typing import Any

import numpy as np

from ._units import estimated_to_per_s, is_reduced
from .config import get_config

logger = logging.getLogger(__name__)


def _jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _jsonable(obj.tolist())
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, float) and not np.isfinite(obj):
        return None
    return obj


def _time_unit(system: str) -> str:
    try:
        return get_config(system).time_unit
    except KeyError:
        return "ps"


def _num(v):
    if v is None:
        return ""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return ""
    return f if np.isfinite(f) else ""


def _rate_rows(result: dict[str, Any], time_unit: str):
    reduced = is_reduced(time_unit)
    methods = dict(result.get("methods", {}))
    for label, rd in result.get("baseline", {}).items():
        methods[f"baseline:{label}"] = rd
    rows = []
    for label, rd in methods.items():
        if not isinstance(rd, dict):
            continue
        k_ab = rd.get("k_AB")
        per_s = (
            None
            if (reduced or k_ab is None or not np.isfinite(k_ab))
            else estimated_to_per_s(float(k_ab), time_unit)
        )
        rows.append(
            {
                "method": label,
                "k_AB": _num(k_ab),
                "k_BA": _num(rd.get("k_BA")),
                "k_AB_per_s": _num(per_s),
                "error": rd.get("error", ""),
            }
        )
    return rows


def write_report(
    result: dict[str, Any], out_dir: str | Path, *, plot: bool = True
) -> dict[str, str]:
    """Write all output artifacts for a sweep result."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    system = result.get("system", "system")
    time_unit = _time_unit(system)
    written: dict[str, str] = {}

    # JSON: drop the heavy per-combo scatter/profile arrays (kept only in the PNGs).
    slim = dict(result)
    slim["combos"] = [
        {k: v for k, v in c.items() if k not in ("scatter", "profiles")}
        for c in result.get("combos", [])
    ]
    json_path = out_dir / "rates_by_method.json"
    with json_path.open("w") as fh:
        json.dump(_jsonable(slim), fh, indent=2)
    written["json"] = str(json_path)

    # Per-window/per-bin profiles persisted separately (the small arrays only, no
    # scatter) so profiles.png can be regenerated without re-running the pipeline
    # (see plotting.replot_profiles).
    prof_payload = {
        "system": system,
        "diagnostics": result.get("diagnostics", {}),
        "combos": [
            {
                "featurization": c.get("featurization"),
                "direction": c.get("direction"),
                "committor": c.get("committor", {}),
                "profiles": c.get("profiles", {}),
            }
            for c in result.get("combos", [])
            if c.get("profiles")
        ],
    }
    profiles_path = out_dir / "profiles.json"
    with profiles_path.open("w") as fh:
        json.dump(_jsonable(prof_payload), fh, indent=2)
    written["profiles_json"] = str(profiles_path)

    rows = _rate_rows(result, time_unit)
    csv_path = out_dir / "rates_by_method.csv"
    with csv_path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["method", "k_AB", "k_BA", "k_AB_per_s", "error"])
        w.writeheader()
        w.writerows(rows)
    written["csv"] = str(csv_path)

    diag_path = out_dir / "diagnostics.json"
    with diag_path.open("w") as fh:
        json.dump(
            _jsonable(
                {
                    "system": system,
                    "mode": result.get("mode"),
                    "reweighting": result.get("reweighting", {}),
                    "diagnostics": result.get("diagnostics", {}),
                    "axes": result.get("axes", {}),
                    "combos": [
                        {
                            "featurization": c.get("featurization"),
                            "direction": c.get("direction"),
                            "committor": c.get("committor", {}),
                        }
                        for c in result.get("combos", [])
                    ],
                }
            ),
            fh,
            indent=2,
        )
    written["diagnostics"] = str(diag_path)

    if plot:
        from .plotting import plot_cv_q_scatter_grid, plot_profiles, plot_rate_comparison

        written["plot_rates"] = plot_rate_comparison(
            result, out_dir / "rate_comparison.png", time_unit=time_unit
        )
        written["plot_scatter"] = plot_cv_q_scatter_grid(result, out_dir / "cv_q_scatter_grid.png")
        written["plot_profiles"] = plot_profiles(result, out_dir / "profiles.png")

    md_path = out_dir / "report.md"
    md_path.write_text(_render_markdown(result, rows, time_unit, written))
    written["markdown"] = str(md_path)

    # Cache the full sweep result (rates + profiles + scatter + committor
    # diagnostics) so every artifact can be regenerated with no recompute -- see
    # :func:`regenerate_report` (instant re-runs for presentation tweaks).
    result_pkl = out_dir / "result.pkl"
    with result_pkl.open("wb") as fh:
        pickle.dump(result, fh)
    written["result_pkl"] = str(result_pkl)

    logger.info("wrote report -> %s", out_dir)
    return written


def regenerate_report(out_dir: str | Path, *, plot: bool = True) -> dict[str, str]:
    """Rewrite all artifacts (JSON/CSV/plots/markdown) from a cached ``result.pkl``.

    The sweep result cached by :func:`write_report` carries everything the
    plotters and tables need (no committor callables required), so this redraws
    the figures and rebuilds the tables in seconds after a labelling / styling /
    layout change -- no pipeline re-run, no re-fitting.
    """
    out_dir = Path(out_dir)
    with (out_dir / "result.pkl").open("rb") as fh:
        result = pickle.load(fh)
    return write_report(result, out_dir, plot=plot)


def _render_markdown(result, rows, time_unit, written) -> str:
    system = result.get("system", "system")
    rw = result.get("reweighting", {})
    diag = result.get("diagnostics", {})
    unit = "reduced" if is_reduced(time_unit) else "1/s"
    reduced = is_reduced(time_unit)
    lines = [
        f"# Kinetics report: {system}",
        "",
        f"- mode: **{result.get('mode')}**",
        f"- reweighting: **{rw.get('method')}** "
        f"(n_eff={rw.get('n_eff', float('nan')):.0f} of {rw.get('n_frames')} frames, "
        f"{rw.get('n_windows')} windows)",
        f"- featurization feature dims: {diag.get('n_features_by_featurization', {})}",
        f"- combos: {diag.get('n_combos')}; dt={diag.get('dt')}; has_dynamics={diag.get('has_dynamics')}",
        "",
        "## Estimators (per-window rate processing)",
        "",
        "Rates are processed PER UMBRELLA WINDOW: each window is one data point "
        "with its own local diffusion and matched local flux (no barrier-band "
        "median is broadcast across windows). The diffusion is the per-window "
        "Hummer estimate (Var/tau_int) throughout, and the SAME reactive current "
        "is read off several COORDINATE-INVARIANT ways on the committor coordinate "
        "(`D_q*pi`): the TPT flux plateau and the Berezhkovskii-Szabo 1D-Smoluchowski "
        "MFPT with the measured q-space D_q; the BS local flux `D_q*pi` at the q=0.5 "
        "surface (no q-integration); and the MFPT (`BS_mfpt_cvmap`) and TPT flux "
        "plateau (`TPT_cvmap`) on the CV-MAPPED diffusion "
        "`D_q(q) = D_s*<|grad q|^2>/<|grad s|^2>`. (The earlier feature-space TPT_cv, "
        "scalar CV-space D times the feature-space `|grad q|^2`, was dropped: it is "
        "not coordinate-invariant.) Kramers = PMF(CV) + CV-space diffusion is "
        "committor-free. "
        "The committor is auto-tuned per feature space (EBMC vs PESB, ranked by "
        "Dirichlet energy). See `rate_comparison.png` (estimators side-by-side), "
        "`cv_q_scatter_grid.png` (committor quality), `profiles.png` (per-window "
        "D/flux/PMF over CV and q).",
        "",
        "## Reference rates",
        "",
    ]
    refs = result.get("references", {})
    if refs:
        lines += ["| label | k | units | source |", "|---|---|---|---|"]
        for label, ref in refs.items():
            lines.append(
                f"| {label} | {ref.get('k'):.4g} | {ref.get('units', '')} | {ref.get('source', '')} |"
            )
    else:
        lines.append("(none)")

    def _fmt(v):
        return f"{v:.4g}" if isinstance(v, float) else "-"

    lines += [
        "",
        f"## Rates by method (k_AB in {unit})",
        "",
        "| method | k_AB | k_BA |",
        "|---|---|---|",
    ]
    rows_sorted = sorted(
        rows, key=lambda r: (r["k_AB"] == "", r["k_AB"] if r["k_AB"] != "" else 0.0)
    )
    for r in rows_sorted:
        kab = r["k_AB_per_s"] if (not reduced and r["k_AB_per_s"] != "") else r["k_AB"]
        method = str(r["method"]).replace("|", "\\|")
        if r["error"]:
            lines.append(f"| {method} | error | {str(r['error'])[:40]} |")
        else:
            lines.append(f"| {method} | {_fmt(kab)} | {_fmt(r['k_BA'])} |")

    for key, fname in (
        ("plot_rates", "rate_comparison.png"),
        ("plot_scatter", "cv_q_scatter_grid.png"),
        ("plot_profiles", "profiles.png"),
    ):
        if key in written:
            lines += ["", f"## {fname}", "", f"![{fname}]({Path(written[key]).name})"]
    lines.append("")
    return "\n".join(lines)
