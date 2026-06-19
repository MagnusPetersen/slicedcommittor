"""Diagnostic plotting for the rate sweep.

Three figures:

* :func:`plot_rate_comparison` - the estimators TPT flux plateau (CV-space D and
  q-space D), Berezhkovskii-Szabo local (q*=0.5) and MFPT (q-space D), plus the
  committor-free Kramers row, one strip per estimator with a marker per
  featurization x direction combo, on a log axis with the reference rate(s). The
  diffusion is the per-window Hummer estimate throughout.
* :func:`plot_cv_q_scatter_grid` - CV-value vs sliced-committor q, in a grid of
  featurization (rows) x direction-sampling (columns) - the committor-quality /
  flattening diagnostic.
* :func:`plot_profiles` - the PER-WINDOW diffusion, flux and PMF profiles over
  the CV and over q (one point per umbrella window, per-window Hummer diffusion),
  with the plateau as the rate.

MD rates are normalised to 1/s; reduced-unit toys are plotted natively. The Agg
backend is used so everything works headless.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from ._units import estimated_to_per_s, is_reduced, reference_to_per_s

logger = logging.getLogger(__name__)

# Estimator display order (top -> bottom) and human-readable labels. The
# diffusion is the per-window Hummer estimate throughout; labels name the SPACE
# it is measured in (q-space for Szabo, CV-space for TPT / Kramers). Kramers is
# committor-independent, so it appears once as a committor-free row.
_ESTIMATOR_ORDER = (
    "Szabo_mfpt",
    "BS_mfpt_cvmap",
    "BS_local",
    "TPT_q",
    "TPT_cvmap",
    "baseline:pmf_kramers",
)
_ESTIMATOR_LABELS = {
    "Szabo_mfpt": "Berezhkovskii-Szabo MFPT\n(q-space D)",
    "BS_mfpt_cvmap": "Berezhkovskii-Szabo MFPT\n(CV-mapped q-space D)",
    "BS_local": "Berezhkovskii-Szabo local, q*=0.5\n(q-space D)",
    "TPT_q": "TPT flux plateau\n(q-space D)",
    "TPT_cvmap": "TPT flux plateau\n(CV-mapped q-space D)",
    "baseline:pmf_kramers": "Kramers, committor-free\n(PMF + CV-space D)",
}
# Estimator rows that do NOT depend on the sliced committor (single mark, no
# featurization/direction).
_COMMITTOR_FREE = ("baseline:pmf_kramers",)


def _beeswarm_offsets(klog, half=0.36, res=0.06):
    """Vertical offsets that spread same-x points apart (deterministic beeswarm).

    Points whose log10(k) round to the same ``res`` bin are fanned out evenly
    across ``[-half, half]`` so identical / near-identical rates do not overlap.
    """
    klog = np.asarray(klog, dtype=float)
    off = np.zeros(klog.shape[0])
    if klog.size == 0:
        return off
    keys = np.round(klog / res).astype(int)
    for k in np.unique(keys):
        idx = np.nonzero(keys == k)[0]
        if idx.size > 1:
            off[idx] = np.linspace(-half, half, idx.size)
    return off


_FEAT_LABELS = {
    "identity": "raw config (x,y)",
    "cv_only": "CV only",
    "dihedrals": "dihedrals (sin/cos)",
    "aligned_cartesian": "aligned cartesian",
    "distance_matrix": "distance matrix",
    "contact_map": "contact map",
    "-": "(n/a)",
}
_DIR_LABELS = {
    "uniform": "uniform",
    "lda": "LDA",
    "pca": "PCA",
    "gcpca": "gcPCA",
    "tica_ema": "TICA-EMA",
    "tica_ema_decomposed": "TICA-EMA (bias-aware)",
    "-": "(n/a)",
}
# Distinct markers per direction-sampling method.
_DIR_MARKERS = {
    "uniform": "o",
    "lda": "P",
    "pca": "s",
    "gcpca": "^",
    "tica_ema": "D",
    "tica_ema_decomposed": "v",
    "-": "X",
}


def _use_agg():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _to_plot_units(k, time_unit):
    if is_reduced(time_unit):
        return float(k)
    return estimated_to_per_s(float(k), time_unit)


def _reference_values(result, time_unit):
    reduced = is_reduced(time_unit)
    out = []
    for label, ref in result.get("references", {}).items():
        k = ref.get("k")
        if k is None:
            continue
        val = float(k) if reduced else reference_to_per_s(float(k), ref.get("units", "1/s"))
        if val is not None and np.isfinite(val) and val > 0:
            out.append((label, val))
    return out


def _parse_method(label: str):
    """``"feat/dir :: ESTIMATOR"`` -> (featurization, direction, estimator)."""
    combo, _, est = label.partition(" :: ")
    feat, _, direction = combo.partition("/")
    return feat, direction, est


def plot_rate_comparison(
    result: dict[str, Any], out_path: str | Path, *, time_unit: str = "ps", title: str | None = None
) -> str:
    """Estimators side-by-side: one labelled row per estimator, a marker per combo.

    Colour = featurization, marker shape = direction-sampling method. Legends sit
    OUTSIDE the axes so they never cover the data or the reference. A shaded band
    marks "within ~3x of the reference", so closeness reads at a glance.
    """
    plt = _use_agg()
    out_path = Path(out_path)
    reduced = is_reduced(time_unit)
    unit_label = "k_AB  (reduced units)" if reduced else "k_AB  (s$^{-1}$)"

    # Collect points per estimator: (featurization, direction, k_plot).
    pts: dict[str, list[tuple[str, str, float]]] = {}
    for label, rd in result.get("methods", {}).items():
        if not isinstance(rd, dict) or rd.get("k_AB") is None or rd.get("k_AB") != rd.get("k_AB"):
            continue
        feat, direction, est = _parse_method(label)
        if est == "Kramers":
            continue  # committor-free; shown once as the PMF-Kramers baseline row
        kp = _to_plot_units(rd["k_AB"], time_unit)
        if kp is not None and np.isfinite(kp) and kp > 0:
            pts.setdefault(est, []).append((feat, direction, kp))
    for blabel, brd in result.get("baseline", {}).items():
        if isinstance(brd, dict) and brd.get("k_AB") == brd.get("k_AB") and brd.get("k_AB"):
            kp = _to_plot_units(brd["k_AB"], time_unit)
            if kp and np.isfinite(kp) and kp > 0:
                pts.setdefault(f"baseline:{blabel}", []).append(("-", "-", kp))

    estimators = [e for e in _ESTIMATOR_ORDER if e in pts] + [
        e for e in pts if e not in _ESTIMATOR_ORDER
    ]
    refs = _reference_values(result, time_unit)
    if not estimators:
        fig, ax = plt.subplots(figsize=(6, 2))
        ax.text(0.5, 0.5, "no finite k_AB to plot", ha="center", va="center")
        ax.axis("off")
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return str(out_path)

    # Featurization/direction legends only from committor-dependent estimators.
    real = {e: v for e, v in pts.items() if e not in _COMMITTOR_FREE}
    feats = sorted({f for v in real.values() for (f, _d, _k) in v})
    dirs = sorted({d for v in real.values() for (_f, d, _k) in v})
    cmap = plt.get_cmap("tab10")
    fcolor = {f: cmap(i % 10) for i, f in enumerate(feats)}

    # rows top -> bottom (reverse y so first estimator is at the top).
    n = len(estimators)
    ypos = {est: n - 1 - i for i, est in enumerate(estimators)}

    fig, ax = plt.subplots(figsize=(11, max(3.0, 0.95 * n + 1.5)))

    # Reference band (within ~3x) + lines.
    if refs:
        rmin = min(v for _l, v in refs) / 3.0
        rmax = max(v for _l, v in refs) * 3.0
        ax.axvspan(rmin, rmax, color="#7fbf7b", alpha=0.12, zorder=0)
    rcolors = ["#C44E52", "#1b7837", "#8172B3", "#CCB974"]
    for i, (rlabel, rval) in enumerate(refs):
        ax.axvline(
            rval,
            color=rcolors[i % len(rcolors)],
            ls="--",
            lw=1.8,
            zorder=1,
            label=f"reference: {rlabel} = {rval:.2g}",
        )

    # Alternating row shading.
    for est in estimators:
        if ypos[est] % 2 == 0:
            ax.axhspan(ypos[est] - 0.5, ypos[est] + 0.5, color="0.95", zorder=0)

    for est in estimators:
        y0 = ypos[est]
        rows = pts[est]
        if est in _COMMITTOR_FREE:
            # Committor-free (Kramers): a single black mark, no feat/direction.
            for _feat, _direction, kp in rows:
                ax.scatter(
                    kp,
                    y0,
                    color="black",
                    marker="*",
                    s=180,
                    zorder=4,
                    edgecolors="white",
                    linewidths=0.5,
                )
            continue
        kvals = np.array([kp for _f, _d, kp in rows], dtype=float)
        offs = _beeswarm_offsets(np.log10(kvals))
        for (feat, direction, kp), off in zip(rows, offs):
            ax.scatter(
                kp,
                y0 + off,
                color=fcolor.get(feat, "k"),
                marker=_DIR_MARKERS.get(direction, "o"),
                s=48,
                alpha=0.9,
                edgecolors="white",
                linewidths=0.4,
                zorder=3,
            )

    ax.set_yticks(range(n))
    ax.set_yticklabels(
        [_ESTIMATOR_LABELS.get(estimators[n - 1 - i], estimators[n - 1 - i]) for i in range(n)],
        fontsize=8,
    )
    ax.set_ylim(-0.6, n - 0.4)
    ax.set_xscale("log")
    ax.set_xlabel(unit_label, fontsize=10)
    ax.grid(axis="x", which="major", ls=":", alpha=0.5)
    ax.set_title(
        title or f"Reaction-rate estimators: {result.get('system', '')} ({result.get('mode', '')})",
        fontsize=12,
    )

    # Legends OUTSIDE on the right: featurization (colour), direction (marker), reference.
    feat_handles = [
        plt.Line2D([], [], marker="o", ls="", color=fcolor[f], label=_FEAT_LABELS.get(f, f))
        for f in feats
    ]
    dir_handles = [
        plt.Line2D(
            [], [], marker=_DIR_MARKERS.get(d, "o"), ls="", color="0.3", label=_DIR_LABELS.get(d, d)
        )
        for d in dirs
    ]
    leg_f = ax.legend(
        handles=feat_handles,
        title="featurization (colour)",
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        fontsize=8,
        title_fontsize=8,
    )
    ax.add_artist(leg_f)
    leg_d = ax.legend(
        handles=dir_handles,
        title="direction sampling (shape)",
        loc="upper left",
        bbox_to_anchor=(1.02, 0.52),
        fontsize=8,
        title_fontsize=8,
    )
    ax.add_artist(leg_d)
    if refs:
        rhandles = [
            plt.Line2D(
                [],
                [],
                color=rcolors[i % len(rcolors)],
                ls="--",
                lw=1.8,
                label=f"{rlabel} = {rval:.2g}",
            )
            for i, (rlabel, rval) in enumerate(refs)
        ]
        ax.legend(
            handles=rhandles,
            title="reference (band = +/-3x)",
            loc="lower left",
            bbox_to_anchor=(1.02, 0.0),
            fontsize=8,
            title_fontsize=8,
        )

    fig.tight_layout(rect=(0, 0, 0.78, 1))
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("wrote rate comparison -> %s", out_path)
    return str(out_path)


def plot_cv_q_scatter_grid(result: dict[str, Any], out_path: str | Path) -> str:
    """Grid of CV-value vs sliced-committor q scatter (featurization x direction)."""
    plt = _use_agg()
    out_path = Path(out_path)
    combos = [c for c in result.get("combos", []) if c.get("scatter", {}).get("q")]
    if not combos:
        fig, ax = plt.subplots(figsize=(5, 2))
        ax.text(0.5, 0.5, "no scatter data", ha="center", va="center")
        ax.axis("off")
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return str(out_path)

    feats = sorted({c["featurization"] for c in combos})
    dirs = sorted({c["direction"] for c in combos})
    by = {(c["featurization"], c["direction"]): c for c in combos}
    nrow, ncol = len(feats), len(dirs)
    fig, axes = plt.subplots(
        nrow,
        ncol,
        figsize=(2.4 * ncol + 1, 2.2 * nrow + 1),
        squeeze=False,
        sharex=True,
        sharey=True,
    )
    for i, f in enumerate(feats):
        for j, d in enumerate(dirs):
            ax = axes[i][j]
            c = by.get((f, d))
            if c is not None:
                cv = np.asarray(c["scatter"]["cv"])
                q = np.asarray(c["scatter"]["q"])
                ax.scatter(cv, q, s=3, alpha=0.25, color="#4C72B0", edgecolors="none")
                vf = c.get("committor", {}).get("valid_fraction")
                if vf is not None:
                    ax.text(0.03, 0.92, f"valid {vf:.0%}", transform=ax.transAxes, fontsize=6)
            else:
                ax.text(0.5, 0.5, "n/a", ha="center", va="center", fontsize=7)
            if i == 0:
                ax.set_title(d, fontsize=8)
            if j == 0:
                ax.set_ylabel(f"{f}\nq_sliced", fontsize=7)
            ax.tick_params(labelsize=6)
    fig.suptitle(
        f"CV vs sliced committor: {result.get('system', '')} "
        "(diffuse = committor not aligned with CV / flattening)",
        fontsize=10,
    )
    fig.supxlabel("CV value", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("wrote CV-vs-q scatter grid -> %s", out_path)
    return str(out_path)


def _arr(seq):
    """JSON profile list (with None for non-finite) -> float array with NaNs."""
    if not seq:
        return None
    return np.asarray([np.nan if v is None else v for v in seq], dtype=float)


def replot_profiles(profiles_json: str | Path, out_path: str | Path | None = None) -> str:
    """Regenerate ``profiles.png`` from a saved ``profiles.json`` (no pipeline re-run).

    ``profiles.json`` is written by :func:`report.write_report` and carries the
    per-window/per-bin profile arrays + diagnostics, so this redraws the figure
    (e.g. after a label/style tweak) in seconds. Writes next to the JSON by default.
    """
    import json

    profiles_json = Path(profiles_json)
    result = json.loads(profiles_json.read_text())
    out_path = Path(out_path) if out_path is not None else profiles_json.with_name("profiles.png")
    return plot_profiles(result, out_path)


def plot_profiles(result: dict[str, Any], out_path: str | Path) -> str:
    """Per-window D / flux / PMF profiles over CV and q (one point per window).

    Each umbrella window is one data point (so a 30-window run has 30 points per
    curve). The diffusion is the per-window Hummer estimate (CV-space for TPT,
    q-space for Szabo); each flux uses its own matched per-window D. Plateau bars
    mark the value actually taken as the rate.
    """
    plt = _use_agg()
    out_path = Path(out_path)
    combos = [c for c in result.get("combos", []) if c.get("profiles")]
    if not combos:
        fig, ax = plt.subplots(figsize=(5, 2))
        ax.text(0.5, 0.5, "no profile data", ha="center", va="center")
        ax.axis("off")
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return str(out_path)

    # One point per committor bin ("bins" mode) or per umbrella window.
    dmode = result.get("diagnostics", {}).get("diffusion_mode", "per_window")
    loc = "bin centre" if dmode == "bins" else "window centre"
    cvx = f"CV ({loc})"
    qx = f"committor q ({loc})"

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    # (ax, space, field, title, xlabel, logy, plateau_key, mark_bs_local, overlay)
    panels = [
        (
            axes[0][0],
            "cv",
            "D_cv",
            "D along CV (CV-space D) -> Kramers / CV-mapped D_q",
            cvx,
            False,
            None,
            False,
            None,
        ),
        (axes[0][1], "cv", "pmf", "PMF(CV) = -log pi  -> Kramers", cvx, False, None, False, None),
        (
            axes[0][2],
            "q",
            "flux_tpt_cvmap",
            "TPT flux Phi(q) = D_q^mapped pi (CV-mapped q-space D; bars=plateau)",
            qx,
            True,
            "tpt_cvmap_plateau",
            False,
            None,
        ),
        (
            axes[1][0],
            "q",
            "D_q",
            "D along q (q-space D; dashed = CV-mapped) -> Berezhkovskii-Szabo",
            qx,
            False,
            None,
            False,
            "D_q_mapped",
        ),
        (
            axes[1][1],
            "q",
            "flux_tpt_q",
            "Flux Phi(q) = D_q pi (q-space D; bars=plateau, star=BS local q*=0.5)",
            qx,
            True,
            "tpt_q_plateau",
            True,
            None,
        ),
        (axes[1][2], "q", "pi", "committor density pi(q)", qx, False, None, False, None),
    ]
    cmap = plt.get_cmap("tab20")
    for ax, space, field, title, xlabel, logy, plateau_key, mark_bs, overlay in panels:
        for idx, c in enumerate(combos):
            prof = c["profiles"].get(space, {})
            lev = _arr(prof.get("levels"))
            val = _arr(prof.get(field))
            color = cmap(idx % 20)
            if lev is None or val is None or not np.isfinite(val).any():
                continue
            # Connect points left-to-right in the PANEL's x-coordinate. The points
            # are stored sorted by committor (q_center), so a CV-axis panel would
            # otherwise zig-zag wherever the committor is non-monotonic in the CV
            # (purely cosmetic: the rate uses the q-sorted profiles, not this order).
            order = np.argsort(lev)
            lev = lev[order]
            val = val[order]
            ax.plot(lev, val, marker="o", ls="-", ms=4, lw=0.8, alpha=0.7, color=color)
            # Optional overlay (e.g. the CV-mapped D_q) in a dashed style.
            if overlay is not None:
                ov = _arr(prof.get(overlay))
                if ov is not None and np.isfinite(ov).any():
                    ax.plot(
                        lev, ov[order], marker="s", ls="--", ms=3, lw=0.8, alpha=0.7, color=color
                    )
            # Plateau bar actually taken as the rate.
            if plateau_key is not None:
                pm = prof.get(plateau_key)
                if pm and pm.get("value", 0) and pm["value"] > 0:
                    ax.plot(
                        [pm["lo"], pm["hi"]],
                        [pm["value"]] * 2,
                        ls="-",
                        color=color,
                        lw=3.0,
                        alpha=0.95,
                    )
            # Berezhkovskii-Szabo local point read at the q=0.5 surface.
            if mark_bs:
                bl = prof.get("bs_local")
                if bl and bl.get("value", 0) and bl["value"] > 0:
                    ax.plot(
                        bl["q_star"],
                        bl["value"],
                        marker="*",
                        ms=11,
                        color=color,
                        markeredgecolor="k",
                        markeredgewidth=0.4,
                        zorder=5,
                    )
        ax.set_title(title, fontsize=8)
        ax.set_xlabel(xlabel, fontsize=8)
        if logy:
            ax.set_yscale("log")
        ax.grid(ls=":", alpha=0.4)
        ax.tick_params(labelsize=6)

    combo_handles = [
        plt.Line2D(
            [],
            [],
            color=cmap(i % 20),
            marker="o",
            ls="-",
            label=f"{_FEAT_LABELS.get(c['featurization'], c['featurization'])}"
            f" / {_DIR_LABELS.get(c['direction'], c['direction'])}",
        )
        for i, c in enumerate(combos)
    ]
    fig.legend(
        handles=combo_handles,
        loc="lower center",
        ncol=min(4, len(combo_handles)),
        fontsize=6,
        title="featurization / direction (colour)",
        title_fontsize=7,
    )
    nb = result.get("diagnostics", {}).get("n_diff_bins")
    dlabel = {
        "bins": f"{nb} committor bins, Kramers-Moyal diffusion",
        "bins_hummer": f"per-window Hummer diffusion on a {nb}-bin rate grid",
    }.get(dmode, "one point per umbrella window, Hummer diffusion")
    fig.suptitle(
        f"Diffusion / flux / PMF profiles: {result.get('system', '')} ({dlabel})", fontsize=11
    )
    fig.tight_layout(rect=(0, 0.08, 1, 0.96))
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("wrote profiles -> %s", out_path)
    return str(out_path)
