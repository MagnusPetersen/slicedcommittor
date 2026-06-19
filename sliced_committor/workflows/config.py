"""Per-system configuration registry for the umbrella-sampling workflow.

Each :class:`SystemConfig` collects everything the pipeline needs that cannot be
read from the raw umbrella files: basin (state) definitions, CV metadata,
temperature, topology hints, featurization defaults, and literature/reference
rates for the comparison plot.

The registry is intentionally small and declarative. Basin membership is computed
by :func:`compute_basins` from a couple of geometric primitives (axis-aligned box,
Euclidean ball) evaluated on either the CV array or the feature array; this covers
1D threshold states (chignolin Q, alanine phi) and 2D ball states (c-Src in the
(alphaC, A-loop) plane, the analytic toys in the (x, y) plane) uniformly.

A sidecar config (see :mod:`cli`) can override any field at run time, so this
registry is a convenience for the known systems, not a hard requirement.
"""

from __future__ import annotations

from typing import Any, NamedTuple

import numpy as np

# Boltzmann constant in kJ/mol/K (GROMACS / PLUMED energy units).
KB_KJ_PER_MOL_K = 0.00831446261815324


def beta_from_temperature(temperature_K: float | None) -> float:
    """Inverse temperature ``1/kT`` in (kJ/mol)^-1, or 1.0 for reduced units.

    Toy systems carry their own dimensionless ``beta`` (typically 1.0) in the
    stored samples, so ``temperature_K=None`` returns 1.0.
    """
    if temperature_K is None:
        return 1.0
    return 1.0 / (KB_KJ_PER_MOL_K * float(temperature_K))


class Region(NamedTuple):
    """A basin region: a box or a ball evaluated on the CV or feature array.

    Attributes:
        kind: ``"box"`` (axis-aligned) or ``"ball"`` (Euclidean).
        on: ``"cvs"`` or ``"features"`` (which per-frame array to test).
        lo: box lower bounds (per tested dim) for ``kind="box"``.
        hi: box upper bounds for ``kind="box"``.
        center: ball center for ``kind="ball"``.
        radius: ball radius for ``kind="ball"``.
        dims: which column indices of the tested array to use (default: all).
    """

    kind: str
    on: str
    lo: tuple[float, ...] | None = None
    hi: tuple[float, ...] | None = None
    center: tuple[float, ...] | None = None
    radius: float | None = None
    dims: tuple[int, ...] | None = None


def _region_mask(region: Region, features: np.ndarray, cvs: np.ndarray) -> np.ndarray:
    arr = cvs if region.on == "cvs" else features
    arr = np.asarray(arr)
    if arr.ndim == 1:
        arr = arr[:, None]
    dims = region.dims if region.dims is not None else tuple(range(arr.shape[1]))
    sub = arr[:, list(dims)]
    if region.kind == "box":
        lo = np.asarray(region.lo, dtype=float)
        hi = np.asarray(region.hi, dtype=float)
        return np.all((sub >= lo[None, :]) & (sub <= hi[None, :]), axis=1)
    if region.kind == "ball":
        center = np.asarray(region.center, dtype=float)
        d = np.linalg.norm(sub - center[None, :], axis=1)
        return d < float(region.radius)
    raise ValueError(f"unknown region kind {region.kind!r}")


def compute_basins(
    features: np.ndarray, cvs: np.ndarray, region_A: Region, region_B: Region
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(in_A, in_B)`` boolean masks, made disjoint (A loses overlap).

    Args:
        features: ``(N, d)`` feature array.
        cvs: ``(N, n_cv)`` CV array.
        region_A: state-A region (committor boundary q=0).
        region_B: state-B region (committor boundary q=1).

    Returns:
        ``(in_A, in_B)`` each ``(N,)`` bool. Any accidental overlap is removed
        from A so the two are disjoint (required by ``compute_sliced_committor``).
    """
    in_A = _region_mask(region_A, features, cvs)
    in_B = _region_mask(region_B, features, cvs)
    in_A = in_A & ~in_B
    return in_A, in_B


class SystemConfig(NamedTuple):
    """Declarative configuration for one umbrella-sampling system.

    Attributes:
        name: registry key.
        n_cv: bias dimensionality (1 or 2).
        cv_names: per-CV label.
        cv_periodic: per-CV ``(min, max)`` period or ``None``.
        temperature_K: simulation temperature, or ``None`` for reduced-unit toys.
        region_A, region_B: basin definitions (see :func:`compute_basins`).
        featurizations: featurization names available for this system. Toy
            systems use ``("identity",)``; proteins the full set.
        featurize_params: per-featurization keyword params (atom selections etc.).
        topology_globs: filename globs to locate a topology for mdtraj.
        colvar_cv_columns: which COLVAR data columns hold the bias CV(s)
            (after ``time``), or ``None`` to auto-detect.
        reference_rates: label -> ``{"k": value, "units": str, "source": str}``
            for the comparison plot. Empty for toys (their references are loaded
            from the stored ``.npz`` instead).
        time_unit: human label for ``dt`` (e.g. ``"ps"``, ``"reduced"``).
        known_diffusion: configurational diffusion D when known a priori (the
            analytic toys, D=1), used by the feature-space rate as the exact
            length scale. ``None`` for real MD (estimate D from the trajectory).
        has_dynamics: whether the stored frames form a real dynamical trajectory
            usable for diffusion estimation. ``False`` for the toys (strided
            umbrella snapshots), ``True`` for MD.
        notes: free-form provenance / caveats.
    """

    name: str
    n_cv: int
    cv_names: tuple[str, ...]
    cv_periodic: tuple[tuple[float, float] | None, ...]
    temperature_K: float | None
    region_A: Region
    region_B: Region
    featurizations: tuple[str, ...]
    featurize_params: dict[str, Any]
    topology_globs: tuple[str, ...]
    colvar_cv_columns: tuple[int, ...] | None
    reference_rates: dict[str, dict[str, Any]]
    time_unit: str
    known_diffusion: float | None = None
    has_dynamics: bool = True
    solver_kwargs: dict[str, Any] | None = None
    notes: str = ""


_TWO_PI = 2.0 * np.pi

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
# Analytic toys: features are the raw 2D configuration (x, y); CV is x. Basins
# are balls in the (x, y) plane. References are loaded from the stored .npz, so
# reference_rates is left empty here.

_DOUBLE_WELL = SystemConfig(
    name="double_well",
    n_cv=1,
    cv_names=("x",),
    cv_periodic=(None,),
    temperature_K=None,
    region_A=Region("ball", "features", center=(-1.0, 0.0), radius=0.3),
    region_B=Region("ball", "features", center=(1.0, 0.0), radius=0.3),
    featurizations=("identity",),
    featurize_params={},
    topology_globs=(),
    colvar_cv_columns=None,
    reference_rates={},
    time_unit="reduced",
    known_diffusion=1.0,
    has_dynamics=True,
    notes="2D double well V=2(x^2-1)^2+y^2, beta=1, D0=1. Regenerated fine-stride US.",
)

_DOUBLE_WELL_HIGH = _DOUBLE_WELL._replace(
    name="double_well_high",
    notes="Higher-barrier (deltaV=20) 2D double well. The user's '2D scaled-up'.",
)

# Wolfe-Quapp: rotated quartic; minima quasi-aligned with x after rotation.
_WOLFE_QUAPP = SystemConfig(
    name="wolfe_quapp",
    n_cv=1,
    cv_names=("x",),
    cv_periodic=(None,),
    temperature_K=None,
    region_A=Region("ball", "features", center=(-1.863, 0.022), radius=0.4),
    region_B=Region("ball", "features", center=(1.887, 0.022), radius=0.4),
    featurizations=("identity",),
    featurize_params={},
    topology_globs=(),
    colvar_cv_columns=None,
    reference_rates={},
    time_unit="reduced",
    known_diffusion=1.0,
    has_dynamics=True,
    notes="Rotated Wolfe-Quapp 2D, D0=1; CV=x is deliberately sub-optimal.",
)

# Wolfe-Quapp "stiff" tube: same potential, but a 2D umbrella that also restrains
# y to the (upper) MEP, so only the channel around the MEP is sampled. n_cv=2 so
# the orthogonal restraint is reweighted; basins/refs are the same as wolfe_quapp.
_WOLFE_QUAPP_STIFF = SystemConfig(
    name="wolfe_quapp_stiff",
    n_cv=2,
    cv_names=("x", "y"),
    cv_periodic=(None, None),
    temperature_K=None,
    region_A=Region("ball", "features", center=(-1.863, 0.022), radius=0.4),
    region_B=Region("ball", "features", center=(1.887, 0.022), radius=0.4),
    featurizations=("identity",),
    featurize_params={},
    topology_globs=(),
    colvar_cv_columns=None,
    reference_rates={},
    time_unit="reduced",
    known_diffusion=1.0,
    has_dynamics=True,
    notes="WQ confined to the upper MEP via a stiff y-restraint (tube); 2D US.",
)

# Alanine dipeptide: 1D phi bias (periodic), vacuum AMBER99SB. C7eq vs C7ax/alpha.
_ALANINE = SystemConfig(
    name="alanine_dipeptide",
    n_cv=1,
    cv_names=("phi",),
    cv_periodic=((-np.pi, np.pi),),
    temperature_K=300.0,
    region_A=Region("box", "cvs", lo=(-_TWO_PI,), hi=(-0.5,), dims=(0,)),
    region_B=Region("box", "cvs", lo=(0.5,), hi=(2.0,), dims=(0,)),
    featurizations=("cv_only", "dihedrals", "aligned_cartesian", "distance_matrix", "contact_map"),
    featurize_params={
        "dihedrals": {"kinds": ("phi", "psi")},
        "aligned_cartesian": {"selection": "not element H"},
        "distance_matrix": {"selection": "not element H"},
        "contact_map": {"selection": "not element H", "r0_nm": 0.5},
    },
    topology_globs=("reference.pdb", "build_output/*.pdb", "*.pdb", "*.gro"),
    colvar_cv_columns=(0,),  # FIELDS time phi psi bb.bias -> phi is column 0
    reference_rates={
        "Bonomi2024": {
            "k": 2.86e5,
            "units": "1/s",
            "source": "Bonomi et al. 2024 arXiv:2401.14237",
        },
    },
    time_unit="ps",
    notes="A={phi<-0.5} (C7eq), B={0.5<phi<2.0} (C7ax/alphaR). Periodic phi.",
)

# Chignolin (CLN025): 1D Best-Hummer Q. A=folded (high Q), B=unfolded (low Q),
# so k_AB is the UNFOLDING rate (matches the plotted literature k_unfold).
_CHIGNOLIN = SystemConfig(
    name="chignolin",
    n_cv=1,
    cv_names=("Q",),
    cv_periodic=(None,),
    temperature_K=340.0,
    region_A=Region("box", "cvs", lo=(0.85,), hi=(np.inf,), dims=(0,)),
    region_B=Region("box", "cvs", lo=(-np.inf,), hi=(0.30,), dims=(0,)),
    featurizations=("cv_only", "dihedrals", "aligned_cartesian", "distance_matrix", "contact_map"),
    featurize_params={
        "dihedrals": {"kinds": ("phi", "psi", "chi1")},
        "aligned_cartesian": {"selection": "name CA"},
        "distance_matrix": {"selection": "name CA"},
        "contact_map": {"selection": "name CA", "r0_nm": 0.8},
    },
    topology_globs=(
        "build_output/_solvated.gro",
        "build_output/*.gro",
        "windows/*/conf.gro",
        "windows/*/*.gro",
        "*.pdb",
    ),
    colvar_cv_columns=(0,),  # FIELDS time q [restr.bias] -> q is column 0
    reference_rates={
        # Only the unfolding reference is plotted (k_fold dropped on request).
        "LL2011_k_unfold": {"k": 4.5e5, "units": "1/s", "source": "Lindorff-Larsen 2011 (k_BA)"},
    },
    time_unit="ps",
    notes="A=folded {Q>0.85}, B=unfolded {Q<0.30}; k_AB = unfolding rate (vs LL2011 k_unfold).",
)

# c-Src activation: 2D (alphaC rotation, A-loop opening). A=inactive (2src),
# B=active (1y57). Reference CVs and basin radius from cSrc_prep/analysis.
_CSRC = SystemConfig(
    name="csrc_activation",
    n_cv=2,
    cv_names=("cv1_alphaC", "cv2_aloop"),
    cv_periodic=(None, None),
    temperature_K=300.0,
    region_A=Region("ball", "cvs", center=(-0.946, 0.316), radius=0.35),
    region_B=Region("ball", "cvs", center=(1.113, 1.078), radius=0.35),
    featurizations=("cv_only", "dihedrals", "aligned_cartesian", "distance_matrix", "contact_map"),
    featurize_params={
        "dihedrals": {"kinds": ("phi", "psi")},
        "aligned_cartesian": {"selection": "name CA"},
        "distance_matrix": {"selection": "name CA"},
        "contact_map": {"selection": "name CA", "r0_nm": 0.8},
    },
    topology_globs=("windows/*/proc.pdb", "windows/*/*.pdb", "windows/*/*.gro", "*.pdb"),
    colvar_cv_columns=(0, 1),  # FIELDS time cv1 cv2 restr.bias
    reference_rates={
        "MengRoux2016": {
            "k": 0.01053,
            "units": "1/us",
            "source": "Meng & Roux 2016 PNAS (MSM MFPT)",
        },
    },
    time_unit="ps",
    notes="A=inactive(2src), B=active(1y57), ball radius 0.35 nm. MBAR OOMs -> WHAM.",
)

REGISTRY: dict[str, SystemConfig] = {
    c.name: c
    for c in (
        _DOUBLE_WELL,
        _DOUBLE_WELL_HIGH,
        _WOLFE_QUAPP,
        _WOLFE_QUAPP_STIFF,
        _ALANINE,
        _CHIGNOLIN,
        _CSRC,
    )
}

# Aliases for folder-name heuristics.
_ALIASES = {
    "wq_stiff_us": "wolfe_quapp_stiff",
    "wq_stiff": "wolfe_quapp_stiff",
    "us_chignolin_gmx": "chignolin",
    "chignolin_c22star": "chignolin",
    "chignolin_gmx": "chignolin",
    "us_csrc_activation_gmx": "csrc_activation",
    "us_csrc_activation": "csrc_activation",
    "src_activation": "csrc_activation",
    "us_alanine_dipeptide": "alanine_dipeptide",
    "alanine-dipeptide": "alanine_dipeptide",
    "ala2": "alanine_dipeptide",
}

TOY_SYSTEMS = ("double_well", "double_well_high", "wolfe_quapp", "wolfe_quapp_stiff")


def get_config(name: str) -> SystemConfig:
    """Look up a :class:`SystemConfig` by registry name or alias."""
    key = name.strip().lower()
    if key in REGISTRY:
        return REGISTRY[key]
    if key in _ALIASES:
        return REGISTRY[_ALIASES[key]]
    raise KeyError(
        f"unknown system {name!r}; known: {sorted(REGISTRY)} (aliases: {sorted(_ALIASES)})"
    )


def match_config(folder_name: str) -> SystemConfig | None:
    """Best-effort match of a folder name to a registry entry (or None)."""
    key = folder_name.strip().lower()
    if key in REGISTRY:
        return REGISTRY[key]
    if key in _ALIASES:
        return REGISTRY[_ALIASES[key]]
    for alias, target in _ALIASES.items():
        if alias in key:
            return REGISTRY[target]
    for name in REGISTRY:
        if name in key:
            return REGISTRY[name]
    return None
