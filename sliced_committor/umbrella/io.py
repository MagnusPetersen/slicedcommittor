"""Reading umbrella-sampling data: PLUMED COLVAR files, restraints, trajectories.

* :func:`read_colvar` parses the plain-text COLVAR files PLUMED writes, taking
  the periodicity of a field from its ``#! SET min_`` / ``#! SET max_``
  directives (the ``[-pi, pi]`` range of a torsion, say).
* :func:`parse_restraint` reads the ``RESTRAINT`` line of a PLUMED input for a
  window's centre and force constant.
* :func:`load_trajectory` wraps mdtraj, and :func:`align_colvar_traj` pairs
  COLVAR rows with trajectory frames by timestamp when PLUMED and the MD
  engine write at different strides.

Only :func:`load_trajectory` needs mdtraj; the rest is numpy.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import NamedTuple

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# COLVAR
# ---------------------------------------------------------------------------
_PI_TOKENS = {"pi": np.pi, "+pi": np.pi, "-pi": -np.pi}


def _parse_set_value(token: str) -> float:
    """A PLUMED ``#! SET`` numeric token, including ``pi`` / ``-pi``."""
    tok = token.strip()
    low = tok.lower()
    if low in _PI_TOKENS:
        return _PI_TOKENS[low]
    return float(tok)


class ColvarData(NamedTuple):
    """Parsed COLVAR contents.

    fields:   ordered field names, including the leading ``time``.
    values:   ``(T, n_fields)`` float array aligned with ``fields``.
    periodic: field name -> ``(min, max)`` for periodic fields, else absent.
    path:     source file path.
    """

    fields: tuple[str, ...]
    values: np.ndarray
    periodic: dict[str, tuple[float, float]]
    path: str

    @property
    def time(self) -> np.ndarray:
        return self.column("time")

    def column(self, name: str) -> np.ndarray:
        """The column for ``name`` (raises KeyError if absent)."""
        try:
            j = self.fields.index(name)
        except ValueError as exc:
            raise KeyError(f"field {name!r} not in COLVAR {self.fields}") from exc
        return self.values[:, j]

    def non_time_fields(self) -> tuple[str, ...]:
        return tuple(f for f in self.fields if f != "time")


def read_colvar(path: str | Path) -> ColvarData:
    """Parse a PLUMED COLVAR file.

    The first ``#! FIELDS`` line defines the columns; repeated header blocks
    (restarts, appends) are ignored for the schema but their data rows are
    kept. Raises ValueError without a ``#! FIELDS`` header or without data.
    """
    path = Path(path)
    fields: list[str] | None = None
    periodic: dict[str, tuple[float, float]] = {}
    set_min: dict[str, float] = {}
    set_max: dict[str, float] = {}

    with path.open() as fh:
        for line in fh:
            if not line.startswith("#"):
                break  # the header is contiguous at the top
            tokens = line.lstrip("#! ").split()
            if not tokens:
                continue
            if tokens[0] == "FIELDS" and fields is None:
                fields = tokens[1:]
            elif tokens[0] == "SET" and len(tokens) >= 3:
                key, val = tokens[1], tokens[2]
                if key.startswith("min_"):
                    set_min[key[4:]] = _parse_set_value(val)
                elif key.startswith("max_"):
                    set_max[key[4:]] = _parse_set_value(val)

    if fields is None:
        raise ValueError(f"no '#! FIELDS' header in {path}")
    for name in set_min:
        if name in set_max:
            periodic[name] = (set_min[name], set_max[name])

    values = np.loadtxt(path, comments="#", ndmin=2)
    if values.size == 0:
        raise ValueError(f"COLVAR {path} has no data rows")
    if values.shape[1] != len(fields):
        raise ValueError(
            f"COLVAR {path}: {values.shape[1]} data columns != {len(fields)} FIELDS {fields}"
        )
    return ColvarData(fields=tuple(fields), values=values, periodic=periodic, path=str(path))


def colvar_dt(data: ColvarData) -> float:
    """Median time step between consecutive COLVAR rows."""
    t = data.time
    if t.size < 2:
        return float("nan")
    return float(np.median(np.diff(t)))


# ---------------------------------------------------------------------------
# restraints
# ---------------------------------------------------------------------------
_RESTRAINT_RE = re.compile(r"\bRESTRAINT\b(.*)", re.IGNORECASE)
_ARG_RE = re.compile(r"\bARG=([^\s]+)", re.IGNORECASE)
_KAPPA_RE = re.compile(r"\bKAPPA=([^\s]+)", re.IGNORECASE)
_AT_RE = re.compile(r"\bAT=([^\s]+)", re.IGNORECASE)


class RestraintSpec(NamedTuple):
    """A harmonic restraint parsed from a PLUMED input.

    arg_names: restrained CV names in order, e.g. ``("cv1", "cv2")``.
    kappa:     ``(n_cv,)`` force constants in PLUMED units (kJ/mol per CV-unit^2).
    at:        ``(n_cv,)`` restraint centres in PLUMED CV units.
    """

    arg_names: tuple[str, ...]
    kappa: np.ndarray
    at: np.ndarray


def _split_floats(token: str) -> np.ndarray:
    return np.array([float(x) for x in token.split(",")], dtype=float)


def parse_restraint(plumed_path: str | Path) -> RestraintSpec | None:
    """The first ``RESTRAINT`` line of a PLUMED input with both ``KAPPA`` and ``AT``.

    Returns None for a missing file or an INCLUDE-only stub. A scalar ``KAPPA``
    is broadcast over the ``AT`` components.
    """
    plumed_path = Path(plumed_path)
    if not plumed_path.is_file():
        return None
    for raw in plumed_path.read_text().splitlines():
        line = raw.split("#", 1)[0]
        m = _RESTRAINT_RE.search(line)
        if not m:
            continue
        rest = m.group(1)
        m_kappa = _KAPPA_RE.search(rest)
        m_at = _AT_RE.search(rest)
        if not (m_kappa and m_at):
            continue
        kappa = _split_floats(m_kappa.group(1))
        at = _split_floats(m_at.group(1))
        m_arg = _ARG_RE.search(rest)
        args = (
            tuple(m_arg.group(1).split(",")) if m_arg else tuple(f"cv{i}" for i in range(len(at)))
        )
        if kappa.size == 1 and at.size > 1:
            kappa = np.full(at.size, kappa[0])
        return RestraintSpec(arg_names=args, kappa=kappa, at=at)
    return None


# ---------------------------------------------------------------------------
# trajectories
# ---------------------------------------------------------------------------
def load_trajectory(traj_path: str | Path, top_path: str | Path, *, stride: int = 1):
    """An ``mdtraj.Trajectory`` from a trajectory file and a topology, read every ``stride`` frames."""
    try:
        import mdtraj as md
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "mdtraj is required to load trajectories; install with "
            "'pip install sliced-committor[umbrella]'."
        ) from exc
    return md.load(str(traj_path), top=str(top_path), stride=stride)


def align_colvar_traj(
    colvar_time: np.ndarray,
    traj_time: np.ndarray | None,
    n_traj_frames: int,
    *,
    tol_frac: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Pair COLVAR rows with trajectory frames by timestamp, at the coarser cadence.

    Both series must increase in time. The coarser one anchors the pairing;
    for each of its times the nearest time of the other within ``tol_frac``
    of the coarser cadence is paired. Without usable trajectory timestamps the
    finer series is subsampled by stride onto the coarser length.

    Returns ``(colvar_idx, traj_idx)``, equal-length index arrays.
    """
    colvar_time = np.asarray(colvar_time, dtype=float).reshape(-1)
    tc = colvar_time.size

    degenerate = (
        traj_time is None
        or np.asarray(traj_time).size != n_traj_frames
        or n_traj_frames < 2
        or float(np.ptp(np.asarray(traj_time, dtype=float))) <= 0.0
    )
    if degenerate:
        n = min(tc, n_traj_frames)
        if tc >= n_traj_frames:
            ci = np.linspace(0, tc - 1, n).round().astype(int)
            ti = np.arange(n)
        else:
            ci = np.arange(n)
            ti = np.linspace(0, n_traj_frames - 1, n).round().astype(int)
        logger.warning(
            "align_colvar_traj: no trajectory timestamps; subsampling by length "
            "(colvar=%d, traj=%d -> %d paired frames)",
            tc,
            n_traj_frames,
            n,
        )
        return ci, ti

    traj_time = np.asarray(traj_time, dtype=float).reshape(-1)
    cad_c = float(np.median(np.diff(colvar_time))) if tc > 1 else np.inf
    cad_t = float(np.median(np.diff(traj_time))) if traj_time.size > 1 else np.inf
    tol = tol_frac * max(cad_c, cad_t)

    if cad_t >= cad_c:  # the trajectory is coarser: anchor on it
        nn = np.clip(np.searchsorted(colvar_time, traj_time), 1, tc - 1)
        left, right = colvar_time[nn - 1], colvar_time[nn]
        ci = np.where(np.abs(traj_time - left) <= np.abs(traj_time - right), nn - 1, nn)
        ti = np.arange(traj_time.size)
        good = np.abs(colvar_time[ci] - traj_time) <= tol
        return ci[good], ti[good]
    nn = np.clip(np.searchsorted(traj_time, colvar_time), 1, traj_time.size - 1)
    left, right = traj_time[nn - 1], traj_time[nn]
    ti = np.where(np.abs(colvar_time - left) <= np.abs(colvar_time - right), nn - 1, nn)
    ci = np.arange(tc)
    good = np.abs(traj_time[ti] - colvar_time) <= tol
    return ci[good], ti[good]
