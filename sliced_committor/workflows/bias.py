"""Harmonic-bias parameter parsing for umbrella windows.

Two sources of per-window restraint centers and force constants:

* the PLUMED input (``plumed.dat``) ``RESTRAINT`` line, e.g.
  ``restr: RESTRAINT ARG=cv1,cv2 KAPPA=2092,2092 AT=-0.125,0.625``; and
* the window directory name, for the c-Src grid which encodes the target CV in
  the folder name (e.g. ``W_C1m001p00_C2p008p00`` = CV1 = -1.00 A, CV2 = 8.00 A,
  i.e. -0.1 nm, 0.8 nm). PLUMED centers are in nm; the name encodes Angstrom, so
  the decoder divides by 10.

The loader (see :mod:`loaders`) tries ``plumed.dat`` first, then the directory
name, then a sidecar / ``target_cv.json``.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import NamedTuple

import numpy as np

_RESTRAINT_RE = re.compile(r"\bRESTRAINT\b(.*)", re.IGNORECASE)
_ARG_RE = re.compile(r"\bARG=([^\s]+)", re.IGNORECASE)
_KAPPA_RE = re.compile(r"\bKAPPA=([^\s]+)", re.IGNORECASE)
_AT_RE = re.compile(r"\bAT=([^\s]+)", re.IGNORECASE)
# Window-name CV token: C<idx><sign><int>p<frac>, e.g. C1m001p00 / C2p008p00.
_NAME_CV_RE = re.compile(r"C(\d+)([mp])(\d+)p(\d+)")


class RestraintSpec(NamedTuple):
    """A harmonic restraint parsed from a PLUMED input.

    Attributes:
        arg_names: restrained CV names in order (e.g. ``("cv1", "cv2")``).
        kappa: ``(n_cv,)`` force constants in PLUMED units (kJ/mol/CV-unit^2).
        at: ``(n_cv,)`` restraint centers in PLUMED CV units.
    """

    arg_names: tuple[str, ...]
    kappa: np.ndarray
    at: np.ndarray


def _split_floats(token: str) -> np.ndarray:
    return np.array([float(x) for x in token.split(",")], dtype=float)


def parse_restraint(plumed_path: str | Path) -> RestraintSpec | None:
    """Parse the first ``RESTRAINT`` line of a PLUMED input.

    Args:
        plumed_path: path to a ``plumed.dat`` (or equivalent) file.

    Returns:
        A :class:`RestraintSpec`, or ``None`` if the file has no RESTRAINT line
        with both ``KAPPA`` and ``AT`` (e.g. an INCLUDE-only stub).
    """
    plumed_path = Path(plumed_path)
    if not plumed_path.is_file():
        return None
    text = plumed_path.read_text()
    for raw in text.splitlines():
        line = raw.split("#", 1)[0]  # strip trailing comments
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


def center_from_window_name(name: str) -> np.ndarray | None:
    """Decode CV centers (in nm) from a c-Src-style window directory name.

    ``W_C1m001p00_C2p008p00`` -> ``[-0.1, 0.8]`` nm (the name encodes Angstrom;
    divide by 10). Returns ``None`` if the name has no ``C<idx>...`` tokens
    (e.g. ``window_07``).

    Args:
        name: the window directory basename.

    Returns:
        ``(n_cv,)`` centers in nm ordered by CV index, or ``None``.
    """
    matches = _NAME_CV_RE.findall(name)
    if not matches:
        return None
    by_idx: dict[int, float] = {}
    for idx_s, sign, int_s, frac_s in matches:
        val_angstrom = float(f"{int_s}.{frac_s}")
        if sign == "m":
            val_angstrom = -val_angstrom
        by_idx[int(idx_s)] = val_angstrom / 10.0  # Angstrom -> nm
    return np.array([by_idx[i] for i in sorted(by_idx)], dtype=float)
