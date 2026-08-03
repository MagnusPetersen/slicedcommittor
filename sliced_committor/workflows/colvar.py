"""PLUMED COLVAR parsing.

Reads the plain-text COLVAR files PLUMED writes, coping with the heterogeneity
across our systems: ``#! FIELDS time q [restr.bias]`` (chignolin, 1D),
``#! FIELDS time phi psi bb.bias`` (alanine, 1D bias on phi with psi also
printed), and ``#! FIELDS time cv1 cv2 restr.bias`` (c-Src, 2D). Periodicity is
taken from ``#! SET min_<field> ...`` / ``#! SET max_<field> ...`` directives
(e.g. the ``[-pi, pi]`` range of a torsion).

Only the standard library and numpy are used.
"""

from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import numpy as np

_PI_TOKENS = {"pi": np.pi, "+pi": np.pi, "-pi": -np.pi}


def _parse_set_value(token: str) -> float:
    """Parse a PLUMED ``#! SET`` numeric token, including ``pi`` / ``-pi``."""
    tok = token.strip()
    low = tok.lower()
    if low in _PI_TOKENS:
        return _PI_TOKENS[low]
    return float(tok)


class ColvarData(NamedTuple):
    """Parsed COLVAR contents.

    Attributes:
        fields: ordered field names (including the leading ``time``).
        values: ``(T, n_fields)`` float array aligned with ``fields``.
        periodic: field name -> ``(min, max)`` for periodic fields, else absent.
        path: source file path.
    """

    fields: tuple[str, ...]
    values: np.ndarray
    periodic: dict[str, tuple[float, float]]
    path: str

    @property
    def time(self) -> np.ndarray:
        return self.column("time")

    def column(self, name: str) -> np.ndarray:
        """Return the column for ``name`` (raises if absent)."""
        try:
            j = self.fields.index(name)
        except ValueError as exc:
            raise KeyError(f"field {name!r} not in COLVAR {self.fields}") from exc
        return self.values[:, j]

    def non_time_fields(self) -> tuple[str, ...]:
        return tuple(f for f in self.fields if f != "time")


def read_colvar(path: str | Path) -> ColvarData:
    """Parse a PLUMED COLVAR file.

    Args:
        path: path to the COLVAR file.

    Returns:
        A :class:`ColvarData`. The first ``#! FIELDS`` line defines the columns;
        any repeated header blocks (from simulation restarts / appends) are
        ignored for the schema but their data rows are kept.

    Raises:
        ValueError: if no ``#! FIELDS`` header is found or the data is empty.
    """
    path = Path(path)
    fields: list[str] | None = None
    periodic: dict[str, tuple[float, float]] = {}
    set_min: dict[str, float] = {}
    set_max: dict[str, float] = {}

    with path.open() as fh:
        for line in fh:
            if not line.startswith("#"):
                break  # header is contiguous at the top; stop at first data row
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

    # numpy treats both '#' and the '#!' lines as comments; restart headers and
    # any stray blank lines are skipped automatically.
    values = np.loadtxt(path, comments="#", ndmin=2)
    if values.size == 0:
        raise ValueError(f"COLVAR {path} has no data rows")
    if values.shape[1] != len(fields):
        raise ValueError(
            f"COLVAR {path}: {values.shape[1]} data columns != {len(fields)} FIELDS {fields}"
        )
    return ColvarData(fields=tuple(fields), values=values, periodic=periodic, path=str(path))


def colvar_dt(data: ColvarData) -> float:
    """Median time step between consecutive COLVAR rows (output cadence)."""
    t = data.time
    if t.size < 2:
        return float("nan")
    return float(np.median(np.diff(t)))
