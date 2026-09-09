"""Rate-unit conversions.

Estimated rates come out in inverse trajectory-time units (``1/dt``, with ``dt``
in the trajectory's ``time_unit``, ``ps`` for MD); reference rates are quoted
in whatever their source used. Both convert to ``1/s`` here. Reduced-unit toys
have no SI conversion and return None.
"""

from __future__ import annotations

# Seconds per one unit of the named time unit.
_SECONDS_PER_TIME_UNIT = {
    "fs": 1e-15,
    "ps": 1e-12,
    "ns": 1e-9,
    "us": 1e-6,
    "ms": 1e-3,
    "s": 1.0,
    "reduced": None,  # dimensionless; no SI conversion
}

# Multiply a rate in these units to get 1/s.
_RATE_UNIT_TO_PER_S = {
    "1/fs": 1e15,
    "1/ps": 1e12,
    "1/ns": 1e9,
    "1/us": 1e6,
    "1/ms": 1e3,
    "1/s": 1.0,
}


def estimated_to_per_s(k_native: float, time_unit: str) -> float | None:
    """Convert an estimated rate (``1/time_unit``) to ``1/s``.

    Returns ``None`` for reduced (dimensionless) units, signalling that no SI
    normalisation applies.
    """
    spu = _SECONDS_PER_TIME_UNIT.get(time_unit)
    if spu is None:
        return None
    return float(k_native) / spu


def reference_to_per_s(k: float, units: str) -> float | None:
    """Convert a reference rate in ``units`` to ``1/s`` (None if reduced)."""
    factor = _RATE_UNIT_TO_PER_S.get(units)
    if factor is None:
        return None
    return float(k) * factor


def is_reduced(time_unit: str) -> bool:
    return _SECONDS_PER_TIME_UNIT.get(time_unit) is None
