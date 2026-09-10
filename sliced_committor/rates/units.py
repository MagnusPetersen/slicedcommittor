"""Rate-unit conversion.

Estimated rates come out in inverse trajectory-time units (``1/dt``, with ``dt``
in the trajectory's ``time_unit``, ``ps`` for MD) and convert to ``1/s`` here.
Reduced-unit toys have no SI conversion and return None.
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


def estimated_to_per_s(k_native: float, time_unit: str) -> float | None:
    """Convert an estimated rate (``1/time_unit``) to ``1/s``.

    Returns ``None`` for reduced (dimensionless) units, signalling that no SI
    normalisation applies.
    """
    spu = _SECONDS_PER_TIME_UNIT.get(time_unit)
    if spu is None:
        return None
    return float(k_native) / spu
