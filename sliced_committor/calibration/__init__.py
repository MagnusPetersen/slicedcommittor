"""Post-hoc calibration of an already-computed sliced committor.

These submodules take a fitted ``SlicedCommittorResult`` + weights and produce
a corrected committor estimate by remapping ``q̂ → f(q̂)``. They do NOT
compute weights themselves (for that, see :func:`sliced_committor.full_gram_weights`
and :func:`sliced_committor.basin_moment_weights`).

  * ``affine``      : affine label-mean calibration via a centered-basis dict.
  * ``abc``         : ABC_v2 global affine calibration ``q̂ = α·q + β``.
  * ``recalibrate`` : 1D-RD recalibration along the aggregate q̄.
"""

from .abc import (
    AffineCalibration,
    apply_affine,
    compute_affine_calibration,
    evaluate_committor_calibrated,
)
from .affine import calibrate_weights_affine
from .recalibrate import (
    apply_recalibration,
    compute_recalibration_curve,
    evaluate_committor_with_recal,
)

__all__ = [
    "AffineCalibration",
    "apply_affine",
    "apply_recalibration",
    "calibrate_weights_affine",
    "compute_affine_calibration",
    "compute_recalibration_curve",
    "evaluate_committor_calibrated",
    "evaluate_committor_with_recal",
]
