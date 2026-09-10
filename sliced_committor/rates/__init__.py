"""Reaction rates from a fitted committor.

The state is the pair ``{D_q(q), pi(q)}``: the diffusion and the density along
the committor coordinate. Their product, the reactive flux ``nu_R(q)``, is
constant for the exact committor and reduces to the rate; how far it is from
constant measures the committor. The density comes from the static ensemble
(:func:`density`, :func:`basin_populations`); the diffusion comes from one of
three constructors:

* measured on the trajectory: :func:`diffusion_profile` (Kramers-Moyal at an
  explicit lag; :func:`lag_scan` shows whether there is one),
  :func:`hummer_diffusion` (per umbrella window), :func:`pooled_acf_diffusion`
  (the paper's, autocorrelations pooled over windows);
* mapped from a collective variable through the Jacobian,
  :func:`committor_diffusion_from_cv` of :func:`committor_grad_sq`, with the
  CV's own mean squared gradient from :func:`linear_response_grad_sq` (or
  :func:`committor_diffusion_from_cv_reparam`, which needs no gradient of
  the CV);
* an assumed configurational ``D0``: the same map with ``cv_grad_sq=1``.

:func:`rate_from_profiles` reduces the pair; :func:`committor_rate` does the
whole thing in one call. :mod:`sliced_committor.rates.baselines` holds the
committor-free Kramers row and :mod:`sliced_committor.rates.units` the
rate-unit conversions.
"""

from ._coordinate import PlateauWindow, Profile, find_plateau, flux_flatness, value_at
from .diffusion import (
    PooledDiffusion,
    diffusion_profile,
    hummer_diffusion,
    lag_scan,
    pooled_acf_diffusion,
)
from .formulas import committor_rate, rate_from_profiles
from .quantities import (
    basin_populations,
    committor_diffusion_from_cv,
    committor_diffusion_from_cv_reparam,
    committor_grad_sq,
    density,
    linear_response_grad_sq,
)

__all__ = [
    # the static ensemble
    "density",
    "committor_grad_sq",
    "basin_populations",
    # the diffusion: measured
    "diffusion_profile",
    "lag_scan",
    "hummer_diffusion",
    "pooled_acf_diffusion",
    "PooledDiffusion",
    # the diffusion: mapped from a collective variable
    "committor_diffusion_from_cv",
    "committor_diffusion_from_cv_reparam",
    "linear_response_grad_sq",
    # the reduction
    "rate_from_profiles",
    "committor_rate",
    # profiles and their reducers
    "Profile",
    "value_at",
    "flux_flatness",
    "find_plateau",
    "PlateauWindow",
]
