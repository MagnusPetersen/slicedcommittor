"""Reaction-rate computations on a callable sliced committor.

Three quantity primitives along the committor coordinate -- diffusion
coefficient, density, and reactive flux -- plus five rate formulas that combine
them. Every function takes the callable committor ``q`` (from
``sliced_committor.fit_committor`` / ``build_committor``) and its data; nothing
needs the underlying result or weights.

Quantities (return a :class:`~sliced_committor.rates._coordinate.Profile`, or a
value when ``at=`` is given)::

    from sliced_committor import density, diffusion_coefficient, reactive_flux

Rates (return ``{nu_R, rho_A, rho_B, k_AB, k_BA, ...}``)::

    from sliced_committor import (
        dirichlet_rate, berezhkovskii_szabo_rate, tpt_rate, kramers_rate,
    )
"""

from ._coordinate import PlateauWindow, Profile, find_plateau, value_at
from .formulas import (
    berezhkovskii_szabo_rate,
    committor_rate,
    dirichlet_rate,
    kramers_rate,
    tpt_rate,
)
from .bridge import (
    bayesian_smoluchowski_diffusion,
    bootstrap_barrier_Ds,
    committor_grad_profile,
    committor_populations,
    conditional_mean,
    constancy_reconstruction,
    flux_reductions,
    hummer_Ds_profile,
    mapped_committor_diffusion_field,
    mapped_committor_diffusion_reparam,
)
from .quantities import (
    BridgeD,
    density,
    diffusion_coefficient,
    mapped_committor_diffusion,
    reactive_flux,
    saddle_bridge_D,
)

__all__ = [
    # Quantities
    "density",
    "diffusion_coefficient",
    "reactive_flux",
    "saddle_bridge_D",
    "mapped_committor_diffusion",
    "BridgeD",
    # Bridge v2 (position-dependent / regressed / UQ)
    "mapped_committor_diffusion_field",
    "mapped_committor_diffusion_reparam",
    "committor_grad_profile",
    "conditional_mean",
    "flux_reductions",
    "committor_populations",
    "hummer_Ds_profile",
    "constancy_reconstruction",
    "bootstrap_barrier_Ds",
    "bayesian_smoluchowski_diffusion",
    # Rates
    "dirichlet_rate",
    "berezhkovskii_szabo_rate",
    "committor_rate",
    "tpt_rate",
    "kramers_rate",
    # Profile helpers
    "Profile",
    "value_at",
    "find_plateau",
    "PlateauWindow",
]
