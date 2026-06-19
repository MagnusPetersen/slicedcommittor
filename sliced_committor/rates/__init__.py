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
