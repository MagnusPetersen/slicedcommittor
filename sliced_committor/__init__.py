"""Sliced committor: sample-based committor approximation via 1D projections.

Given samples and basin labels (`in_A`, `in_B`), compute the committor q(x)
via random 1D projections + 1D reaction-diffusion solves + weighted
recombination. The committor is returned as a callable JAX function ``q(x)``;
because it is a pure function, ``jax.grad(q)`` (wrapped as ``committor_gradient``)
gives the spatial gradient, and the rate computations build on it directly.

Quickstart::

    import jax
    jax.config.update("jax_enable_x64", True)   # required for EBMC weights

    from sliced_committor import fit_committor

    # samples : (N, dim) array; in_A, in_B : (N,) bool basin labels
    q = fit_committor(samples, in_A=in_A, in_B=in_B, n_directions=256)
    vals = q(points)                 # committor at arbitrary points

Rates (build on the callable committor)::

    from sliced_committor import dirichlet_rate, berezhkovskii_szabo_rate
    rate = dirichlet_rate(q, samples, D=D)        # {nu_R, rho_A, rho_B, k_AB, k_BA}

Notes:
    * No `beta` argument. The committor is β-invariant given fixed samples;
      β goes in, β comes out, leaves no trace. Internally the library uses
      β = 1 so that `result.free_energies` stores `-log ρ` directly.
    * Basin labels are bool arrays, not callable state functions. Define
      states however you want (clustering, RMSD thresholds, geometric
      expressions) and pass the resulting masks.
"""

from .core.solver import (
    SlicedCommittorResult,
    WeightingContext,
    compute_basin_moment_weights,
    compute_enriched_basin_moment_weights,
    compute_enriched_basin_moment_weights_power,
    compute_full_gram_weights,
    compute_sliced_committor,
    compute_weights_multi,
    make_weighting_context,
    summarize_gram_diagnostics,
    why_masked,
)

from .core.committor import (
    CommittorFit,
    build_committor,
    committor_gradient,
    fit_committor,
)

from .core.weights import (
    compute_epsilon,
    compute_epsilon_equilibrium,
    compute_epsilon_flux1d,
    compute_epsilon_rms,
    corrected_dirichlet_inv_rd,
    full_gram_weights,
    get_all_weight_functions,
    get_default_weight_functions,
)

from .core._bmc import basin_moment_weights
from .core._bmc_enriched import (
    EnrichedBMCRepresentationError,
    enriched_basin_moment_weights,
    enriched_basin_moment_weights_power,
)

from .core.directions import (
    DirectionSamplingConfig,
    compute_lda_axis,
    directions_gcpca,
    directions_pca,
    directions_tica_ema,
    directions_tica_ema_decomposed,
    directions_uniform,
    gcpca_basis,
    pca_basis,
    sample_directions,
    sample_power_spherical_mixture,
)

from .rates import (
    BridgeD,
    PlateauWindow,
    Profile,
    berezhkovskii_szabo_rate,
    committor_rate,
    density,
    diffusion_coefficient,
    dirichlet_rate,
    find_plateau,
    kramers_rate,
    reactive_flux,
    saddle_bridge_D,
    tpt_rate,
    value_at,
)

__all__ = [
    # Committor model (callable + one-shot fit)
    "fit_committor",
    "build_committor",
    "committor_gradient",
    "CommittorFit",
    # Solver internals (results + diagnostics)
    "compute_sliced_committor",
    "SlicedCommittorResult",
    "WeightingContext",
    "make_weighting_context",
    # Weight solvers, ordered from recommended-default to specialised:
    # enriched BMC (default), PESB-EBMC (higher-order softmix basis),
    # plain BMC, full-Gram simplex, diagonal RD.
    "compute_enriched_basin_moment_weights",
    "compute_enriched_basin_moment_weights_power",
    "compute_basin_moment_weights",
    "compute_full_gram_weights",
    "corrected_dirichlet_inv_rd",
    "full_gram_weights",
    "basin_moment_weights",
    "enriched_basin_moment_weights",
    "enriched_basin_moment_weights_power",
    "EnrichedBMCRepresentationError",
    "compute_weights_multi",
    "get_all_weight_functions",
    "get_default_weight_functions",
    # Diagnostics
    "why_masked",
    "summarize_gram_diagnostics",
    # Boundary-error estimators
    "compute_epsilon_equilibrium",
    "compute_epsilon_rms",
    "compute_epsilon_flux1d",
    "compute_epsilon",
    # Directions
    "DirectionSamplingConfig",
    "sample_directions",
    "sample_power_spherical_mixture",
    "compute_lda_axis",
    "directions_uniform",
    "directions_tica_ema",
    "directions_tica_ema_decomposed",
    # Geometric basis factories (PCA, gcPCA):
    "pca_basis",
    "directions_pca",
    "gcpca_basis",
    "directions_gcpca",
    # Rate quantities + formulas
    "density",
    "diffusion_coefficient",
    "reactive_flux",
    "saddle_bridge_D",
    "BridgeD",
    "dirichlet_rate",
    "berezhkovskii_szabo_rate",
    "committor_rate",
    "tpt_rate",
    "kramers_rate",
    "Profile",
    "value_at",
    "find_plateau",
    "PlateauWindow",
]

__version__ = "0.4.0"
