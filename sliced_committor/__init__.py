"""Sliced committor: sample-based committor approximation via 1D projections.

Given samples and basin labels (`in_A`, `in_B`), compute the committor q(x)
via random 1D projections + 1D reaction-diffusion solves + weighted recombination.

Quickstart::

    import jax.numpy as jnp
    from sliced_committor import (
        compute_sliced_committor, evaluate_committor,
        compute_weights_multi, get_default_weight_functions,
    )

    # samples : (N, dim) array
    # in_A, in_B : (N,) bool arrays, basin labels
    result = compute_sliced_committor(samples, in_A=in_A, in_B=in_B,
                                       n_directions=256)
    weights = compute_weights_multi(result, get_default_weight_functions())
    q = evaluate_committor(result, points, weights['corrected_dirichlet_inv_rd'])

Notes:
    * No `beta` argument. The committor is β-invariant given fixed samples;
      β goes in, β comes out, leaves no trace. Internally the library uses
      β = 1 so that `result.free_energies` stores `-log ρ` directly.
    * Basin labels are bool arrays, not callable state functions. Define
      states however you want (clustering, RMSD thresholds, geometric
      expressions) and pass the resulting masks.
"""

from .solver import (
    SlicedCommittorResult,
    WeightingContext,
    compute_sliced_committor,
    evaluate_committor,
    make_weighting_context,
    compute_weights_multi,
    compute_full_gram_weights,
    compute_basin_moment_weights,
    compute_enriched_basin_moment_weights,
    compute_enriched_basin_moment_weights_power,
    why_masked,
    summarize_gram_diagnostics,
)

from .weights import (
    corrected_dirichlet_inv_rd,
    full_gram_weights,
    compute_epsilon_equilibrium,
    compute_epsilon_rms,
    compute_epsilon_flux1d,
    compute_epsilon,
    get_all_weight_functions,
    get_default_weight_functions,
)

from ._bmc import basin_moment_weights
from ._bmc_enriched import (
    enriched_basin_moment_weights,
    enriched_basin_moment_weights_power,
    EnrichedBMCRepresentationError,
)

from .directions import (
    DirectionSamplingConfig,
    sample_directions,
    sample_power_spherical_mixture,
    compute_lda_axis,
    directions_uniform,
    directions_tica_ema,
    directions_tica_ema_decomposed,
    pca_basis,
    directions_pca,
    gcpca_basis,
    directions_gcpca,
)

__all__ = [
    # Solver
    "compute_sliced_committor",
    "evaluate_committor",
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
]

__version__ = "0.4.0"
