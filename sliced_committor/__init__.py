"""Sliced committor: sample-based committor approximation via 1D projections.

Given samples and basin labels (``in_A``, ``in_B``), the committor ``q(x)`` (the
probability of reaching B before A from ``x``) is built from many 1D
reaction-diffusion committors along random directions, recombined with weights
that minimise the Dirichlet form under the basin-moment constraints. The
result is a callable JAX function, so ``jax.grad`` gives its gradient and the
rate functionals build on it directly.

Quickstart::

    import jax
    jax.config.update("jax_enable_x64", True)   # the weight solve needs float64

    from sliced_committor import fit_committor
    q = fit_committor(samples, in_A=in_A, in_B=in_B, n_directions=256)
    vals = q(points)

Rates::

    from sliced_committor import committor_rate
    rate = committor_rate(q, samples, trajectory=traj, dt=dt, lag=lag)

No ``beta`` argument: the committor is beta-invariant given the samples.
Basin labels are bool arrays; define the states however you like.
"""

from .core._ebmc import Bootstrap, RepresentationError, Weights, bootstrap_weights, solve_weights
from .core.committor import (
    CommittorFit,
    build_committor,
    committor_gradient,
    fit_committor,
    rescale_transition,
)
from .core.directions import (
    DirectionSamplingConfig,
    compute_lda_axis,
    directions_uniform,
    sample_power_spherical_mixture,
)
from .core.metric import AngleSign, sincos_pullback_metric
from .core.solver import SlicedCommittorResult, compute_sliced_committor
# RATES_IMPORT_PLACEHOLDER (Phase 2 restores this)

__all__ = [
    # the committor
    "fit_committor",
    "build_committor",
    "committor_gradient",
    "rescale_transition",
    "CommittorFit",
    "compute_sliced_committor",
    "SlicedCommittorResult",
    # the weights
    "solve_weights",
    "Weights",
    "bootstrap_weights",
    "Bootstrap",
    "RepresentationError",
    # directions
    "DirectionSamplingConfig",
    "directions_uniform",
    "compute_lda_axis",
    "sample_power_spherical_mixture",
    # the feature-space metric
    "sincos_pullback_metric",
    "AngleSign",
    # rates: the {D_q, pi} pair and its reductions
    "density",
    "committor_grad_sq",
    "basin_populations",
    "diffusion_profile",
    "lag_scan",
    "hummer_diffusion",
    "pooled_acf_diffusion",
    "committor_diffusion_from_cv",
    "committor_diffusion_from_cv_reparam",
    "linear_response_grad_sq",
    "committor_rate",
    "rate_from_profiles",
    "Profile",
    "value_at",
    "find_plateau",
    "PlateauWindow",
]

__version__ = "1.0.0"
