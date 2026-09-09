"""RECOVAR transfers for the sliced committor.

CryoEM (RECOVAR) reconstruction ideas ported to the sliced-committor method:
halfset-eigen Gram regularization, held-out dual diagnostics with contiguous
block splits, adaptive per-slice profile bandwidth, iterative direction
sampling (IDS), a graph-Laplacian resolution metric, and UQ on the rate.

Provenance: JAX port of the numpy prototype vendored at ``recovar/_reference``
(see ``_reference/REPORT.md`` and ``docs/recovar_transfers.md`` in the repo).
This package calls ``sliced_committor`` for shared core machinery and does not
modify it. All validated 2D results use beta=1.0 (see ``systems``).
"""

import jax

# The Gram/solve chain is float64 throughout (same contract as EBMC in
# sliced_committor, which raises without x64). Enabled here at import,
# following the src/noneq precedent.
jax.config.update("jax_enable_x64", True)

from .basis import (  # noqa: E402
    SliceBasis,
    build_basis,
    isotropic_directions,
    lda_directions,
    rd_profile,
)
from .assemble import (  # noqa: E402
    assemble,
    assemble_slice_values,
    clip01,
    dual_objective,
    evaluate,
    fit,
    solve_dual,
    transition_rmse,
)
from .splits import block_split, iid_split  # noqa: E402
from .regularize import (  # noqa: E402
    halfset_eigen_regularize,
    halfset_grams_from_arrays,
    halfset_grams_recovar,
    solve_with_G,
)
from .dual import cv_dual, nested_M_curve  # noqa: E402
from .bandwidth import make_adaptive_bw_fn, make_halfset_bw_fn  # noqa: E402
from .ids import (  # noqa: E402
    directions_exact_floor,
    feature_importance,
    ids_fit,
    ids_path_exact2,
    masked_directions,
    path_directions_exact,
    path_labels,
    path_scatter_w,
    select_rank_perm_exact,
    select_rank_permutation,
)
from .adapters import (  # noqa: E402
    evaluate_handle,
    halfset_committor_pair,
    ids_protocol_metadata,
    make_lib_fit,
    make_recovar_fit,
)
from .resolution import laplacian_shell_correlation  # noqa: E402
from .uq import uq_bootstrap, uq_delta  # noqa: E402
from .diagnostics import GramOperator, gram_diagonal, nystrom_gram  # noqa: E402

__all__ = [
    "SliceBasis", "build_basis", "isotropic_directions", "lda_directions",
    "rd_profile",
    "assemble", "assemble_slice_values", "solve_dual", "dual_objective",
    "evaluate", "fit", "transition_rmse", "clip01",
    "iid_split", "block_split",
    "halfset_eigen_regularize", "halfset_grams_recovar",
    "halfset_grams_from_arrays", "solve_with_G",
    "cv_dual", "nested_M_curve",
    "make_halfset_bw_fn", "make_adaptive_bw_fn",
    "path_labels", "path_scatter_w", "select_rank_permutation",
    "path_directions_exact", "select_rank_perm_exact",
    "directions_exact_floor", "ids_fit", "ids_path_exact2",
    "feature_importance", "masked_directions",
    "make_lib_fit", "make_recovar_fit", "evaluate_handle",
    "halfset_committor_pair", "ids_protocol_metadata",
    "laplacian_shell_correlation",
    "uq_bootstrap", "uq_delta",
    "GramOperator", "gram_diagonal", "nystrom_gram",
]
