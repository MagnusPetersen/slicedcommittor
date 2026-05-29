"""Validation against the closed-form 1D Ornstein-Uhlenbeck committor.

For the Smoluchowski dynamics ``dx = -V'(x) dt + sqrt(2) dW`` with
quadratic potential ``V(x) = x^2 / 2`` and absorbing boundaries at
``x = a`` (A) and ``x = b`` (B > a), the committor is

    q(x) = (erf(x / sqrt(2)) - erf(a / sqrt(2)))
         / (erf(b / sqrt(2)) - erf(a / sqrt(2))).

Samples are drawn directly from the equilibrium density of the dynamics
restricted to ``[a, b]`` (a truncated Gaussian), so no MD simulation is
needed.
"""

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.special import erf

jax.config.update("jax_enable_x64", True)

import sliced_committor as sc


def _ou_committor(x: np.ndarray, a: float, b: float) -> np.ndarray:
    """Closed-form OU committor on ``[a, b]`` with absorbing boundaries."""
    return (erf(x / math.sqrt(2.0)) - erf(a / math.sqrt(2.0))) / (
        erf(b / math.sqrt(2.0)) - erf(a / math.sqrt(2.0))
    )


def _ou_samples(n: int, a: float, b: float, seed: int = 0) -> np.ndarray:
    """Draw samples from N(0,1) restricted to ``[a, b]`` via rejection."""
    rng = np.random.default_rng(seed)
    out = []
    while len(out) < n:
        draw = rng.standard_normal(size=n)
        out.extend(draw[(draw > a) & (draw < b)].tolist())
    return np.asarray(out[:n], dtype=np.float64)


@pytest.fixture(scope="module")
def ou_data():
    a, b = -2.0, 2.0
    x = _ou_samples(n=4000, a=a, b=b, seed=0)
    # Lift to a 1D feature with the basin labels living in [a, a+0.2] / [b-0.2, b].
    samples = jnp.asarray(x[:, None])
    in_A = jnp.asarray(x < a + 0.2)
    in_B = jnp.asarray(x > b - 0.2)
    return samples, in_A, in_B, a, b


def _interior_mask(x_flat, a, b, pad=0.3):
    return (x_flat > a + pad) & (x_flat < b - pad)


def _run_with_weights(samples, in_A, in_B, weight_fn_name):
    """Compute sliced committor and the named weight vector."""
    result = sc.compute_sliced_committor(
        samples, in_A=in_A, in_B=in_B, n_directions=8, n_bins=80, seed=42
    )

    if weight_fn_name == "diag":
        weights = sc.compute_weights_multi(result, [sc.corrected_dirichlet_inv_rd])[
            "corrected_dirichlet_inv_rd"
        ]
        return result, weights
    if weight_fn_name == "full_gram":
        d = sc.compute_full_gram_weights(result, samples)
        return result, d["w"]
    if weight_fn_name == "ebmc":
        d = sc.compute_enriched_basin_moment_weights(result, samples)
        return result, d
    raise ValueError(weight_fn_name)


# Plain BMC is not parametrised here: in d=1 the only directions are +/-1
# and the basin-moment vectors a, b collapse onto a single line, triggering
# the rank-deficiency gate. The same algebra is tested in
# ``test_weights_bmc.py`` against >=2-D data; we keep this file focused on
# the closed-form validation.
@pytest.mark.parametrize("weight_fn_name", ["diag", "full_gram", "ebmc"])
def test_recovers_ou_closed_form(ou_data, weight_fn_name):
    samples, in_A, in_B, a, b = ou_data
    result, weights = _run_with_weights(samples, in_A, in_B, weight_fn_name)

    # Evaluate on a grid in the interior.
    x_eval = np.linspace(a + 0.3, b - 0.3, 120)
    points = jnp.asarray(x_eval[:, None])
    q_true = _ou_committor(x_eval, a, b)

    in_A_pts = jnp.asarray(x_eval < a + 0.2)
    in_B_pts = jnp.asarray(x_eval > b - 0.2)
    q_pred = sc.evaluate_committor(result, points, weights, in_A=in_A_pts, in_B=in_B_pts)
    q_pred = np.asarray(q_pred)

    rmse = float(np.sqrt(np.mean((q_pred - q_true) ** 2)))
    # The 1D OU problem is the easiest possible setting; all four solvers
    # should converge to within a few percent on the interior.
    assert rmse < 0.05, f"{weight_fn_name}: RMSE={rmse:.3f}"
