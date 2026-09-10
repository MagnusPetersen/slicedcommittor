"""Choosing settings without a reference, and error bars, on the Wolfe-Quapp potential.

Two label-free numbers compare fits of the same data. The Dirichlet energy
``w^T G w`` is the variational objective and ranks fits within one trial
space. The held-out Dirichlet cap is the same objective read on folds the
solve never saw; it ranks trial spaces, and its gap against the in-sample
value is the overfitting guard. This script fits a grid of ``n_directions``
x ``n_bins`` with ``heldout_cap=True``, picks the lowest held-out cap, and
then puts a block-bootstrap error bar on the winner. The PDE RMSE is printed
as the outcome only; it never selects.

Runtime on CPU: about 1-2 minutes.

Run:
    python 03_model_selection_wolfe_quapp.py
"""

from __future__ import annotations

import itertools
import time

import jax
import jax.numpy as jnp
import numpy as np
from scipy.interpolate import RegularGridInterpolator

jax.config.update("jax_enable_x64", True)

from _wolfe_quapp import (
    BETA,
    CENTRE_A,
    CENTRE_B,
    SEED,
    SOLVER_KWARGS,
    STATE_RADIUS,
    boltzmann_samples,
    jacobi_committor_2d,
    potential,
)

import sliced_committor as sc

N_SAMPLES = 20_000
GRID = {"n_directions": (64, 128, 256), "n_bins": (100, 200)}
N_BOOT = 30


def main():
    samples = boltzmann_samples(n_samples=N_SAMPLES)
    in_A = np.linalg.norm(samples - CENTRE_A, axis=1) < STATE_RADIUS
    in_B = np.linalg.norm(samples - CENTRE_B, axis=1) < STATE_RADIUS
    X = jnp.asarray(samples)

    # the outcome, never the selector: the PDE committor at the samples of the transition region
    q_pde, X_pde, Y_pde, _, _ = jacobi_committor_2d(n_grid=150, tol=1e-7)
    interp = RegularGridInterpolator(
        (Y_pde[:, 0], X_pde[0, :]), q_pde, bounds_error=False, fill_value=np.nan
    )
    q_ref = interp(samples[:, ::-1])
    rho = np.exp(-BETA * potential(samples[:, 0], samples[:, 1]))
    score_mask = (rho > 1e-3 * rho.max()) & np.isfinite(q_ref) & ~in_A & ~in_B

    print(f"[1] {N_SAMPLES:,} samples; grid {GRID}\n")
    print(
        f"  {'M':>5s} {'bins':>5s} {'energy':>10s} {'held-out cap':>13s} {'se':>8s} {'gap':>9s} {'RMSE':>7s} {'s':>5s}"
    )
    print("  " + "-" * 68)
    fits = {}
    for M, bins in itertools.product(GRID["n_directions"], GRID["n_bins"]):
        t0 = time.time()
        q, fit = sc.fit_committor(
            X,
            in_A=in_A,
            in_B=in_B,
            n_directions=M,
            n_bins=bins,
            seed=SEED,
            heldout_cap=True,
            return_details=True,
            **SOLVER_KWARGS,
        )
        cap = fit.weights.heldout_cap
        rmse = float(np.sqrt(np.mean((np.asarray(q(X)) - q_ref)[score_mask] ** 2)))
        fits[(M, bins)] = (fit, cap["cap"], rmse)
        print(
            f"  {M:>5d} {bins:>5d} {fit.dirichlet_energy:>10.5f} {cap['cap']:>13.5f} {cap['se']:>8.5f} "
            f"{cap['gap']:>+9.5f} {rmse:>7.4f} {time.time() - t0:>5.1f}"
        )

    best = min(fits, key=lambda k: fits[k][1])
    fit, cap, rmse = fits[best]
    print(
        f"\n[2] lowest held-out cap: M = {best[0]}, n_bins = {best[1]} (RMSE {rmse:.4f}); "
        f"lowest RMSE: M = {min(fits, key=lambda k: fits[k][2])[0]}, "
        f"n_bins = {min(fits, key=lambda k: fits[k][2])[1]}"
    )
    print(
        "    (gap = held-out minus in-sample cap; a gap that grows with M at fixed N is overfitting)"
    )

    # ------------------------------------------------------------------------
    # error bars on the winner: a block bootstrap over frames, basis held fixed.
    # The samples are independent draws, so block_len = 1; on MD frames the
    # block must reach the integrated autocorrelation time.
    # ------------------------------------------------------------------------
    probe = jnp.asarray(
        [[0.0, 0.0], CENTRE_A + np.array([0.6, 0.0]), CENTRE_B - np.array([0.6, 0.0])]
    )
    boot = sc.bootstrap_weights(
        fit.result, fit.weights, n_boot=N_BOOT, block_len=1, seed=0, points=probe
    )
    q_hat = np.asarray(fit.committor(probe))
    print(f"\n[3] block bootstrap ({boot.n_ok} replicates):")
    print(f"    Dirichlet energy {fit.dirichlet_energy:.5f} +- {boot.dirichlet_energy.std():.5f}")
    for pt, qv, se in zip(np.asarray(probe), q_hat, boot.q.std(axis=0)):
        print(f"    q({pt[0]:+.2f}, {pt[1]:+.2f}) = {qv:.4f} +- {se:.4f}")
    print("    (variance only: the bootstrap cannot see the bias of an undersampled barrier)")


if __name__ == "__main__":
    main()
