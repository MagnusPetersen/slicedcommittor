"""Label-free settings sweep on the rotated Wolfe-Quapp 2D potential.

Companion to ``01_wolfe_quapp_2d.py``. Shows the variational-objective workflow:

    1. ``fit_committor(..., return_details=True)`` reports the fit's Dirichlet
       energy ``𝓓[q̂] = <D|∇q̂|²>`` -- a label-free, ground-truth-free quality
       score (lower is closer to the true committor, by the variational
       principle).
    2. ``sweep_committor(...)`` fits a Cartesian grid of settings and keeps the
       best by that objective; ``res.summary()`` tabulates every combination.

No PDE baseline and no ground truth are used here: model selection is driven
purely by the variational principle. Runtime on CPU: ~1-2 minutes.

Run:
    python 03_sweep_wolfe_quapp.py
"""

from __future__ import annotations

import jax
import numpy as np

# Float64 is required for the EBMC / PESB / full-Gram constrained solves.
jax.config.update("jax_enable_x64", True)

# Shared Wolfe-Quapp scaffolding (potential, paper settings, Boltzmann sampler)
# lives in _wolfe_quapp, reused across the examples without duplication.
from _wolfe_quapp import (
    CENTRE_A,
    CENTRE_B,
    N_BINS,
    SEED,
    STATE_RADIUS,
    boltzmann_samples,
)

import sliced_committor as sc


def main():
    # ------------------------------------------------------------------------
    # Step 1: samples + basin masks. Subsample to keep the multi-fit sweep
    # brisk for an example (the full 100k run is in example 01).
    # ------------------------------------------------------------------------
    print("[1] Generating samples...")
    samples = boltzmann_samples()
    rng = np.random.default_rng(0)
    idx = rng.choice(samples.shape[0], size=min(20000, samples.shape[0]), replace=False)
    samples = samples[idx]
    in_A = np.linalg.norm(samples - CENTRE_A, axis=1) < STATE_RADIUS
    in_B = np.linalg.norm(samples - CENTRE_B, axis=1) < STATE_RADIUS
    print(f"    samples.shape = {samples.shape}, |A| = {in_A.sum()}, |B| = {in_B.sum()}")

    # ------------------------------------------------------------------------
    # Step 2: a single fit, and its variational objective. return_details=True
    # gives back the CommittorFit, whose .dirichlet_energy is the label-free
    # quality score 𝓓[q̂] (lower = closer to the true committor).
    # ------------------------------------------------------------------------
    print("\n[2] Single fit + its Dirichlet energy:")
    _, fit = sc.fit_committor(
        samples, in_A=in_A, in_B=in_B, weights="ebmc",
        n_directions=256, n_bins=N_BINS, seed=SEED, return_details=True,
    )
    print(f"    dirichlet_energy = {fit.dirichlet_energy:.4g}")

    # ------------------------------------------------------------------------
    # Step 3: sweep a grid of settings; keep the best by the Dirichlet energy.
    # The grid is one Gram-family of solvers so the energies are comparable
    # (a sweep that also varied onto the diagonal solver would warn).
    # ------------------------------------------------------------------------
    print("\n[3] sweep_committor over a Gram-family grid...")
    res = sc.sweep_committor(
        samples, in_A=in_A, in_B=in_B,
        grid={"n_directions": [128, 256], "weights": ["ebmc", "pesb", "full_gram"]},
        n_bins=N_BINS, seed=SEED,
    )
    print(res.summary())
    best = res.configs[res.best_idx]
    print(f"\n    best config: {best}  (D[q] = {res.dirichlet_energies[res.best_idx]:.4g})")

    # ------------------------------------------------------------------------
    # Step 4: the winner is a ready-to-call committor q(x).
    # ------------------------------------------------------------------------
    print("\n[4] best_committor at three probe points:")
    q_best = res.best_committor
    probe = np.array(
        [[0.0, 0.0], CENTRE_A + np.array([0.5, 0.0]), CENTRE_B + np.array([-0.5, 0.0])]
    )
    for pt, qv in zip(probe, np.asarray(q_best(probe))):
        print(f"      q({pt[0]:+.2f}, {pt[1]:+.2f}) = {qv:.4f}")


if __name__ == "__main__":
    main()
