# slicedcommittor

[![CI](https://github.com/MagnusPetersen/slicedcommittor/actions/workflows/test.yml/badge.svg)](https://github.com/MagnusPetersen/slicedcommittor/actions/workflows/test.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

**slicedcommittor** computes the committor `q(x)`, the probability that a
trajectory started at `x` reaches state B before state A, from equilibrium
samples and two basin labels, and turns it into reaction rates. The
committor is the reaction coordinate of a rare transition: its level sets
are the transition-state ensemble, and the flux through them is the rate.

The library implements the sliced committor of

> Petersen, M., Lichtinger, S. and Covino, R.
> *Committors and Reaction Rates from Trial Functions That Violate the Boundary Conditions*
> (2026, manuscript submitted).

The committor is written as a sum of one-dimensional committors along random
directions,

    q(x) = c + sum_j w_j q_j(theta_j . x),

each `q_j` from a 1D reaction-diffusion solve on the projected samples and
the weights from one symmetric linear solve. No clustering, no basis, no
training; the cost is linear in the number of samples and in the dimension.
The paper validates the method on the AIB9 peptide, villin HP-35 and
chignolin from full torsion-angle representations, with no pre-specified
reaction coordinate, and reads the chignolin folding rate off
umbrella-sampling data.

## Install

```bash
pip install sliced-committor              # jax, numpy, scipy
pip install sliced-committor[umbrella]    # + mdtraj and pymbar, for umbrella-sampling data
```

Pick your own JAX build (`jax[cpu]`, `jax[cuda12]`, ...); the library does
not pin one. Float64 is required; enable it before anything else.

## The committor

```python
import jax

jax.config.update("jax_enable_x64", True)

from sliced_committor import fit_committor

# samples: (N, dim) equilibrium configurations; in_A, in_B: (N,) bool basin labels
q = fit_committor(samples, in_A=in_A, in_B=in_B, n_directions=256)
q_vals = q(points)  # a plain function of x; jax.grad works on it
```

`fit_committor` builds the slices, solves the weights and returns the
callable committor. With `return_details=True` it also returns the slice basis
and the solved `Weights`, whose `dirichlet_energy` is the variational
objective of the fit: lower is closer to the true committor, and no reference
is needed to compare fits of the same data.

The defaults are the paper's settings. What the paper changed per system:

| setting | default | the paper |
|---|---|---|
| `n_directions` | 256 | 512 (AIB9), 2048 (villin), 256 (chignolin, Wolfe-Quapp) |
| `n_bins` | 200 | 2000 (AIB9), 500 (villin), 100 (chignolin) |
| `binning_method` | `"quantile"` | `"equal_width"` for the 2D benchmark and the umbrella data |
| `tikhonov` | `"halfset_eigen"` | the same everywhere: the half-set spectral filter has no constant to tune |
| `direction_sampling` | uniform on the sphere | the Fisher-LDA cone, `DirectionSamplingConfig(mode="lda", mu=..., alpha=0.2)`, on the proteins |
| `feature_metric` | the identity | the sin/cos pull-back metric `sincos_pullback_metric` on torsion features |
| `boundary_quantile` | 1.0 | 0.99 (AIB9), 0.98 (villin) |

## Rates

The rate is a reduction of the committor-coordinate flux
`nu_R(q) = D_q(q) pi(q)`, which is constant in `q` for the exact committor.
The density comes from the samples; the diffusion along the committor comes
from one of three constructors: measured on a trajectory, mapped from a
collective variable through the Jacobian (the paper's route, below), or an
assumed configurational `D0`.

```python
from sliced_committor import committor_diffusion_from_cv, committor_grad_sq, committor_rate

g_q = committor_grad_sq(q, samples)  # <|grad q|^2>_q, the co-area profile
D_q = committor_diffusion_from_cv(g_q, D_s=D_s, cv_grad_sq=g_s)  # D_s measured along a CV
rate = committor_rate(q, samples, D_q=D_q, in_A=in_A, in_B=in_B)  # the plateau of D_q pi
k_AB, k_BA, flatness = rate["k_AB"], rate["k_BA"], rate["flatness"]
```

`docs/rates.md` walks through the constructors, the reductions (plateau,
arithmetic, harmonic, local) and what their spread says about the committor.

## Umbrella sampling

```python skip
from sliced_committor.umbrella import USDataset, fit_and_rate, reweight

dataset = USDataset(features=..., cvs=..., window_ids=..., window_centers=...,
                    window_kappa=..., beta=..., dt=..., in_A=..., in_B=...,
                    cv_periodic=(None,), meta={})
w = reweight(dataset).sample_weights  # MBAR, with a WHAM fallback
out = fit_and_rate(dataset, dataset.features, w)
k_AB = out["rates"]["cvmap"]["plateau"]["k_AB"]
```

`fit_and_rate` fits the committor on the reweighted ensemble, measures the
diffusion along the biased coordinate by the pooled-autocorrelation Hummer
estimator, maps it into committor space and reads off the rates, with the
committor-free Kramers baseline beside them. `docs/umbrella.md` has the
dataset contract and the reading helpers for PLUMED and mdtraj files.

## Diagnostics

* `fit.dirichlet_energy`: the variational objective of a fit.
* `fit_committor(..., heldout_cap=True)`: the out-of-sample Dirichlet cap and
  its train-versus-held-out `gap`, the overfitting guard; the paper selects
  the LDA cone's `(mu, alpha)` with it.
* `bootstrap_weights(result, weights, block_len=...)`: block-bootstrap error
  bars on the weights and the energy (variance only; blocks never cross
  independent runs).
* `rate["flatness"]` and `flux_flatness(rate["nu"], (0.2, 0.8))`: how
  constant the flux is; the label-free committor-quality score.

## What is not here

`docs/design_decisions.md` records what was tried and removed: the other
weight solvers, ridge rules, direction samplers and rate estimators, each
with the evidence. The code lives at tag `v0.6.0`.

## Citation

If you use `slicedcommittor` in published work, please cite the paper:

> Petersen, M., Lichtinger, S. and Covino, R.
> *Committors and Reaction Rates from Trial Functions That Violate the Boundary Conditions*
> (2026, manuscript submitted).

A machine-readable [`CITATION.cff`](CITATION.cff) sits at the repository
root; GitHub's "Cite this repository" button exports it as BibTeX, APA,
EndNote or RIS.

```bibtex
@article{petersen2026sliced,
  title   = {Committors and Reaction Rates from Trial Functions That Violate the Boundary Conditions},
  author  = {Petersen, Magnus and Lichtinger, Simon and Covino, Roberto},
  year    = {2026},
  note    = {Manuscript submitted; journal and DOI to be filled in on acceptance.},
}
```
