# slicedcommittor

[![CI](https://github.com/MagnusPetersen/slicedcommittor/actions/workflows/test.yml/badge.svg)](https://github.com/MagnusPetersen/slicedcommittor/actions/workflows/test.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

**slicedcommittor** computes the committor `q(x)`, the probability that a
trajectory started at configuration `x` reaches state B before state A,
from equilibrium samples and two state labels, and turns it into reaction
rates. The committor is the reaction coordinate of a rare transition: its
level sets are the transition-state ensemble, and the flux through them is
the rate.

The library implements the sliced committor of

> Petersen, M., Lichtinger, S. and Covino, R.
> *Committors and Reaction Rates from Trial Functions That Violate the Boundary Conditions*
> (2026, manuscript submitted).

The committor is written as a sum of one-dimensional committors along
random directions,

    q(x) = c + sum_j w_j q_j(theta_j . x),

each `q_j` solved on the projection of the samples onto its direction and
the weights from one symmetric linear solve. No clustering, no basis
functions, no training, and no dynamics: the fit reads the equilibrium
density and the labels, and its cost is linear in the number of samples
and in the dimension. The paper validates the method on the AIB9 peptide,
villin HP-35 and chignolin from full torsion-angle representations, with no
pre-specified reaction coordinate, and reads the chignolin folding rate off
umbrella-sampling data.

## What it needs, and what it returns

* **Samples.** An `(N, dim)` float64 array of configurations from the
  equilibrium ensemble, in any feature space: Cartesian coordinates,
  torsion sines and cosines, distances. A biased ensemble works with
  per-frame weights that sum to one (`sample_weights`; MBAR weights for
  umbrella sampling).
* **States.** Two `(N,)` boolean masks, `in_A` and `in_B`, marking the
  frames inside the two metastable states. They must be disjoint and both
  populated; how you define the states is yours to decide.
* **Time order.** The default regularisation and the error bars deal the
  frames into contiguous blocks, so consecutive frames must be consecutive
  in time. Independent runs go end to end, never shuffled.
* **For a rate, dynamics.** The committor needs none. A rate needs the
  diffusion along the reaction coordinate, measured on a time-ordered
  series with its frame spacing `dt`, or taken from a known diffusion
  coefficient.

Back comes a callable `q(x)`, a pure JAX function of `x` that `jax.grad`
differentiates, with the diagnostics that judge the fit without a
reference committor.

## The assumption you accept

For a reversible diffusion the committor minimises the Dirichlet form
`<grad q^T D grad q>` over the equilibrium density, under `q = 0` on A and
`q = 1` on B. The density is in the samples and the boundary values are in
the labels, which is why static samples suffice. What the samples cannot
tell is the shape of the diffusion tensor `D` in your feature space. The
library takes it to be the identity, which holds for Cartesian coordinates
in explicit solvent, and accepts a metric when it does not: for torsion
features that is the pull-back metric `sincos_pullback_metric`, the
paper's setting on all three proteins.

## Install

```bash
pip install sliced-committor              # jax, numpy, scipy
pip install sliced-committor[umbrella]    # + mdtraj and pymbar, for umbrella-sampling data
```

Pick your own JAX build (`jax[cpu]`, `jax[cuda12]`, ...); the library does
not pin one. Float64 is required; enable it before anything else.

## Fit, check, use

```python
import jax

jax.config.update("jax_enable_x64", True)

from sliced_committor import committor_gradient, fit_committor

# samples: (N, dim) equilibrium configurations; in_A, in_B: (N,) bool state labels
q, fit = fit_committor(samples, in_A=in_A, in_B=in_B, n_directions=256, return_details=True)

q_vals = q(points)  # (P,) committor values in [0, 1]
q_snapped = q(samples, in_A=in_A, in_B=in_B)  # exactly 0 on A and 1 on B
grad_q = committor_gradient(q, points)  # (P, dim), by autodiff

print(fit.result.summary())  # the slice basis: directions, bins, boundary errors
print(fit.weights.dirichlet_energy, fit.weights.cond)  # the objective and the representation gate
```

Three numbers say how the fit went, and none needs a reference:

* `fit.result.boundary_errors`, one per direction: how far that slice's 1D
  committor is from 0 on A and 1 on B. A large value means the direction
  separates the states badly; the weights suppress such slices, and a fit
  in which every slice is bad points at features that do not resolve the
  states, or at states that overlap.
* `fit.weights.cond`, the representation gate. It collapses, and the solve
  raises, when no combination of the slices separates the states.
* `fit.weights.dirichlet_energy`, the variational objective `w^T G w`.
  Lower is closer to the true committor, so it ranks fits of the same
  data. With `heldout_cap=True` the same objective is read on folds the
  solve did not see, which also ranks different trial spaces.

A rate adds a fourth: the flatness of the reactive flux along `q`, which
is constant for the exact committor.

## Settings

The defaults are the paper's settings; the table gives the per-system
values the paper uses.

| setting | what it controls | default | the paper |
|---|---|---|---|
| `n_directions` | the number of slices `M`. The cost is linear in `M`, and the default regularisation keeps a large `M` from over-fitting | 256 | 512 (AIB9), 2048 (villin), 256 (chignolin, Wolfe-Quapp) |
| `n_bins`, `binning_method` | the 1D histogram of each slice: its resolution, and equal-count (`"quantile"`) or `"equal_width"` bins | 200, `"quantile"` | 2000 (AIB9), 500 (villin), 100 equal-width (chignolin), 200 equal-width (Wolfe-Quapp) |
| `direction_sampling` | uniform on the sphere, or a cone around the Fisher discriminant axis between the states | uniform | the cone, `DirectionSamplingConfig(mode="lda", mu=0.8 / 0.7 / 0.5, alpha=0.2)` on AIB9 / villin / chignolin |
| `feature_metric` | the shape of the diffusion tensor in feature space, in the weight solve | identity | `sincos_pullback_metric` on the torsion features of the proteins |
| `boundary_quantile` | where the absorbing boundaries of the 1D solves sit inside the projected states; below 1 it removes the halo that many nuisance dimensions inflate | 1.0 | 0.99 (AIB9), 0.98 (villin) |
| `sample_weights` | per-frame weights of a biased ensemble | uniform | MBAR on chignolin |
| `tikhonov` | the regularisation of the Gram solve | `"halfset_eigen"` | the same everywhere: it has no constant to tune |

## Rates

The rate is a reduction of the committor-coordinate flux
`nu_R(q) = D_q(q) pi(q)`, constant in `q` for the exact committor. The
density `pi(q)` comes from the samples. The diffusion `D_q` along the
committor is the one thing the static ensemble cannot supply, and where
it comes from depends on your data:

| you have | `D_q` comes from | functions |
|---|---|---|
| a long unbiased trajectory in which the committor diffuses | the Kramers-Moyal estimate on `q` itself, at a lag read off a lag scan | `lag_scan`, `diffusion_profile`, or `committor_rate(trajectory=...)` |
| umbrella sampling along a collective variable `s` | the diffusion along `s` by the pooled-autocorrelation Hummer estimator, mapped into committor space through the Jacobian: the paper's route | `pooled_acf_diffusion`, `committor_grad_sq`, `linear_response_grad_sq`, `committor_diffusion_from_cv`; together in `umbrella.fit_and_rate` |
| a model with a known mobility `D0` | the same map with `cv_grad_sq=1` | `committor_diffusion_from_cv(g_q, D_s=D0, cv_grad_sq=1.0)` |

The paper's route, given the diffusion `D_s` measured along `s` and the
mean squared gradient `g_s` of `s` in the feature space:

```python
from sliced_committor import committor_diffusion_from_cv, committor_grad_sq, committor_rate

g_q = committor_grad_sq(q, samples)  # <|grad q|^2> on the iso-committor surfaces
D_q = committor_diffusion_from_cv(g_q, D_s=D_s, cv_grad_sq=g_s)  # D_q(q) = D_s g_q(q) / g_s
rate = committor_rate(q, samples, D_q=D_q, in_A=in_A, in_B=in_B)  # the plateau of D_q pi
k_AB, k_BA, flatness = rate["k_AB"], rate["k_BA"], rate["flatness"]
```

`flatness` is the relative spread of the flux over the transition region,
the committor-quality score that needs no reference.
[docs/rates.md](docs/rates.md) walks through the three routes, the four
reductions and how to read their spread.

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
diffusion along the biased coordinate from the windows' time series, maps
it into committor space and reads off the rates, with the committor-free
Kramers baseline beside them. [docs/umbrella.md](docs/umbrella.md) gives
the dataset contract, the readers for PLUMED and mdtraj files, and the same
computation step by step.

## Cost

The fit holds the projection of every sample onto every direction, an
`(M, N)` float64 array, and a few working arrays of that size;
`direction_batch_size` bounds the transient. Runtime is linear in `N`,
`dim` and `M`: the 2D example (`N = 100,000`, `M = 256`) fits in under a
minute on a CPU, the paper's largest fit (villin: `N = 200,000`,
`dim = 350`, `M = 2048`) in a few minutes on a workstation. The same code
runs on a GPU.

## Documentation

In reading order:

1. [quickstart](docs/quickstart.md): the inputs and why each is needed,
   the one call, what comes back and how to read it.
2. [theory](docs/theory.md): the model, the sliced ansatz, the weight
   solve and the diagnostics it yields.
3. [settings](docs/settings.md): every setting, what it does, how to
   choose it without a reference, and error bars.
4. [rates](docs/rates.md): from the committor to a rate, by three routes.
5. [umbrella sampling](docs/umbrella.md): the dataset contract and the
   rate bundle.
6. [reproducibility](docs/reproducibility.md) and
   [design decisions](docs/design_decisions.md): what is pinned, and what
   was tried and is not here.

`examples/` holds three self-contained walkthroughs on the paper's 2D
benchmark: the committor step by step, the rate against an exact
reference, and model selection with error bars.

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
