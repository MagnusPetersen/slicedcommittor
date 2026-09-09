# Design decisions: what was tried and is not here

Version 1.0 keeps one way to do each thing. This page records what the
research program behind the paper tried and removed, with the evidence, so
that nobody re-implements it without knowing. The code of every item is at
tag `v0.6.0` of this repository.

The invariants the surviving API follows: no argument with one legal value;
no overloaded arguments; typed results; the committor callable is a pure
function of `x`; every quantity is a profile and every reduction is a
separate, named step; silent fallbacks are errors.

## Weight solvers

| removed | what it was | why |
|---|---|---|
| plain BMC | the two-moment constrained solve without the free bias `c` | EBMC (the surviving solve) has a strictly lower Dirichlet energy, by construction; the paper's Eq. (w*) is EBMC |
| PESB-EBMC | each slice basis enriched with smoothstep functions `v^n / (v^n + (1 - v)^n)` | never won the label-free selection against EBMC on the molecular systems; a wider Gram for no gain |
| full-Gram simplex | `sum w = 1` with a self-consistent boundary-error loop | its committors are compressed toward the middle; the boundary-error estimators (equilibrium, RMS, 1D flux) existed only to serve it |
| diagonal RD weights | `w_j ~ (1 - eps_j)_+ / D_j`, no constraints | the diagonal approximation the Gram solve was built to remove |
| Nitsche boundary conditions | weak enforcement of the basin values in the solve | its own docstring: the motivating premise did not survive testing; 11% on the villin control |
| post-hoc calibration (0.5.0) | affine and 1D recalibration of the evaluated committor | did not reliably improve the estimate |

## Ridge rules

| removed | why |
|---|---|
| `tikhonov="cv"`, `"cv_refit"` (the held-out cap as the ridge selector) | the weakest rule on every system tested; costs 14 percentage points where it bites and gains nothing |
| `tikhonov="auto_lambda"` (`1.8e-4 (1e5 / N_eff) M geomean(diag G)`) | the M-scaling is derived but the constant is fitted, with per-system optima spanning 8x |
| `tikhonov=("lambda", v)`, `("ridge_abs", r)`, relative floats, `"auto_m"` | spellings of one number; a float is now the absolute ridge |

The half-set spectral filter has no constant and nothing to select, and
regularises band by band, which a scalar cannot; the held-out cap survives
as a diagnostic (`heldout_cap=True`) and as the trial-space selector of the
LDA cone.

## Direction samplers

| removed | why |
|---|---|
| PCA and generalised-contrastive PCA cones | no user, no result |
| TICA-EMA (single trajectory and bias-aware) | needs dynamics the static fit does not; the Fisher cone is supervised, dynamics-free and the paper's |
| LDA by mean difference | the Fisher axis is the discriminant |
| kernel axes (diffusion maps, random-feature kernel PCA) | unstable |
| iterative direction search (IDS) | withdrawn on four independent measures; the held-out cap preferred it while every accuracy measure preferred the cone |

## Slice construction and the Gram matrix

| removed | why |
|---|---|
| adaptive per-slice bandwidth | after the villin metric fix it regresses on one peptide while helping the other |
| the fine-grid slice basis of the RECOVAR port | anti-correlated between the peptides: villin's best arm and AIB9's worst |
| two-pass feature mask | refuted |
| Laplacian resolution metric | null result |
| matrix-free Gram assembly | refuted; the assembly is not the bottleneck |
| MBAR weights inside IDS | refuted with IDS |
| `absorption_quantile` decoupled from `boundary_quantile` | never decoupled in any published run |
| `quantile_subsample`, `store_projected_samples`, `gram_dtype` | one legal value each: the internal rule, always stored, float64 |
| overlap detection in `valid_mask` | never implemented; the moment gap zeroes such slices on its own. `valid_mask` now means the 1D solve is finite |

What the RECOVAR program did transfer is in `core/_halfset.py`: the half-set
spectral filter, contiguous basin-stratified folds, the held-out cap with its
train-versus-held-out gap, and the block bootstrap. The iid delta-method
error bar was not ported: 13x too small on molecular data.

## Rates

| removed | why |
|---|---|
| `dirichlet_rate`, `tpt_rate` (the feature-space forms with a length-scale `D`) | by the co-area identity they are `committor_rate` with `D_q = D0 <|grad q|^2>_q`; one functional and three constructors replace two families |
| `saddle_bridge_D` | computed `D = D_q(q*) / <|grad q|^2>_{q*}` and handed it to `tpt_rate`, which multiplied it straight back |
| `berezhkovskii_szabo_rate` | a named alias of the harmonic and local reductions |
| `reactive_flux` | the product of two survivors; the flux profile is returned by every rate |
| the lag scan's argmax as the default lag | upward-biased, and on a coordinate without a diffusive plateau it returns the short-time bounce; `lag` is explicit and `lag_scan` shows the plateau |
| per-window Kramers-Moyal | the window-stratified estimator and Hummer cover it |
| the barrier-band median of per-window Hummer values as `D_s` | the pooled-autocorrelation estimator removes the per-window `tau_int` noise it was averaging over; the paper's figure and rate table now use one `D_s` |
| the `at=` selector (`None` / scalar / range / `"auto"`) | one overloaded argument with three type errors and one silent wrong answer; each reduction now has its own named parameter |
| `flux_reductions`, `mapped_committor_diffusion_field`, `smooth_Ds_profile`, `constancy_reconstruction`, `bootstrap_barrier_Ds` | the bridge-v2 research module; the field bridge reduces to the scalar map when `D_s` is constant, and chignolin turned out committor-quality-limited, not `D`-limited |
| Bayesian Smoluchowski `(F, D)` inference | recovers the local short-lag diffusion, which is not the memory-integrated quantity the rate needs |
| `hist`, `kernel` and `pspline` conditional means | only the local-linear regression survived, inside the reparametrisation route |
| `kramers_rate` on the committor with a fabricated ramp committor and a `nu_R` derived from the rate | `pmf_kramers_rate(pi, D)` takes profiles along a physical coordinate and reports what it computes |

## The umbrella workflow

The orchestrator half of the `workflows` package (a registry of six lab
systems matched by folder name, loaders, a sweep over featurisations and
direction modes, plotting, reports, a command-line entry point) described
one laboratory's directory layout. The estimator half survives as
`sliced_committor.umbrella`: the dataset contract, reweighting, the reading
helpers, and `fit_and_rate`.
