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
| `tikhonov=("lambda", v)`, `("ridge_abs", r)`, relative floats, `"auto_m"` | spellings of one number; the absolute ridge is a float in the units of `G` |

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
| overlap detection in `valid_mask` | never implemented; the moment gap zeroes such slices on its own. `valid_mask` marks the slices whose 1D solve is finite |

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
| the barrier-band median of per-window Hummer values as `D_s` | the pooled-autocorrelation estimator removes the per-window `tau_int` noise it was averaging over; the paper's figure and rate table use one `D_s` |
| the `at=` selector (`None` / scalar / range / `"auto"`) | one overloaded argument with three type errors and one silent wrong answer; each reduction has its own named parameter |
| `flux_reductions`, `mapped_committor_diffusion_field`, `smooth_Ds_profile`, `constancy_reconstruction`, `bootstrap_barrier_Ds` | the bridge-v2 research module; the field bridge reduces to the scalar map when `D_s` is constant, and chignolin turned out committor-quality-limited, not `D`-limited |
| Bayesian Smoluchowski `(F, D)` inference | recovers the local short-lag diffusion, which is not the memory-integrated quantity the rate needs |
| `hist`, `kernel` and `pspline` conditional means | only the local-linear regression survived, inside the reparametrisation route |
| `kramers_rate` on the committor with a fabricated ramp committor and a `nu_R` derived from the rate | `pmf_kramers_rate(pi, D)` takes profiles along a physical coordinate and reports what it computes |

## Tried in the 2026-10 audit of the core

Tried on a copy of the library and not adopted, under one rule: a change
must be better on each of the paper's four systems (the 2D Wolfe-Quapp
potential, AIB9, villin, chignolin) or a draw. The numbers are paired
relative changes of the transition RMSE against each system's reference
(2D 10 seeds, AIB9 5, villin 2 to 3), and of the held-out cap for
chignolin, which has none. Two measures recur in the verdicts: the
held-out cap preferred several changes that the references reject, and on
chignolin every slice overlaps its states, so the cap and the read-off's
`E / nu - 1` both rest on the basin moments and disagree with each other.

| tried | result |
|---|---|
| the half-set filter on the unit-diagonal Gram `D^-1/2 G D^-1/2` | regularises less: the in-sample energy falls on every 2D, AIB9 and villin seed (chignolin's fit does not move). AIB9 -6.7% (5 of 5); undersampled 2D -15% to -35% (N = 2000 and 5000, 5 of 5); but 2D +8% and +17% at N = 20000 (4 of 5 worse), a coin flip at the paper's N = 1e5 (5 of 10), and villin +0.6% on the binned reference at every binning (a draw on the raw labels). The held-out cap prefers it everywhere, the cases it loses included. On a 500-sample fixture whose slice energies span 2.6e5 the plain filter drops nine of twelve bands, so this is the open lead |
| a soft slice profile by default (`rd_stiffness`) | kept as an option; no value is better everywhere: `0` (closed form) AIB9 -8.4%, a 52-D embedded Wolfe-Quapp -32%, villin +45%; `1e4` AIB9 -6.8%, villin +1.5%; `1e5` AIB9 +3.3% |
| the Fisher axis from the weighted class statistics on biased data | identical on unbiased data. On chignolin, at each axis's cap-selected cone, the cap falls 35% while `E / nu - 1` rises 31%, and the folding rate moves 41% |
| cross-validated eigenvalues in place of the half-set filter (k-fold or half-split) | 2D +87%, eightfold on one seed; AIB9 -6.8% |
| the harmonic mean of the bin densities as the slice conductance | 2D +2.3%; AIB9 unchanged |
| absorption over a fixed fraction of each state's shadow | a constant to choose (2D -4% at one half, +6% at one quarter) that failed on a 2D umbrella benchmark |
| zero slope outside each slice's sampled range | an exact no-op |
| bin edges half at quantiles, half equally spaced | 2D +27%; chignolin cap +45% |
| bands of fixed width (256/12 directions) instead of twelve bands | AIB9 -6.0% but cap +50%; 2D unchanged |
| the band correlation averaged over 16 balanced splits of the folds | no change beyond 0.4% on any system |
| the read-off's recalibration `q_tilde = f(q)` as the committor | 2D +7% to +8.5% (100 and 400 bins) |
| averaging the committor over five direction seeds | AIB9 -1.5% at five times the cost |

## The umbrella workflow

The orchestrator half of the `workflows` package (a registry of six lab
systems matched by folder name, loaders, a sweep over featurisations and
direction modes, plotting, reports, a command-line entry point) described
one laboratory's directory layout. The estimator half survives as
`sliced_committor.umbrella`: the dataset contract, reweighting, the reading
helpers, and `fit_and_rate`.
