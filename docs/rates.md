# Rates

## The pair {D_q, pi}

Projected onto the committor coordinate the dynamics is a 1D diffusion with
density `pi(q)` and diffusion `D_q(q)`. Its reactive flux `nu_R(q) = D_q(q)
pi(q)` is constant in `q` for the exact committor (current conservation) and
gives the rate constants `k_AB = nu_R / rho_A`, `k_BA = nu_R / rho_B`, with
`rho_B = E[q]`. Every rate in the library is built from this pair: the
density from the samples, the diffusion from one of three constructors, and a
reduction of the flux to one number. Rates come out in inverse units of the
trajectory's `dt`; `sliced_committor.rates.units` converts them.

## The density and the populations

```python
from sliced_committor import basin_populations, density

qx = q(samples)  # (N,) committor values
pi = density(qx, n_bins=25)  # a Profile on [0, 1]; pass sample_weights= for biased data
rho_A, rho_B = basin_populations(qx, in_A=in_A, in_B=in_B)  # snapped to the basins
```

Every profiled quantity is a `Profile(levels, values, counts)`, and
`value_at(profile, level)` reads it at a level, at an array of levels, or
averaged over a `(lo, hi)` range.

## Three constructors of D_q

### Measured on a trajectory

The Kramers-Moyal (mean squared displacement) estimator on the committor
itself needs a time-ordered trajectory and a lag at which the committor
diffuses. The lag is explicit on purpose: on a coordinate without a diffusive
regime, which is the committor of a slow system, there is no `D` to estimate,
and a rule that picked the lag with the largest `D` would return the
short-time bounce. Look before choosing:

```python
from sliced_committor import diffusion_profile, lag_scan

qt = q(trajectory)  # (T,) committor along the trajectory
scan = lag_scan(qt, dt=dt)  # levels are lags; a plateau across them is a diffusive regime
D_q = diffusion_profile(qt, dt=dt, lag=lag, window_ids=window_ids)  # window_ids: never pair across windows
```

For umbrella sampling the honest measurement is along the biased coordinate
`s`, which the restraint confines: Hummer's `Var(s) / tau_int` needs no lag.
`hummer_diffusion` gives one value per window; `pooled_acf_diffusion`, the
paper's estimator, averages the windows' autocorrelation functions before
integrating, which removes the noise of a single window's `tau_int`, and
selects windows by their mean coordinate, never by the committor:

```python
from sliced_committor import hummer_diffusion, pooled_acf_diffusion

D_per_window = hummer_diffusion(s_traj, window_ids, dt=dt)  # Profile at the window means
pooled = pooled_acf_diffusion(s_traj, window_ids, dt=dt, run_ids=run_ids, n_boot=50)
D_s, tau_int, ci = pooled.D, pooled.tau_int, pooled.ci
```

`run_ids` mark independent contiguous runs (replicates): an autocorrelation
is only defined within one, and splicing runs inflates `tau_int`. The
Kramers-Moyal estimator takes the same `run_ids=` and never pairs frames
across a join.

### Mapped from a collective variable: the Jacobian route

This is the paper's route. A diffusion measured along `s` and the diffusion
along the committor are one configurational diffusion tensor contracted
along two gradients, so with `D = D0 M0` they read `D_s = D0 <|grad s|^2>`
and `D_q(q) = D0 <|grad q|^2>_q`, and

    D_q(q) = D_s <|grad q|^2>_q / <|grad s|^2> .

`committor_grad_sq` supplies the iso-committor mean squared gradient (one
autodiff pass; by the co-area identity it is `Phi_{D=1}(q) / pi(q)`), and
`linear_response_grad_sq` the CV's mean squared gradient:

```python
from sliced_committor import committor_diffusion_from_cv, committor_grad_sq, linear_response_grad_sq

g_q = committor_grad_sq(q, samples, n_bins=25)  # <|grad q|^2>_q; metric= for a feature-space metric
g_s = linear_response_grad_sq(s, samples)  # <|grad s|^2> in the same feature space and metric
D_q = committor_diffusion_from_cv(g_q, D_s=D_s, cv_grad_sq=g_s)
```

Both gradients must live in the same feature space and metric: the map is
linear in one and inverse-linear in the other. `linear_response_grad_sq`
carries a caveat: it fits `s ~ a . x` and reads `a^T a`, which is exact
when `s` is linear in the features and depends on the fitting window
otherwise (on chignolin the global fit gives 0.222 and per-window fits
0.0066). When `s = f(x)` is differentiable, differentiate it instead. When
it is not, the reparametrisation route needs no gradient of `s` at all; it
assumes that `q` is a monotone function of `s`, gated by the rank
correlation:

```python
from sliced_committor import committor_diffusion_from_cv_reparam

D_q_reparam = committor_diffusion_from_cv_reparam(qx, s, D_s, n_bins=25, r2_min=0.5)
```

### An assumed configurational D0

Toy systems and Langevin models have a known mobility. The same map with
`cv_grad_sq=1` converts it:

```python
D_q_assumed = committor_diffusion_from_cv(g_q, D_s=0.05, cv_grad_sq=1.0)
```

## The reductions

```python
from sliced_committor import rate_from_profiles

rate = rate_from_profiles(pi, D_q, rho_A, rho_B, reduction="plateau", band=(0.3, 0.7))
```

| `reduction` | `nu_R` | its parameter | what it is |
|---|---|---|---|
| `"plateau"` (default) | median of `D_q pi` over `band` | `band=(0.3, 0.7)`, or `"auto"` | a robust location estimate of the constant, read away from the basins where committor error inflates the flux; the published reduction |
| `"arithmetic"` | `int_0^1 D_q pi dq` | none | the Dirichlet form, the variational upper bound on the rate; `pi`-weighted, so the basins dominate |
| `"harmonic"` | (`k = 1 / MFPT`) | `window=(lo, hi)` | the exact mean first-passage time of the projected 1D diffusion; bottleneck-dominated |
| `"local"` | `D_q(q*) pi(q*)` | `q_star=0.5` | the Berezhkovskii-Szabo single-surface reading |

Each reduction accepts only its own parameter and raises on a foreign one.
All coincide for the exact committor; every result also carries the
arithmetic and harmonic values (`k_AB_arithmetic`, `k_AB_harmonic`), the flux
profile `nu`, and `flatness`, the relative spread of the flux over the band.
Their spread is the committor-quality diagnostic. The harmonic value is not a
bound: for an approximate committor it can lie on either side of the
arithmetic one.

`band="auto"` locates the flattest band with `find_plateau` and warns when
nothing is flat to tolerance; `plateau_ok` records it. The paper's
label-free score `flux_cv` is the same flatness on the fixed band
`(0.2, 0.8)`, so that fits can be compared on one band:

```python
from sliced_committor import flux_flatness

flux_cv = flux_flatness(rate["nu"], (0.2, 0.8))
```

## In one call

`committor_rate` builds the pair and reduces it. The diffusion comes from
exactly one of the two routes:

```python
from sliced_committor import committor_rate

r_mapped = committor_rate(q, samples, D_q=D_q, in_A=in_A, in_B=in_B, n_bins=25)
r_measured = committor_rate(q, samples, trajectory=trajectory, dt=dt, lag=lag, n_bins=25)
```

## The Kramers baseline

`sliced_committor.rates.baselines.pmf_kramers_rate(pi_s, D_s)` is the
overdamped Kramers estimate on a physical coordinate: harmonic fits to the
wells and the barrier of `F(s) = -log pi(s)` and the diffusion at the
barrier. It is a labelled comparison row, not an estimator of this library:
it needs interior wells, a barrier well above `kT` and a diffusive
coordinate, and it knows nothing about the committor.
