# Umbrella sampling

## What umbrella data give you

Umbrella-sampling data reach the committor as a static ensemble with
reweighting weights, and reach the diffusion as per-window time series of
the biased coordinate. The `sliced_committor.umbrella` subpackage holds
both views in one dataset, supplies the weights, and fits the committor and
reads off the rates in one call. It orchestrates nothing else: reading
files and defining the states is the caller's job, with helpers for the
pieces. Install the optional dependencies with
`pip install sliced-committor[umbrella]` (pymbar for MBAR, mdtraj for
trajectories); WHAM and the COLVAR parser need neither.

What you need per frame: the features the committor is fitted on, the
biased collective variable, the window it belongs to, and whether it lies
in state A or B; per window, the restraint centre and force constant; and
for the whole set, `beta`, the frame spacing `dt` and the periodicity of the
collective variables.

## The dataset

`USDataset` keeps frames in per-window, time-ordered layout: windows
concatenated, frames within a window in simulation order (the diffusion
estimators read the time series of each window, so the order matters). Its
fields are the features `(N, d)` the committor is fitted on, the biased
collective variables `cvs (N, n_cv)`, the per-frame `window_ids`, the
restraints `window_centers (K, n_cv)` and `window_kappa (K, n_cv)` with
`beta` (so the reduced bias is `0.5 beta kappa (cv - center)^2`), the frame
spacing `dt`, the state masks `in_A` and `in_B`, `cv_periodic` (a
`(min, max)` per periodic CV, else None), and a free-form `meta` dict.

```python
from sliced_committor.umbrella import USDataset

print(dataset.n_frames, dataset.n_windows, dataset.n_cv)
subset, kept = dataset.subsample(20_000)  # even temporal thinning per window
```

The reading helpers: `read_colvar` parses PLUMED COLVAR files (with the
periodicity from their `#! SET` lines), `parse_restraint` reads the
`RESTRAINT` line of a PLUMED input, `load_trajectory` wraps mdtraj, and
`align_colvar_traj` pairs COLVAR rows with trajectory frames written at a
different stride.

## Reweighting

```python
from sliced_committor.umbrella import reweight

rw = reweight(dataset)  # MBAR; WHAM above 128 windows or when MBAR fails
w = rw.sample_weights  # (N,), summing to one
print(rw.method, rw.n_eff)  # the engine that ran, and the Kish effective sample size
```

`reweight(dataset, method="wham", n_bins=80)` forces the binned engine;
`mbar_weights` and `wham_weights` take the arrays directly. `n_eff`, the
effective sample size of the reweighted ensemble, is the number to keep in
mind when judging the fit: a set with `n_eff` far below `N` is a small
sample in disguise.

## The rate bundle

```python
from sliced_committor.umbrella import fit_and_rate

out = fit_and_rate(dataset, dataset.features, w, n_directions=64, n_bins=60, n_diff_bins=25)
k_AB = out["rates"]["cvmap"]["plateau"]["k_AB"]
D_s, tau_int = out["D_s"]["value"], out["D_s"]["tau_int"]
```

`fit_and_rate` does, in order:

1. fits the committor on the reweighted ensemble with `fit_committor`
   (binning defaults to `equal_width`: quantile bins under-resolve the
   sparsely sampled barrier and inflate the rate); the solver settings,
   `direction_sampling`, `feature_metric` and `boundary_quantile` among
   them, ride along as keyword arguments;
2. evaluates the committor and its gradient on every frame, and builds the
   density `pi(q)` with the weights;
3. measures the diffusion `D_s` along the progress coordinate with
   `pooled_acf_diffusion` over all windows (pass `D_s=` for a known value,
   or measure it yourself with a window selection, below);
4. builds the pair `{D_q, pi}` for every constructor in `diffusion=` and
   reduces it every way in `reductions=`:
   * `"cvmap"`, the default and the paper's route: `D_s` mapped into
     committor space through the Jacobian, with the CV's gradient from
     `linear_response_grad_sq`; `bridge_metric=` applies one metric to
     both gradients;
   * `"hummer_q"`: Hummer's `Var / tau_int` of the committor per window, a
     diagnostic (the restraint confines `s`, not `q`);
   * `"km_q"`: the Kramers-Moyal estimate on the committor at `lag`, which
     needs a diffusive regime the committor of a slow system lacks;
   * `reductions=("plateau", "harmonic", "arithmetic")` by default,
     `"local"` on request;
5. computes the committor-free Kramers baseline along the progress
   coordinate.

`run_ids=` marks independent replicates; no estimator reads a displacement
or an autocorrelation across a join. Like every fit, `fit_and_rate` needs
`jax_enable_x64` on in the calling process.

The bundle carries the fit diagnostics (`committor`), the per-sample
committor (`q_samples`), the profiles on the `n_diff_bins` grid
(`levels`, `counts`, `pi`, `grad_sq`, and per constructor `D_q` and
`flux`), the rates per constructor and reduction, the label-free `flux_cv`
per constructor, the Kramers baseline (`kramers`, also available as
`kramers_baseline(dataset, weights)`), and `errors`. With `strict=True`
(the default) the first failing estimator raises; with `strict=False` an
estimator's `ValueError` or `RuntimeError` is recorded under `errors` and
the others carry on.

## The same computation, step by step

The bundle is composed of public pieces, and taking them one at a time is
how you depart from it: select the windows that set `D_s`, pass a metric,
or use a diffusion measured elsewhere.

```python
from sliced_committor import (
    committor_diffusion_from_cv,
    committor_grad_sq,
    committor_rate,
    fit_committor,
    linear_response_grad_sq,
    pooled_acf_diffusion,
)
from sliced_committor.umbrella import progress_coordinate

s = progress_coordinate(dataset)  # the biased CV, oriented from A to B
pooled = pooled_acf_diffusion(s, dataset.window_ids, dt=dataset.dt)  # window_band=(lo, hi) selects windows
q_us = fit_committor(
    dataset.features, in_A=dataset.in_A, in_B=dataset.in_B, sample_weights=w,
    n_directions=64, n_bins=60, binning_method="equal_width",
)
g_q = committor_grad_sq(q_us, dataset.features, sample_weights=w, n_bins=25)
g_s = linear_response_grad_sq(s, dataset.features, w)
D_q = committor_diffusion_from_cv(g_q, D_s=pooled.D, cv_grad_sq=g_s)
rate = committor_rate(
    q_us, dataset.features, D_q=D_q, sample_weights=w, in_A=dataset.in_A, in_B=dataset.in_B, n_bins=25
)
```

`progress_coordinate` is the biased CV for a 1D bias and, for a 2D bias,
the projection onto the axis between the state means, oriented from A to
B. `pooled_acf_diffusion(..., window_band=(lo, hi))` restricts the
diffusion estimate to the windows whose mean coordinate lies in the band,
the paper's committor-free barrier selection; `run_ids=` keeps replicates
apart. With torsion features pass the pull-back metric to the fit as
`feature_metric=` and to both gradients as `metric=`
([settings](settings.md)).

## Units and the baseline

Rates are in inverse units of `dataset.dt`:

```python
from sliced_committor.rates.units import estimated_to_per_s

k_per_s = estimated_to_per_s(k_AB, "ps")
```

The Kramers baseline reads the free-energy profile along the progress
coordinate and the per-window Hummer diffusion at the barrier. It is a
comparison row: it needs interior wells and a barrier well above `kT`, and
it knows nothing about the committor.
