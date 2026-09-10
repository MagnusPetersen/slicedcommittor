# Umbrella sampling

Umbrella-sampling data reach the committor as a static ensemble with
reweighting weights and reach the diffusion as per-window time series. The
`sliced_committor.umbrella` subpackage holds both views in one dataset,
supplies the weights, and fits the committor and reads off the rates in
one call. It orchestrates nothing else: reading files and defining basins
is the caller's job, with helpers for the pieces. Install the optional
dependencies with `pip install sliced-committor[umbrella]` (pymbar for
MBAR, mdtraj for trajectories); WHAM and the COLVAR parser need neither.

## The dataset

`USDataset` keeps frames in per-window, time-ordered layout: windows
concatenated, frames within a window in simulation order. Its fields are
the features `(N, d)` the committor is fitted on, the biased collective
variables `cvs (N, n_cv)`, the per-frame `window_ids`, the restraints
`window_centers (K, n_cv)` and `window_kappa (K, n_cv)` with `beta` (so the
reduced bias is `0.5 beta kappa (cv - center)^2`), the frame spacing `dt`,
the basin masks `in_A` and `in_B`, `cv_periodic` (a `(min, max)` per
periodic CV, else None), and a free-form `meta` dict.

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
`mbar_weights` and `wham_weights` take the arrays directly.

## The rate bundle

```python
from sliced_committor.umbrella import fit_and_rate

out = fit_and_rate(dataset, dataset.features, w, n_directions=64, n_bins=60, n_diff_bins=25)
k_AB = out["rates"]["cvmap"]["plateau"]["k_AB"]
D_s, tau_int = out["D_s"]["value"], out["D_s"]["tau_int"]
```

`fit_and_rate` fits the committor on the reweighted ensemble (binning
defaults to `equal_width`: quantile bins under-resolve the sparsely
sampled barrier and inflate the rate), then builds the pair `{D_q, pi}` for
every diffusion constructor asked for and reduces it every way asked for:

* `diffusion=("cvmap",)`, the default and the paper's route: `D_s` along the
  progress coordinate by `pooled_acf_diffusion` over all windows (pass
  `D_s=` for a known value, or measure it yourself with a window selection),
  mapped into committor space through the Jacobian with the CV's gradient
  from `linear_response_grad_sq`. `bridge_metric=` applies one metric to
  both gradients.
* `"hummer_q"`: Hummer's `Var / tau_int` of the committor per window, a
  diagnostic (the restraint confines `s`, not `q`).
* `"km_q"`: the Kramers-Moyal estimate on the committor at `lag`, which
  needs a diffusive regime the committor of a slow system lacks.

`run_ids=` marks independent replicates; no estimator reads a displacement
or an autocorrelation across a join. Like every fit, `fit_and_rate` needs
`jax_enable_x64` to be on in your process; it does not switch it on for you.
* `reductions=("plateau", "harmonic", "arithmetic")` by default, `"local"`
  on request.

The bundle carries the fit diagnostics (`committor`), the per-sample
committor (`q_samples`), the profiles on the `n_diff_bins` grid
(`levels`, `counts`, `pi`, `grad_sq`, and per constructor `D_q` and
`flux`), the rates per constructor and reduction, the label-free `flux_cv`
per constructor, the committor-free Kramers baseline along the progress
coordinate (`kramers`, also available as `kramers_baseline(dataset,
weights)`), and `errors`. With `strict=True` (the default) the first
failing estimator raises; with `strict=False` an estimator's `ValueError`
or `RuntimeError` is recorded under `errors` and the others carry on.

Rates are in inverse units of `dataset.dt`:

```python
from sliced_committor.rates.units import estimated_to_per_s

k_per_s = estimated_to_per_s(k_AB, "ps")
```

## The progress coordinate

`progress_coordinate(dataset)` is the biased CV for a 1D bias and, for a
2D bias, the projection onto the axis between the basin means, oriented
from A to B. The diffusion along it is what the Jacobian map starts from,
and the Kramers baseline reads its free-energy profile.
