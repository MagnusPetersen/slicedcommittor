# Recipes

Short snippets for the most common variations from the default pipeline.

## Switching the boundary-error estimator

```python
from sliced_committor import compute_full_gram_weights
from sliced_committor.core.weights import compute_epsilon_rms

result = compute_sliced_committor(samples, in_A=in_A, in_B=in_B, n_directions=256)
gram = compute_full_gram_weights(result, samples, epsilon_fn=compute_epsilon_rms)
# Equivalent string form:
gram = compute_full_gram_weights(result, samples, epsilon_fn="rms")
```

## Using a custom direction sampler (PCA / gcPCA)

```python
from sliced_committor import DirectionSamplingConfig

result = compute_sliced_committor(
    samples, in_A=in_A, in_B=in_B,
    n_directions=256,
    direction_sampling=DirectionSamplingConfig(mode="pca", n_bias_axes=4),
)
```

For label-aware bias use `mode="gcpca"`. To supply your own directions:

```python
import jax.numpy as jnp
my_axes = jnp.asarray(...)  # (M, dim) unit vectors
result = compute_sliced_committor(
    samples, in_A=in_A, in_B=in_B,
    n_directions=my_axes.shape[0],
    directions=my_axes,
)
```

## High-dimensional halo mitigation

```python
result = compute_sliced_committor(
    samples, in_A=in_A, in_B=in_B,
    n_directions=1024,
    boundary_quantile=0.97,   # paired by default with absorption_quantile
)
```

## Memory-light mode

```python
result = compute_sliced_committor(
    samples, in_A=in_A, in_B=in_B,
    n_directions=4096,
    store_projected_samples=False,   # diagonal RD only; no full-Gram / BMC
)
```

## Inspecting masked slices

```python
from sliced_committor import why_masked
for j, ok in enumerate(result.valid_mask):
    if not ok:
        print(j, why_masked(result, j))
```

## Computing rates

Every rate function takes the callable committor from `fit_committor` (or
`build_committor`) plus your data; nothing needs the underlying result or
weights. `traj` is a time-ordered `(T, dim)` trajectory (only the diffusion
estimate needs it). See
[`examples/02_rates_wolfe_quapp.py`](../examples/02_rates_wolfe_quapp.py) for a
full worked comparison against an exact reference rate.

```python
import sliced_committor as sc

q = sc.fit_committor(samples, in_A=in_A, in_B=in_B, n_directions=256)

# Quantity profiles (a Profile when at=None, else the value at that level/range).
pi = sc.density(q, samples)                     # equilibrium density π(q̄)
phi = sc.reactive_flux(q, samples, D=0.05)      # TPT reactive flux Φ(q̄)
D = sc.diffusion_coefficient(q, traj, dt=dt)    # position-dependent D from a trajectory

# Rate constants -> {nu_R, rho_A, rho_B, k_AB, k_BA, ...}.
sc.dirichlet_rate(q, samples, D=0.05, in_A=in_A, in_B=in_B)        # variational ⟨D|∇q̄|²⟩_π
sc.tpt_rate(q, samples, D=0.05, in_A=in_A, in_B=in_B)             # flux through a surface
sc.berezhkovskii_szabo_rate(q, samples, traj, dt=dt, mode="mfpt")  # committor-coordinate MFPT
sc.kramers_rate(q, samples, traj, dt=dt, coordinate=cv, traj_coordinate=cv_t)  # harmonic estimate
```

`D` may be a scalar, a callable `level -> D`, or a diffusion `Profile`. Pass
`at=q_star` (a point) or `at=(lo, hi)` (a range) to a quantity or to
`dirichlet_rate` to query a single iso-committor level instead of the full
`[0, 1]` profile. `coordinate=` (diffusion / density) profiles along an
arbitrary collective variable instead of the committor: along a Cartesian CV the
Kramers-Moyal estimate recovers the configurational diffusion, whereas along the
committor it returns `D(q̄) = D·|∇q̄|²` (the quantity Berezhkovskii-Szabo uses).

### Lag-robust diffusion (Hummer / Var·τ_int⁻¹)

`diffusion_coefficient` defaults to a drift-corrected Kramers-Moyal estimate,
which is *lag-sensitive* on slow coordinates (no clean diffusive plateau). For
**umbrella-sampling / restrained** data, `method="hummer"` instead uses the
Kramers/Hummer per-window `D = Var(level)/(τ_int·dt)` with a Geyer integrated
autocorrelation time — lag-free, no lag scan. It assumes the coordinate is
locally confined within each window, so it **requires `window_ids`**:

```python
D = sc.diffusion_coefficient(q, traj, dt=dt, window_ids=wid, method="hummer")
# and through the rates:
sc.berezhkovskii_szabo_rate(q, samples, traj, dt=dt, mode="mfpt",
                            window_ids=wid, diffusion_method="hummer")
```

### Coordinate-general rates: `committor_rate`

`dirichlet_rate` / `tpt_rate` are **feature-space** forms — they multiply a
length-scale `D` by the autodiff `|∇q̄|²`, so they need `D` in the committor's
own coordinate units (well-defined for Cartesian features, abstract for
dimensionless torsions, and prone to overcount with an approximate committor).
Use them only when a *trusted* configurational `D` is known a priori (e.g. a
2D toy with prescribed `D₀`).

`committor_rate` is the **coordinate-invariant** alternative: every rate is a
functional of the same pair `{D_q(q), π(q)}` — the committor-coordinate
diffusion (from the trajectory) and density (from the ensemble), with local flux
`ν_R(q) = D_q(q)·π(q)`. Nothing depends on the feature space (Cartesian,
torsions, TICA, … all give the same `D_q`/`π`). Pick the reduction:

```python
sc.committor_rate(q, samples, traj, dt=dt, reduction="harmonic")    # 1/∫ dq/(D_q π)  — exact 1-D rate, bottleneck-dominated (default)
sc.committor_rate(q, samples, traj, dt=dt, reduction="arithmetic")  # ∫ D_q π dq      — the Dirichlet form, π-weighted
sc.committor_rate(q, samples, traj, dt=dt, reduction="plateau")     # median_{[0.3,0.7]} D_q π — transition-region flux
sc.committor_rate(q, samples, traj, dt=dt, reduction="local")       # D_q(q*)·π(q*)
```

For the exact committor all reductions agree (`D_q·π` is constant); their spread
on an approximate committor is a quality diagnostic. `harmonic`/`local` reproduce
`berezhkovskii_szabo_rate` mfpt/local; pass `diffusion_method="hummer"` to use the
lag-robust D throughout.

### Automatic plateau range: `find_plateau` / `at="auto"`

The feature-space flux `Φ(c)` (and the `D_q·π` flux) is constant in `c` for the
true committor and flat on a *saddle plateau* near `c≈0.5` for an approximate
one, inflating in the basins. Instead of hard-coding a window (`[0.2, 0.8]`),
let the flux flatness locate the trustworthy range:

```python
pw = sc.find_plateau(sc.reactive_flux(q, samples, D=1.0))   # PlateauWindow(lo, hi, flatness, value, n_valid, ok)
sc.tpt_rate(q, samples, D=D, at="auto")                     # auto plateau; result carries plateau / plateau_flatness / plateau_ok
sc.committor_rate(q, samples, traj, dt=dt, reduction="plateau", at="auto")
```

`find_plateau` returns the **widest** window whose relative flatness `σ/|μ|` is
≤ `tol` (basins excluded by `margin`); `ok=False` flags that no window met `tol`
(treat the rate as indicative). `≲ 0.2` flatness ⇒ well-conserved.

### Calibrated feature-space `D`: `saddle_bridge_D` (the geometric / q-stratified rate)

When you want the feature-space `Φ(c)` form (e.g. the q-stratified flux profile
for diagnostics) but have no *trusted* configurational `D`, calibrate it from the
trajectory's own committor-coordinate diffusion. The iso-`q` identity fixes the
scalar so the geometric estimator reproduces the committor-coordinate value:

```python
bridge = sc.saddle_bridge_D(q, samples, traj, dt=dt, q_star=0.5,   # bridge.D = D_q(q*) / ⟨|∇q̄|²⟩_{q*}
                            window_ids=wid, diffusion_method="hummer")
sc.tpt_rate(q, samples, D=bridge, at="auto", sample_weights=w)     # calibrated q-stratified plateau rate
```

`saddle_bridge_D` returns a `BridgeD` — a **callable** constant-`D` (drops into
any `D=` argument) carrying the calibrated scalar `D` plus its intermediates
(`D_q_ref`, `g_ref`). This is the principled version of the q-stratified flux
estimator (the plateau of `D·⟨|∇q̄|²⟩·π`): a raw local-atomic `D` mismatches the
committor's coordinate units and over/under-counts, whereas `bridge.D` is matched
to `D_q`.
`mode="volume_average"` uses the π-averaged `⟨D_q⟩_π / ⟨|∇q̄|²⟩_π` instead of a
single `q*`. Prefer `committor_rate` (no gradient, no calibration) as the clean
default; reach for the bridge to reproduce / cross-check a geometric pipeline.
