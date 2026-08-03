# Umbrella-sampling kinetics (`sliced-committor-us`)

The `workflows` subpackage turns a folder of umbrella-sampling (US) molecular-dynamics
data into reaction rates: it discovers the per-window files, reweights to the unbiased
ensemble, fits a sliced committor in several feature spaces, and reads the rate off
several coordinate-invariant estimators. It ships as a console command,
`sliced-committor-us`, and a programmatic driver, `run_pipeline`.

The optional dependencies live under the `workflows` extra and are imported lazily, so
`import sliced_committor` stays light:

```bash
pip install -e .[workflows]   # mdtraj, pymbar, matplotlib, scipy, pyyaml
```

## Quickstart

Point it at a US campaign folder. Everything is auto-discovered; no copying or config
editing is required for a registered system:

```bash
sliced-committor-us US_data/US_chignolin_gmx --mode exhaustive
# from inside the folder this also works:
cd US_data/US_chignolin_gmx && sliced-committor-us ./
```

Add `-v` to see every discovery decision, and `--reference` to draw a literature rate:

```bash
sliced-committor-us US_data/US_chignolin_gmx -v --reference expt=4.5e5:1/s
```

Results land in `./sc_us_out/<system>/` (override with `--out`).

## What the pipeline does

```
folder ──▶ discover (windows, COLVAR, trajectory, topology, restraint)
       ──▶ reweight (MBAR, WHAM fallback)              -> unbiased weights
       ──▶ for each (featurization x direction mode):
              fit sliced committor q(x)                -> auto-tuned EBMC/PESB
              estimate rate from {D_q(q), pi(q)}       -> several estimators
       ──▶ write report (JSON/CSV + plots + markdown)
```

The same flow runs through `run_pipeline` programmatically:

```python
from sliced_committor.workflows import run_pipeline

result = run_pipeline(
    "US_data/US_chignolin_gmx",   # a US folder path, or a toy system name
    mode="exhaustive",            # "fast" (default) or "exhaustive"
    references={"expt": {"k": 4.5e5, "units": "1/s"}},  # optional
    out_dir="runs/chignolin",
)
```

## Command-line reference

| flag | default | meaning |
|---|---|---|
| `target` | (required) | path to a US folder, or a toy name (`double_well`, ...) |
| `--mode {fast,exhaustive}` | `fast` | sweep breadth (see [the sweep](#the-sweep-fast-vs-exhaustive)) |
| `--system NAME` | auto | force a registry entry instead of folder-name matching |
| `--reweight {auto,mbar,wham}` | `auto` | reweighting engine (`auto` picks WHAM past 128 windows or if MBAR fails) |
| `--n-directions N` | `256` | projection directions per committor fit |
| `--out DIR` | `./sc_us_out/<system>` | output directory |
| `--sidecar FILE` | none | YAML overrides for an unregistered folder ([schema](#the-sidecar-file)) |
| `--reference NAME=VALUE[:UNITS]` | none | reference rate(s) to plot; repeatable; `UNITS` default `1/s` |
| `--traj-stride N` | `1` | read every N-th trajectory frame |
| `--max-frames N` | none | cap total frames (stratified per window) after loading |
| `--seed S` | `0` | RNG seed |
| `--no-plot` | off | skip the PNG plots (JSON/CSV/report still written) |
| `-v`, `--verbose` | off | log every discovery and reweighting decision |

## Auto-discovery

Pointing at a folder triggers a best-effort scan (run with `-v` to watch it):

- **Windows.** Subdirectories holding a COLVAR file, found under `windows/` or the root
  (patterns like `window_NN`, `W_*`, `w*`).
- **COLVAR.** The per-window PLUMED COLVAR; its `#! FIELDS` header names the columns and
  the bias CV column(s) are taken from the registry (or `cv_columns` in a sidecar).
- **Trajectory.** `prod.{xtc,trr,dcd,h5}` preferred, else the first matching trajectory in
  the window directory.
- **Topology.** A `.gro/.tpr/.pdb/.prmtop/.psf` near the windows, chosen so its atom count
  matches the trajectory (a protein-only `proc.pdb` is skipped if the trajectory is the
  full solvated system, etc.).
- **Restraint.** `KAPPA` and the center `AT` are parsed from `plumed.dat`; failing that
  the center is decoded from the window directory name (e.g. `W_C1m006p50_C2p008p00` ->
  `[-0.65, 0.80]`) and `kappa` is taken from a sidecar.
- **Time step.** `dt` is the spacing of the frames actually kept (after `--traj-stride` and
  COLVAR/trajectory reconciliation), not the raw COLVAR cadence.

All three bundled GROMACS datasets (chignolin, alanine dipeptide, c-Src activation) are
fully discoverable from a bare folder path.

## The system registry

What discovery matches against is a **registry of `SystemConfig` entries baked into the
package** at `sliced_committor/workflows/config.py`. It is pure metadata (the package never
ships or generates data): for each known system it stores the basin definitions, which
COLVAR columns are the bias CV(s), the temperature (-> beta), the available featurizations,
literature reference rates, and the time unit. A `SystemConfig` has the fields:

| field | meaning |
|---|---|
| `name` | registry key |
| `temperature_K` | simulation temperature (or `None` for reduced-unit toys) -> beta |
| `region_A`, `region_B` | basin definitions (`Region`, see below); the committor boundaries `q=0` / `q=1` |
| `colvar_cv_columns` | which COLVAR data columns hold the bias CV(s) |
| `featurizations` | featurization names to sweep (`cv_only`, `dihedrals`, ... or `identity` for toys) |
| `reference_rates` | `{label: {"k": value, "units": str, "source": str}}` |
| `time_unit` | human label for `dt` (`"ps"`, `"reduced"`, ...) |

Registered systems: `double_well`, `double_well_high`, `wolfe_quapp`, `wolfe_quapp_stiff`,
`alanine_dipeptide`, `chignolin`, `csrc_activation`. A folder is matched to one by
lower-casing its name and looking it up against the registry names and a table of aliases
(e.g. `US_chignolin_gmx` -> `chignolin`, `US_cSrc_activation_gmx` -> `csrc_activation`),
including a substring fallback. Use `--system NAME` to force a specific entry.

There is **no per-folder config file to create for a registered system**. The only config
you ever author is the sidecar, and only for a folder the registry does not know.

## The sidecar file

If discovery cannot match a registry entry (an unregistered folder), the run stops with a
message naming exactly what to provide. Supply it as a YAML file via `--sidecar`. The
sidecar provides the things discovery cannot infer (the basins and which CV columns to
use) and can override any auto-discovered piece. Recognized keys:

| key | type | meaning |
|---|---|---|
| `region_A`, `region_B` | dict | basin definitions (the committor boundaries); see below |
| `cv_columns` | list[int] | which COLVAR columns are the bias CV(s) |
| `kappa` | float or list | restraint force constant(s), if not in `plumed.dat` |
| `beta` | float | inverse temperature `1/kT` in the data's energy units |
| `topology` | path | explicit topology file (overrides the discovered one) |

A `region` is a box or a ball, evaluated on either the CV array or the feature array:

- **box** (axis-aligned): `{kind: box, on: cvs|features, lo: [...], hi: [...], dims: [...]}`
  where `lo`/`hi` are per-dimension bounds and `dims` selects the array columns to test.
- **ball** (Euclidean): `{kind: ball, on: cvs|features, center: [...], radius: <float>}`.

A complete example for an unregistered 1D folder whose COLVAR is `time cv bias` (CV in
column 0), folded at `cv > 0.85`, unfolded at `cv < 0.30`, restrained with kappa 5000 at
300 K:

```yaml
# my_system.yaml
cv_columns: [0]
beta: 0.4009          # = 1 / (0.00831446 kJ/mol/K * 300 K)
kappa: 5000.0         # only needed if plumed.dat has no RESTRAINT line
region_A:             # state A, committor q = 0  (here: folded)
  kind: box
  on: cvs
  lo: [0.85]
  hi: [.inf]
  dims: [0]
region_B:             # state B, committor q = 1  (here: unfolded)
  kind: box
  on: cvs
  lo: [-.inf]
  hi: [0.30]
  dims: [0]
# topology: /abs/path/to/conf.gro   # only if discovery cannot find one
```

```bash
sliced-committor-us /path/to/my_us_folder --sidecar my_system.yaml -v
```

A 2D ball-basin example (e.g. an activation landscape with CVs in columns 0 and 1):

```yaml
cv_columns: [0, 1]
beta: 0.4009
region_A: {kind: ball, on: cvs, center: [-0.95, 0.32], radius: 0.35}
region_B: {kind: ball, on: cvs, center: [ 1.11, 1.08], radius: 0.35}
```

## Reweighting

Umbrella weights are removed with MBAR (pymbar). Because MBAR runs out of memory on very
large window counts, `--reweight auto` (the default) switches to a binned WHAM solver when
there are more than 128 windows or MBAR raises. Force either with `--reweight mbar` /
`--reweight wham`. The chosen engine and the effective sample size are reported.

## Featurizations

The committor is fit in one or more feature spaces (the sweep tries each):

- `cv_only` -- the umbrella CV(s) themselves (1D or 2D).
- `dihedrals` -- backbone/side-chain dihedrals as sin/cos.
- `aligned_cartesian` -- Kabsch-aligned Cartesian coordinates.
- `distance_matrix` -- pairwise heavy-atom/CA distances.
- `contact_map` -- soft native-contact map.
- `identity` -- the raw coordinates (analytic toy systems only).

## Direction sampling

Each committor is a weighted sum of 1D committors along projection directions. The sweep
varies how those directions are chosen:

- `uniform` -- isotropic on the sphere (the robust baseline).
- `lda` -- a single supervised discriminant axis from the A/B labels. **The fast-mode
  informed default**: it needs only the basin labels (no dynamics), so it is cheaper than
  the TICA modes, which require per-window trajectories. On a 1D feature (`cv_only`)
  direction sampling is a no-op, so `lda` there is identical to `uniform`.
- `pca` / `gcpca` -- geometric / contrastive bias axes (label-free / label-aware).
- `tica_ema` / `tica_ema_decomposed` -- EMA-integrated slow modes; the decomposed variant
  is bias-aware (built for US/MBAR data).

### The sweep: fast vs exhaustive

| `--mode` | featurizations | direction modes |
|---|---|---|
| `fast` | `cv_only`, `dihedrals` (proteins); `identity` (toys) | `uniform`, `lda` |
| `exhaustive` | `cv_only` + all protein featurizations | `uniform`, `lda`, `pca`, `gcpca`, `tica_ema`, `tica_ema_decomposed` |

For each `(featurization, direction)` cell the committor settings (EBMC vs PESB) are
auto-tuned by the label-free Dirichlet-energy objective and the best fit is kept.

## Rate estimators

Every estimator is a functional of the pair `{D_q(q), pi(q)}` -- the committor-coordinate
diffusion and density -- so all of them are **coordinate-invariant** (independent of the
arbitrary scaling of the features). For the exact committor they agree; their spread on an
approximate committor is a quality diagnostic.

| key | label | reduction | diffusion |
|---|---|---|---|
| `Szabo_mfpt` | Berezhkovskii-Szabo MFPT | harmonic (1D Smoluchowski MFPT) | measured `D_q` |
| `TPT_q` | TPT flux plateau | flux plateau | measured `D_q` |
| `BS_local` | BS local flux at `q*=0.5` | local (no q-integration) | measured `D_q` |
| `BS_mfpt_cvmap` | Berezhkovskii-Szabo MFPT | harmonic | CV-mapped `D_q` |
| `TPT_cvmap` | TPT flux plateau | flux plateau | CV-mapped `D_q` |
| `baseline:pmf_kramers` | Kramers (committor-free) | -- | PMF along the CV + CV diffusion |

The **CV-mapped** diffusion maps the cleanly-measurable umbrella-CV diffusion `D_s` onto the
committor coordinate, `D_q(q) = D_s * <|grad q|^2>_q / <|grad s|^2>`, instead of measuring
`D_q` directly on the (barrier-unstable) committor.

```{note}
An earlier feature-space estimator, `TPT_cv` (a scalar CV-space `D` times the
feature-space `|grad q|^2`), was **removed** because it is *not* coordinate-invariant: it
scales as `1/alpha^2` under a feature rescaling `f -> alpha*f`. `TPT_cvmap` is its
coordinate-invariant replacement -- the configurational scale `D0 = D_s/<|grad s|^2>`
transforms as `alpha^2` and cancels the gradient.
```

The committor-coordinate diffusion itself is estimated with `diffusion_mode="bins_hummer"`
(the default): a per-window Hummer `Var/tau_int` estimate interpolated onto the rate grid.
This is confinement-unbiased, the right choice for restrained US data. `"bins"` (per-bin
Kramers-Moyal) and `"per_window"` are also available through `run_pipeline`.

## Reference rates

Known systems carry their literature rate(s) in the registry and plot them automatically.
For an unregistered folder, or to override the registry, pass `--reference`:

```bash
sliced-committor-us <folder> --reference expt=0.0105:1/us --reference msm=4.5e5
```

`UNITS` defaults to `1/s` and is ignored for reduced-unit toys. With no `--reference`, a
known system shows its registry reference and an unregistered folder simply plots none.

## Outputs

Written to `--out` (default `./sc_us_out/<system>/`):

| file | contents |
|---|---|
| `rates_by_method.json` / `.csv` | every `feat/dir :: estimator` rate (`k_AB`, `k_BA`, `k_AB_per_s`) |
| `profiles.json` | per-window/per-bin `D_q`, `pi`, and flux profiles |
| `diagnostics.json` | reweighting engine, committor diagnostics, sweep axes |
| `rate_comparison.png` | all estimators vs the reference band (the headline plot) |
| `cv_q_scatter_grid.png` | committor quality: `(CV, q)` scatter per cell |
| `profiles.png` | `D_q(q)`, `pi(q)`, and the flux plateaus per window |
| `report.md` | a human-readable summary of the run |
| `result.pkl` | the full result dict for re-plotting |
