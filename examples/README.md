# Examples

Two self-contained walkthroughs of the public API on the rotated Wolfe-Quapp
2D potential (the paper figure 1 benchmark; Bonati et al., PNAS 2019). The
examples do not ship data: samples are drawn from the analytic Boltzmann
density so the library itself stays usage-focused. Bring your own samples
when calling the library on a real problem. Both share the potential, sampler,
and PDE baseline via `_wolfe_quapp.py`.

| Script | What it demonstrates | Runtime (CPU) |
|---|---|---|
| [01_wolfe_quapp_2d.py](01_wolfe_quapp_2d.py) | The end-to-end workflow: input contract (`(N, dim)` samples + boolean basin masks), `compute_sliced_committor` and its `SlicedCommittorResult`, `result.summary()` / `why_masked` diagnostics, four weight solvers (diagonal RD, full Gram, BMC, EBMC) with `summarize_gram_diagnostics`, `build_committor` returning a callable committor evaluated on a grid and at named points, and a cross-check against a converged 300x300 Jacobi PDE baseline. Headline settings reproduce paper figure 1 (100 000 samples, 256 directions, full-Gram weighting). | ~ 2-3 min |
| [02_rates_wolfe_quapp.py](02_rates_wolfe_quapp.py) | The rate API on the same benchmark: `fit_committor` (the one-shot callable committor), the three quantity primitives (`density`, `diffusion_coefficient` from a window-stratified Langevin swarm, `reactive_flux`), the feature-space rate formulas (`dirichlet_rate`, `tpt_rate`), the coordinate-invariant `committor_rate` (whose harmonic / local reductions are `berezhkovskii_szabo_rate` mfpt / local), and `kramers_rate`, all compared against an exact PDE reference rate. Prints a method-vs-reference table with PASS/FAIL self-checks and saves a four-panel figure (π, D, Φ profiles + a rate bar chart). | ~ 2-3 min |

Install the example extras to get matplotlib + scipy:

```bash
pip install -e .[examples]
```

Then run directly:

```bash
python 01_wolfe_quapp_2d.py
```
