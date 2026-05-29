# Examples

One self-contained walkthrough of the public API on the rotated Wolfe-Quapp
2D potential (the paper figure 1 benchmark; Bonati et al., PNAS 2019). The
example does not ship data: samples are drawn from the analytic Boltzmann
density so the library itself stays usage-focused. Bring your own samples
when calling the library on a real problem.

| Script | What it demonstrates | Runtime (CPU) |
|---|---|---|
| [01_wolfe_quapp_2d.py](01_wolfe_quapp_2d.py) | The end-to-end workflow: input contract (`(N, dim)` samples + boolean basin masks), `compute_sliced_committor` and its `SlicedCommittorResult`, `result.summary()` / `why_masked` diagnostics, four weight solvers (diagonal RD, full Gram, BMC, EBMC) with `summarize_gram_diagnostics`, `evaluate_committor` on a grid and at named points, and a cross-check against a converged 300x300 Jacobi PDE baseline. Headline settings reproduce paper figure 1 (100 000 samples, 256 directions, full-Gram weighting). | ~ 2-3 min |

Install the example extras to get matplotlib + scipy:

```bash
pip install -e .[examples]
```

Then run directly:

```bash
python 01_wolfe_quapp_2d.py
```
