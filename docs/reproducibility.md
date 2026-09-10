# Reproducibility

## The determinism contract

| held fixed | same result |
|---|---|
| `seed`, one JAX build, one machine | bit-exact on the golden fixtures; a large `(M, M)` solve can differ between processes by about `1e-13` relative from the BLAS reduction order, with `G`, `a` and `b` identical |
| `seed`, another JAX build or another machine, CPU | the slice basis to `1e-13`; the committor to about `1e-5` with the half-set filter (`1e-3` with the scalar ridge on a small fixture with duplicated frames); individual weights to about `1e-2` relative; energies and held-out caps to `1e-4` relative; the band signal-to-noise ratios and the per-fold caps by a few percent |
| `seed`, GPU versus CPU | about `1e-9` to `1e-7` relative in float64 on the slice basis, from non-associative reductions, amplified into the weights as in the row above |

Two mechanisms carry rounding into the weights, and neither is a bug. A
sample whose projection coincides with a grid point of its slice takes the
slope of one adjacent bin or the other, decided by the last bit of the
projection; quantile bins are built from the sample values, so duplicated
frames (a Monte Carlo rejection, a repeated snapshot) put bin centres
exactly on samples, and a change of XLA version flips a handful of them,
which moves a few Gram entries by order one. And the half-set filter reads
a band correlation off a nearly degenerate spectrum, so `1e-13` in the Gram
matrix becomes `1e-3` in individual weights while the committor itself
moves far less. XLA compiles for the host CPU and the BLAS kernels are
chosen by it, so another machine rounds the last bit differently even with
the same versions, and the same mechanisms apply. The numbers in the table
are measured between JAX 0.5.3, 0.6.2 and 0.10.2 on the golden fixtures.

Two facts about the numerics are load-bearing and pinned by the golden
tests under `sliced_committor/tests/golden/`:

* uniform sample weights are exactly `ones(N) / N`. A renormalisation by
  one ulp was amplified to `4e-4` in the half-set weights, through the
  saturation of the per-band signal-to-noise ratio on a Gram matrix with
  condition number `1e12`;
* the `(M, M)` solve is scipy's Cholesky factorisation with
  `w = w_dual / R`. The JAX Cholesky differs from it by `1e-4` at that
  condition number, so the solve stays on scipy to keep the published
  numbers.

The half-set filter needs the frames in time order and deals them into
contiguous, basin-stratified folds; a permutation of the frames changes the
regularisation and therefore the weights.

## Golden references

`tests/golden/` holds the references the release is gated on: the slice
basis, the weights under both ridge rules, the half-set regularised Gram
matrices and the held-out cap on a two-basin fixture and on the paper's
Wolfe-Quapp benchmark. The script that produced them,
`freeze_1_0_references.py`, sits beside them as their provenance, and they
were produced with JAX 0.5.3. On that machine with that JAX, and only
there, the tests hold the half-set weights to the bit and the scalar-ridge
weights to `1e-13`: the strict tier, selected with
`SLICED_COMMITTOR_GOLDEN_TIER=strict`, the release gate. No probe can
certify another machine in advance (a GitHub runner reproduced the
32-direction fixture bit for bit and the 64-direction solve at `1e-3`,
from the BLAS kernels), so by default, CI included, the drift tier runs:
it compares the committor values, the half-set weights and the scalar
summaries at the measured drift of the table above, with a margin, so a
real change of the numerics still fails, and leaves the ill-conditioned
intermediates alone. CI runs it on the newest JAX and on the oldest
supported one.

## The paper

The paper's figures and rate table are reproduced by the Zenodo record's
self-contained package, which vendors this library, pins its environment
(JAX 0.6.2) and checks every number against its reference outputs (`2e-3` relative on the scalars, `5e-3`
absolute on the arrays, and the headline RMSE values to the last digit the
paper prints). Pin the
library version in a downstream regression suite:

```toml
dependencies = ["sliced-committor==1.0.0", "jax==0.5.3"]
```

## Why no `beta` argument?

The committor is beta-invariant given fixed samples ([theory.md](theory.md)),
so an inverse-temperature argument would be cosmetic. The library fixes
`beta = 1` and `result.free_energies` stores `-log rho` per slice; multiply
by `1/beta_physical` for physical units.
