# Reproducibility

## The determinism contract

| held fixed | same result |
|---|---|
| `seed`, JAX minor version, accelerator | bit-exact |
| `seed`, different JAX minor versions, CPU | committors and weights to about `1e-12` relative; XLA fusion moves the 1D solves by about `1e-13` |
| `seed`, GPU versus CPU | about `1e-9` to `1e-7` relative in float64, from non-associative reductions |

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
Wolfe-Quapp benchmark, frozen from version 0.6.0 by
`freeze_1_0_references.py` (kept for provenance; it cannot run against 1.0).
The half-set weights reproduce them to the bit; the scalar-ridge weights to
`1e-13`.

## The paper

The paper's figures and rate table are reproduced by the Zenodo record's
self-contained package, which vendors this library and checks every number
against its reference outputs (209 checks at `2e-3` relative). Pin the
library version in a downstream regression suite:

```toml
dependencies = ["sliced-committor==1.0.0", "jax==0.5.3"]
```

## Why no `beta` argument?

The committor is beta-invariant given fixed samples ([theory.md](theory.md)),
so an inverse-temperature argument would be cosmetic. The library fixes
`beta = 1` and `result.free_energies` stores `-log rho` per slice; multiply
by `1/beta_physical` for physical units.
